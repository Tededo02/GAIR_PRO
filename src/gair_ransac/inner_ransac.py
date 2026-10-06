from dataclasses import dataclass
from typing import Optional
import time
import numpy as np
from src.superquadrics.superquadric_param import SuperQuadricParams
from scipy.optimize import least_squares
from .consensus import compute_consensus
from .metal_superquadric import fit_superquadric_metal, metal_available
from src.superquadrics.superquadric_residual import superquadric_radial_residual_and_jacobian


ROBUST_LOSS_SCALE_FACTOR = 0.02
AXIS_UPPER_FACTOR = 1.6
AXIS_UPPER_FLOOR_FACTOR = 0.18


@dataclass
class InnerRansacResult:
    best_model: SuperQuadricParams
    best_inlier_count: int
    best_inliers_mask: np.ndarray

def pca_initialization(points: np.ndarray) -> SuperQuadricParams:
    point_array = np.asarray(points, dtype=np.float64)

    center = point_array.mean(axis=0)
    centered_points = point_array - center
    covariance = centered_points.T @ centered_points / max(point_array.shape[0] - 1, 1)
    _, principal_axes = np.linalg.eigh(covariance)
    rotation_matrix = principal_axes[:, ::-1]
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[:, 2] *= -1.0

    pca_coordinates = centered_points @ rotation_matrix
    lower_quantile = np.percentile(pca_coordinates, 5.0, axis=0)
    upper_quantile = np.percentile(pca_coordinates, 95.0, axis=0)
    semi_axes = np.maximum(0.525 * (upper_quantile - lower_quantile), 1e-3)

    sin_pitch = np.clip(-rotation_matrix[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) < 1e-8:
        yaw = 0.0
        roll = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])
    else:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return SuperQuadricParams(
        a1=float(semi_axes[0]),
        a2=float(semi_axes[1]),
        a3=float(semi_axes[2]),
        e1=1.5,
        e2=1.5,
        rot=np.array([yaw, pitch, roll], dtype=np.float64),
        t=center.astype(np.float64, copy=False),
    )


def _reference_axis_upper_bounds(
    reference_axes: np.ndarray,
    reference_diagonal: float,
    min_axis_length: float,
    global_max_axis_length: float,
) -> np.ndarray:
    axis_floor = max(min_axis_length, AXIS_UPPER_FLOOR_FACTOR * reference_diagonal)
    upper_bounds = np.maximum(AXIS_UPPER_FACTOR * reference_axes, axis_floor)
    return np.clip(upper_bounds, min_axis_length, global_max_axis_length)


def fit_superquadric_ls(
    points: np.ndarray,
    error_metric: str = "radial",
    bounds_reference_points: np.ndarray | None = None,
    backend: str = "auto",
) -> SuperQuadricParams:
    """Fit on Metal when available, or select the explicit 'metal'/'cpu' backend.

    Metal runs bounded, damped Gauss-Newton with the same radial soft_l1 objective.
    PCA initialization and optimization bounds are prepared on the CPU.
    """
    del error_metric
    if backend not in {"auto", "metal", "cpu"}:
        raise ValueError("backend must be 'auto', 'metal' or 'cpu'")
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3 or not np.isfinite(point_array).all():
        raise ValueError("points must be a finite array with shape (N, 3)")
    if point_array.shape[0] < 11:
        raise ValueError("too few points for a stable superqadric fit")

    # Build a robust initial guess from PCA.
    initial_model = pca_initialization(point_array)

    # Use the requested reference points to define stable optimization bounds.
    reference_points = point_array if bounds_reference_points is None else np.asarray(bounds_reference_points, dtype=np.float64)
    if reference_points.ndim != 2 or reference_points.shape[1] != 3 or not np.isfinite(reference_points).all():
        raise ValueError("bounds_reference_points must be a finite array with shape (N, 3)")
    if reference_points.shape[0] == 0:
        raise ValueError("bounds_reference_points must contain at least one point")
    reference_model = initial_model if bounds_reference_points is None else pca_initialization(reference_points)
    reference_axes = np.array(
        [reference_model.a1, reference_model.a2, reference_model.a3],
        dtype=np.float64,
    )

    reference_min = reference_points.min(axis=0)
    reference_max = reference_points.max(axis=0)
    reference_diagonal = float(np.linalg.norm(reference_max - reference_min) + 1e-12)
    min_axis_length = max(1e-3, 5e-2 * reference_diagonal)
    global_max_axis_length = max(1e-2, 1.2 * reference_diagonal)
    max_axis_lengths = _reference_axis_upper_bounds(
        reference_axes,
        reference_diagonal,
        min_axis_length,
        global_max_axis_length,
    )
    exponent_min, exponent_max = 0.06, 4.5
    angle_min, angle_max = -np.pi, np.pi
    translation_margin = 0.25 * reference_diagonal

    lower_bounds = np.array(
        [
            min_axis_length,
            min_axis_length,
            min_axis_length,
            exponent_min,
            exponent_min,
            angle_min,
            angle_min,
            angle_min,
            reference_min[0] - translation_margin,
            reference_min[1] - translation_margin,
            reference_min[2] - translation_margin,
        ],
        dtype=np.float64,
    )
    upper_bounds = np.array(
        [
            max_axis_lengths[0],
            max_axis_lengths[1],
            max_axis_lengths[2],
            exponent_max,
            exponent_max,
            angle_max,
            angle_max,
            angle_max,
            reference_max[0] + translation_margin,
            reference_max[1] + translation_margin,
            reference_max[2] + translation_margin,
        ],
        dtype=np.float64,
    )

    initial_parameters = np.array(
        [
            initial_model.a1,
            initial_model.a2,
            initial_model.a3,
            initial_model.e1,
            initial_model.e2,
            initial_model.rot[0],
            initial_model.rot[1],
            initial_model.rot[2],
            initial_model.t[0],
            initial_model.t[1],
            initial_model.t[2],
        ],
        dtype=np.float64,
    )
    np.clip(initial_parameters, lower_bounds, upper_bounds, out=initial_parameters)
    robust_loss_scale = max(1e-3, ROBUST_LOSS_SCALE_FACTOR * reference_diagonal)

    if backend == "metal" or (backend == "auto" and metal_available()):
        parameters = fit_superquadric_metal(
            point_array,
            initial_parameters,
            lower_bounds,
            upper_bounds,
            robust_loss_scale,
            reference_diagonal,
        )
        return SuperQuadricParams(
            *parameters[:5],
            rot=parameters[5:8],
            t=parameters[8:11],
        )

    radial_cache: dict[str, np.ndarray | None] = {"parameters": None, "residuals": None, "jacobian": None}

    def radial_residuals(parameters: np.ndarray) -> np.ndarray:
        cached_parameters = radial_cache["parameters"]
        if cached_parameters is None or not np.array_equal(cached_parameters, parameters):
            current_model = SuperQuadricParams(
                a1=parameters[0],
                a2=parameters[1],
                a3=parameters[2],
                e1=parameters[3],
                e2=parameters[4],
                rot=np.array(parameters[5:8], dtype=np.float64),
                t=np.array(parameters[8:11], dtype=np.float64),
            )
            residuals, jacobian = superquadric_radial_residual_and_jacobian(
                current_model,
                point_array,
            )
            radial_cache["parameters"] = np.array(parameters, dtype=np.float64, copy=True)
            radial_cache["residuals"] = residuals
            radial_cache["jacobian"] = jacobian
        return radial_cache["residuals"]

    def radial_jacobian(parameters: np.ndarray) -> np.ndarray:
        radial_residuals(parameters)
        return radial_cache["jacobian"]

    optimization_result = least_squares(
        fun=radial_residuals,
        jac=radial_jacobian,
        x0=initial_parameters,
        method="trf",
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        f_scale=robust_loss_scale,
        max_nfev=1000,
    )

    if not optimization_result.success:
        raise RuntimeError(f"least_squares failed: {optimization_result.message}")

    a1, a2, a3, e1, e2, yaw, pitch, roll, px, py, pz = optimization_result.x
    return SuperQuadricParams(
        a1=a1,
        a2=a2,
        a3=a3,
        e1=e1,
        e2=e2,
        rot=np.array([yaw, pitch, roll], dtype=np.float64),  # (yaw=z, pitch=y, roll=x)
        t=np.array([px, py, pz], dtype=np.float64),
    )


def inner_ransac(
    point_cloud: np.ndarray,
    refined_set_index: np.ndarray,
    actual_set_index: np.ndarray | None,
    threshold: float,
    normals: np.ndarray | None = None,
    error_metric: str = "radial",
    consensus_metric: str | None = None,
    n_iters: int = 50,
    random_seed: int | None = None,
    deadline: float | None = None,
) -> InnerRansacResult:
    """deadline: an absolute time.perf_counter() value. When set, the iteration loop below
    stops immediately once it passes, returning whatever best result was found so far.
    None (default) preserves normal, iteration-count-bounded behavior for run.py."""
    point_cloud = np.asarray(point_cloud, dtype=np.float64)
    refined_set_index = np.asarray(refined_set_index, dtype=np.int64)
    actual_set_index = None if actual_set_index is None else np.asarray(actual_set_index, dtype=np.int64)
    actual_points = point_cloud if actual_set_index is None else point_cloud[actual_set_index]
    if normals is None:
        actual_normals = None
    else:
        normals = np.asarray(normals, dtype=np.float64)
        actual_normals = normals if actual_set_index is None else normals[actual_set_index]
    bounds_reference_points = point_cloud[refined_set_index]
    sample_size: int = 30
    rng = np.random.default_rng(random_seed)
    best_model: Optional[SuperQuadricParams] = None
    best_inliers: np.ndarray = np.zeros(actual_points.shape[0], dtype=bool)
    best_count: int = -1
    if consensus_metric is None:
        consensus_metric = error_metric
    if refined_set_index.size == 0 or actual_points.shape[0] == 0:
        return InnerRansacResult(
            best_model=SuperQuadricParams(1, 1, 1, 1, 1, [0, 0, 0], [0, 0, 0]),
            best_inlier_count=0,
            best_inliers_mask=np.empty((0,), dtype=bool),
        )
    size_sample = min(np.size(refined_set_index), sample_size)
    for _ in range(n_iters):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        sample_idx = rng.choice(refined_set_index, size=size_sample, replace=False)
        sampled_points = point_cloud[sample_idx]
        try:
            candidate_model = fit_superquadric_ls(
                sampled_points,
                error_metric=error_metric,
                bounds_reference_points=bounds_reference_points,
            )
        except Exception:
            continue

        candidate_inlier_mask = compute_consensus(
            candidate_model,
            actual_points,
            threshold,
            error_metric=consensus_metric,
            normals=actual_normals,
        )
        candidate_inlier_count = int(np.count_nonzero(candidate_inlier_mask))
        if candidate_inlier_count > best_count:
            best_model = candidate_model
            best_count = candidate_inlier_count
            best_inliers = candidate_inlier_mask.astype(bool, copy=False)

    if best_count < 0 or best_model is None:
        return InnerRansacResult(
            best_model=SuperQuadricParams(1, 1, 1, 1, 1, [0, 0, 0], [0, 0, 0]),
            best_inlier_count=0,
            best_inliers_mask=np.empty((0,), dtype=bool),
        )
    inlier_points = actual_points[best_inliers]
    if inlier_points.shape[0] >= 11:
        try:
            refined_model = fit_superquadric_ls(
                inlier_points,
                error_metric=error_metric,
                bounds_reference_points=inlier_points,
            )
            refined_inlier_mask = compute_consensus(
                refined_model,
                actual_points,
                threshold,
                error_metric=consensus_metric,
                normals=actual_normals,
            )
            refined_inlier_count = int(np.count_nonzero(refined_inlier_mask))
            if refined_inlier_count >= best_count:
                best_model = refined_model
                best_count = refined_inlier_count
                best_inliers = refined_inlier_mask.astype(bool, copy=False)
        except Exception:
            pass
    return InnerRansacResult(best_model=best_model, best_inlier_count=best_count, best_inliers_mask=best_inliers)
