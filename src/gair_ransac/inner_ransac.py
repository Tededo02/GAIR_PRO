from dataclasses import dataclass
from typing import Iterator, Optional
import time
import numpy as np
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.deformable_superquadric import DeformableSuperQuadricParams
from src.superquadrics.model_family import parameter_count, validate_model_family
from scipy.optimize import least_squares
from .consensus import compute_consensus
from .metal_superquadric import fit_superquadric_metal, fit_superquadric_metal_batch, metal_available
from .metal_consensus import MetalConsensusContext, create_metal_consensus_context
from .metal_superflex import SuperflexMetalError
from src.superquadrics.superquadric_residual import superquadric_radial_residual_and_jacobian
from .axis_regularization import (
    DEFAULT_AXIS_PENALTY_WEIGHT, axis_penalty_residual_and_jacobian,
    axis_support_box, validate_axis_penalty_weight,
)
from .interior_consensus import (
    InteriorPenaltyContext, evaluate_model_consensus, prefer_consensus_model,
)


ROBUST_LOSS_SCALE_FACTOR = 0.02
AXIS_UPPER_FACTOR = 1.6
AXIS_UPPER_FLOOR_FACTOR = 0.18
INNER_RANSAC_BATCH_SIZE = 80


@dataclass
class InnerRansacResult:
    best_model: SuperQuadricParams
    best_inlier_count: int
    best_inliers_mask: np.ndarray
    best_score: float | None = None
    interior_mass: float = 0.0

def pca_initialization(points: np.ndarray) -> SuperQuadricParams:
    parameters = _pca_initial_parameters(np.asarray(points, dtype=np.float64)[None, :, :])[0]
    return _model_from_parameters(parameters)


def _model_from_parameters(parameters: np.ndarray) -> SuperQuadricParams:
    return SuperQuadricParams(*parameters[:5], rot=parameters[5:8], t=parameters[8:11])


def _pca_initial_parameters(point_batches: np.ndarray) -> np.ndarray:
    # NumPy batches the small eigendecompositions and percentile calculations for all hypotheses.
    centers = point_batches.mean(axis=1)
    centered_points = point_batches - centers[:, None, :]
    covariance = centered_points.swapaxes(1, 2) @ centered_points / max(point_batches.shape[1] - 1, 1)
    _, principal_axes = np.linalg.eigh(covariance)
    rotation_matrix = principal_axes[:, :, ::-1].copy()
    rotation_matrix[np.linalg.det(rotation_matrix) < 0, :, 2] *= -1.0

    pca_coordinates = centered_points @ rotation_matrix
    lower_quantile, upper_quantile = np.percentile(pca_coordinates, [5.0, 95.0], axis=1)
    semi_axes = np.maximum(0.525 * (upper_quantile - lower_quantile), 1e-3)

    angles = _euler_from_rotation_matrices(rotation_matrix)
    return np.column_stack((semi_axes, np.full((len(point_batches), 2), 1.5), angles, centers))


def _euler_from_rotation_matrices(rotation_matrix: np.ndarray) -> np.ndarray:
    sin_pitch = np.clip(-rotation_matrix[:, 2, 0], -1.0, 1.0)
    pitch = np.arcsin(sin_pitch)
    cos_pitch = np.cos(pitch)
    gimbal_lock = np.abs(cos_pitch) < 1e-8
    yaw = np.where(gimbal_lock, 0.0, np.arctan2(rotation_matrix[:, 1, 0], rotation_matrix[:, 0, 0]))
    roll = np.where(
        gimbal_lock,
        np.arctan2(-rotation_matrix[:, 0, 1], rotation_matrix[:, 1, 1]),
        np.arctan2(rotation_matrix[:, 2, 1], rotation_matrix[:, 2, 2]),
    )
    return np.column_stack((yaw, pitch, roll))


def _reference_axis_upper_bounds(
    reference_axes: np.ndarray,
    reference_diagonal: float,
    min_axis_length: float,
    global_max_axis_length: float,
) -> np.ndarray:
    axis_floor = max(min_axis_length, AXIS_UPPER_FLOOR_FACTOR * reference_diagonal)
    upper_bounds = np.maximum(AXIS_UPPER_FACTOR * reference_axes, axis_floor)
    return np.clip(upper_bounds, min_axis_length, global_max_axis_length)


def _optimization_bounds(
    reference_points: np.ndarray,
    reference_model: SuperQuadricParams | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if reference_points.ndim != 2 or reference_points.shape[1] != 3 or not np.isfinite(reference_points).all():
        raise ValueError("bounds_reference_points must be a finite array with shape (N, 3)")
    if reference_points.shape[0] == 0:
        raise ValueError("bounds_reference_points must contain at least one point")
    reference_model = pca_initialization(reference_points) if reference_model is None else reference_model
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
    robust_loss_scale = max(1e-3, ROBUST_LOSS_SCALE_FACTOR * reference_diagonal)
    return lower_bounds, upper_bounds, robust_loss_scale, reference_diagonal


def fit_superquadric_ls(
    points: np.ndarray,
    error_metric: str = "radial",
    bounds_reference_points: np.ndarray | None = None,
    backend: str = "auto",
    axis_penalty_weight: float = DEFAULT_AXIS_PENALTY_WEIGHT,
    model_family: str = "rigid",
    initial_model: SuperQuadricParams | None = None,
) -> SuperQuadricParams:
    """Fit on Metal when available, or select the explicit 'metal'/'cpu' backend.

    Metal and CPU share the radial soft_l1 objective and the selected family's axis penalty.
    PCA initialization and optimization bounds are prepared on the CPU.
    """
    validate_model_family(model_family)
    axis_penalty_weight = validate_axis_penalty_weight(axis_penalty_weight)
    if backend not in {"auto", "metal", "cpu"}:
        raise ValueError("backend must be 'auto', 'metal' or 'cpu'")
    if model_family == "superflex":
        from .deformable_fitting import fit_deformable_superquadric_ls
        return fit_deformable_superquadric_ls(points, bounds_reference_points, axis_penalty_weight, initial_model, backend=backend)
    del error_metric
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3 or not np.isfinite(point_array).all():
        raise ValueError("points must be a finite array with shape (N, 3)")
    if point_array.shape[0] < 11:
        raise ValueError("too few points for a stable superquadric fit")

    # Share initialization and bound preparation with the batched fitting path.
    initial_parameters = _pca_initial_parameters(point_array[None, :, :])[0]
    if initial_model is not None:
        initial_parameters = np.r_[initial_model.a1, initial_model.a2, initial_model.a3, initial_model.e1, initial_model.e2, initial_model.rot, initial_model.t]
    reference_points = point_array if bounds_reference_points is None else np.asarray(bounds_reference_points, dtype=np.float64)
    reference_model = _model_from_parameters(initial_parameters) if bounds_reference_points is None else None
    lower_bounds, upper_bounds, robust_loss_scale, reference_diagonal = _optimization_bounds(reference_points, reference_model)
    if reference_model is None:
        reference_model = pca_initialization(reference_points)
    support_box = axis_support_box(reference_points, reference_model, lower_bounds[0], robust_loss_scale)
    np.clip(initial_parameters, lower_bounds, upper_bounds, out=initial_parameters)

    if backend == "metal" or (backend == "auto" and metal_available()):
        parameters = fit_superquadric_metal(
            point_array,
            initial_parameters,
            lower_bounds,
            upper_bounds,
            robust_loss_scale,
            reference_diagonal,
            axis_penalty_weight=axis_penalty_weight,
            axis_support=support_box,
        )
        return _model_from_parameters(parameters)

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
            if axis_penalty_weight > 0.0:
                # Scale by sqrt(N) so regularization has the same strength for MSS and full refits.
                penalty, penalty_jacobian = axis_penalty_residual_and_jacobian(
                    parameters, support_box,
                    np.sqrt(len(point_array) * axis_penalty_weight) * robust_loss_scale,
                )
                residuals = np.concatenate((residuals, penalty))
                jacobian = np.vstack((jacobian, penalty_jacobian))
            radial_cache["parameters"] = np.array(parameters, dtype=np.float64, copy=True)
            radial_cache["residuals"] = residuals
            radial_cache["jacobian"] = jacobian
        return radial_cache["residuals"]

    def radial_jacobian(parameters: np.ndarray) -> np.ndarray:
        radial_residuals(parameters)
        return radial_cache["jacobian"]

    def regularized_loss(z: np.ndarray) -> np.ndarray:
        # Keep soft_l1 on data residuals and a quadratic loss on the three axis residuals.
        root = np.sqrt(1.0 + z)
        rho = np.vstack((2.0 * z / (root + 1.0), 1.0 / root, -0.5 / root**3))
        rho[0, -3:] = z[-3:]
        rho[1, -3:] = 1.0
        rho[2, -3:] = 0.0
        return rho

    optimization_result = least_squares(
        fun=radial_residuals,
        jac=radial_jacobian,
        x0=initial_parameters,
        method="trf",
        bounds=(lower_bounds, upper_bounds),
        loss=regularized_loss if axis_penalty_weight > 0.0 else "soft_l1",
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
    n_iters: int = 80,
    random_seed: int | None = None,
    deadline: float | None = None,
    consensus_context: MetalConsensusContext | None = None,
    axis_penalty_weight: float = DEFAULT_AXIS_PENALTY_WEIGHT,
    interior_context: InteriorPenaltyContext | None = None,
    min_inliers: int = 0,
    model_family: str = "rigid",
) -> InnerRansacResult:
    """Fit and score independent 40-point hypotheses in batches of up to 80.

    deadline is an absolute time.perf_counter() value, checked before preparing
    and launching each batch. In-flight GPU fitting and consensus dispatches
    finish before returning the best result found so far.
    """
    validate_model_family(model_family)
    axis_penalty_weight = validate_axis_penalty_weight(axis_penalty_weight)
    if interior_context is None and consensus_context is not None:
        interior_context = consensus_context.interior_context
    if model_family == "superflex":
        from .metal_superflex import require_superflex_metal
        require_superflex_metal()
    score_enabled = interior_context is not None and interior_context.weight > 0.0
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
    sample_size: int = 40
    rng = np.random.default_rng(random_seed)
    best_model: Optional[SuperQuadricParams] = None
    best_inliers: np.ndarray = np.zeros(actual_points.shape[0], dtype=bool)
    best_count: int = -1
    best_score: float = -1.0
    best_mass: float = 0.0
    if consensus_metric is None:
        consensus_metric = error_metric
    if refined_set_index.size == 0 or actual_points.shape[0] == 0:
        return InnerRansacResult(
            best_model=(SuperQuadricParams if model_family == "rigid" else DeformableSuperQuadricParams)(1, 1, 1, 1, 1),
            best_inlier_count=0,
            best_inliers_mask=np.zeros(actual_points.shape[0], dtype=bool),
        )
    size_sample = min(np.size(refined_set_index), sample_size)
    if consensus_context is not None:
        if consensus_metric != "radial":
            raise ValueError("Metal consensus supports the radial residual metric")
        consensus_context.check_inputs(actual_points, actual_normals)
        if score_enabled and consensus_context.interior_context is not interior_context:
            raise ValueError("Metal consensus must share the requested original-cloud interior context")
    for candidate_models in _inner_candidate_model_batches(
        point_cloud, refined_set_index, bounds_reference_points,
        size_sample, n_iters, rng, error_metric, deadline,
        axis_penalty_weight,
        model_family,
    ):
        if not candidate_models:
            continue
        if consensus_context is None and metal_available():
            active_indices = None
            if score_enabled:
                active_indices = interior_context.active_indices
                if actual_set_index is not None:
                    active_indices = active_indices[actual_set_index]
            consensus_context = create_metal_consensus_context(
                actual_points, actual_normals, consensus_metric,
                interior_context=interior_context if score_enabled else None, active_indices=active_indices,
                model_family=model_family,
            )
        # Keep the first completed candidate when a fitting dispatch overruns the deadline.
        if deadline is not None and time.perf_counter() >= deadline:
            candidate_models = candidate_models[:1]
        if consensus_context is not None:
            consensus = consensus_context.evaluate(candidate_models, threshold)
            eligible = np.flatnonzero(consensus.counts >= min_inliers)
            if not len(eligible):
                continue
            winner = int(eligible[0])
            for index in eligible[1:]:
                if prefer_consensus_model(
                    candidate_models[index], int(consensus.counts[index]), float(consensus.scores[index]),
                    candidate_models[winner], int(consensus.counts[winner]), float(consensus.scores[winner]),
                    axis_penalty_weight, score_enabled,
                ):
                    winner = int(index)
            candidate_count = int(consensus.counts[winner])
            candidate_score = float(consensus.scores[winner])
            if prefer_consensus_model(
                candidate_models[winner], candidate_count, candidate_score,
                best_model, best_count, best_score, axis_penalty_weight, score_enabled,
            ):
                best_model = candidate_models[winner]
                best_count = candidate_count
                best_score = candidate_score
                best_mass = 0.0 if consensus.interior_masses is None else float(consensus.interior_masses[winner])
                best_inliers = consensus.mask(winner)
        else:
            for candidate_model in candidate_models:
                candidate_inlier_mask, candidate_count, candidate_score, candidate_mass = evaluate_model_consensus(
                    candidate_model, actual_points, threshold, consensus_metric, actual_normals,
                    None, interior_context, compute_consensus,
                )
                if candidate_count >= min_inliers and prefer_consensus_model(
                    candidate_model, candidate_count, candidate_score,
                    best_model, best_count, best_score, axis_penalty_weight, score_enabled,
                ):
                    best_model = candidate_model
                    best_count = candidate_count
                    best_score = candidate_score
                    best_mass = candidate_mass
                    best_inliers = candidate_inlier_mask.astype(bool, copy=False)
                if deadline is not None and time.perf_counter() >= deadline:
                    break
        if deadline is not None and time.perf_counter() >= deadline:
            break

    if best_count < 0 or best_model is None:
        return InnerRansacResult(
            best_model=(SuperQuadricParams if model_family == "rigid" else DeformableSuperQuadricParams)(1, 1, 1, 1, 1),
            best_inlier_count=0,
            best_inliers_mask=np.zeros(actual_points.shape[0], dtype=bool),
        )
    inlier_points = actual_points[best_inliers]
    if inlier_points.shape[0] >= parameter_count(model_family) and (deadline is None or time.perf_counter() < deadline):
        try:
            refined_model = fit_superquadric_ls(
                inlier_points,
                error_metric=error_metric,
                bounds_reference_points=inlier_points,
                axis_penalty_weight=axis_penalty_weight,
                model_family=model_family,
                initial_model=best_model if model_family == "superflex" else None,
            )
            refined_inlier_mask, refined_inlier_count, refined_score, refined_mass = evaluate_model_consensus(
                refined_model, actual_points, threshold, consensus_metric, actual_normals,
                consensus_context, interior_context, compute_consensus,
            )
            if refined_inlier_count >= min_inliers and (prefer_consensus_model(
                refined_model, refined_inlier_count, refined_score,
                best_model, best_count, best_score, axis_penalty_weight, score_enabled,
            ) or (
                not score_enabled and axis_penalty_weight == 0.0 and refined_inlier_count == best_count
            )):
                best_model = refined_model
                best_count = refined_inlier_count
                best_score = refined_score
                best_mass = refined_mass
                best_inliers = refined_inlier_mask.astype(bool, copy=False)
        except SuperflexMetalError:
            raise
        except Exception:
            pass
    return InnerRansacResult(
        best_model, best_count, best_inliers, best_score, best_mass,
    )


def _inner_candidate_models(
    point_cloud: np.ndarray,
    refined_set_index: np.ndarray,
    bounds_reference_points: np.ndarray,
    size_sample: int,
    n_iters: int,
    rng: np.random.Generator,
    error_metric: str,
    deadline: float | None,
    axis_penalty_weight: float = DEFAULT_AXIS_PENALTY_WEIGHT,
    model_family: str = "rigid",
) -> Iterator[SuperQuadricParams]:
    # Preserve the individual-candidate iterator for callers and validation tools.
    for models in _inner_candidate_model_batches(
        point_cloud, refined_set_index, bounds_reference_points, size_sample,
        n_iters, rng, error_metric, deadline, axis_penalty_weight, model_family,
    ):
        for model in models:
            yield model
            if deadline is not None and time.perf_counter() >= deadline:
                return


def _inner_candidate_model_batches(
    point_cloud: np.ndarray,
    refined_set_index: np.ndarray,
    bounds_reference_points: np.ndarray,
    size_sample: int,
    n_iters: int,
    rng: np.random.Generator,
    error_metric: str,
    deadline: float | None,
    axis_penalty_weight: float = DEFAULT_AXIS_PENALTY_WEIGHT,
    model_family: str = "rigid",
) -> Iterator[list[SuperQuadricParams]]:
    if n_iters <= 0 or size_sample < parameter_count(model_family):
        return
    if deadline is not None and time.perf_counter() >= deadline:
        return
    if model_family == "superflex":
        from .metal_superflex import require_superflex_metal, superflex_optimization_bounds, fit_superflex_hypothesis_batch
        from .deformable_fitting import _unpack
        require_superflex_metal()
        lower, upper, loss_scale, diagonal, support_box = superflex_optimization_bounds(bounds_reference_points, axis_penalty_weight)
        for start in range(0, n_iters, INNER_RANSAC_BATCH_SIZE):
            if deadline is not None and time.perf_counter() >= deadline:
                return
            sample_indices = [rng.choice(refined_set_index, size=size_sample, replace=False) for _ in range(min(INNER_RANSAC_BATCH_SIZE, n_iters - start))]
            sampled_points = point_cloud[np.stack(sample_indices)]
            initial_parameters = _pca_initial_parameters(sampled_points)
            if deadline is not None and time.perf_counter() >= deadline:
                return
            result = fit_superflex_hypothesis_batch(
                sampled_points, initial_parameters, lower, upper, loss_scale, diagonal,
                support_box, axis_penalty_weight,
            )
            yield [_unpack(parameters, diagonal) for parameters in result.parameters[result.success]]
        return
    if not metal_available():
        # Preserve the sequential SciPy path on machines without Metal.
        for _ in range(n_iters):
            if deadline is not None and time.perf_counter() >= deadline:
                return
            sample_idx = rng.choice(refined_set_index, size=size_sample, replace=False)
            try:
                model = fit_superquadric_ls(
                    point_cloud[sample_idx], error_metric=error_metric,
                    bounds_reference_points=bounds_reference_points, backend="cpu",
                    axis_penalty_weight=axis_penalty_weight,
                    model_family=model_family,
                )
            except Exception:
                continue
            yield [model]
        return

    try:
        # All samples use the same support bounds; compute them once for the whole search.
        reference_model = pca_initialization(bounds_reference_points)
        lower, upper, loss_scale, diagonal = _optimization_bounds(bounds_reference_points, reference_model)
        support_box = axis_support_box(bounds_reference_points, reference_model, lower[0], loss_scale)
    except Exception:
        return
    for start in range(0, n_iters, INNER_RANSAC_BATCH_SIZE):
        if deadline is not None and time.perf_counter() >= deadline:
            return
        sample_indices = []
        for _ in range(min(INNER_RANSAC_BATCH_SIZE, n_iters - start)):
            if deadline is not None and time.perf_counter() >= deadline:
                return
            sample_indices.append(rng.choice(refined_set_index, size=size_sample, replace=False))
        sampled_points = point_cloud[np.stack(sample_indices)]
        try:
            initial_parameters = np.clip(_pca_initial_parameters(sampled_points), lower, upper)
            if deadline is not None and time.perf_counter() >= deadline:
                return
            result = fit_superquadric_metal_batch(
                sampled_points, initial_parameters, lower, upper, loss_scale, diagonal,
                axis_penalty_weight=axis_penalty_weight, axis_support=support_box,
            )
        except Exception:
            continue
        # Preserve candidate order and tie-breaking, and discard only fits that failed to converge.
        yield [_model_from_parameters(parameters) for parameters in result.parameters[result.success]]
