from typing import Optional
import time
import numpy as np

from src.superquadrics.superquadric_param import SuperQuadricParams
from .consensus import compute_consensus, expanded_removal_mask
from .energy_strategies import FullGairEnergy, GairEnergyStrategy
from .inner_ransac import inner_ransac, fit_superquadric_ls, InnerRansacResult
from .gair import gair
from .initgraph import build_knn_graph
from .mss import (
    AdaptiveLocalFpsSamplerContext,
    adaptive_local_fps_mss,
    build_adaptive_local_fps_sampler_context,
)
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray  = NDArray[np.bool_]
IntArray   = NDArray[np.int64]


def compare_consensus(prev_mask: np.ndarray, new_mask: np.ndarray, min_gain: int = 1) -> bool:
    return int(new_mask.sum()) >= int(prev_mask.sum()) + min_gain


def _point_subset_mask(points: FloatArray, subset_points: np.ndarray, atol: float = 1e-12) -> BoolArray:
    mask = np.zeros(points.shape[0], dtype=bool)
    if subset_points.size == 0:
        return mask
    subset_points = np.asarray(subset_points)
    if subset_points.dtype == bool:
        if subset_points.shape[0] != points.shape[0]:
            raise ValueError(f"Boolean mask must have length {points.shape[0]}, got {subset_points.shape[0]}")
        return subset_points.astype(bool, copy=False)
    subset_points = np.asarray(subset_points, dtype=np.float64).reshape(-1, 3)
    for sample_point in subset_points:
        mask |= np.all(np.isclose(points, sample_point, atol=atol, rtol=0.0), axis=1)
    return mask


def _induced_subgraph_edges(edges: IntArray, remaining_indices: IntArray, n_points: int) -> IntArray:
    if edges.size == 0 or remaining_indices.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    global_to_local = np.full(n_points, -1, dtype=np.int64)
    global_to_local[remaining_indices] = np.arange(remaining_indices.size, dtype=np.int64)
    local_edges = global_to_local[edges]
    keep_edges = np.all(local_edges >= 0, axis=1)
    return local_edges[keep_edges]


def _center_opposes_inlier_normals(
    model: SuperQuadricParams,
    inlier_points: FloatArray,
    inlier_normals: FloatArray,
    min_fraction: float = 0.70,
) -> bool:
    if inlier_points.shape[0] == 0:
        return False
    center_direction = np.asarray(model.t, dtype=np.float64) - inlier_points
    normal_dot = np.einsum("ij,ij->i", center_direction, inlier_normals, optimize=True)
    opposite_fraction = float(np.mean(normal_dot < 0.0))
    return opposite_fraction >= min_fraction


def _sample_superquadric_surface(model: "SuperQuadricParams", n: int = 1000) -> FloatArray:
    """Sample n points on the superquadric surface parametrically."""
    eta   = np.linspace(-np.pi / 2, np.pi / 2, int(n ** 0.5) + 1)
    omega = np.linspace(-np.pi, np.pi, int(n ** 0.5) + 1)
    eta, omega = np.meshgrid(eta, omega)
    eta, omega = eta.ravel(), omega.ravel()
    def _sp(val, exp):
        return np.sign(val) * (np.abs(val) ** exp)
    x = model.a1 * _sp(np.cos(eta), model.e1) * _sp(np.cos(omega), model.e2)
    y = model.a2 * _sp(np.cos(eta), model.e1) * _sp(np.sin(omega), model.e2)
    z = model.a3 * _sp(np.sin(eta), model.e1)
    pts = np.stack([x, y, z], axis=1)
    R   = model.rotation_matrix()
    return (pts @ R.T) + model.t


def gair_ransac(
    point_cloud: np.ndarray,
    normals: np.ndarray | None = None,
    threshold: float = 0.1,
    max_models: int = 1,
    max_iterations: int = 300,
    m_neighbors: int = 6,
    radius: float = 0.06,
    radius_is_relative: bool = True,
    sample_size: int = 50,
    min_inliers: int = 30,
    min_gain: int = 1,
    error_metric: str = "radial",
    consensus_metric: str = "radial",
    inner_iterations: int = 50,
    random_seed: int | None = None,
    min_coverage: float = 0.0,
    use_normal_coherence: bool | None = None,
    energy_strategy: GairEnergyStrategy | None = None,
    deadline: float | None = None,
) -> tuple[list[SuperQuadricParams], list[BoolArray], FloatArray | None, int]:
    """deadline: an absolute time.perf_counter() value. When set, the per-model iteration
    loop, the graph-cut local-optimization loop, and the inner_ransac refinement step all
    stop immediately once it passes, returning whatever best result was found so far.
    None (default) preserves normal, iteration-count-bounded behavior for run.py."""
    total_best_mss_used: FloatArray | None = None
    total_local_opts: int = 0
    if energy_strategy is None:
        energy_strategy = FullGairEnergy()

    point_cloud: FloatArray = np.asarray(point_cloud, dtype=np.float64)
    if use_normal_coherence is None:
        use_normal_coherence = normals is not None
    # MSS always gets normals when available, independent of the GAIR coherence flag
    mss_normals: FloatArray | None = np.asarray(normals, dtype=np.float64) if normals is not None else None
    normals: FloatArray = np.asarray(normals, dtype=np.float64) if normals is not None else np.zeros((point_cloud.shape[0], 3), dtype=np.float64)

    rng = np.random.default_rng(random_seed)
    n_points: int = point_cloud.shape[0]
    remaining_indices: IntArray = np.arange(n_points, dtype=np.int64)
    _, full_edge = build_knn_graph(
        point_cloud,
        m_neighbors=m_neighbors,
    )
    full_edge: IntArray = np.asarray(full_edge, dtype=np.int64)
    models_set: list[SuperQuadricParams] = []
    inliers_set: list[BoolArray] = []

    for k in range(max_models):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if remaining_indices.size < max(sample_size, min_inliers):
            break

        current_point_cloud: FloatArray = point_cloud[remaining_indices]
        V: FloatArray     = normals[remaining_indices]
        V_mss: FloatArray | None = mss_normals[remaining_indices] if mss_normals is not None else None

        best_model: Optional[SuperQuadricParams] = None
        best_inliers: BoolArray = np.zeros(current_point_cloud.shape[0], dtype=bool)

        edge: IntArray = _induced_subgraph_edges(full_edge, remaining_indices, n_points)
        best_mss_used: FloatArray | None = None
        mss_sampler_context: AdaptiveLocalFpsSamplerContext | None = None

        for j in range(max_iterations):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            if mss_sampler_context is None:
                mss_sampler_context = build_adaptive_local_fps_sampler_context(
                    current_point_cloud,
                    V_mss,
                )
            M_j = np.asarray(
                adaptive_local_fps_mss(
                    current_point_cloud,
                    normals=V_mss,
                    sample_size=sample_size,
                    initial_k=300,
                    candidate_multiplier=12,
                    rng=rng,
                    sampler_context=mss_sampler_context,
                ),
                dtype=np.float64,
            )
            try:
                H_j: SuperQuadricParams = fit_superquadric_ls(M_j, error_metric=error_metric)
            except Exception:
                continue

            candidate_inliers: BoolArray = np.asarray(
                compute_consensus(
                    H_j,
                    current_point_cloud,
                    threshold,
                    error_metric=consensus_metric,
                    normals=V if use_normal_coherence else None,
                ),
                dtype=bool,
            )
            candidate_count: int = int(np.count_nonzero(candidate_inliers))
            best_count: int     = int(np.count_nonzero(best_inliers))
            if candidate_count < best_count + 1:
                continue

            current_model:   SuperQuadricParams = H_j
            current_inliers: BoolArray          = candidate_inliers.copy()
            current_count:   int                = candidate_count
            terminate:       bool               = False
            local_iteration: int                = 0

            while not terminate:
                if deadline is not None and time.perf_counter() >= deadline:
                    terminate = True
                    continue
                total_local_opts += 1
                refined_inliers: BoolArray = np.asarray(
                    gair(
                        points=current_point_cloud,
                        edges=edge,
                        normals=V,
                        model=current_model,
                        eps=threshold,
                        error_metric=consensus_metric,
                        energy_strategy=energy_strategy,
                    ),
                    dtype=bool,
                )
                local_iteration += 1
                refined_count: int = int(np.count_nonzero(refined_inliers))
                if refined_count < min_inliers:
                    terminate = True
                    continue

                refined_set_index: IntArray = np.flatnonzero(refined_inliers).astype(np.int64)
                inner_result: InnerRansacResult = inner_ransac(
                    current_point_cloud,
                    refined_set_index,
                    None,
                    threshold,
                    normals=V if use_normal_coherence else None,
                    error_metric=error_metric,
                    consensus_metric=consensus_metric,
                    n_iters=inner_iterations,
                    random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                    deadline=deadline,
                )
                if inner_result.best_inlier_count <= 0:
                    terminate = True
                    continue

                # Update using inner-RANSAC consensus (c_hat): more stable than using
                # the GAIR-refined set directly, which is the strict paper variant.
                new_inliers: BoolArray = np.asarray(inner_result.best_inliers_mask, dtype=bool)
                if compare_consensus(current_inliers, new_inliers, min_gain=min_gain):
                    current_inliers = new_inliers
                    current_count   = int(np.count_nonzero(new_inliers))
                    current_model   = inner_result.best_model
                else:
                    terminate = True

            if current_count > int(np.count_nonzero(best_inliers)):
                best_model    = current_model
                best_inliers  = current_inliers
                best_mss_used = M_j.copy()

        if best_model is None:
            break
        best_count: int = int(np.count_nonzero(best_inliers))
        if best_count < min_inliers:
            break

        # Final refit on the full inlier set before extracting the model
        best_points = current_point_cloud[best_inliers]
        if best_points.shape[0] >= 11:
            try:
                refit_model = fit_superquadric_ls(
                    best_points,
                    error_metric=error_metric,
                    bounds_reference_points=best_points,
                )
                refit_inliers = np.asarray(
                    compute_consensus(
                        refit_model,
                        current_point_cloud,
                        threshold,
                        error_metric=consensus_metric,
                        normals=V if use_normal_coherence else None,
                    ),
                    dtype=bool,
                )
                refit_count = int(np.count_nonzero(refit_inliers))
                if refit_count >= best_count:
                    best_model   = refit_model
                    best_inliers = refit_inliers
                    best_count   = refit_count
            except Exception:
                pass

        if mss_normals is not None and not _center_opposes_inlier_normals(
            best_model,
            current_point_cloud[best_inliers],
            V[best_inliers],
        ):
            remaining_indices = remaining_indices[~best_inliers]
            continue

        # Reject models whose superquadric surface is sparsely supported by inliers
        if min_coverage > 0.0:
            from scipy.spatial import cKDTree as _cKDTree
            surface_samples = _sample_superquadric_surface(best_model, n=1000)
            inlier_pts      = current_point_cloud[best_inliers]
            tree_inliers    = _cKDTree(inlier_pts)
            dists, _        = tree_inliers.query(surface_samples, k=1)
            coverage        = float((dists < 4*threshold).mean())
            if coverage < min_coverage:
                continue

        global_inliers: BoolArray = np.zeros(n_points, dtype=bool)
        global_inliers[remaining_indices[best_inliers]] = True
        models_set.append(best_model)
        inliers_set.append(global_inliers)
        if best_mss_used is not None:
            total_best_mss_used = (
                best_mss_used
                if total_best_mss_used is None
                else np.vstack((total_best_mss_used, best_mss_used))
            )

        remove_mask = expanded_removal_mask(
            best_model,
            current_point_cloud,
            threshold,
            factor=3.0,
            error_metric=consensus_metric,
            normals=None
        )
        remaining_indices = remaining_indices[~remove_mask]

    return models_set, inliers_set, total_best_mss_used, total_local_opts
