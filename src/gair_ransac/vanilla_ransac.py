from typing import Optional
import numpy as np

from src.superquadrics.superquadric_param import SuperQuadricParams
from .consensus import compute_consensus, expanded_removal_mask
from .inner_ransac import fit_superquadric_ls
from .mss import adaptive_local_fps_mss
from .gair_ransac import _sample_superquadric_surface
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def vanilla_ransac(
    point_cloud: np.ndarray,
    normals: np.ndarray,
    threshold: float,
    max_models: int = 1,
    max_iterations: int = 300,
    sample_size: int = 30,
    min_inliers: int = 30,
    consensus_metric: str = "radial",
    error_metric: str = "radial",
    random_seed: int | None = None,
    min_coverage: float = 0.0,
) -> tuple[list[SuperQuadricParams], list[BoolArray]]:
    """
    Pure sequential RANSAC — no local optimization, no graph-cut, no inner RANSAC.
    Each iteration: sample MSS → fit → consensus → keep best.
    """
    point_cloud: FloatArray = np.asarray(point_cloud, dtype=np.float64)
    normals: FloatArray = np.asarray(normals, dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    n_points: int = point_cloud.shape[0]
    remaining_indices: IntArray = np.arange(n_points, dtype=np.int64)
    models_set: list[SuperQuadricParams] = []
    inliers_set: list[BoolArray] = []

    for _ in range(max_models):
        if remaining_indices.size < max(sample_size, min_inliers):
            break

        current_point_cloud: FloatArray = point_cloud[remaining_indices]
        V: FloatArray = normals[remaining_indices]
        best_model: Optional[SuperQuadricParams] = None
        best_inliers: BoolArray = np.zeros(current_point_cloud.shape[0], dtype=bool)

        for _ in range(max_iterations):
            M_j: FloatArray = np.asarray(
                adaptive_local_fps_mss(
                    current_point_cloud,
                    V,
                    sample_size=sample_size,
                    seed_tries=12,
                    candidate_multiplier=20.0,
                    initial_k=512,
                    rng=rng,
                ),
                dtype=np.float64,
            )

            try:
                H_j: SuperQuadricParams = fit_superquadric_ls(M_j, error_metric="radial")
            except Exception:
                continue

            candidate_inliers: BoolArray = np.asarray(
                compute_consensus(H_j, current_point_cloud, threshold, error_metric=consensus_metric),
                dtype=bool,
            )
            candidate_count: int = int(np.count_nonzero(candidate_inliers))
            best_count: int = int(np.count_nonzero(best_inliers))

            if candidate_count > best_count:
                best_model = H_j
                best_inliers = candidate_inliers.copy()

        if best_model is None:
            break

        best_count = int(np.count_nonzero(best_inliers))
        if best_count < min_inliers:
            break

        # Final refit on all inliers
        best_points = current_point_cloud[best_inliers]
        if best_points.shape[0] >= 11:
            try:
                refit_model = fit_superquadric_ls(
                    best_points,
                    error_metric=error_metric,
                    bounds_reference_points=best_points,
                )
                refit_inliers = np.asarray(
                    compute_consensus(refit_model, current_point_cloud, threshold, error_metric=consensus_metric),
                    dtype=bool,
                )
                refit_count = int(np.count_nonzero(refit_inliers))
                if refit_count >= best_count:
                    best_model = refit_model
                    best_inliers = refit_inliers
                    best_count = refit_count
            except Exception:
                pass

        # Coverage check
        if min_coverage > 0.0:
            from scipy.spatial import cKDTree as _cKDTree
            surface_samples = _sample_superquadric_surface(best_model, n=1000)
            inlier_pts = current_point_cloud[best_inliers]
            tree_inliers = _cKDTree(inlier_pts)
            dists, _ = tree_inliers.query(surface_samples, k=1)
            coverage = float((dists < threshold).mean())
            if coverage < min_coverage:
                remaining_indices = remaining_indices[~best_inliers]
                continue

        global_inliers: BoolArray = np.zeros(n_points, dtype=bool)
        global_inliers[remaining_indices[best_inliers]] = True

        models_set.append(best_model)
        inliers_set.append(global_inliers)

        remove_mask = expanded_removal_mask(
            best_model, current_point_cloud, threshold, factor=1.3, error_metric=consensus_metric
        )
        remaining_indices = remaining_indices[~remove_mask]

    return models_set, inliers_set
