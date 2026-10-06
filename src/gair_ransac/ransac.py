from typing import Optional
import time
import numpy as np

from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.model_family import parameter_count, validate_model_family
from .consensus import compute_consensus, expanded_removal_mask
from .inner_ransac import inner_ransac, fit_superquadric_ls, InnerRansacResult
from .metal_consensus import create_metal_consensus_context
from .metal_superflex import SuperflexMetalError
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def compare_consensus(prev_mask: np.ndarray, new_mask: np.ndarray, min_gain: int = 1) -> bool:
    return int(new_mask.sum()) >= int(prev_mask.sum()) + min_gain


def ransac(
    point_cloud: np.ndarray,
    threshold: float,
    max_models: int = 1,
    max_iterations: int = 300,
    sample_size: int = 30,
    min_inliers: int = 30,
    error_metric: str = "radial",
    consensus_metric: str = "radial",
    inner_iterations: int = 80,
    random_seed: int | None = None,
    local_optimization: bool = True,
    deadline: float | None = None,
    model_family: str = "rigid",
) -> tuple[list[SuperQuadricParams], list[BoolArray], int]:
    """deadline: an absolute time.perf_counter() value. When set, both the per-model
    iteration loop and the inner_ransac local-optimization step stop immediately once it
    passes, returning whatever best result was found so far. None (default) preserves
    normal, iteration-count-bounded behavior for run.py."""
    validate_model_family(model_family)
    if sample_size < parameter_count(model_family):
        raise ValueError(f"sample_size must be at least {parameter_count(model_family)} for {model_family}")
    point_cloud: FloatArray = np.asarray(point_cloud, dtype=np.float64)
    rng = np.random.default_rng(random_seed)
    n_points: int = point_cloud.shape[0]
    remaining_indices: IntArray = np.arange(n_points, dtype=np.int64)
    models_set: list[SuperQuadricParams] = []
    inliers_set: list[BoolArray] = []
    total_local_opts: int = 0

    for _ in range(max_models):
        if deadline is not None and time.perf_counter() >= deadline:
            break
        if remaining_indices.size < max(sample_size, min_inliers):
            break

        current_point_cloud: FloatArray = point_cloud[remaining_indices]
        consensus_context = create_metal_consensus_context(current_point_cloud, error_metric=consensus_metric, model_family=model_family)
        best_model: Optional[SuperQuadricParams] = None
        best_inliers: BoolArray = np.zeros(current_point_cloud.shape[0], dtype=bool)

        for _ in range(max_iterations):
            if deadline is not None and time.perf_counter() >= deadline:
                break
            idx = rng.choice(current_point_cloud.shape[0], size=min(sample_size, current_point_cloud.shape[0]), replace=False)
            sample_pts: FloatArray = current_point_cloud[idx]

            try:
                H_j: SuperQuadricParams = fit_superquadric_ls(sample_pts, error_metric=error_metric, model_family=model_family)
            except SuperflexMetalError:
                raise
            except Exception:
                continue

            candidate_inliers: BoolArray = np.asarray(
                compute_consensus(H_j, current_point_cloud, threshold, error_metric=consensus_metric, metal_context=consensus_context),
                dtype=bool,
            )
            candidate_count: int = int(np.count_nonzero(candidate_inliers))
            best_count: int = int(np.count_nonzero(best_inliers))

            if candidate_count < best_count + 1:
                continue

            current_model: SuperQuadricParams = H_j
            current_inliers: BoolArray = candidate_inliers.copy()
            current_count: int = candidate_count

            if local_optimization:
                refined_set_index: IntArray = np.flatnonzero(candidate_inliers).astype(np.int64)
                if refined_set_index.size >= min_inliers:
                    total_local_opts += 1
                    inner_result: InnerRansacResult = inner_ransac(
                        current_point_cloud,
                        refined_set_index,
                        None,
                        threshold,
                        normals=None,
                        error_metric=error_metric,
                        consensus_metric=consensus_metric,
                        n_iters=inner_iterations,
                        random_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                        deadline=deadline,
                        consensus_context=consensus_context,
                        model_family=model_family,
                    )
                    if inner_result.best_inlier_count > 0:
                        current_inliers = np.asarray(inner_result.best_inliers_mask, dtype=bool)
                        current_count = int(np.count_nonzero(current_inliers))
                        current_model = inner_result.best_model

            if current_count > int(np.count_nonzero(best_inliers)):
                best_model = current_model
                best_inliers = current_inliers

        if best_model is None:
            break

        best_count: int = int(np.count_nonzero(best_inliers))
        if best_count < min_inliers:
            break

        # Final refit on all inliers
        best_points = current_point_cloud[best_inliers]
        if best_points.shape[0] >= parameter_count(model_family):
            try:
                refit_model = fit_superquadric_ls(
                    best_points,
                    error_metric=error_metric,
                    bounds_reference_points=best_points,
                    model_family=model_family,
                    initial_model=best_model if model_family == "superflex" else None,
                )
                refit_inliers = np.asarray(
                    compute_consensus(refit_model, current_point_cloud, threshold, error_metric=consensus_metric, metal_context=consensus_context),
                    dtype=bool,
                )
                refit_count = int(np.count_nonzero(refit_inliers))
                if refit_count >= best_count:
                    best_model = refit_model
                    best_inliers = refit_inliers
                    best_count = refit_count
            except SuperflexMetalError:
                raise
            except Exception:
                pass

        global_inliers: BoolArray = np.zeros(n_points, dtype=bool)
        global_inliers[remaining_indices[best_inliers]] = True

        models_set.append(best_model)
        inliers_set.append(global_inliers)

        remove_mask = expanded_removal_mask(best_model, current_point_cloud, threshold, factor=1.3, error_metric=consensus_metric)
        remaining_indices = remaining_indices[~remove_mask]

    return models_set, inliers_set, total_local_opts
