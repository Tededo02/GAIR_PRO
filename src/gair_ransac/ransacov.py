# RansaCov: maximum-coverage ILP for multi-model selection.
# Based on equation (5) of "RansaCov: Multi-model fitting with coverage
# constraints" by Magri and Fusiello.
# Given a candidate pool and a point cloud, it selects at most k models that
# cover the maximum number of inlier points (residual < threshold).

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix, eye, hstack, vstack


ResidualFunction = Callable[[Any, np.ndarray], np.ndarray]


def _build_coverage_matrix(
    candidates: Sequence[Any],
    points: np.ndarray,
    threshold: float,
    residual_fn: ResidualFunction,
) -> np.ndarray:
    # Build the binary coverage matrix P[i, j]. P[i, j] is true when point i
    # is an inlier of candidate model j.
    coverage = np.zeros((points.shape[0], len(candidates)), dtype=bool)
    for candidate_index, model in enumerate(candidates):
        residuals = np.asarray(residual_fn(model, points), dtype=np.float64).reshape(-1)
        if residuals.shape[0] != points.shape[0]:
            raise ValueError(
                "residual_fn must return one residual for every input point"
            )
        coverage[:, candidate_index] = np.isfinite(residuals) & (residuals < threshold)
    return coverage


def _remove_dominated_candidates(coverage: np.ndarray) -> list[int]:
    # Sort by cardinality and remove a candidate only when one retained model
    # covers all its points. This preserves maximum-coverage solutions.
    cardinalities = coverage.sum(axis=0)
    order = np.argsort(-cardinalities, kind="stable")
    kept_indices: list[int] = []

    for candidate_index in order:
        candidate_coverage = coverage[:, candidate_index]
        is_dominated = any(
            np.all(candidate_coverage <= coverage[:, kept_index])
            for kept_index in kept_indices
        )
        if not is_dominated:
            kept_indices.append(int(candidate_index))

    return kept_indices


def ransacov(
    candidates: Sequence[Any],
    points: np.ndarray,
    k: int,
    threshold: float,
    residual_fn: ResidualFunction,
) -> tuple[list[int], int]:
    # Parameters:
    # candidates: model objects such as SuperQuadricParams.
    # points: (N, 3) point cloud to cover.
    # k: maximum number of models to select.
    # threshold: inlier distance threshold.
    # residual_fn: callable returning one unsigned residual per point.
    #
    # Returns:
    # selected_indices: indices into candidates for the selected models.
    # n_covered: number of points covered by the selected models.
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.ndim != 2 or point_array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if k <= 0:
        raise ValueError("k must be positive")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")
    if len(candidates) == 0 or point_array.shape[0] == 0:
        return [], 0

    coverage = _build_coverage_matrix(
        candidates,
        point_array,
        threshold,
        residual_fn,
    )
    kept_indices = _remove_dominated_candidates(coverage)
    if not kept_indices:
        return [], 0

    kept_coverage = coverage[:, kept_indices]
    # Points with the same candidate-membership pattern share one auxiliary
    # variable weighted by their multiplicity. This keeps the ILP exact while
    # reducing its size on large real point clouds.
    coverage_patterns, pattern_weights = np.unique(
        kept_coverage,
        axis=0,
        return_counts=True,
    )
    covered_patterns = coverage_patterns.any(axis=1)
    coverage_patterns = coverage_patterns[covered_patterns]
    pattern_weights = pattern_weights[covered_patterns].astype(np.float64)
    if coverage_patterns.shape[0] == 0:
        return [], 0

    model_count = len(kept_indices)
    pattern_count = coverage_patterns.shape[0]
    variable_count = model_count + pattern_count
    effective_k = min(k, model_count)

    # Sparse MILP formulation:
    # variables: [z_0, ..., z_M-1, y_0, ..., y_U-1]
    # minimize: model_cost * sum(z_j) - sum(weight_i * y_i)
    # subject to:
    #   (1) sum(z_j) <= k
    #   (2) sum(P[i, j] * z_j) - y_i >= 0 for every coverage pattern i
    #   z_j in {0, 1}, 0 <= y_i <= 1
    # A sub-unit total model cost breaks equal-coverage ties in favor of fewer models.
    objective = np.empty(variable_count, dtype=np.float64)
    objective[:model_count] = 0.5 / (model_count + 1.0)
    objective[model_count:] = -pattern_weights

    integrality = np.zeros(variable_count, dtype=np.uint8)
    # Only z variables are binary; y variables remain continuous in [0, 1].
    integrality[:model_count] = 1
    variable_bounds = Bounds(
        lb=np.zeros(variable_count, dtype=np.float64),
        ub=np.ones(variable_count, dtype=np.float64),
    )

    # Constraint (1): sum(z_j) <= k.
    model_limit = csc_matrix(
        (
            np.ones(model_count, dtype=np.float64),
            (
                np.zeros(model_count, dtype=np.int64),
                np.arange(model_count, dtype=np.int64),
            ),
        ),
        shape=(1, variable_count),
    )
    # Constraints (2): P[i, :] z - y_i >= 0.
    coverage_constraints = hstack(
        (
            csc_matrix(coverage_patterns, dtype=np.float64),
            -eye(pattern_count, format="csc", dtype=np.float64),
        ),
        format="csc",
    )
    constraint_matrix = vstack(
        (model_limit, coverage_constraints),
        format="csc",
    )
    constraint_lower_bounds = np.concatenate(
        ((-np.inf,), np.zeros(pattern_count, dtype=np.float64))
    )
    constraint_upper_bounds = np.concatenate(
        ((float(effective_k),), np.full(pattern_count, np.inf, dtype=np.float64))
    )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=variable_bounds,
        constraints=LinearConstraint(
            constraint_matrix,
            constraint_lower_bounds,
            constraint_upper_bounds,
        ),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        print(f"  [ransacov] ILP solver failed: {result.message}")
        return [], 0

    # Map the retained-column solution back to the original candidate indices.
    selected_local_indices = np.flatnonzero(result.x[:model_count] > 0.5)
    selected_indices = [kept_indices[index] for index in selected_local_indices]
    if not selected_indices:
        return [], 0

    covered_points = np.any(coverage[:, selected_indices], axis=1)
    return selected_indices, int(np.count_nonzero(covered_points))
