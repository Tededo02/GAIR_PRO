# For each edge, p_ij = c_ij(1 - (rho_i + rho_j) / 2) and w_ij = c_ij - p_ij / 2.
# The equivalent graph-cut term is p_ij for two outliers, 0 for two inliers, and c_ij otherwise.

import numpy as np

from .pairwise_costs import PairwiseCosts


def build_residual_aware_pairwise_costs(
    *,
    point_count: int,
    residual: np.ndarray,
    eps: float,
    outlier_scale: float,
    edge_sources: np.ndarray,
    edge_targets: np.ndarray,
    coherence: np.ndarray,
) -> PairwiseCosts:
    if outlier_scale <= 0.0:
        raise ValueError("outlier_scale must be positive")

    edge_sources = np.asarray(edge_sources, dtype=np.int64)
    edge_targets = np.asarray(edge_targets, dtype=np.int64)
    coherence = np.asarray(coherence, dtype=np.float64)
    if edge_targets.shape != edge_sources.shape:
        raise ValueError("edge_sources and edge_targets must have the same shape")
    if coherence.shape != edge_sources.shape:
        raise ValueError("coherence must contain one value per edge")

    normalized_residual = np.clip(
        np.asarray(residual, dtype=np.float64)
        / (outlier_scale * eps + 1e-12),
        0.0,
        1.0,
    )
    outlier_pair_cost = coherence * (
        1.0
        - 0.5
        * (
            normalized_residual[edge_sources]
            + normalized_residual[edge_targets]
        )
    )
    np.maximum(outlier_pair_cost, 0.0, out=outlier_pair_cost)

    edge_weights = coherence - 0.5 * outlier_pair_cost
    np.maximum(edge_weights, 0.0, out=edge_weights)

    outlier_correction = np.zeros(point_count, dtype=np.float64)
    endpoint_correction = 0.5 * outlier_pair_cost
    np.add.at(outlier_correction, edge_sources, endpoint_correction)
    np.add.at(outlier_correction, edge_targets, endpoint_correction)

    return PairwiseCosts(
        edge_sources=edge_sources,
        edge_targets=edge_targets,
        edge_weights=edge_weights,
        outlier_correction=outlier_correction,
    )
