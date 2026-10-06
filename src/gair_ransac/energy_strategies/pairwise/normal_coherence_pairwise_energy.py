# For c_ij > coherence_min, C_i = 1/2 sum_j p_ij and V_ij = w_ij [y_i != y_j].
# c_ij = (1 + <n_i, n_j>) / 2, p_ij = c_ij(1 - (rho_i + rho_j) / 2).
# rho_i = clip(d_i / (outlier_scale eps), 0, 1), w_ij = c_ij - p_ij / 2.

from dataclasses import dataclass

import numpy as np

from ..context import EnergyContext
from .pairwise_costs import PairwiseCosts
from .pairwise_energy_term import PairwiseEnergyTerm
from .residual_aware_pairwise import build_residual_aware_pairwise_costs


COHERENCE_MIN = 0.9
OUTLIER_SCALE = 2.5


@dataclass(frozen=True, slots=True)
class NormalCoherencePairwiseEnergy(PairwiseEnergyTerm):
    coherence_min: float = COHERENCE_MIN
    outlier_scale: float = OUTLIER_SCALE

    def __post_init__(self) -> None:
        if not 0.0 <= self.coherence_min <= 1.0:
            raise ValueError("coherence_min must be between 0 and 1")
        if self.outlier_scale <= 0.0:
            raise ValueError("outlier_scale must be positive")

    def build(
        self,
        context: EnergyContext,
        residual: np.ndarray,
    ) -> PairwiseCosts:
        if context.normals is None:
            raise ValueError(
                "NormalCoherencePairwiseEnergy requires point normals"
            )

        point_count = context.points.shape[0]
        if not context.edges.size:
            return PairwiseCosts.empty(point_count)

        edge_sources = context.edges[:, 0]
        edge_targets = context.edges[:, 1]
        coherence = 0.5 * (
            1.0
            + np.einsum(
                "ij,ij->i",
                context.normals[edge_sources],
                context.normals[edge_targets],
                optimize=True,
            )
        )

        valid_edges = coherence > self.coherence_min
        edge_sources = edge_sources[valid_edges]
        edge_targets = edge_targets[valid_edges]
        coherence = coherence[valid_edges]
        if not edge_sources.size:
            return PairwiseCosts.empty(point_count)

        return build_residual_aware_pairwise_costs(
            point_count=point_count,
            residual=residual,
            eps=context.eps,
            outlier_scale=self.outlier_scale,
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            coherence=coherence,
        )
