# C_ij = 1, p_ij = 1 - (rho_i + rho_j) / 2, and V_ij = w_ij [y_i != y_j].
# rho_i = clip(d_i / (outlier_scale eps), 0, 1), w_ij = 1 - p_ij / 2.

from dataclasses import dataclass

import numpy as np

from ..context import EnergyContext
from .normal_coherence_pairwise_energy import OUTLIER_SCALE
from .pairwise_costs import PairwiseCosts
from .pairwise_energy_term import PairwiseEnergyTerm
from .residual_aware_pairwise import build_residual_aware_pairwise_costs


@dataclass(frozen=True, slots=True)
class ConstantCoherencePairwiseEnergy(PairwiseEnergyTerm):
    outlier_scale: float = OUTLIER_SCALE

    def __post_init__(self) -> None:
        if self.outlier_scale <= 0.0:
            raise ValueError("outlier_scale must be positive")

    def build(
        self,
        context: EnergyContext,
        residual: np.ndarray,
    ) -> PairwiseCosts:
        point_count = context.points.shape[0]
        if not context.edges.size:
            return PairwiseCosts.empty(point_count)

        edge_sources = context.edges[:, 0]
        edge_targets = context.edges[:, 1]
        coherence = np.ones(edge_sources.shape[0], dtype=np.float64)
        return build_residual_aware_pairwise_costs(
            point_count=point_count,
            residual=residual,
            eps=context.eps,
            outlier_scale=self.outlier_scale,
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            coherence=coherence,
        )
