# Edges are retained when C_ij > coherence_min, then assigned C_ij = 1 in the energy.
# p_ij = 1 - (rho_i + rho_j) / 2 and w_ij = 1 - p_ij / 2 for retained edges.

from dataclasses import dataclass

import numpy as np

from ..context import EnergyContext
from .normal_coherence_pairwise_energy import COHERENCE_MIN, OUTLIER_SCALE
from .pairwise_costs import PairwiseCosts
from .pairwise_energy_term import PairwiseEnergyTerm
from .residual_aware_pairwise import build_residual_aware_pairwise_costs


@dataclass(frozen=True, slots=True)
class ThresholdedConstantCoherencePairwiseEnergy(PairwiseEnergyTerm):
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
                "ThresholdedConstantCoherencePairwiseEnergy requires point "
                "normals"
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
        if not edge_sources.size:
            return PairwiseCosts.empty(point_count)

        constant_coherence = np.ones(
            edge_sources.shape[0],
            dtype=np.float64,
        )
        return build_residual_aware_pairwise_costs(
            point_count=point_count,
            residual=residual,
            eps=context.eps,
            outlier_scale=self.outlier_scale,
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            coherence=constant_coherence,
        )
