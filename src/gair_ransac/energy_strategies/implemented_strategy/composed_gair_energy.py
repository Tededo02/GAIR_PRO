# E(y) = sum_i (U_i(y_i) + C_i [y_i = outlier]) + sum_(i,j) w_ij [y_i != y_j].
# U is supplied by the unary term; C and w are supplied by the pairwise term.
# costruisci energia come vuoi
from dataclasses import dataclass

from ...consensus import distance_err
from ..context import EnergyContext
from ..gair_energy_strategy import GairEnergyStrategy
from ..graph_cut_energy import GraphCutEnergy
from ..pairwise.pairwise_energy_term import PairwiseEnergyTerm
from ..unary.unary_energy_term import UnaryEnergyTerm


@dataclass(frozen=True, slots=True)
class ComposedGairEnergy(GairEnergyStrategy):
    unary: UnaryEnergyTerm
    pairwise: PairwiseEnergyTerm

    def build(self, context: EnergyContext) -> GraphCutEnergy:
        residual = distance_err(
            context.model,
            context.points,
            error_metric=context.error_metric,
        )
        unary_costs = self.unary.build(context, residual)
        pairwise_costs = self.pairwise.build(context, residual)

        if unary_costs.inlier.shape[0] != context.points.shape[0]:
            raise ValueError("unary costs must contain one value per point")
        if (
            pairwise_costs.outlier_correction.shape
            != unary_costs.outlier.shape
        ):
            raise ValueError(
                "pairwise outlier correction must contain one value per point"
            )

        return GraphCutEnergy(
            inlier_cost=unary_costs.inlier,
            outlier_cost=(
                unary_costs.outlier
                + pairwise_costs.outlier_correction
            ),
            edge_sources=pairwise_costs.edge_sources,
            edge_targets=pairwise_costs.edge_targets,
            edge_weights=pairwise_costs.edge_weights,
        )
