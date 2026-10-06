# E(y) = sum_i U_i(y_i) + 1/2 sum_(i,j) [y_i != y_j], with r_i = clip(d_i / eps, 0, 1).
# U_i(inlier) = r_i + 1/2 sum_j ((r_i + r_j) / 2).
# U_i(outlier) = 1 + 1/2 sum_j (1 - (r_i + r_j) / 2).
# energia per gc_ransac
from dataclasses import dataclass

import numpy as np

from ...consensus import distance_err
from ..context import EnergyContext
from ..gair_energy_strategy import GairEnergyStrategy
from ..graph_cut_energy import GraphCutEnergy
from ..unary.residual_unary_energy import ResidualUnaryEnergy
from ..unary.unary_costs import UnaryCosts


@dataclass(frozen=True, slots=True)
class GcRansacEnergy(GairEnergyStrategy):
    def _build_unary_costs(
        self,
        context: EnergyContext,
        residual: np.ndarray,
        residual_unary_costs: UnaryCosts,
    ) -> UnaryCosts:
        return residual_unary_costs

    def build(self, context: EnergyContext) -> GraphCutEnergy:
        residual = distance_err(
            context.model,
            context.points,
            error_metric=context.error_metric,
        )
        residual_unary_costs = ResidualUnaryEnergy().build(context, residual)
        unary_costs = self._build_unary_costs(
            context,
            residual,
            residual_unary_costs,
        )
        normalized_residual = residual_unary_costs.inlier
        inlier_cost = unary_costs.inlier.copy()
        outlier_cost = unary_costs.outlier.copy()
        if not context.edges.size:
            return GraphCutEnergy(
                inlier_cost=inlier_cost,
                outlier_cost=outlier_cost,
                edge_sources=np.empty(0, dtype=np.int64),
                edge_targets=np.empty(0, dtype=np.int64),
                edge_weights=np.empty(0, dtype=np.float64),
            )

        edge_sources = context.edges[:, 0]
        edge_targets = context.edges[:, 1]
        same_inlier_cost = 0.5 * (
            normalized_residual[edge_sources]
            + normalized_residual[edge_targets]
        )
        same_outlier_cost = 1.0 - same_inlier_cost

        np.add.at(inlier_cost, edge_sources, 0.5 * same_inlier_cost)
        np.add.at(inlier_cost, edge_targets, 0.5 * same_inlier_cost)
        np.add.at(outlier_cost, edge_sources, 0.5 * same_outlier_cost)
        np.add.at(outlier_cost, edge_targets, 0.5 * same_outlier_cost)

        edge_weights = np.full(
            edge_sources.shape[0],
            0.5,
            dtype=np.float64,
        )
        return GraphCutEnergy(
            inlier_cost=inlier_cost,
            outlier_cost=outlier_cost,
            edge_sources=edge_sources,
            edge_targets=edge_targets,
            edge_weights=edge_weights,
        )
