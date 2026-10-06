# U_i(inlier) = clip(d_i / eps, 0, 1), U_i(outlier) = 1.

from dataclasses import dataclass

import numpy as np

from ..context import EnergyContext
from .unary_costs import UnaryCosts
from .unary_energy_term import UnaryEnergyTerm


@dataclass(frozen=True, slots=True)
class ResidualUnaryEnergy(UnaryEnergyTerm):
    def build(
        self,
        context: EnergyContext,
        residual: np.ndarray,
    ) -> UnaryCosts:
        normalized_residual = np.clip(
            residual / (context.eps + 1e-12),
            0.0,
            1.0,
        )
        return UnaryCosts(
            inlier=normalized_residual,
            outlier=np.ones(context.points.shape[0], dtype=np.float64),
        )
