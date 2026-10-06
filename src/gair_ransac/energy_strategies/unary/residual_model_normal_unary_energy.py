# U_i(inlier) = clip(d_i / eps, 0, 1) + clip((1 - <n_i, n_model_i>) / 2, 0, 1).
# U_i(outlier) = 1.

import numpy as np

from ...consensus import normal_alignment_score
from ..context import EnergyContext
from .residual_unary_energy import ResidualUnaryEnergy
from .unary_costs import UnaryCosts


class ResidualModelNormalUnaryEnergy(ResidualUnaryEnergy):
    __slots__ = ()

    def build(
        self,
        context: EnergyContext,
        residual: np.ndarray,
    ) -> UnaryCosts:
        if context.normals is None:
            raise ValueError(
                "ResidualModelNormalUnaryEnergy requires point normals"
            )

        residual_costs = super().build(context, residual)
        alignment = normal_alignment_score(
            context.model,
            context.points,
            context.normals,
        )
        normal_penalty = np.clip(
            0.5 * (1.0 - alignment),
            0.0,
            1.0,
        )
        return UnaryCosts(
            inlier=residual_costs.inlier + normal_penalty,
            outlier=residual_costs.outlier,
        )
