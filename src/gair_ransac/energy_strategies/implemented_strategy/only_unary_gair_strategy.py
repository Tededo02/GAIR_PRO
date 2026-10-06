# E(y) uses the model-normal unary term and the unchanged GC-RANSAC pairwise term.
# U_i(inlier) = r_i + q_i + 1/2 sum_j ((r_i + r_j) / 2).
# U_i(outlier) = 1 + 1/2 sum_j (1 - (r_i + r_j) / 2), with pairwise weight 1/2.
# r_i = clip(d_i / eps, 0, 1), q_i = clip((1 - <n_i, n_model_i>) / 2, 0, 1).
# solo unary term con normal model, pairwise term come in gc_ransac
import numpy as np

from ..context import EnergyContext
from ..unary.residual_model_normal_unary_energy import (
    ResidualModelNormalUnaryEnergy,
)
from ..unary.unary_costs import UnaryCosts
from .gc_ransac_energy import GcRansacEnergy


class OnlyUnaryGairStrategy(GcRansacEnergy):
    __slots__ = ()

    def _build_unary_costs(
        self,
        context: EnergyContext,
        residual: np.ndarray,
        _residual_unary_costs: UnaryCosts,
    ) -> UnaryCosts:
        return ResidualModelNormalUnaryEnergy().build(context, residual)
