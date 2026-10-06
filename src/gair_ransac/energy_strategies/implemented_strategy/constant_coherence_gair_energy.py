# E(y) = sum_i U_i(y_i) + sum_(i,j) w_ij [y_i != y_j], with C_ij = 1.
# U_i(inlier) = clip(d_i / eps, 0, 1) + clip((1 - <n_i, n_model_i>) / 2, 0, 1).
# U_i(outlier) = 1 + 1/2 sum_j p_ij, with p_ij = 1 - (rho_i + rho_j) / 2.
# rho_i = clip(d_i / (outlier_scale eps), 0, 1), w_ij = 1 - p_ij / 2.
# gair completo nuovo con C==1
from ..pairwise.constant_coherence_pairwise_energy import (
    ConstantCoherencePairwiseEnergy,
)
from ..pairwise.normal_coherence_pairwise_energy import OUTLIER_SCALE
from ..unary.residual_model_normal_unary_energy import (
    ResidualModelNormalUnaryEnergy,
)
from .composed_gair_energy import ComposedGairEnergy


class ConstantCoherenceGairEnergy(ComposedGairEnergy):
    __slots__ = ()

    def __init__(
        self,
        outlier_scale: float = OUTLIER_SCALE,
    ) -> None:
        super().__init__(
            unary=ResidualModelNormalUnaryEnergy(),
            pairwise=ConstantCoherencePairwiseEnergy(
                outlier_scale=outlier_scale,
            ),
        )
