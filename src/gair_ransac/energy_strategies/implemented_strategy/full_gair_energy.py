# E(y) = sum_i U_i(y_i) + sum_(i,j) w_ij [y_i != y_j].
# U_i(inlier) = clip(d_i / eps, 0, 1) + clip((1 - <n_i, n_model_i>) / 2, 0, 1).
# U_i(outlier) = 1 + 1/2 sum_j p_ij, with p_ij = c_ij(1 - (rho_i + rho_j) / 2).
# c_ij = (1 + <n_i, n_j>) / 2, rho_i = clip(d_i / (outlier_scale eps), 0, 1).
# w_ij = c_ij - p_ij / 2.
# gair completo nuovo con C==c_ij
from .composed_gair_energy import ComposedGairEnergy
from ..pairwise.normal_coherence_pairwise_energy import (
    COHERENCE_MIN,
    OUTLIER_SCALE,
    NormalCoherencePairwiseEnergy,
)
from ..unary.residual_model_normal_unary_energy import (
    ResidualModelNormalUnaryEnergy,
)


class FullGairEnergy(ComposedGairEnergy):
    __slots__ = ()

    def __init__(
        self,
        coherence_min: float = COHERENCE_MIN,
        outlier_scale: float = OUTLIER_SCALE,
    ) -> None:
        super().__init__(
            unary=ResidualModelNormalUnaryEnergy(),
            pairwise=NormalCoherencePairwiseEnergy(
                coherence_min=coherence_min,
                outlier_scale=outlier_scale,
            ),
        )
