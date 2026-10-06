# E(y) = sum_i U_i(y_i) + sum_(i,j) w_ij [y_i != y_j].
# U_i(inlier) = clip(d_i / eps, 0, 1), U_i(outlier) = 1 + 1/2 sum_j p_ij.
# p_ij = c_ij(1 - (rho_i + rho_j) / 2), c_ij = (1 + <n_i, n_j>) / 2.
# rho_i = clip(d_i / (outlier_scale eps), 0, 1), w_ij = c_ij - p_ij / 2.
# gair come nel paper vecchio senza normal model nella unary, con c_ij e rho_i come sopra dove rho indica la distanza normalizzata dal modello, e c_ij indica la coerenza tra le normali dei punti i e j.
from .composed_gair_energy import ComposedGairEnergy
from ..pairwise.normal_coherence_pairwise_energy import (
    COHERENCE_MIN,
    OUTLIER_SCALE,
    NormalCoherencePairwiseEnergy,
)
from ..unary.residual_unary_energy import ResidualUnaryEnergy


class OldPaperGairEnergy(ComposedGairEnergy):
    __slots__ = ()

    def __init__(
        self,
        coherence_min: float = COHERENCE_MIN,
        outlier_scale: float = OUTLIER_SCALE,
    ) -> None:
        super().__init__(
            unary=ResidualUnaryEnergy(),
            pairwise=NormalCoherencePairwiseEnergy(
                coherence_min=coherence_min,
                outlier_scale=outlier_scale,
            ),
        )
