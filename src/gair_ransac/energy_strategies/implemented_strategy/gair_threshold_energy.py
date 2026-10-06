# Normal coherence selects edges, while retained edges use C_ij = 1 in E(y).
# The unary term is identical to FullGairEnergy for a controlled ablation.

from ..pairwise.normal_coherence_pairwise_energy import (
    COHERENCE_MIN,
    OUTLIER_SCALE,
)
from ..pairwise.thresholded_constant_coherence_pairwise_energy import (
    ThresholdedConstantCoherencePairwiseEnergy,
)
from ..unary.residual_model_normal_unary_energy import (
    ResidualModelNormalUnaryEnergy,
)
from .composed_gair_energy import ComposedGairEnergy


class GairThresholdEnergy(ComposedGairEnergy):
    __slots__ = ()

    def __init__(
        self,
        coherence_min: float = COHERENCE_MIN,
        outlier_scale: float = OUTLIER_SCALE,
    ) -> None:
        super().__init__(
            unary=ResidualModelNormalUnaryEnergy(),
            pairwise=ThresholdedConstantCoherencePairwiseEnergy(
                coherence_min=coherence_min,
                outlier_scale=outlier_scale,
            ),
        )
