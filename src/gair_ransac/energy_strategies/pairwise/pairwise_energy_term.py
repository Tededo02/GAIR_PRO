# A pairwise term builds C_i [y_i = outlier] + sum_(i,j) w_ij [y_i != y_j].

from abc import ABC, abstractmethod

import numpy as np

from ..context import EnergyContext
from .pairwise_costs import PairwiseCosts


class PairwiseEnergyTerm(ABC):
    @abstractmethod
    def build(
        self,
        context: EnergyContext,
        residual: np.ndarray,
    ) -> PairwiseCosts:
        pass
