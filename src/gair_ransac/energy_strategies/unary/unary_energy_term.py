# A unary term builds sum_i U_i(y_i) for the binary inlier/outlier labels.

from abc import ABC, abstractmethod

import numpy as np

from ..context import EnergyContext
from .unary_costs import UnaryCosts


class UnaryEnergyTerm(ABC):
    @abstractmethod
    def build(
        self,
        context: EnergyContext,
        residual: np.ndarray,
    ) -> UnaryCosts:
        pass
