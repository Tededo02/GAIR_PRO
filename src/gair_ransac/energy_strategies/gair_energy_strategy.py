# A strategy builds E(y) = sum_i U_i(y_i) + sum_(i,j) w_ij [y_i != y_j].

from abc import ABC, abstractmethod

from .context import EnergyContext
from .graph_cut_energy import GraphCutEnergy


class GairEnergyStrategy(ABC):
    @abstractmethod
    def build(self, context: EnergyContext) -> GraphCutEnergy:
        pass
