# E(y) = sum_i U_i(y_i) + sum_(i,j) w_ij [y_i != y_j].
# U_i is selected from inlier_cost or outlier_cost according to the binary label y_i.

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class GraphCutEnergy:
    inlier_cost: FloatArray
    outlier_cost: FloatArray
    edge_sources: IntArray
    edge_targets: IntArray
    edge_weights: FloatArray

    def __post_init__(self) -> None:
        if self.inlier_cost.ndim != 1:
            raise ValueError("inlier_cost must be one-dimensional")
        if self.outlier_cost.shape != self.inlier_cost.shape:
            raise ValueError(
                "inlier_cost and outlier_cost must have the same shape"
            )
        if self.edge_sources.ndim != 1:
            raise ValueError("edge_sources must be one-dimensional")

        edge_count = self.edge_sources.shape[0]
        if self.edge_targets.shape != (edge_count,):
            raise ValueError(
                "edge_sources and edge_targets must have the same shape"
            )
        if self.edge_weights.shape != (edge_count,):
            raise ValueError("each pairwise edge must have one weight")
