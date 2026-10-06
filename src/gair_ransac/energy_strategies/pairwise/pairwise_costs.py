# Pairwise costs represent C_i [y_i = outlier] + sum_(i,j) w_ij [y_i != y_j].

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class PairwiseCosts:
    edge_sources: IntArray
    edge_targets: IntArray
    edge_weights: FloatArray
    outlier_correction: FloatArray

    def __post_init__(self) -> None:
        if self.edge_sources.ndim != 1:
            raise ValueError("edge_sources must be one-dimensional")
        edge_count = self.edge_sources.shape[0]
        if self.edge_targets.shape != (edge_count,):
            raise ValueError(
                "edge_sources and edge_targets must have the same shape"
            )
        if self.edge_weights.shape != (edge_count,):
            raise ValueError("each pairwise edge must have one weight")
        if self.outlier_correction.ndim != 1:
            raise ValueError("outlier correction must be one-dimensional")

    @classmethod
    def empty(cls, point_count: int) -> "PairwiseCosts":
        return cls(
            edge_sources=np.empty(0, dtype=np.int64),
            edge_targets=np.empty(0, dtype=np.int64),
            edge_weights=np.empty(0, dtype=np.float64),
            outlier_correction=np.zeros(point_count, dtype=np.float64),
        )
