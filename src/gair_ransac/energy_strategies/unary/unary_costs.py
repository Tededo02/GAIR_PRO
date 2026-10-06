# Unary costs represent sum_i U_i(y_i), with one inlier and one outlier cost per point.

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class UnaryCosts:
    inlier: FloatArray
    outlier: FloatArray

    def __post_init__(self) -> None:
        if self.inlier.ndim != 1:
            raise ValueError("inlier unary costs must be one-dimensional")
        if self.outlier.shape != self.inlier.shape:
            raise ValueError(
                "inlier and outlier unary costs must have the same shape"
            )
