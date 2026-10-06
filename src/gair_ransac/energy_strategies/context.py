# The context provides the points, graph, normals, model, and eps used to build E(y).

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.superquadrics.superquadric_param import SuperQuadricParams


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class EnergyContext:
    points: FloatArray
    edges: IntArray
    normals: FloatArray | None
    model: SuperQuadricParams
    eps: float
    error_metric: str

    @classmethod
    def create(
        cls,
        points: np.ndarray,
        edges: np.ndarray,
        normals: np.ndarray | None,
        model: SuperQuadricParams,
        eps: float,
        error_metric: str,
    ) -> EnergyContext:
        point_array = np.asarray(points, dtype=np.float64)
        edge_array = np.asarray(edges, dtype=np.int64)
        normal_array = (
            None
            if normals is None
            else np.asarray(normals, dtype=np.float64)
        )

        if point_array.ndim != 2 or point_array.shape[1] != 3:
            raise ValueError(
                f"points must have shape (N, 3), got {point_array.shape}"
            )
        if normal_array is not None and normal_array.shape != point_array.shape:
            raise ValueError(
                f"normals must have shape {point_array.shape}, "
                f"got {normal_array.shape}"
            )
        if edge_array.ndim != 2 or edge_array.shape[1] != 2:
            raise ValueError(
                f"edges must have shape (E, 2), got {edge_array.shape}"
            )
        if edge_array.size and (
            int(edge_array.min()) < 0
            or int(edge_array.max()) >= point_array.shape[0]
        ):
            raise ValueError(
                "edges contain point indices outside the valid range"
            )
        if eps <= 0.0:
            raise ValueError(f"eps must be positive, got {eps}")

        return cls(
            points=point_array,
            edges=edge_array,
            normals=normal_array,
            model=model,
            eps=float(eps),
            error_metric=error_metric,
        )
