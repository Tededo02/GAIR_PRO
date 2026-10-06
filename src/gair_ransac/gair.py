import maxflow
import numpy as np

from src.superquadrics.superquadric_param import SuperQuadricParams
from .energy_strategies.context import EnergyContext
from .energy_strategies.gair_energy_strategy import GairEnergyStrategy


def gair(
    points: np.ndarray,
    edges: np.ndarray,
    normals: np.ndarray | None,
    model: SuperQuadricParams,
    eps: float,
    energy_strategy: GairEnergyStrategy,
    error_metric: str = "radial",
) -> np.ndarray:
    context = EnergyContext.create(
        points=points,
        edges=edges,
        normals=normals,
        model=model,
        eps=eps,
        error_metric=error_metric,
    )
    energy = energy_strategy.build(context)

    graph = maxflow.Graph[float](
        context.points.shape[0],
        int(energy.edge_sources.shape[0] * 2),
    )
    node_ids = graph.add_nodes(context.points.shape[0])

    if energy.edge_sources.size:
        graph.add_edges(
            node_ids[energy.edge_sources],
            node_ids[energy.edge_targets],
            energy.edge_weights,
            energy.edge_weights,
        )

    graph.add_grid_tedges(node_ids, energy.inlier_cost, energy.outlier_cost)
    graph.maxflow()

    # Segment 0 is SOURCE/outlier and segment 1 is SINK/inlier.
    return graph.get_grid_segments(node_ids)
