from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.superquadrics.superquadric_param import SuperQuadricParams
from .axis_regularization import prefer_compact_model
from .initgraph import build_knn_graph


DEFAULT_INTERIOR_PENALTY_WEIGHT = 25.0
SCORE_RELATIVE_TOLERANCE = 1e-6


def validate_interior_penalty_weight(weight: float) -> float:
    weight = float(weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("interior_penalty_weight must be finite and non-negative")
    return weight


def consensus_scores(counts, masses, weight: float) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    ratio = masses / np.maximum(counts, 1.0)
    return counts / (1.0 + weight * ratio**2)


def best_score_index(scores: np.ndarray) -> int:
    if not len(scores):
        raise ValueError("An empty consensus batch has no winner")
    maximum = float(np.max(scores))
    tolerance = SCORE_RELATIVE_TOLERANCE * max(abs(maximum), 1.0)
    return int(np.flatnonzero(scores >= maximum - tolerance)[0])


def prefer_consensus_model(
    candidate: SuperQuadricParams,
    count: int,
    score: float,
    best: SuperQuadricParams | None,
    best_count: int,
    best_score: float,
    axis_penalty_weight: float,
    score_enabled: bool,
    min_gain: int = 1,
) -> bool:
    if not score_enabled:
        return count >= best_count + max(min_gain, 1) or (
            axis_penalty_weight > 0.0 and prefer_compact_model(candidate, count, best, best_count, min_gain)
        )
    if best is None:
        return count > 0
    tolerance = SCORE_RELATIVE_TOLERANCE * max(abs(score), abs(best_score), 1.0)
    if score > best_score + tolerance:
        return True
    return abs(score - best_score) <= tolerance and count == best_count and (
        axis_penalty_weight > 0.0 and prefer_compact_model(candidate, count, best, best_count)
    )


def evaluate_model_consensus(
    model, points, threshold, error_metric, normals, metal_context,
    interior_context, consensus_function,
) -> tuple[np.ndarray, int, float, float]:
    if metal_context is not None and metal_context.interior_context is not None:
        metal_context.check_inputs(points, normals)
        result = metal_context.evaluate([model], threshold)
        return result.mask(0), int(result.counts[0]), float(result.scores[0]), float(result.interior_masses[0])
    mask = np.asarray(consensus_function(
        model, points, threshold, error_metric=error_metric,
        normals=normals, metal_context=metal_context,
    ), dtype=bool)
    count = int(np.count_nonzero(mask))
    if interior_context is not None and interior_context.weight > 0.0:
        score, mass = interior_context.score(model, count, threshold)
        return mask, count, score, mass
    return mask, count, float(count), 0.0


def interior_strength(model: SuperQuadricParams, points: np.ndarray, threshold: float) -> np.ndarray:
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("interior scoring requires a finite, positive threshold")
    canonical = (np.asarray(points, dtype=np.float64) - model.t) @ model.rotation_matrix()
    axes = np.array([model.a1, model.a2, model.a3])
    radius = np.linalg.norm(canonical, axis=1)
    with np.errstate(divide="ignore"):
        logs = np.log(np.abs(canonical / axes))
    log_xy = np.logaddexp((2.0 / model.e2) * logs[:, 0], (2.0 / model.e2) * logs[:, 1])
    log_shape = np.logaddexp((model.e2 / model.e1) * log_xy, (2.0 / model.e1) * logs[:, 2])
    depth = np.zeros(len(points), dtype=np.float64)
    inside = (log_shape < 0.0) & (radius > 1e-12)
    # Use the unclamped implicit value so deep interior points retain their true radial depth.
    depth[inside] = np.exp(np.log(radius[inside]) - 0.5 * model.e1 * log_shape[inside]) - radius[inside]
    center = radius <= 1e-12
    # A conservative inscribed radius gives finite penetration when the ray direction is undefined.
    depth[center] = axes.min() * 2.0**(-0.5 * (max(model.e1 - 1.0, 0.0) + max(model.e2 - 1.0, 0.0)))
    excess = np.maximum(depth - threshold, 0.0)
    return (excess / np.hypot(excess, threshold))**2


@dataclass
class InteriorPenaltyContext:
    """Original-cloud geometry and neighbors shared by all candidate scores."""

    points: np.ndarray
    neighbors: Sequence[Sequence[int]] | None = None
    weight: float = DEFAULT_INTERIOR_PENALTY_WEIGHT
    normals: np.ndarray | None = None
    neighbor_matrix: np.ndarray = field(init=False, repr=False)
    _neighbor_counts: np.ndarray = field(init=False, repr=False)
    active_indices: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.weight = validate_interior_penalty_weight(self.weight)
        self.points = np.asarray(self.points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not np.isfinite(self.points).all():
            raise ValueError("interior reference points must be a finite array with shape (N, 3)")
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64)
            if self.normals.shape != self.points.shape or not np.isfinite(self.normals).all():
                raise ValueError("reference normals must be finite and have the same shape as points")
        n = len(self.points)
        if self.neighbors is None:
            self.neighbors, _ = build_knn_graph(self.points, m_neighbors=8)
        if len(self.neighbors) != n:
            raise ValueError("neighbors must contain one row per reference point")
        counts = np.array([len(row) for row in self.neighbors], dtype=np.int64)
        width = int(counts.max()) if n else 0
        if n and np.all(counts == width):
            matrix = np.asarray(self.neighbors, dtype=np.int64).reshape(n, width)
        else:
            matrix = np.full((n, width), n, dtype=np.int64)
            for i, row in enumerate(self.neighbors):
                matrix[i, :len(row)] = row
        valid = np.arange(width)[None, :] < counts[:, None]
        if np.any(valid & ((matrix < 0) | (matrix >= n) | (matrix == np.arange(n)[:, None]))):
            raise ValueError("neighbors must contain valid indices other than the point itself")
        self.neighbor_matrix = matrix.astype(np.uint32)
        self._neighbor_counts = counts
        self.active_indices = np.arange(n, dtype=np.int64)

    def mass(self, model: SuperQuadricParams, threshold: float) -> float:
        strength = interior_strength(model, self.points, threshold)
        if not self.neighbor_matrix.shape[1]:
            return float(strength.sum())
        padded = np.append(strength, 0.0)
        counts = self._neighbor_counts
        local_mean = np.sum(padded[self.neighbor_matrix], axis=1) / np.maximum(counts, 1)
        local_mean[counts == 0] = 1.0
        return float(np.dot(strength, local_mean))

    def score(self, model: SuperQuadricParams, count: int, threshold: float) -> tuple[float, float]:
        mass = self.mass(model, threshold)
        return float(consensus_scores(count, mass, self.weight)), mass
