from dataclasses import dataclass, field
from copy import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.model_family import validate_model_family
from .metal_superquadric import _metal_runtime, metal_available
from .interior_consensus import InteriorPenaltyContext, best_score_index, consensus_scores


CONSENSUS_THREADS = 256


@lru_cache(maxsize=2)
def _consensus_kernel(superflex: bool = False):
    mx = _metal_runtime()
    if mx is None:
        raise RuntimeError("Metal consensus requires Apple silicon, an accessible GPU and MLX")
    directory = Path(__file__).parent
    return mx.fast.metal_kernel(
        name="superflex_radial_consensus_batch" if superflex else "superquadric_radial_consensus_batch",
        input_names=["points", "normals", "parameters", "options", "active"],
        output_names=["masks", "partials", "interior"],
        header=(directory / "superquadric_fit_helpers.metal").read_text() + "\n" + (directory / "superflex_geometry.metal").read_text() + "\n" + """
            float sq_logadd(float a, float b) {
                float m = max(a, b);
                return m == -INFINITY ? m : m + log1p(exp(-abs(a-b)));
            }
        """,
        source=Path(__file__).with_name("superquadric_consensus.metal").read_text(),
    )


@lru_cache(maxsize=1)
def _interior_mass_kernel():
    return _metal_runtime().fast.metal_kernel(
        name="superquadric_coherent_interior_mass",
        input_names=["interior", "neighbors"],
        output_names=["mass_partials"],
        source=Path(__file__).with_name("superquadric_interior_mass.metal").read_text(),
    )


@lru_cache(maxsize=1)
def _uncertain_indices_kernel():
    return _metal_runtime().fast.metal_kernel(
        name="superquadric_consensus_uncertain_indices",
        input_names=["masks", "prefix"],
        output_names=["indices"],
        source="""
            uint i = thread_position_in_grid.x;
            if (i < uint(masks_shape[0]) && (masks[i] & 2)) {
                indices[prefix[i]-1] = i;
            }
        """,
    )


@dataclass
class MetalConsensusResult:
    counts: np.ndarray
    _masks: Any = field(repr=False)
    _corrections: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict, repr=False)
    scores: np.ndarray | None = None
    interior_masses: np.ndarray | None = None
    _active_indices: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self.scores is None:
            self.scores = self.counts.astype(np.float64)

    @property
    def best_index(self) -> int:
        if not self.counts.size:
            raise ValueError("An empty consensus batch has no winner")
        if self.interior_masses is not None:
            return best_score_index(self.scores)
        # NumPy's first maximum preserves RANSAC candidate order for ties.
        return int(np.argmax(self.counts))

    def mask(self, index: int) -> np.ndarray:
        if not 0 <= index < self.counts.size:
            raise IndexError("consensus candidate index is out of range")
        mx = _metal_runtime()
        row = self._masks[index]
        if self._active_indices is not None:
            row = row[self._active_indices]
        winner_mask = row == 1
        mx.eval(winner_mask)
        result = np.array(winner_mask, dtype=bool)
        if index in self._corrections:
            point_indices, values = self._corrections[index]
            result[point_indices] = values
        return result


class MetalConsensusContext:
    """Keep one cloud and its normalized normals resident on the Metal GPU.

    A batch returns only counts and sparse float64 boundary corrections to the
    CPU. Its masks remain on the GPU until the winning mask is requested.
    Subset views share the original cloud and its neighbor graph for interior scoring.
    """

    def __init__(
        self, points: np.ndarray, normals: np.ndarray | None = None,
        interior_context: InteriorPenaltyContext | None = None,
    ):
        mx = _metal_runtime()
        if mx is None:
            raise RuntimeError("Metal consensus requires Apple silicon, an accessible GPU and MLX")
        if interior_context is not None and interior_context.weight == 0.0:
            interior_context = None
        self._source_points = np.asarray(points)
        self.points = np.asarray(self._source_points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not np.isfinite(self.points).all():
            raise ValueError("points must be a finite array with shape (N, 3)")
        self._source_normals = None if normals is None else np.asarray(normals)
        self.normals = None if normals is None else np.asarray(self._source_normals, dtype=np.float64)
        if self.normals is not None and (self.normals.shape != self.points.shape or not np.isfinite(self.normals).all()):
            raise ValueError("normals must be finite and have the same shape as points")
        self.point_count = len(self.points)
        self.interior_context = interior_context
        self._reference_points = self.points
        self._reference_normals = self.normals
        self._reference_count = self.point_count
        self._global_to_local = None
        self._gpu_active_indices = None
        self._gpu_active = mx.ones((self.point_count if interior_context is not None else 1,), dtype=mx.uint8)
        self._gpu_neighbors = None
        if interior_context is not None:
            if not np.array_equal(self.points, interior_context.points):
                raise ValueError("Interior scoring must reference the original point cloud")
            self._gpu_neighbors = mx.array(interior_context.neighbor_matrix)
        # Center in float64 before conversion, including clouds far from the world origin.
        self.origin = self.points.mean(axis=0) if self.point_count else np.zeros(3)
        self.scale = max(float(np.linalg.norm(np.ptp(self.points, axis=0))), 1e-12) if self.point_count else 1.0
        self._gpu_points = mx.array(((self.points - self.origin) / self.scale).astype(np.float32))
        if self.normals is None:
            self._gpu_normals = mx.array(np.zeros((1, 3), dtype=np.float32))
        else:
            normal_lengths = np.maximum(np.linalg.norm(self.normals, axis=1, keepdims=True), 1e-9)
            self._gpu_normals = mx.array((self.normals / normal_lengths).astype(np.float32))
        mx.eval(self._gpu_points, self._gpu_normals, self._gpu_active)
        if self._gpu_neighbors is not None:
            mx.eval(self._gpu_neighbors)

    def for_subset(self, points: np.ndarray, normals: np.ndarray | None, indices: np.ndarray):
        if self.interior_context is None:
            raise ValueError("Subset views require an original-cloud interior context")
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) != len(points) or np.any(indices < 0) or np.any(indices >= self._reference_count):
            raise ValueError("Subset indices must map every active point to the original cloud")
        if len(np.unique(indices)) != len(indices) or not np.array_equal(points, self._reference_points[indices]):
            raise ValueError("Subset points must match their original-cloud indices without duplicates")
        if (normals is None) != (self._reference_normals is None) or (
            normals is not None and not np.array_equal(normals, self._reference_normals[indices])
        ):
            raise ValueError("Subset normals must match the original-cloud normals")
        view = copy(self)
        view._source_points = np.asarray(points)
        view.points = np.asarray(points, dtype=np.float64)
        view._source_normals = None if normals is None else np.asarray(normals)
        view.normals = None if normals is None else np.asarray(normals, dtype=np.float64)
        view.point_count = len(indices)
        view._global_to_local = np.full(self._reference_count, -1, dtype=np.int64)
        view._global_to_local[indices] = np.arange(len(indices))
        mx = _metal_runtime()
        view._gpu_active = mx.array((view._global_to_local >= 0).astype(np.uint8))
        view._gpu_active_indices = mx.array(indices.astype(np.uint32))
        mx.eval(view._gpu_active, view._gpu_active_indices)
        return view

    def check_inputs(self, points: np.ndarray, normals: np.ndarray | None) -> None:
        def same_array(left, right):
            return left is right or (
                left is not None and right is not None
                and left.shape == right.shape and left.strides == right.strides
                and left.__array_interface__["data"][0] == right.__array_interface__["data"][0]
            )
        point_array = np.asarray(points)
        normal_array = None if normals is None else np.asarray(normals)
        if not (same_array(self.points, point_array) or same_array(self._source_points, point_array)):
            raise ValueError("Metal consensus context belongs to a different point cloud")
        if not (same_array(self.normals, normal_array) or same_array(self._source_normals, normal_array)):
            raise ValueError("Metal consensus context belongs to different normals")

    def evaluate(
        self,
        models: Sequence[SuperQuadricParams],
        threshold: float,
        normal_cos_threshold: float = 0.0,
    ) -> MetalConsensusResult:
        if any(model.parameter_count not in (11, 19) for model in models):
            raise ValueError("Metal consensus supports rigid and SuperFlex models")
        superflex = any(model.parameter_count == 19 for model in models)
        mx = _metal_runtime()
        batch_size = len(models)
        if not batch_size or not self.point_count:
            masses = None if self.interior_context is None else np.zeros(batch_size)
            return MetalConsensusResult(
                np.zeros(batch_size, dtype=np.int64), mx.zeros((batch_size, self.point_count), dtype=mx.uint8),
                interior_masses=masses,
            )
        parameter_rows = []
        for model in models:
            row = np.concatenate((
                np.array([model.a1, model.a2, model.a3]) / self.scale,
                [model.e1, model.e2],
                (model.t - self.origin) / self.scale,
                model.rotation_matrix().ravel(),
            ))
            if superflex:
                deformation = np.r_[model.taper, (model.bending_components() * self.scale).ravel()] if model.parameter_count == 19 else np.zeros(8)
                row = np.r_[row, deformation]
            parameter_rows.append(row)
        parameters = np.stack(parameter_rows).astype(np.float32)
        score_interior = self.interior_context is not None and self.interior_context.weight > 0.0
        if score_interior and (not np.isfinite(threshold) or threshold <= 0.0):
            raise ValueError("interior scoring requires a finite, positive threshold")
        point_count = self._reference_count
        tiles = (point_count + CONSENSUS_THREADS - 1) // CONSENSUS_THREADS
        options = np.array([
            threshold / self.scale, normal_cos_threshold,
            1e-12 / self.scale, np.log(self.scale),
        ], dtype=np.float32)
        masks, partials, interior = _consensus_kernel(superflex)(
            inputs=[self._gpu_points, self._gpu_normals, mx.array(parameters), mx.array(options), self._gpu_active],
            template=[("THREADS", CONSENSUS_THREADS), ("HAS_NORMALS", self.normals is not None), ("SCORE_INTERIOR", score_interior), ("SUPERFLEX", superflex)],
            grid=(tiles * CONSENSUS_THREADS, batch_size, 1),
            threadgroup=(CONSENSUS_THREADS, 1, 1),
            output_shapes=[(batch_size, point_count), (batch_size, tiles, 2), (batch_size, point_count) if score_interior else (1,)],
            output_dtypes=[mx.uint8, mx.uint32, mx.float32], stream=mx.gpu,
        )
        reduced = mx.sum(partials, axis=1)
        masses = None
        if score_interior:
            mass_partials, = _interior_mass_kernel()(
                inputs=[interior, self._gpu_neighbors], template=[("THREADS", CONSENSUS_THREADS)],
                grid=(tiles * CONSENSUS_THREADS, batch_size, 1), threadgroup=(CONSENSUS_THREADS, 1, 1),
                output_shapes=[(batch_size, tiles)], output_dtypes=[mx.float32], stream=mx.gpu,
            )
            masses = mx.sum(mass_partials, axis=1)
            mx.eval(reduced, masses)
            masses = np.array(masses, dtype=np.float64)
        else:
            mx.eval(reduced)
        host_counts = np.array(reduced, dtype=np.int64)
        counts = host_counts[:, 0].copy()
        corrections = {}
        uncertain_count = int(host_counts[:, 1].sum())
        if uncertain_count and not superflex:
            # Compact only ambiguous indices on the GPU; never transfer all candidate masks.
            flat_masks = masks.reshape(-1)
            prefix = mx.cumsum(((flat_masks & 2) != 0).astype(mx.uint32))
            indices, = _uncertain_indices_kernel()(
                inputs=[flat_masks, prefix],
                grid=(batch_size * point_count, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(uncertain_count,)], output_dtypes=[mx.uint32], stream=mx.gpu,
            )
            mx.eval(indices)
            indices = np.array(indices, dtype=np.int64)
            candidate_indices = indices // point_count
            point_indices = indices % point_count
            from .consensus import compute_consensus
            for candidate in np.flatnonzero(host_counts[:, 1]):
                selected = point_indices[candidate_indices == candidate]
                values = compute_consensus(
                    models[candidate], self._reference_points[selected], threshold,
                    normals=None if self._reference_normals is None else self._reference_normals[selected],
                    normal_cos_threshold=normal_cos_threshold,
                )
                counts[candidate] += np.count_nonzero(values)
                local_indices = selected if self._global_to_local is None else self._global_to_local[selected]
                corrections[int(candidate)] = (local_indices, values)
        scores = None if masses is None else consensus_scores(counts, masses, self.interior_context.weight)
        return MetalConsensusResult(counts, masks, corrections, scores, masses, self._gpu_active_indices)


def create_metal_consensus_context(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    error_metric: str = "radial",
    interior_context: InteriorPenaltyContext | None = None,
    active_indices: np.ndarray | None = None,
    model_family: str = "rigid",
) -> MetalConsensusContext | None:
    validate_model_family(model_family)
    # Other residual metrics and platforms retain their existing CPU implementation.
    if model_family == "superflex":
        from .metal_superflex import require_superflex_metal
        require_superflex_metal()
        if error_metric != "radial":
            raise ValueError("SuperFlex Metal consensus supports the radial residual metric")
    if error_metric != "radial" or not metal_available():
        return None
    if interior_context is None or interior_context.weight == 0.0:
        return MetalConsensusContext(points, normals)
    reference_normals = interior_context.normals
    if reference_normals is None and normals is not None:
        if active_indices is None:
            reference_normals = normals
        else:
            reference_normals = np.zeros_like(interior_context.points)
            reference_normals[active_indices] = normals
    context = MetalConsensusContext(interior_context.points, reference_normals, interior_context)
    return context if active_indices is None else context.for_subset(points, normals, active_indices)
