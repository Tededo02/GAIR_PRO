from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.superquadrics.superquadric_param import SuperQuadricParams
from .metal_superquadric import _metal_runtime, metal_available


CONSENSUS_THREADS = 256


@lru_cache(maxsize=1)
def _consensus_kernel():
    mx = _metal_runtime()
    if mx is None:
        raise RuntimeError("Metal consensus requires Apple silicon, an accessible GPU and MLX")
    return mx.fast.metal_kernel(
        name="superquadric_radial_consensus_batch",
        input_names=["points", "normals", "parameters", "options"],
        output_names=["masks", "partials"],
        header="""
            float sq_logadd(float a, float b) {
                float m = max(a, b);
                return m == -INFINITY ? m : m + log1p(exp(-abs(a-b)));
            }
        """,
        source=Path(__file__).with_name("superquadric_consensus.metal").read_text(),
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

    @property
    def best_index(self) -> int:
        if not self.counts.size:
            raise ValueError("An empty consensus batch has no winner")
        # NumPy's first maximum preserves RANSAC candidate order for ties.
        return int(np.argmax(self.counts))

    def mask(self, index: int) -> np.ndarray:
        if not 0 <= index < self.counts.size:
            raise IndexError("consensus candidate index is out of range")
        mx = _metal_runtime()
        winner_mask = self._masks[index] == 1
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
    Create a new context after changing the cloud or its normals.
    """

    def __init__(self, points: np.ndarray, normals: np.ndarray | None = None):
        mx = _metal_runtime()
        if mx is None:
            raise RuntimeError("Metal consensus requires Apple silicon, an accessible GPU and MLX")
        self._source_points = np.asarray(points)
        self.points = np.asarray(self._source_points, dtype=np.float64)
        if self.points.ndim != 2 or self.points.shape[1] != 3 or not np.isfinite(self.points).all():
            raise ValueError("points must be a finite array with shape (N, 3)")
        self._source_normals = None if normals is None else np.asarray(normals)
        self.normals = None if normals is None else np.asarray(self._source_normals, dtype=np.float64)
        if self.normals is not None and (self.normals.shape != self.points.shape or not np.isfinite(self.normals).all()):
            raise ValueError("normals must be finite and have the same shape as points")
        self.point_count = len(self.points)
        # Center in float64 before conversion, including clouds far from the world origin.
        self.origin = self.points.mean(axis=0) if self.point_count else np.zeros(3)
        self.scale = max(float(np.linalg.norm(np.ptp(self.points, axis=0))), 1e-12) if self.point_count else 1.0
        self._gpu_points = mx.array(((self.points - self.origin) / self.scale).astype(np.float32))
        if self.normals is None:
            self._gpu_normals = mx.array(np.zeros((1, 3), dtype=np.float32))
        else:
            normal_lengths = np.maximum(np.linalg.norm(self.normals, axis=1, keepdims=True), 1e-9)
            self._gpu_normals = mx.array((self.normals / normal_lengths).astype(np.float32))
        mx.eval(self._gpu_points, self._gpu_normals)

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
        mx = _metal_runtime()
        batch_size = len(models)
        if not batch_size or not self.point_count:
            return MetalConsensusResult(np.zeros(batch_size, dtype=np.int64), mx.zeros((batch_size, self.point_count), dtype=mx.uint8))
        parameters = np.stack([
            np.concatenate((
                np.array([model.a1, model.a2, model.a3]) / self.scale,
                [model.e1, model.e2],
                (model.t - self.origin) / self.scale,
                model.rotation_matrix().ravel(),
            ))
            for model in models
        ]).astype(np.float32)
        tiles = (self.point_count + CONSENSUS_THREADS - 1) // CONSENSUS_THREADS
        options = np.array([
            threshold / self.scale, normal_cos_threshold,
            1e-12 / self.scale, np.log(self.scale),
        ], dtype=np.float32)
        masks, partials = _consensus_kernel()(
            inputs=[self._gpu_points, self._gpu_normals, mx.array(parameters), mx.array(options)],
            template=[("THREADS", CONSENSUS_THREADS), ("HAS_NORMALS", self.normals is not None)],
            grid=(tiles * CONSENSUS_THREADS, batch_size, 1),
            threadgroup=(CONSENSUS_THREADS, 1, 1),
            output_shapes=[(batch_size, self.point_count), (batch_size, tiles, 2)],
            output_dtypes=[mx.uint8, mx.uint32], stream=mx.gpu,
        )
        reduced = mx.sum(partials, axis=1)
        mx.eval(reduced)
        host_counts = np.array(reduced, dtype=np.int64)
        counts = host_counts[:, 0].copy()
        corrections = {}
        uncertain_count = int(host_counts[:, 1].sum())
        if uncertain_count:
            # Compact only ambiguous indices on the GPU; never transfer all candidate masks.
            flat_masks = masks.reshape(-1)
            prefix = mx.cumsum(((flat_masks & 2) != 0).astype(mx.uint32))
            indices, = _uncertain_indices_kernel()(
                inputs=[flat_masks, prefix],
                grid=(batch_size * self.point_count, 1, 1), threadgroup=(256, 1, 1),
                output_shapes=[(uncertain_count,)], output_dtypes=[mx.uint32], stream=mx.gpu,
            )
            mx.eval(indices)
            indices = np.array(indices, dtype=np.int64)
            candidate_indices = indices // self.point_count
            point_indices = indices % self.point_count
            from .consensus import compute_consensus
            for candidate in np.flatnonzero(host_counts[:, 1]):
                selected = point_indices[candidate_indices == candidate]
                values = compute_consensus(
                    models[candidate], self.points[selected], threshold,
                    normals=None if self.normals is None else self.normals[selected],
                    normal_cos_threshold=normal_cos_threshold,
                )
                counts[candidate] += np.count_nonzero(values)
                corrections[int(candidate)] = (selected, values)
        return MetalConsensusResult(counts, masks, corrections)


def create_metal_consensus_context(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    error_metric: str = "radial",
) -> MetalConsensusContext | None:
    # Other residual metrics and platforms retain their existing CPU implementation.
    if error_metric != "radial" or not metal_available():
        return None
    return MetalConsensusContext(points, normals)
