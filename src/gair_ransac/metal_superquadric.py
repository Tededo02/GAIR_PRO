from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import platform

import numpy as np


@dataclass
class MetalBatchFitResult:
    parameters: np.ndarray
    diagnostics: np.ndarray

    @property
    def success(self) -> np.ndarray:
        return (
            (self.diagnostics[:, 0] > 0)
            & np.isfinite(self.diagnostics[:, 2])
            & np.isfinite(self.parameters).all(axis=1)
        )


@lru_cache(maxsize=1)
def _metal_runtime():
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    try:
        import mlx.core as mx
    except ImportError:
        return None
    return mx if mx.metal.is_available() else None


def metal_available() -> bool:
    return _metal_runtime() is not None


@lru_cache(maxsize=1)
def _fit_kernel():
    mx = _metal_runtime()
    if mx is None:
        raise RuntimeError("Metal fitting requires Apple silicon, an accessible GPU and MLX; run uv sync")
    directory = Path(__file__).parent
    return mx.fast.metal_kernel(
        name="superquadric_bounded_soft_l1_batch_fit",
        input_names=["points", "initial", "bounds", "options"],
        output_names=["fitted", "diagnostics"],
        header=(directory / "superquadric_fit_helpers.metal").read_text(),
        source=(directory / "superquadric_fit.metal").read_text(),
    )


def fit_superquadric_metal_batch(
    points: np.ndarray,
    initial_parameters: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    robust_loss_scale: float | np.ndarray,
    length_scale: float | np.ndarray,
    max_nfev: int | np.ndarray = 1000,
) -> MetalBatchFitResult:
    """Optimize independent (B, N, 3) samples in one Metal dispatch.

    Bounds may be shared (11,) or supplied per fit (B, 11). Failed fits are
    identified by result.success without discarding successful batch entries.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 11:
        raise ValueError("points must have shape (B, N, 3) with at least 11 points per fit")
    batch_size, point_count, _ = points.shape
    initial_parameters = np.asarray(initial_parameters, dtype=np.float64)
    if initial_parameters.shape != (batch_size, 11):
        raise ValueError("initial_parameters must have shape (B, 11)")
    if batch_size == 0:
        return MetalBatchFitResult(np.empty((0, 11)), np.empty((0, 3)))
    scales = np.broadcast_to(np.asarray(length_scale, dtype=np.float64), (batch_size,)).copy()
    loss_scales = np.broadcast_to(np.asarray(robust_loss_scale, dtype=np.float64), (batch_size,))
    limits = np.broadcast_to(np.asarray(max_nfev, dtype=np.float64), (batch_size,))
    if not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("length_scale must be finite and positive")
    if not np.isfinite(loss_scales).all() or np.any(loss_scales <= 0):
        raise ValueError("robust_loss_scale must be finite and positive")
    if not np.isfinite(limits).all() or np.any(limits < 1) or np.any(limits != np.floor(limits)):
        raise ValueError("max_nfev must contain positive integers")
    lower_bounds = np.broadcast_to(np.asarray(lower_bounds, dtype=np.float64), (batch_size, 11))
    upper_bounds = np.broadcast_to(np.asarray(upper_bounds, dtype=np.float64), (batch_size, 11))

    # Normalize before converting to float32 to preserve small shapes at large world coordinates.
    origins = initial_parameters[:, 8:11].copy()
    normalized_points = (points - origins[:, None, :]) / scales[:, None, None]
    normalized_initial = initial_parameters.copy()
    normalized_lower = lower_bounds.copy()
    normalized_upper = upper_bounds.copy()
    for parameters in (normalized_initial, normalized_lower, normalized_upper):
        parameters[:, :3] /= scales[:, None]
        parameters[:, 8:11] = (parameters[:, 8:11] - origins) / scales[:, None]
    normalized_initial = np.clip(normalized_initial, normalized_lower, normalized_upper)

    # Forty-point samples use two complete SIMD groups; each fit owns its threadgroup state.
    threads = 32 if point_count <= 32 else 64
    options = np.column_stack((loss_scales / scales, 1e-12 / scales, limits)).astype(np.float32)
    mx = _metal_runtime()
    if mx is None:
        raise RuntimeError("Metal fitting requires Apple silicon, an accessible GPU and MLX; run uv sync")
    kernel = _fit_kernel()
    fitted, diagnostics = kernel(
        inputs=[
            mx.array(normalized_points.astype(np.float32)),
            mx.array(normalized_initial.astype(np.float32)),
            mx.array(np.concatenate((normalized_lower, normalized_upper), axis=1).astype(np.float32)),
            mx.array(options),
        ],
        template=[("THREADS", threads)],
        grid=(threads, batch_size, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(batch_size, 11), (batch_size, 3)],
        output_dtypes=[mx.float32, mx.float32],
        stream=mx.gpu,
    )
    mx.eval(fitted, diagnostics)
    parameters = np.array(fitted, dtype=np.float64)
    parameters[:, :3] *= scales[:, None]
    parameters[:, 8:11] = parameters[:, 8:11] * scales[:, None] + origins
    return MetalBatchFitResult(np.clip(parameters, lower_bounds, upper_bounds), np.array(diagnostics))


def fit_superquadric_metal(
    points: np.ndarray,
    initial_parameters: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    robust_loss_scale: float,
    length_scale: float,
    max_nfev: int = 1000,
) -> np.ndarray:
    result = fit_superquadric_metal_batch(
        np.asarray(points)[None, :, :],
        np.asarray(initial_parameters)[None, :],
        lower_bounds,
        upper_bounds,
        robust_loss_scale,
        length_scale,
        max_nfev,
    )
    status, evaluations, _ = result.diagnostics[0]
    if not result.success[0]:
        reason = "maximum function evaluations exceeded" if status == 0 else "non-finite optimization state"
        raise RuntimeError(f"Metal least-squares failed after {int(evaluations)} evaluations: {reason}")
    return result.parameters[0]
