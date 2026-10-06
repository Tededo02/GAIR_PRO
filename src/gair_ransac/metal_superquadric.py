from functools import lru_cache
from pathlib import Path
import platform

import numpy as np


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
        name="superquadric_bounded_soft_l1_fit",
        input_names=["points", "initial", "bounds", "options"],
        output_names=["fitted", "diagnostics"],
        header=(directory / "superquadric_fit_helpers.metal").read_text(),
        source=(directory / "superquadric_fit.metal").read_text(),
    )


def fit_superquadric_metal(
    points: np.ndarray,
    initial_parameters: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    robust_loss_scale: float,
    length_scale: float,
    max_nfev: int = 1000,
) -> np.ndarray:
    mx = _metal_runtime()
    kernel = _fit_kernel()

    # Normalize before converting to float32 to preserve small shapes at large world coordinates.
    origin = initial_parameters[8:11].copy()
    normalized_points = (points - origin) / length_scale
    normalized_initial = initial_parameters.copy()
    normalized_lower = lower_bounds.copy()
    normalized_upper = upper_bounds.copy()
    for parameters in (normalized_initial, normalized_lower, normalized_upper):
        parameters[:3] /= length_scale
        parameters[8:11] = (parameters[8:11] - origin) / length_scale
    normalized_initial = np.clip(normalized_initial, normalized_lower, normalized_upper)

    # A single SIMD group keeps every optimization iteration on the GPU, including the linear solve.
    fitted, diagnostics = kernel(
        inputs=[
            mx.array(normalized_points.astype(np.float32)),
            mx.array(normalized_initial.astype(np.float32)),
            mx.array(np.concatenate((normalized_lower, normalized_upper)).astype(np.float32)),
            mx.array([robust_loss_scale / length_scale, 1e-12 / length_scale, max_nfev], dtype=mx.float32),
        ],
        grid=(32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(11,), (3,)],
        output_dtypes=[mx.float32, mx.float32],
        stream=mx.gpu,
    )
    mx.eval(fitted, diagnostics)
    status, evaluations, cost = np.asarray(diagnostics)
    if status <= 0 or not np.isfinite(cost):
        reason = "maximum function evaluations exceeded" if status == 0 else "non-finite optimization state"
        raise RuntimeError(f"Metal least-squares failed after {int(evaluations)} evaluations: {reason}")
    parameters = np.array(fitted, dtype=np.float64)
    parameters[:3] *= length_scale
    parameters[8:11] = parameters[8:11] * length_scale + origin
    if not np.isfinite(parameters).all():
        raise RuntimeError("Metal least-squares returned non-finite parameters")
    return np.clip(parameters, lower_bounds, upper_bounds)
