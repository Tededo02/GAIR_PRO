# SuperFlex keeps dimensionless Cartesian curvature in the six final fit coordinates.

import numpy as np

from .metal_superquadric import MetalBatchFitResult, fit_superquadric_metal_batch, metal_available
from .superflex_bounds import extend_superflex_bounds


class SuperflexMetalError(RuntimeError):
    pass


def require_superflex_metal() -> None:
    if not metal_available():
        raise SuperflexMetalError("SuperFlex requires an accessible Metal GPU and MLX; backend='cpu' is available only when explicitly requested")


def _gpu_fit(*args, **kwargs):
    try:
        return fit_superquadric_metal_batch(*args, **kwargs)
    except RuntimeError as error:
        raise SuperflexMetalError(f"SuperFlex GPU dispatch failed: {error}") from error


def fit_superflex_metal_batch(
    points, initial_parameters, lower_bounds, upper_bounds,
    robust_loss_scale, length_scale, max_nfev=400,
    axis_penalty_weight=0.0, axis_support=None,
) -> MetalBatchFitResult:
    require_superflex_metal()
    return _gpu_fit(
        points, initial_parameters, lower_bounds, upper_bounds,
        robust_loss_scale, length_scale, max_nfev,
        axis_penalty_weight=axis_penalty_weight, axis_support=axis_support,
        model_family="superflex",
    )


def superflex_optimization_bounds(reference_points, axis_penalty_weight):
    from .inner_ransac import _optimization_bounds, pca_initialization
    from .axis_regularization import axis_support_box, validate_axis_penalty_weight

    validate_axis_penalty_weight(axis_penalty_weight)
    lower, upper, loss_scale, diagonal = _optimization_bounds(reference_points)
    reference_model = pca_initialization(reference_points)
    support_box = axis_support_box(reference_points, reference_model, lower[0], loss_scale)
    upper[:3] = max(1e-2, 1.2 * diagonal)
    lower, upper = extend_superflex_bounds(lower, upper)
    return lower, upper, loss_scale, diagonal, support_box


def superflex_axis_initializations(rigid_parameters):
    from .inner_ransac import _euler_from_rotation_matrices, _model_from_parameters

    rigid_parameters = np.asarray(rigid_parameters, dtype=np.float64)
    batch_size = len(rigid_parameters)
    rotations = np.stack([_model_from_parameters(p).rotation_matrix() for p in rigid_parameters])
    initial = np.zeros((batch_size, 3, 19), dtype=np.float64)
    for variant, order in enumerate(([0, 1, 2], [1, 2, 0], [2, 0, 1])):
        initial[:, variant, :11] = rigid_parameters
        initial[:, variant, :3] = rigid_parameters[:, order]
        initial[:, variant, 5:8] = _euler_from_rotation_matrices(rotations[:, :, order])
    return initial


def fit_superflex_hypothesis_batch(
    points, initial_rigid_parameters, lower, upper, loss_scale, diagonal,
    support_box, axis_penalty_weight,
) -> MetalBatchFitResult:
    require_superflex_metal()
    batch_size = len(points)
    if batch_size == 0:
        return MetalBatchFitResult(np.empty((0, 19)), np.empty((0, 3)))
    warm = _gpu_fit(
        points, initial_rigid_parameters, lower[:11], upper[:11], loss_scale, diagonal,
        axis_penalty_weight=axis_penalty_weight, axis_support=support_box,
    )
    # Failed rigid warm starts retain PCA seeds; no optimization is retried on the CPU.
    rigid = np.where(warm.success[:, None], warm.parameters, initial_rigid_parameters)
    initial = superflex_axis_initializations(rigid)
    extended = fit_superflex_metal_batch(
        np.repeat(points, 3, axis=0), initial.reshape(-1, 19), lower, upper, loss_scale, diagonal,
        axis_penalty_weight=axis_penalty_weight, axis_support=support_box,
    )
    costs = np.where(extended.success, extended.diagnostics[:, 2], np.inf).reshape(batch_size, 3)
    winner = np.argmin(costs, axis=1)
    indices = 3 * np.arange(batch_size) + winner
    return MetalBatchFitResult(extended.parameters[indices], extended.diagnostics[indices])
