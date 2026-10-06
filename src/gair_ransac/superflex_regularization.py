# SuperFlex adds rotation-invariant axis size costs to the existing excess penalty.

import numpy as np

from src.superquadrics.superquadric_residual import _rotation_matrix_and_derivatives


SUPERFLEX_COMPACTNESS_WEIGHT = 0.01
SUPERFLEX_OVERSIZE_WEIGHT = 0.1


def superflex_axis_penalty_residual_and_jacobian(
    parameters: np.ndarray,
    support_box: np.ndarray,
    residual_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotation, *derivatives = _rotation_matrix_and_derivatives(parameters[5:8])
    projections = support_box.T @ rotation
    support = np.sum(np.abs(projections), axis=0)
    excess = np.maximum(parameters[:3] / support - 1.0, 0.0)
    reference_radius = np.linalg.norm(support_box)
    size_ratio = parameters[:3] / reference_radius
    size_cost = size_ratio * np.sqrt(SUPERFLEX_COMPACTNESS_WEIGHT + SUPERFLEX_OVERSIZE_WEIGHT * size_ratio**2)
    penalty = np.hypot(excess, size_cost)
    size_gradient = (SUPERFLEX_COMPACTNESS_WEIGHT * size_ratio + 2.0 * SUPERFLEX_OVERSIZE_WEIGHT * size_ratio**3) / reference_radius
    jacobian = np.zeros((3, 19), dtype=np.float64)
    jacobian[np.arange(3), np.arange(3)] = residual_scale * (excess / support + size_gradient) / penalty
    for column, derivative in enumerate(derivatives, start=5):
        support_derivative = np.sum(np.sign(projections) * (support_box.T @ derivative), axis=0)
        jacobian[:, column] = -residual_scale * excess * parameters[:3] * support_derivative / (penalty * support**2)
    # The squared residual gives quadratic size cost and quartic growth for oversized axes.
    return residual_scale * penalty, jacobian
