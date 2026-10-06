import numpy as np

from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.superquadric_residual import _rotation_matrix_and_derivatives


DEFAULT_AXIS_PENALTY_WEIGHT = 0.4


def validate_axis_penalty_weight(weight: float) -> float:
    weight = float(weight)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("axis_penalty_weight must be finite and non-negative")
    return weight


def axis_support_box(
    points: np.ndarray,
    reference_model: SuperQuadricParams,
    min_axis_length: float,
    loss_scale: float,
) -> np.ndarray:
    # Project the reference bounding box into each fitted axis as its rotation changes.
    rotation = reference_model.rotation_matrix()
    local_points = (points - reference_model.t) @ rotation
    half_spans = 0.5 * np.ptp(local_points, axis=0)
    free_axes = np.maximum(1.1 * half_spans + 2.0 * loss_scale, min_axis_length)
    return rotation * free_axes[None, :]


def axis_penalty_residual_and_jacobian(
    parameters: np.ndarray,
    support_box: np.ndarray,
    residual_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    rotation, *derivatives = _rotation_matrix_and_derivatives(parameters[5:8])
    projections = support_box.T @ rotation
    support = np.sum(np.abs(projections), axis=0)
    excess = np.maximum(parameters[:3] / support - 1.0, 0.0)
    active = excess > 0.0
    jacobian = np.zeros((3, 11), dtype=np.float64)
    jacobian[np.arange(3), np.arange(3)] = residual_scale * active / support
    for column, derivative in enumerate(derivatives, start=5):
        support_derivative = np.sum(np.sign(projections) * (support_box.T @ derivative), axis=0)
        jacobian[:, column] = -residual_scale * active * parameters[:3] * support_derivative / support**2
    return residual_scale * excess, jacobian


def prefer_compact_model(
    candidate: SuperQuadricParams,
    candidate_count: int,
    best: SuperQuadricParams | None,
    best_count: int,
    min_gain: int = 1,
) -> bool:
    if candidate_count > best_count and candidate_count >= best_count + min_gain:
        return True
    if candidate_count != best_count or best is None:
        return False
    # A relative tolerance prevents equal-consensus local refinements from cycling on roundoff.
    candidate_size = candidate.a1**2 + candidate.a2**2 + candidate.a3**2
    best_size = best.a1**2 + best.a2**2 + best.a3**2
    return candidate_size < best_size * (1.0 - 1e-6)
