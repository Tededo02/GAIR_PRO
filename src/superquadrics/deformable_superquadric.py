# SuperFlex geometry: https://arxiv.org/html/2607.01015, Section 3 and Appendix A.1.

from dataclasses import dataclass, field

import numpy as np

from .superquadric_param import SuperQuadricParams


def _axis_order(axis: int) -> list[int]:
    return [(axis + 1) % 3, (axis + 2) % 3, axis]


def bend(points: np.ndarray, components: np.ndarray, axis: int) -> np.ndarray:
    order = _axis_order(axis)
    local = np.asarray(points, dtype=np.float64)[:, order]
    curvature = float(np.linalg.norm(components))
    if curvature == 0.0:
        return np.asarray(points, dtype=np.float64).copy()
    direction = components / curvature
    radial = local[:, :2] @ direction
    angle = curvature * local[:, 2]
    # sinc and the half-angle identity preserve the zero-curvature limit.
    sine_over_k = local[:, 2] * np.sinc(angle / np.pi)
    one_minus_cos_over_k = 0.5 * curvature * local[:, 2]**2 * np.sinc(angle / (2 * np.pi))**2
    shift = one_minus_cos_over_k - radial * (2.0 * np.sin(angle / 2.0)**2)
    result = local.copy()
    result[:, :2] += shift[:, None] * direction
    result[:, 2] = sine_over_k - radial * np.sin(angle)
    return result[:, np.argsort(order)]


def inverse_bend(
    points: np.ndarray,
    components: np.ndarray,
    axis: int,
    derivatives: bool = False,
):
    order = _axis_order(axis)
    local = np.asarray(points, dtype=np.float64)[:, order]
    components = np.asarray(components, dtype=np.float64)
    curvature = float(np.linalg.norm(components))
    xy, z = local[:, :2], local[:, 2]
    result = local.copy()
    spatial = np.broadcast_to(np.eye(3), (len(local), 3, 3)).copy() if derivatives else None
    parameter = np.zeros((len(local), 3, 2)) if derivatives else None
    if curvature * max(float(np.max(np.abs(local), initial=0.0)), 1.0) < 1e-5:
        # Cartesian curvature removes the undefined bending angle at k=0.
        dot = xy @ components
        k2 = curvature**2
        result[:, :2] -= 0.5 * z[:, None]**2 * components * (1.0 + dot[:, None])
        result[:, 2] = z * (1.0 + dot + dot**2) - k2 * z**3 / 3.0
        if derivatives:
            spatial[:, :2, :2] -= 0.5 * z[:, None, None]**2 * np.outer(components, components)
            spatial[:, :2, 2] = -z[:, None] * components * (1.0 + dot[:, None])
            spatial[:, 2, :2] = z[:, None] * (1.0 + 2.0 * dot[:, None]) * components
            spatial[:, 2, 2] = 1.0 + dot + dot**2 - k2 * z**2
            parameter[:, :2, :] = -0.5 * z[:, None, None]**2 * (
                (1.0 + dot[:, None, None]) * np.eye(2) + components[None, :, None] * xy[:, None, :]
            )
            parameter[:, 2, :] = z[:, None] * (1.0 + 2.0 * dot[:, None]) * xy - (2.0 / 3.0) * z[:, None]**3 * components
    else:
        direction = components / curvature
        perpendicular = np.array([-direction[1], direction[0]])
        radial = xy @ direction
        h = 1.0 - curvature * radial
        length = np.maximum(np.hypot(h, curvature * z), 1e-12)
        numerator = 2.0 * radial - curvature * (radial**2 + z**2)
        unbent_radial = numerator / (1.0 + length)
        shift = unbent_radial - radial
        angle = np.arctan2(curvature * z, h)
        result[:, :2] += shift[:, None] * direction
        result[:, 2] = angle / curvature
        if derivatives:
            shift_r = h / length - 1.0
            shift_z = -curvature * z / length
            longitudinal_r = curvature * z / length**2
            longitudinal_z = h / length**2
            spatial[:, :2, :2] += shift_r[:, None, None] * np.outer(direction, direction)
            spatial[:, :2, 2] = shift_z[:, None] * direction
            spatial[:, 2, :2] = longitudinal_r[:, None] * direction
            spatial[:, 2, 2] = longitudinal_z
            length_k = (-radial * h + curvature * z**2) / length
            shift_k = -(radial**2 + z**2) / (1.0 + length) - numerator * length_k / (1.0 + length)**2
            longitudinal_k = (curvature * z / length**2 - angle) / curvature**2
            radial_alpha = xy @ perpendicular
            alpha_derivative = np.column_stack((
                shift_r[:, None] * radial_alpha[:, None] * direction + shift[:, None] * perpendicular,
                longitudinal_r * radial_alpha,
            ))
            k_derivative = np.column_stack((shift_k[:, None] * direction, longitudinal_k))
            parameter = k_derivative[:, :, None] * direction + alpha_derivative[:, :, None] * perpendicular / curvature
    inverse_order = np.argsort(order)
    result = result[:, inverse_order]
    if not derivatives:
        return result
    return result, spatial[:, inverse_order][:, :, inverse_order], parameter[:, inverse_order]


@dataclass
class DeformableSuperQuadricParams(SuperQuadricParams):
    taper: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    bending: np.ndarray = field(default_factory=lambda: np.zeros((3, 2), dtype=np.float64))

    def __post_init__(self) -> None:
        self.taper = np.asarray(self.taper, dtype=np.float64).reshape(2).copy()
        # Each row stores (curvature, plane angle), in x, y, z order.
        self.bending = np.asarray(self.bending, dtype=np.float64).reshape(3, 2).copy()
        super().__post_init__()

    def validate(self) -> None:
        super().validate()
        if not np.isfinite(self.taper).all() or np.any(np.abs(self.taper) >= 1.0):
            raise ValueError("taper must contain two finite coefficients strictly between -1 and 1")
        if not np.isfinite(self.bending).all():
            raise ValueError("bending must contain finite (curvature, angle) pairs")

    @property
    def parameter_count(self) -> int:
        return 19

    def bending_components(self) -> np.ndarray:
        return self.bending[:, :1] * np.column_stack((np.cos(self.bending[:, 1]), np.sin(self.bending[:, 1])))

    def deform(self, canonical_points: np.ndarray) -> np.ndarray:
        points = np.asarray(canonical_points, dtype=np.float64).copy()
        points[:, :2] *= 1.0 + points[:, 2:3] * self.taper / self.a3
        components = self.bending_components()
        # Reverse the inverse composition specified by SuperFlex: taper, y, x, z.
        for axis in (1, 0, 2):
            points = bend(points, components[axis], axis)
        return points

    def inverse_deformation(self, local_points: np.ndarray, derivatives: bool = False):
        points = np.asarray(local_points, dtype=np.float64)
        spatial = np.broadcast_to(np.eye(3), (len(points), 3, 3)).copy() if derivatives else None
        parameter = np.zeros((len(points), 3, 19)) if derivatives else None
        components = self.bending_components()
        for axis in (2, 0, 1):
            if derivatives:
                points, step_spatial, step_parameter = inverse_bend(points, components[axis], axis, True)
                spatial = step_spatial @ spatial
                parameter = step_spatial @ parameter
                parameter[:, :, 13 + 2 * axis:15 + 2 * axis] += step_parameter
            else:
                points = inverse_bend(points, components[axis], axis)
        raw_denominator = 1.0 + points[:, 2:3] * self.taper / self.a3
        denominator = np.maximum(raw_denominator, 1e-6)
        result = points.copy()
        result[:, :2] /= denominator
        if not derivatives:
            return result
        active = raw_denominator > 1e-6
        step_spatial = np.broadcast_to(np.eye(3), (len(points), 3, 3)).copy()
        step_spatial[:, 0, 0] = 1.0 / denominator[:, 0]
        step_spatial[:, 1, 1] = 1.0 / denominator[:, 1]
        step_spatial[:, :2, 2] = -(active * points[:, :2] * self.taper / self.a3 / denominator**2)
        spatial = step_spatial @ spatial
        parameter = step_spatial @ parameter
        parameter[:, :2, 2] += active * points[:, :2] * self.taper * points[:, 2:3] / self.a3**2 / denominator**2
        for axis in (0, 1):
            parameter[:, axis, 11 + axis] -= active[:, axis] * points[:, axis] * points[:, 2] / self.a3 / denominator[:, axis]**2
        return result, spatial, parameter

    def inverse_deform(self, local_points: np.ndarray) -> np.ndarray:
        return self.inverse_deformation(local_points)

    def gradient_to_world(self, local_points: np.ndarray, canonical_gradient: np.ndarray) -> np.ndarray:
        _, spatial, _ = self.inverse_deformation(local_points, derivatives=True)
        local_gradient = np.einsum("ni,nij->nj", canonical_gradient, spatial)
        return local_gradient @ self.rotation_matrix().T


def deformable_radial_residual_and_jacobian(model, points, eps=1e-12):
    from .superquadric_residual import _rotation_matrix_and_derivatives, superquadric_radial_residual_and_jacobian

    centered = np.asarray(points, dtype=np.float64) - model.t
    rotation, *rotation_derivatives = _rotation_matrix_and_derivatives(model.rot)
    local = centered @ rotation
    canonical, spatial, parameter = model.inverse_deformation(local, derivatives=True)
    rigid = SuperQuadricParams(model.a1, model.a2, model.a3, model.e1, model.e2)
    canonical_residual, canonical_jacobian = superquadric_radial_residual_and_jacobian(rigid, canonical, eps)
    canonical_radius = np.maximum(np.linalg.norm(canonical, axis=1), eps)
    radius = np.maximum(np.linalg.norm(local, axis=1), eps)
    factor = canonical_residual / canonical_radius
    radius_gradient = np.where((canonical_radius > eps)[:, None], canonical / canonical_radius[:, None], 0.0)
    factor_gradient = (-canonical_jacobian[:, 8:11] - factor[:, None] * radius_gradient) / canonical_radius[:, None]
    local_parameter = np.zeros_like(parameter)
    for column, derivative in enumerate(rotation_derivatives, start=5):
        local_parameter[:, :, column] = centered @ derivative
    local_parameter[:, :, 8:11] = -rotation.T
    parameter += spatial @ local_parameter
    jacobian = radius[:, None] * np.einsum("ni,nip->np", factor_gradient, parameter)
    jacobian[:, :5] += (radius / canonical_radius)[:, None] * canonical_jacobian[:, :5]
    jacobian[:, 8:11] -= factor[:, None] * np.where((radius > eps)[:, None], centered / radius[:, None], 0.0)
    return radius * factor, jacobian
