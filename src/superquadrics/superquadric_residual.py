import numpy as np
from .superquadric_param import SuperQuadricParams, rotz, roty, rotx


def _drotz(yaw: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[-sy, -cy, 0.0],
         [cy, -sy, 0.0],
         [0.0, 0.0, 0.0]],
        dtype=float,
    )


def _droty(pitch: float) -> np.ndarray:
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array(
        [[-sp, 0.0, cp],
         [0.0, 0.0, 0.0],
         [-cp, 0.0, -sp]],
        dtype=float,
    )


def _drotx(roll: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [[0.0, 0.0, 0.0],
         [0.0, -sr, -cr],
         [0.0, cr, -sr]],
        dtype=float,
    )


def _rotation_matrix_and_derivatives(rot: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yaw, pitch, roll = rot
    rz = rotz(yaw)
    ry = roty(pitch)
    rx = rotx(roll)
    r = rz @ ry @ rx
    dr_dyaw = _drotz(yaw) @ ry @ rx
    dr_dpitch = rz @ _droty(pitch) @ rx
    dr_droll = rz @ ry @ _drotx(roll)
    return r, dr_dyaw, dr_dpitch, dr_droll


def _points_in_canonical_frame(
    model: SuperQuadricParams,
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rotation_matrix = model.rotation_matrix()
    centered_points = np.asarray(points, dtype=np.float64) - model.t
    return rotation_matrix, centered_points @ rotation_matrix


def _superquadric_shape_value_from_pc(
    model: SuperQuadricParams,
    pc: np.ndarray,
) -> np.ndarray:
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    a1, a2, a3, e1, e2 = model.a1, model.a2, model.a3, model.e1, model.e2
    return (
        (np.abs(x / a1) ** (2.0 / e2) + np.abs(y / a2) ** (2.0 / e2)) ** (e2 / e1)
        + np.abs(z / a3) ** (2.0 / e1)
    )


# compute gradient
def _superquadric_gradient_canonical_from_pc(
    model: SuperQuadricParams,
    pc: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    a1, a2, a3 = model.a1, model.a2, model.a3
    e1, e2 = model.e1, model.e2

    p_xy = 2.0 / e2
    p_z = 2.0 / e1
    k = e2 / e1

    ax = np.maximum(np.abs(x / a1), eps)
    ay = np.maximum(np.abs(y / a2), eps)
    az = np.maximum(np.abs(z / a3), eps)

    u = ax ** p_xy + ay ** p_xy
    u = np.maximum(u, eps)

    du_dx = p_xy * (ax ** (p_xy - 1.0)) * (np.sign(x) / a1)
    du_dy = p_xy * (ay ** (p_xy - 1.0)) * (np.sign(y) / a2)
    dv_du = k * (u ** (k - 1.0))

    dfdx = dv_du * du_dx
    dfdy = dv_du * du_dy
    dfdz = p_z * (az ** (p_z - 1.0)) * (np.sign(z) / a3)
    return np.stack([dfdx, dfdy, dfdz], axis=1)

# compute the implicit function value for each point
# return an array of shape (N,) with the distance from each point to the superquadric surface
# the distance is computed as the absolute value of the implicit function value, which is zero on the surface and positive outside, negative inside
def implicit_f_superquadric_residual(model: SuperQuadricParams, points: np.ndarray) -> np.ndarray:
    _, points_canonical = _points_in_canonical_frame(model, points)
    return _superquadric_shape_value_from_pc(model, points_canonical) - 1.0

# compute the first order residual for each point, which is the implicit function value divided by the norm of the gradient of the implicit function
# this gives a better approximation of the distance to the surface, especially for points close to the surface, and is more stable for optimization
def superquadric_first_order_residual(model: SuperQuadricParams, points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    _, pc = _points_in_canonical_frame(model, points)
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    a1, a2, a3 = model.a1, model.a2, model.a3
    e1, e2 = model.e1, model.e2
    #build the function and its gradient in a way that is stable for optimization
    # powers
    p_xy = 2.0 / e2
    p_z = 2.0 / e1
    k = e2 / e1

    # safe abs to avoid 0**(neg) when exponents go < 1
    ax = np.maximum(np.abs(x / a1), eps)
    ay = np.maximum(np.abs(y / a2), eps)
    az = np.maximum(np.abs(z / a3), eps)
    u = ax ** p_xy + ay ** p_xy
    u = np.maximum(u, eps)

    v = u ** k
    w = az ** p_z
    f = v + w - 1.0

    grad_canonical = _superquadric_gradient_canonical_from_pc(model, pc, eps)
    grad_norm = np.linalg.norm(grad_canonical, axis=1)
    grad_norm = np.maximum(grad_norm, 1e-9)
    return f / grad_norm

# return the normal vector of the superquadric at each point, computed as the normalized gradient of the implicit function
def superquadric_normal_world(
    model: SuperQuadricParams,
    points: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    R, pc = _points_in_canonical_frame(model, points)
    grad_canonical = _superquadric_gradient_canonical_from_pc(model, pc, eps)
    grad_world = grad_canonical @ R.T
    grad_norm = np.linalg.norm(grad_world, axis=1, keepdims=True)
    grad_norm = np.maximum(grad_norm, 1e-9)
    return grad_world / grad_norm

# computes the ray from the center of the superquadric to each point in space
# then finds the respective point on the surface
# then computes the length of the segment from the surface to the point
def  superquadric_radial_residual(model: SuperQuadricParams, points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    _, pc = _points_in_canonical_frame(model, points)
    r = np.maximum(np.linalg.norm(pc, axis=1), eps)
    shape_value = _superquadric_shape_value_from_pc(model, pc)
    r_surface = r * np.maximum(shape_value, eps) ** (-model.e1 / 2.0)
    return r - r_surface

# Compute the radial residual together with its analytic Jacobian w.r.t. the 11
# superquadric parameters. This is used by least_squares to avoid finite-difference
# derivatives, reducing the cost of each fit while keeping the same objective.
def superquadric_radial_residual_and_jacobian(
    model: SuperQuadricParams,
    points: np.ndarray,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    R, dR_dyaw, dR_dpitch, dR_droll = _rotation_matrix_and_derivatives(model.rot)
    centered = points - model.t
    pc = centered @ R
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]

    a1, a2, a3 = model.a1, model.a2, model.a3
    e1, e2 = model.e1, model.e2

    qx = x / a1
    qy = y / a2
    qz = z / a3
    abs_qx = np.abs(qx)
    abs_qy = np.abs(qy)
    abs_qz = np.abs(qz)

    active_ax = abs_qx > eps
    active_ay = abs_qy > eps
    active_az = abs_qz > eps
    ax = np.where(active_ax, abs_qx, eps)
    ay = np.where(active_ay, abs_qy, eps)
    az = np.where(active_az, abs_qz, eps)

    p_xy = 2.0 / e2
    p_z = 2.0 / e1
    k = e2 / e1
    h = -0.5 * e1

    A = ax ** p_xy
    B = ay ** p_xy
    u_raw = A + B
    active_u = u_raw > eps
    u = np.where(active_u, u_raw, eps)
    v = u ** k
    w = az ** p_z
    s_raw = v + w
    active_s = s_raw > eps
    s = np.where(active_s, s_raw, eps)

    r_raw = np.sqrt(x * x + y * y + z * z)
    active_r = r_raw > eps
    r = np.where(active_r, r_raw, eps)
    sh = s ** h
    residual = r * (1.0 - sh)

    scale_u = active_u.astype(np.float64)
    scale_s = active_s.astype(np.float64)
    inv_u = np.where(active_u, 1.0 / u, 0.0)
    inv_s = np.where(active_s, 1.0 / s, 0.0)
    inv_r = np.where(active_r, 1.0 / r_raw, 0.0)
    grad_r_pc = pc * inv_r[:, None]

    sign_qx = np.sign(qx)
    sign_qy = np.sign(qy)
    sign_qz = np.sign(qz)
    sign_x_over_a1 = np.where(active_ax, sign_qx / a1, 0.0)
    sign_y_over_a2 = np.where(active_ay, sign_qy / a2, 0.0)
    sign_z_over_a3 = np.where(active_az, sign_qz / a3, 0.0)

    coeff_A = np.where(active_ax, A * (p_xy / ax), 0.0)
    coeff_B = np.where(active_ay, B * (p_xy / ay), 0.0)
    coeff_W = np.where(active_az, w * (p_z / az), 0.0)
    log_ax = np.log(ax)
    log_ay = np.log(ay)
    log_az = np.log(az)
    log_u = np.log(u)
    log_s = np.log(s)

    def _dres_from_dpc(dpc: np.ndarray) -> np.ndarray:
        dr = np.einsum("ij,ij->i", grad_r_pc, dpc, optimize=True)
        dax = sign_x_over_a1 * dpc[:, 0]
        day = sign_y_over_a2 * dpc[:, 1]
        daz = sign_z_over_a3 * dpc[:, 2]
        dA = coeff_A * dax
        dB = coeff_B * day
        du = scale_u * (dA + dB)
        dv = v * (k * inv_u) * du
        dw = coeff_W * daz
        ds = scale_s * (dv + dw)
        dsh = sh * (h * inv_s) * ds
        return dr * (1.0 - sh) - r * dsh

    jac = np.empty((points.shape[0], 11), dtype=np.float64)

    dax_da1 = np.where(active_ax, -abs_qx / a1, 0.0)
    dA_da1 = coeff_A * dax_da1
    du_da1 = scale_u * dA_da1
    dv_da1 = v * (k * inv_u) * du_da1
    ds_da1 = scale_s * dv_da1
    jac[:, 0] = -r * sh * (h * inv_s) * ds_da1

    day_da2 = np.where(active_ay, -abs_qy / a2, 0.0)
    dB_da2 = coeff_B * day_da2
    du_da2 = scale_u * dB_da2
    dv_da2 = v * (k * inv_u) * du_da2
    ds_da2 = scale_s * dv_da2
    jac[:, 1] = -r * sh * (h * inv_s) * ds_da2

    daz_da3 = np.where(active_az, -abs_qz / a3, 0.0)
    dw_da3 = coeff_W * daz_da3
    ds_da3 = scale_s * dw_da3
    jac[:, 2] = -r * sh * (h * inv_s) * ds_da3

    dk_de1 = -k / e1
    dn_de1 = -p_z / e1
    dh_de1 = -0.5
    dv_de1 = v * (dk_de1 * log_u)
    dw_de1 = w * (dn_de1 * log_az)
    ds_de1 = scale_s * (dv_de1 + dw_de1)
    dsh_de1 = sh * (dh_de1 * log_s + (h * inv_s) * ds_de1)
    jac[:, 3] = -r * dsh_de1

    dp_xy_de2 = -p_xy / e2
    dk_de2 = 1.0 / e1
    dA_de2 = A * (dp_xy_de2 * log_ax)
    dB_de2 = B * (dp_xy_de2 * log_ay)
    du_de2 = scale_u * (dA_de2 + dB_de2)
    dv_de2 = v * (dk_de2 * log_u + (k * inv_u) * du_de2)
    ds_de2 = scale_s * dv_de2
    jac[:, 4] = -r * sh * (h * inv_s) * ds_de2

    jac[:, 5] = _dres_from_dpc(centered @ dR_dyaw)
    jac[:, 6] = _dres_from_dpc(centered @ dR_dpitch)
    jac[:, 7] = _dres_from_dpc(centered @ dR_droll)
    jac[:, 8] = _dres_from_dpc(np.broadcast_to(-R[0], pc.shape))
    jac[:, 9] = _dres_from_dpc(np.broadcast_to(-R[1], pc.shape))
    jac[:, 10] = _dres_from_dpc(np.broadcast_to(-R[2], pc.shape))
    return residual, jac

# uses first-order residual for points whose direction is close to one of the 3 axes
# (horizontal/vertical), and radial residual from center for diagonal points.
# axis-alignment is measured in ellipsoid-normalised coordinates: if one component
# dominates (> AXIS_THRESHOLD after normalisation) the point is considered axis-aligned.
def superquadric_combo(model: SuperQuadricParams, points: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    R = model.rotation_matrix()
    a1, a2, a3 = model.a1, model.a2, model.a3
    AXIS_THRESHOLD = 0.7 # sensitivity of the axis distance (1 is full radial, 0 is full first order)

    pc = (points - model.t) @ R 

    # direction normalised by semi-axes so the measure is shape-aware
    n = np.stack([np.abs(pc[:, 0] / a1),
                  np.abs(pc[:, 1] / a2),
                  np.abs(pc[:, 2] / a3)], axis=1)
    n_norm = np.maximum(np.linalg.norm(n, axis=1, keepdims=True), eps)
    n = n / n_norm  # each row is a unit vector; component > threshold → near that axis

    use_axis   = n.max(axis=1) > AXIS_THRESHOLD
    use_center = ~use_axis

    residuals = np.zeros(len(points), dtype=np.float64)

    if use_center.any():
        residuals[use_center] = superquadric_radial_residual(model, points[use_center], eps)

    if use_axis.any():
        residuals[use_axis] = superquadric_first_order_residual(model, points[use_axis], eps)

    return residuals


def superquadric_residual_vector(
    model: SuperQuadricParams,
    points: np.ndarray,
    metric: str = "radial",
    eps: float = 1e-12,
) -> np.ndarray:
    metric_name = metric.lower().replace("-", "_").replace(" ", "_")

    if metric_name == "radial":
        return superquadric_radial_residual(model, points, eps)

    if metric_name == "first_order":
        return superquadric_first_order_residual(model, points, eps)

    if metric_name == "mix":
        return superquadric_combo(model, points, eps)

    raise ValueError(f"Unsupported residual metric: {metric}")
