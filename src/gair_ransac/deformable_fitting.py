# Bounded SuperFlex fitting uses analytic derivatives and Cartesian bending coordinates.

import numpy as np

from src.superquadrics.deformable_superquadric import DeformableSuperQuadricParams
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.superquadric_residual import superquadric_radial_residual_and_jacobian
from .axis_regularization import DEFAULT_AXIS_PENALTY_WEIGHT, axis_penalty_residual_and_jacobian, axis_support_box, validate_axis_penalty_weight


def _pack(model: SuperQuadricParams, length_scale: float) -> np.ndarray:
    parameters = np.zeros(19, dtype=np.float64)
    parameters[:11] = np.r_[model.a1, model.a2, model.a3, model.e1, model.e2, model.rot, model.t]
    if isinstance(model, DeformableSuperQuadricParams):
        parameters[11:13] = model.taper
        parameters[13:19] = model.bending_components().ravel() * length_scale
    return parameters


def _unpack(parameters: np.ndarray, length_scale: float) -> DeformableSuperQuadricParams:
    components = parameters[13:19].reshape(3, 2) / length_scale
    bending = np.column_stack((np.linalg.norm(components, axis=1), np.arctan2(components[:, 1], components[:, 0])))
    return DeformableSuperQuadricParams(
        *parameters[:5], rot=parameters[5:8], t=parameters[8:11],
        taper=parameters[11:13], bending=bending,
    )


def _rigid_axis_permutations(model: SuperQuadricParams):
    from .inner_ransac import _euler_from_rotation_matrices
    axes = np.array([model.a1, model.a2, model.a3])
    rotation = model.rotation_matrix()
    for order in ([0, 1, 2], [1, 2, 0], [2, 0, 1]):
        # Tapering acts along local z, so each principal axis needs an initialization.
        yield SuperQuadricParams(
            *axes[order], model.e1, model.e2,
            rot=_euler_from_rotation_matrices(rotation[:, order][None])[0], t=model.t,
        )


def fit_deformable_superquadric_ls(
    points: np.ndarray,
    bounds_reference_points: np.ndarray | None = None,
    axis_penalty_weight: float = DEFAULT_AXIS_PENALTY_WEIGHT,
    initial_model: SuperQuadricParams | None = None,
    backend: str = "auto",
) -> DeformableSuperQuadricParams:
    if backend not in {"auto", "metal", "cpu"}:
        raise ValueError("backend must be 'auto', 'metal' or 'cpu'")
    if backend == "cpu":
        return _fit_deformable_cpu(points, bounds_reference_points, axis_penalty_weight, initial_model)
    from .inner_ransac import _pca_initial_parameters
    from .metal_superflex import require_superflex_metal, superflex_optimization_bounds, fit_superflex_metal_batch, fit_superflex_hypothesis_batch

    axis_penalty_weight = validate_axis_penalty_weight(axis_penalty_weight)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite array with shape (N, 3)")
    if len(points) < 19:
        raise ValueError("at least 19 points are required for a SuperFlex fit")
    require_superflex_metal()
    reference = points if bounds_reference_points is None else np.asarray(bounds_reference_points, dtype=np.float64)
    lower, upper, loss_scale, diagonal, support_box = superflex_optimization_bounds(reference, axis_penalty_weight)
    if initial_model is None:
        result = fit_superflex_hypothesis_batch(
            points[None], _pca_initial_parameters(points[None]), lower, upper,
            loss_scale, diagonal, support_box, axis_penalty_weight,
        )
    else:
        initial_model.validate()
        result = fit_superflex_metal_batch(
            points[None], _pack(initial_model, diagonal)[None], lower, upper, loss_scale, diagonal,
            axis_penalty_weight=axis_penalty_weight, axis_support=support_box,
        )
    if not result.success[0]:
        raise RuntimeError(f"SuperFlex Metal fitting failed after {int(result.diagnostics[0, 1])} evaluations")
    return _unpack(result.parameters[0], diagonal)


def _fit_deformable_cpu(points, bounds_reference_points, axis_penalty_weight, initial_model):
    from scipy.optimize import least_squares
    from .inner_ransac import _optimization_bounds, fit_superquadric_ls, pca_initialization

    axis_penalty_weight = validate_axis_penalty_weight(axis_penalty_weight)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite array with shape (N, 3)")
    if len(points) < 19:
        raise ValueError("at least 19 points are required for a SuperFlex fit")
    reference = points if bounds_reference_points is None else np.asarray(bounds_reference_points, dtype=np.float64)
    lower, upper, loss_scale, diagonal = _optimization_bounds(reference)
    reference_model = pca_initialization(reference)
    support_box = axis_support_box(reference, reference_model, lower[0], loss_scale)
    # Axis identity changes across the three seeds; allow the existing global axis bound.
    upper[:3] = max(1e-2, 1.2 * diagonal)
    lower = np.r_[lower, [-0.999, -0.999], np.full(6, -4.0)]
    upper = np.r_[upper, [0.999, 0.999], np.full(6, 4.0)]
    if initial_model is not None:
        initial_model.validate()
        seeds = [initial_model]
    else:
        try:
            rigid = fit_superquadric_ls(points, bounds_reference_points=reference, backend="cpu", axis_penalty_weight=axis_penalty_weight)
        except (RuntimeError, np.linalg.LinAlgError):
            # A difficult rigid fit should not prevent the extended family from being tried.
            rigid = pca_initialization(points)
        seeds = list(_rigid_axis_permutations(rigid))
    penalty_scale = np.sqrt(len(points) * axis_penalty_weight) * loss_scale
    best_parameters, best_cost = None, np.inf

    for seed in seeds:
        cached_parameters, cached_residual, cached_jacobian = None, None, None

        def residual(parameters):
            nonlocal cached_parameters, cached_residual, cached_jacobian
            if cached_parameters is None or not np.array_equal(parameters, cached_parameters):
                model = _unpack(parameters, diagonal)
                data, jacobian = superquadric_radial_residual_and_jacobian(model, points)
                jacobian[:, 13:19] /= diagonal
                if axis_penalty_weight > 0.0:
                    penalty, penalty_jacobian = axis_penalty_residual_and_jacobian(parameters[:11], support_box, penalty_scale)
                    data = np.r_[data, penalty]
                    jacobian = np.vstack((jacobian, np.pad(penalty_jacobian, ((0, 0), (0, 8)))))
                cached_parameters = parameters.copy()
                cached_residual, cached_jacobian = data, jacobian
            return cached_residual

        def jacobian(parameters):
            residual(parameters)
            return cached_jacobian

        def loss(z):
            root = np.sqrt(1.0 + z)
            rho = np.vstack((2.0 * z / (root + 1.0), 1.0 / root, -0.5 / root**3))
            rho[:, -3:] = np.vstack((z[-3:], np.ones(3), np.zeros(3)))
            return rho

        x0 = np.clip(_pack(seed, diagonal), lower, upper)
        try:
            result = least_squares(
                residual, x0, jac=jacobian, bounds=(lower, upper),
                loss=loss if axis_penalty_weight > 0.0 else "soft_l1",
                f_scale=loss_scale, x_scale="jac", max_nfev=400,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            # An invalid numerical step in one orientation does not discard the other seeds.
            continue
        if result.success and np.isfinite(result.cost) and result.cost < best_cost:
            best_parameters, best_cost = result.x.copy(), float(result.cost)
        if best_cost < len(points) * (diagonal * 1e-6)**2:
            break
    if best_parameters is None:
        raise RuntimeError("SuperFlex least-squares fitting failed to converge")
    return _unpack(best_parameters, diagonal)
