import numpy as np
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.superquadric_residual import superquadric_normal_world,superquadric_residual_vector
from .metal_consensus import MetalConsensusContext

DEFAULT_NORMAL_COS_THRESHOLD = 0.0


def distance_err(
    model: SuperQuadricParams,
    points: np.ndarray,
    error_metric: str = "radial",
) -> np.ndarray:
    d = superquadric_residual_vector(model, points, metric=error_metric)
    return np.abs(d)

def normal_alignment_score(
    model: SuperQuadricParams,
    points: np.ndarray,
    normals: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    model_normals = superquadric_normal_world(model, points)
    normal_norm = np.linalg.norm(normals, axis=1, keepdims=True)
    normal_norm = np.maximum(normal_norm, 1e-9)
    normals_unit = normals / normal_norm
    return np.clip(np.einsum("ij,ij->i", model_normals, normals_unit), -1.0, 1.0)


def compute_consensus(
    model: SuperQuadricParams,
    points: np.ndarray,
    threshold: float,
    error_metric: str = "radial",
    normals: np.ndarray | None = None,
    normal_cos_threshold: float | None = None,
    metal_context: MetalConsensusContext | None = None,
) -> np.ndarray[bool]:
    if metal_context is not None:
        if error_metric != "radial":
            raise ValueError("Metal consensus supports the radial residual metric")
        metal_context.check_inputs(points, normals)
        cos_threshold = DEFAULT_NORMAL_COS_THRESHOLD if normal_cos_threshold is None else float(normal_cos_threshold)
        return metal_context.evaluate([model], threshold, cos_threshold).mask(0)
    err = distance_err(model, points, error_metric=error_metric)
    inliers = err < threshold
    
    if normals is not None:
        cos_threshold = DEFAULT_NORMAL_COS_THRESHOLD if normal_cos_threshold is None else float(normal_cos_threshold)
        _filter_normal_alignment(inliers, model, points, normals, cos_threshold)
    
    return inliers


def _filter_normal_alignment(mask, model, points, normals, cos_threshold):
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    if normals.shape != points.shape:
        raise ValueError("normals must have the same shape as points")
    # Model normals are needed only for points that already passed the residual test.
    selected = np.flatnonzero(mask)
    if selected.size:
        mask[selected] &= normal_alignment_score(model, points[selected], normals[selected]) >= cos_threshold

def expanded_removal_mask(
    model: SuperQuadricParams,
    points: np.ndarray,
    threshold: float,
    factor: float = 1.5,
    error_metric: str = "radial",
    normals: np.ndarray | None = None,
    normal_cos_threshold: float | None = None,
) -> np.ndarray:
    err = distance_err(model, points, error_metric=error_metric)
    remove_mask = err <= factor * threshold
    if normals is not None:
        cos_threshold = DEFAULT_NORMAL_COS_THRESHOLD if normal_cos_threshold is None else float(normal_cos_threshold)
        _filter_normal_alignment(remove_mask, model, points, normals, cos_threshold)
    return remove_mask
