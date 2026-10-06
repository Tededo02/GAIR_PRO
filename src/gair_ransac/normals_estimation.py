import numpy as np


def _as_points(point_cloud: np.ndarray) -> np.ndarray:
    """Return point coordinates from an array-like, Open3D, or trimesh point cloud."""
    if hasattr(point_cloud, "points"):
        points = np.asarray(point_cloud.points, dtype=np.float64)
    elif hasattr(point_cloud, "vertices"):
        points = np.asarray(point_cloud.vertices, dtype=np.float64)
    else:
        points = np.asarray(point_cloud, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"point_cloud must contain points with shape (N, 3), got {points.shape}")
    if points.shape[0] < 3:
        raise ValueError(f"Need at least 3 points to estimate normals, got {points.shape[0]}")
    return points


def _normalized(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms > 0.0, norms, 1.0)


def _orient_normals_global_outward(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    center = np.median(points, axis=0)
    radial = points - center
    score = np.einsum("ij,ij->i", normals, radial, optimize=True)
    finite_score = score[np.isfinite(score)]
    if finite_score.size and float(np.median(finite_score)) < 0.0:
        return -normals
    return normals


def estimate_normals_open3d_consistent(
    point_cloud: np.ndarray,
    k_neighbors: int,
) -> np.ndarray:
    """
    Estimate and orient point-cloud normals with Open3D.

    This follows the normal-estimation path used in the plot branch:
    Open3D estimates local normals with k-NN and propagates a consistent tangent
    plane orientation. There is intentionally no PCA fallback.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required to estimate normals consistently; no PCA fallback is used."
        ) from exc

    points = _as_points(point_cloud)
    k = min(max(int(k_neighbors), 3), points.shape[0])

    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(points)
    point_cloud_o3d.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k)
    )
    point_cloud_o3d.orient_normals_consistent_tangent_plane(k)

    normals = np.asarray(point_cloud_o3d.normals, dtype=np.float64)
    if normals.shape != points.shape:
        raise RuntimeError(
            f"Open3D returned normals with shape {normals.shape}, expected {points.shape}"
        )
    if not np.isfinite(normals).all():
        raise RuntimeError("Open3D returned non-finite normal values")

    normals = _normalized(normals)
    return _orient_normals_global_outward(points, normals)
