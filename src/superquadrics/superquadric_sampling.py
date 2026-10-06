import numpy as np
from trimesh import Trimesh
import trimesh


def _make_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _next_surface_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int32).max))


# sample points on the surface of the superquadric meshes
# uniformly in area. each mesh in the list is sampled with n_points points
# the result is a list of arrays of points
def sampling_sq(
    mesh_list: list[Trimesh],
    n_points: int = 1000,
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = _make_rng(seed)
    points_list: list[np.ndarray] = []
    normals_list: list[np.ndarray] = []
    for mesh in mesh_list:
        pts, face_idx = trimesh.sample.sample_surface(
            mesh,
            n_points,
            seed=_next_surface_seed(rng),
        ) # pts are type numpy array of shape (n_points,3) and coordinates are type float64
        normals = mesh.face_normals[face_idx].astype(np.float64, copy=False)
        points_list.append(pts)
        normals_list.append(normals)
    return points_list, normals_list

def sampling_sq_random(mesh_list: list[Trimesh], n_points: int = 1000, seed: int | None = None) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = _make_rng(seed)
    points_list: list[np.ndarray] = []
    normals_list: list[np.ndarray] = []
    for mesh in mesh_list:
        pts, face_idx = trimesh.sample.sample_surface(
            mesh,
            n_points,
            seed=_next_surface_seed(rng),
        ) # pts are type numpy array of shape (n_points,3) and coordinates are type float64
        normals = mesh.face_normals[face_idx].astype(np.float64, copy=False)
        points_list.append(pts)
        normals_list.append(normals)
    return points_list, normals_list

# sample points on the surface of the superquadric meshes with noise along the normal direction.
# Returned normals can also be perturbed with Gaussian noise around the exact direction.
def sampling_sq_noisy(mesh_list: list[Trimesh],n_points: int = 1000,noise_std: float = 0.01,normal_noise_std: float | None = None,clip_k: float | None = 3.0,seed: int | None = None,) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = _make_rng(seed)
    points_list: list[np.ndarray] = []
    normals_list: list[np.ndarray] = []
    normal_noise_std = 0.0 if normal_noise_std is None else normal_noise_std

    for mesh in mesh_list:
        # sample points on surface + get which face each point came from
        pts, face_idx = trimesh.sample.sample_surface(
            mesh,
            n_points,
            seed=_next_surface_seed(rng),
        )  # pts: (N,3), face_idx: (N,)
        pts = pts.astype(np.float64, copy=False)
        # get the corresponding face normals (one normal per sampled point)
        normals = mesh.face_normals[face_idx].astype(np.float64, copy=False)  # (N,3), already unit-length
        # daw 1D Gaussian noise amplitudes (one scalar per point)
        alpha = rng.normal(loc=0.0, scale=noise_std, size=(n_points, 1)).astype(np.float64)  # (N,1)
        # clip to avoid rare large jumps
        if clip_k is not None:
            alpha = np.clip(alpha, -clip_k * noise_std, clip_k * noise_std)
        # move each point only along its normal
        pts_noisy = pts + normals * alpha  # (N,3)
        # perturb normals with additive Gaussian noise centered on the exact direction
        if normal_noise_std > 0.0:
            normal_delta = rng.normal(loc=0.0, scale=normal_noise_std, size=normals.shape).astype(np.float64)
            if clip_k is not None:
                normal_delta = np.clip(normal_delta, -clip_k * normal_noise_std, clip_k * normal_noise_std)
            normals_noisy = normals + normal_delta
            normals_noisy /= np.linalg.norm(normals_noisy, axis=1, keepdims=True) + 1e-12
            normals_noisy = normals_noisy.astype(np.float64, copy=False)
        else:
            normals_noisy = normals

        points_list.append(pts_noisy)
        normals_list.append(normals_noisy)

    return points_list, normals_list



#Returns outliers sampled inside the global bounding box of the input meshes.
#The sampling distribution is independent from the mesh geometry (only bbox is used).
def sampling_outliers(meshes: list[trimesh.Trimesh],n_out: int = 400,margin: float = 0.10,
    mode: str = "uniform",  # "uniform" , "mog"
    n_clusters: int = 5,
    cluster_std_frac: float = 0.1,
    seed: int | None = None,
) ->tuple[ np.ndarray[np.ndarray], np.ndarray[np.ndarray]]:
    if n_out <= 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    rng = np.random.default_rng(seed)

    mins = np.min([m.bounds[0] for m in meshes], axis=0).astype(np.float64)
    maxs = np.max([m.bounds[1] for m in meshes], axis=0).astype(np.float64)

    diag = np.linalg.norm(maxs - mins)
    pad = margin * diag
    low = mins - pad
    high = maxs + pad
    size = high - low
    mode = mode.lower()

    #generate n_out normals for the outliers randomly oriented in the 3D space
    outlier_normals = rng.normal(size=(n_out, 3)).astype(np.float64)
    outlier_normals /= np.linalg.norm(outlier_normals, axis=1, keepdims=True) + 1e-12  # normalize to unit length

    if mode == "uniform":
        pts = rng.uniform(low=low, high=high, size=(n_out, 3))
        return pts.astype(np.float64),outlier_normals

    if mode == "mog":
        if n_clusters <= 0:
            raise ValueError("n_clusters must be >= 1.")
        if cluster_std_frac <= 0:
            raise ValueError("cluster_std_frac must be > 0.")

        centers_u = rng.uniform(0.0, 1.0, size=(n_clusters, 3))
        centers = low + centers_u * size

        std = cluster_std_frac * diag
        if std <= 0:
            std = 1e-6

        idx = rng.integers(0, n_clusters, size=(n_out,))
        pts = centers[idx] + rng.normal(0.0, std, size=(n_out, 3))

        # clip to bbox so points always stay inside
        pts = np.clip(pts, low, high)
        return pts.astype(np.float64),outlier_normals

    raise ValueError(f"Unknown mode '{mode}'. Use: 'uniform', 'mog'.")
