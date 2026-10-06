from dataclasses import dataclass
import numpy as np
from scipy.spatial import KDTree, cKDTree
import pyvista as pv

# Constants
_CLOSED_FORM_SIZES = {
    'plane':  3,
    'sphere': 4,
    'circle': 3,
}

# How many points the algorithm picks for each patch
_PTS_PER_PATCH = 5

# Voxel side = median NN-spacing × this factor (spatial_walk_mss only).
# Really important, you can choose how big the single voxel cell is
_VOXEL_SPACING_FACTOR = 10


@dataclass(frozen=True)
class AdaptiveLocalFpsSamplerContext:
    data: np.ndarray
    points: np.ndarray
    normals: np.ndarray | None
    tree: cKDTree
    base_scale: float


def _normalize_normals(
    normals: np.ndarray | None,
    points_shape: tuple[int, ...],
) -> np.ndarray | None:
    if normals is None:
        return None

    normals_arr = np.asarray(normals, dtype=np.float64)
    if normals_arr.shape != points_shape:
        raise ValueError(f"normals must have shape {points_shape}, got {normals_arr.shape}")
    return normals_arr / (np.linalg.norm(normals_arr, axis=1, keepdims=True) + 1e-12)


def build_adaptive_local_fps_sampler_context(
    D: np.ndarray,
    normals: np.ndarray | None = None,
) -> AdaptiveLocalFpsSamplerContext:
    data = np.asarray(D)
    points = np.asarray(D, dtype=np.float64)
    normals_arr = _normalize_normals(normals, points.shape)
    tree = cKDTree(points)

    # Reuse the median nearest-neighbor spacing as a scale for sample compactness.
    n_points = int(points.shape[0])
    if n_points <= 1:
        base_scale = 1e-9
    else:
        nn_dists, _ = tree.query(points, k=2)
        nn_dists = np.asarray(nn_dists, dtype=np.float64)
        base_scale = max(float(np.median(nn_dists[:, 1])), 1e-9)

    return AdaptiveLocalFpsSamplerContext(
        data=data,
        points=points,
        normals=normals_arr,
        tree=tree,
        base_scale=base_scale,
    )

# this function is used to spread the sample inside the local pool
def _farthest_point_indices(
    points: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
    start_idx: int | None = None,
) -> np.ndarray:
    n_points = int(points.shape[0])
    if n_points <= sample_size:
        return np.arange(n_points, dtype=int)

    if start_idx is None:
        start_idx = int(rng.integers(n_points))

    selected = np.empty(sample_size, dtype=int)
    selected[0] = int(start_idx)

    diff = points - points[selected[0]]
    min_d2 = np.einsum("ij,ij->i", diff, diff)
    min_d2[selected[0]] = -1.0

    for i in range(1, sample_size):
        next_idx = int(np.argmax(min_d2))
        selected[i] = next_idx
        diff = points - points[next_idx]
        d2 = np.einsum("ij,ij->i", diff, diff)
        min_d2 = np.minimum(min_d2, d2)
        min_d2[selected[: i + 1]] = -1.0

    return selected

# This function gathers a local pool of candidate points around a seed,
#  using both spatial proximity and normal coherence.
def _coherent_local_pool_indices(points: np.ndarray,tree: cKDTree,seed_idx: int,target_pool_size: int,initial_k: int,normals: np.ndarray | None,
    normal_cos_min: float,
    tangent_cos_max: float,
) -> np.ndarray:
    n_points = int(points.shape[0])
    seed_point = points[seed_idx]
    query_k = min(max(initial_k, target_pool_size), n_points)

    if normals is None:
        _, idx = tree.query(seed_point, k=query_k)
        idx = np.atleast_1d(idx).astype(np.int64, copy=False)
        return idx[:target_pool_size]

    seed_normal = normals[seed_idx]
    relax_normal = [normal_cos_min, normal_cos_min - 0.2, normal_cos_min - 0.4, -1.0]
    relax_tangent = [tangent_cos_max, tangent_cos_max + 0.1, tangent_cos_max + 0.2, 1.0]

    for normal_thr, tangent_thr in zip(relax_normal, relax_tangent):
        current_k = query_k
        while True:
            _, idx = tree.query(seed_point, k=current_k)
            idx = np.atleast_1d(idx).astype(np.int64, copy=False)

            other_idx = idx[idx != seed_idx]
            if other_idx.size == 0:
                filtered = np.array([seed_idx], dtype=np.int64)
            else:
                disp = points[other_idx] - seed_point
                dist = np.linalg.norm(disp, axis=1)
                disp_unit = np.zeros_like(disp)
                valid = dist > 1e-12
                disp_unit[valid] = disp[valid] / dist[valid, None]

                normal_align = normals[other_idx] @ seed_normal
                tangent_align = np.abs(disp_unit @ seed_normal)
                keep = (normal_align >= normal_thr) & (tangent_align <= tangent_thr)
                filtered = np.concatenate(([seed_idx], other_idx[keep]))

            if filtered.size >= target_pool_size or current_k >= n_points:
                return filtered[:target_pool_size]
            current_k = min(current_k * 2, n_points)

    _, idx = tree.query(seed_point, k=min(n_points, target_pool_size))
    idx = np.atleast_1d(idx).astype(np.int64, copy=False)
    return idx[:target_pool_size]

# This function scores a local sample based on compactness and normal coherence.
# Lower is better. The compactness term encourages the sample to be spatially tight,
# while the coherence penalty encourages normals to be aligned with the seed normal.
# this is the formula to score: score = (mean distance to seed) / base_scale + 0.75 * (1 - mean normal alignment)
# The base_scale is typically the median nearest neighbor distance in the whole point cloud, which normalizes the compactness term to be scale-invariant.
# The normal alignment is the cosine of the angle between the sample normals and the seed normal, averaged over the sample. A value of 1 means perfect alignment,
# while 0 means orthogonal and -1 means opposite direction. The coherence penalty is then 1 minus this average alignment, 
# so it ranges from 0 (perfectly coherent) to 2 (completely incoherent).
def _local_sample_score(
    points: np.ndarray,
    normals: np.ndarray | None,
    sample_idx: np.ndarray,
    seed_idx: int,
    base_scale: float,
) -> float:
    seed_point = points[seed_idx]
    radial_dist = np.linalg.norm(points[sample_idx] - seed_point, axis=1)
    compactness = float(np.mean(radial_dist) / (base_scale + 1e-12))

    if normals is None:
        return compactness

    seed_normal = normals[seed_idx]
    normal_align = np.clip(normals[sample_idx] @ seed_normal, -1.0, 1.0)
    coherence_penalty = float(1.0 - normal_align.mean())
    return compactness + 0.75 * coherence_penalty

# Main function: adaptive local MSS with farthest-point sampling
def adaptive_local_fps_mss(
    D: np.ndarray,
    normals: np.ndarray | None = None,
    primitive_type: str = "superquadric",
    sample_size: int = 20,
    seed_tries: int = 40,
    candidate_multiplier: float = 4.0,
    initial_k: int = 96,
    normal_cos_min: float = 0.7,
    tangent_cos_max: float = 0.55,
    rng: np.random.Generator | None = None,
    random_seed: int | None = None,
    sampler_context: AdaptiveLocalFpsSamplerContext | None = None,
) -> np.ndarray:
    """
    Robust local MSS for detached, touching, and mildly overlapping shapes.

    Strategy:
    1. draw several random seeds
    2. gather a compact local k-NN pool around each seed
    3. filter neighbors using normal coherence and tangent compatibility
    4. spread the final MSS inside that pool with farthest-point sampling

    The sampler is local by construction, so unlike component-wide growth it
    does not cross an entire connected structure when two superquadrics touch.
    """
    if rng is None:
        rng = np.random.default_rng(random_seed)

    context = sampler_context if sampler_context is not None else build_adaptive_local_fps_sampler_context(D, normals)
    data = context.data
    points = context.points
    normals_arr = context.normals
    tree = context.tree
    base_scale = context.base_scale
    N = int(points.shape[0])

    if primitive_type in _CLOSED_FORM_SIZES:
        k = _CLOSED_FORM_SIZES[primitive_type]
        return data[rng.choice(N, size=min(k, N), replace=False)]

    if N <= sample_size:
        return data.copy()

    target_pool_size = min(N, max(sample_size, int(round(candidate_multiplier * sample_size))))
    query_k = min(N, max(initial_k, target_pool_size))

    best_score = np.inf
    best_sample_idx: np.ndarray | None = None
    n_trials = max(int(seed_tries), 1)

    for _ in range(n_trials):
        seed_idx = int(rng.integers(N))
        pool_idx = _coherent_local_pool_indices(
            points=points,
            tree=tree,
            seed_idx=seed_idx,
            target_pool_size=target_pool_size,
            initial_k=query_k,
            normals=normals_arr,
            normal_cos_min=normal_cos_min,
            tangent_cos_max=tangent_cos_max,
        )

        if pool_idx.size < sample_size:
            _, fallback_idx = tree.query(points[seed_idx], k=sample_size)
            pool_idx = np.atleast_1d(fallback_idx).astype(np.int64, copy=False)

        if pool_idx.size > target_pool_size:
            pool_idx = pool_idx[:target_pool_size]

        local_points = points[pool_idx]
        seed_pos = np.flatnonzero(pool_idx == seed_idx)
        start_idx = int(seed_pos[0]) if seed_pos.size > 0 else 0
        chosen_local = _farthest_point_indices(
            local_points,
            sample_size=sample_size,
            rng=rng,
            start_idx=start_idx,
        )
        sample_idx = pool_idx[chosen_local]
        score = _local_sample_score(
            points=points,
            normals=normals_arr,
            sample_idx=sample_idx,
            seed_idx=seed_idx,
            base_scale=base_scale,
        )
        if score < best_score:
            best_score = score
            best_sample_idx = sample_idx

    if best_sample_idx is None:
        chosen = rng.choice(N, size=sample_size, replace=False)
        best_sample_idx = np.asarray(chosen, dtype=np.int64)

    return data[best_sample_idx]


# Both functions return M_j
def spatial_walk_mss(
    D: np.ndarray,
    normals: np.ndarray | None = None,
    primitive_type: str = 'superquadric',
    sample_size: int = 50,
    pts_per_patch: int = _PTS_PER_PATCH,
    voxel_spacing_factor: int = _VOXEL_SPACING_FACTOR,
    rng: np.random.Generator | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    MSS via density-weighted spatial walk on a voxel grid.

    Picks a random anchor voxel (weighted by number of points), then expands
    to neighbouring voxels, again choosing randomly weighting by number of points until
    sample_size is reached.

    """
    if rng is None:
        rng = np.random.default_rng(random_seed)

    N = len(D)

    # closed-form primitives: just pick the minimal sample (for instance pick 3 points for circles)
    if primitive_type in _CLOSED_FORM_SIZES:
        k = _CLOSED_FORM_SIZES[primitive_type]
        return D[rng.choice(N, size=min(k, N), replace=False)]

    if N <= sample_size:
        return D.copy()


    # k=2: index 0 is self (dist=0), index 1 is true nearest neighbour
    nn_dists, _ = KDTree(D).query(D, k=2)
    voxel_size = max(float(np.median(nn_dists[:, 1])) * voxel_spacing_factor, 1e-9)

    # assign each point to a voxel (integer grid key)
    voxel_coords = np.floor(D / voxel_size).astype(int)
    voxel_map: dict[tuple, list[int]] = {}
    for i in range(N):
        key = tuple(voxel_coords[i])
        if key not in voxel_map:
            voxel_map[key] = []
        voxel_map[key].append(i)

    non_empty = list(voxel_map.keys())

    # anchor: pick seed voxel weighted by density
    counts  = np.array([len(voxel_map[v]) for v in non_empty], dtype=float)
    seed    = non_empty[rng.choice(len(non_empty), p=counts / counts.sum())]
    selected_indices: set[int]   = set()
    visited:          set[tuple] = set()
    frontier:         set[tuple] = {seed}

    while len(selected_indices) < sample_size and frontier:
        # pick next voxel from frontier, weighted by density
        # in the first loop frontier is just the seed, so it forces the algorithm to pick it.
        fl      = sorted(frontier)
        fc      = np.array([len(voxel_map[v]) for v in fl], dtype=float)
        voxel   = fl[rng.choice(len(fl), p=fc / fc.sum())]
        frontier.discard(voxel)

        if voxel in visited:
            continue
        visited.add(voxel)

        # sample up to pts_per_patch new points from this voxel
        cands = [i for i in voxel_map[voxel] if i not in selected_indices]
        n_add = min(pts_per_patch, len(cands), sample_size - len(selected_indices))
        if n_add > 0:
            selected_indices.update(int(i) for i in rng.choice(cands, size=n_add, replace=False))

        # expand frontier to 26-connected non-empty, unvisited neighbours (9 + 9 + 8, the 3D neighbors)
        vi, vj, vk = voxel
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di == dj == dk == 0:
                        continue
                    nbr = (vi + di, vj + dj, vk + dk)
                    if nbr in voxel_map and nbr not in visited:
                        frontier.add(nbr)

    # pad if the frontier ran out before reaching sample_size
    if len(selected_indices) < sample_size:
        remaining = [i for i in range(N) if i not in selected_indices]
        n_pad = min(sample_size - len(selected_indices), len(remaining))
        if n_pad > 0:
            selected_indices.update(int(i) for i in rng.choice(remaining, size=n_pad, replace=False))

    return D[np.array(sorted(selected_indices)[:sample_size], dtype=int)]


def uniform_partition_mss(
    D: np.ndarray,
    pts_per_patch: int = 50,
    rng: np.random.Generator | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Partitions the point cloud into equal-count patches, sample _PTS_PER_PATCH from each.

    Steps: Pick a random unassigned point, then collect its
    pts_per_patch nearest unassigned neighbours to generate a patch. 
    
    In short: Denser regions produce spatially smaller patches, 
    so coverage is more uniform over the surface.
    """
    if rng is None:
        rng = np.random.default_rng(random_seed)

    N = len(D)
    if N < _PTS_PER_PATCH:
        return D.copy()

    tree      = KDTree(D)
    unassigned = np.ones(N, dtype=bool)
    patches: list[list[int]] = []

    while unassigned.sum() >= pts_per_patch:
        seed_idx = int(rng.choice(np.where(unassigned)[0]))

        _, nn = tree.query(D[seed_idx], k=N)
        patch = [int(i) for i in nn if unassigned[i]][:pts_per_patch]

        if len(patch) < pts_per_patch:
            break

        patches.append(patch)
        unassigned[patch] = False

    selected: list[int] = []
    for patch in patches:
        chosen = rng.choice(patch, size=min(_PTS_PER_PATCH, len(patch)), replace=False)
        selected.extend(int(i) for i in chosen)

    return D[np.array(selected, dtype=int)]


def visualize_mss(D: np.ndarray, Mj: np.ndarray) -> None:
    """
    Show the point cloud with MSS points highlighted.

    Green : selected MSS points
    Red   : rest of the cloud, dimmed
    """
    pl = pv.Plotter()
    pl.set_background("white")

    mss_mask = np.zeros(len(D), dtype=bool)
    _, idx = KDTree(D).query(Mj, k=1)
    mss_mask[idx] = True

    point_size = 6

    if (~mss_mask).any():
        pl.add_points(
            D[~mss_mask].astype(np.float64),
            render_points_as_spheres=True,
            point_size=point_size,
            color="red",
            opacity=0.4,
        )

    if mss_mask.any():
        pl.add_points(
            D[mss_mask].astype(np.float64),
            render_points_as_spheres=True,
            point_size=point_size,
            color="green",
        )

    pl.show_axes()
    pl.add_axes()
    pl.show_grid(location="outer", ticks="outside", grid="front", all_edges=True)
    pl.show_bounds(
        grid="back", location="outer", all_edges=True, ticks="outside",
        xtitle="X", ytitle="Y", ztitle="Z", font_size=10,
    )
    pl.add_text(f"MSS: {mss_mask.sum()} / {len(D)} points", position="upper_left", font_size=10)
    pl.reset_camera()
    pl.enable_eye_dome_lighting()
    pl.show()
