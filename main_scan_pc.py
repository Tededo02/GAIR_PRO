import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from src.gair_ransac.energy_strategies import FullGairEnergy, GcRansacEnergy
from src.gair_ransac.gair_ransac import gair_ransac
from src.gair_ransac.normals_estimation import estimate_normals_open3d_consistent
from src.superquadrics import superquadric_mesh as supmesh
from src.visualizations import plot as vis


PC_NAME = "etp_no_floor.ply"


ALGORITHM_NAME = "gair"
ROOT = Path(__file__).resolve().parent
PC_DIR = ROOT / "test_objects" / "real_pc"
TEST_OBJECTS_DIR = ROOT / "test_objects"
SUPPORTED_INPUT_EXTENSIONS = {".ply", ".stl"}
K_NEIGHBORS = 90
THRESHOLD = 0.010 # if 0 use point spacing to compute effective threshold
THRESHOLD_SPACING_FACTOR = 2.0
M_NEIGHBORS = 8
MAX_MODELS = 8
MAX_ITERATIONS = 40
INNER_ITERATIONS = 40
SAMPLE_SIZE = 30
MIN_INLIERS = 600
MIN_COVERAGE = 0.3
MSS_MAX_POOL_FRACTION = 0.18
RANDOM_SEED = 12345777
MESH_SAMPLE_COUNT = 8000


def resolve_input_path(input_file: str | Path) -> Path:
    input_path = Path(input_file).expanduser()
    candidates = [input_path] if input_path.is_absolute() else [
        ROOT / input_path,
        PC_DIR / input_path,
        TEST_OBJECTS_DIR / input_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    available_inputs = sorted(
        path.relative_to(ROOT)
        for extension in SUPPORTED_INPUT_EXTENSIONS
        for path in ROOT.rglob(f"*{extension}")
    )
    available_hint = ", ".join(str(path) for path in available_inputs[:10])
    raise FileNotFoundError(
        f"Input file not found: {input_path}\n"
        f"Available supported inputs in repo: {available_hint}"
    )


def load_trimesh_geometry(path: Path) -> trimesh.Trimesh | trimesh.PointCloud:
    geometry = trimesh.load(str(path), process=False)
    if isinstance(geometry, trimesh.Scene):
        if not geometry.geometry:
            raise ValueError(f"Scene has no geometry: {path}")
        geometry = trimesh.util.concatenate(tuple(geometry.geometry.values()))
    return geometry


def sample_mesh_surface(
    mesh: trimesh.Trimesh,
    n_points: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if n_points <= 0:
        raise ValueError(f"mesh sample count must be positive, got {n_points}")
    if len(mesh.faces) == 0:
        raise ValueError("STL input has no faces to sample")

    points, face_indices = trimesh.sample.sample_surface(
        mesh,
        n_points,
        seed=seed,
    )
    normals = np.asarray(mesh.face_normals[face_indices], dtype=np.float64)
    return np.asarray(points, dtype=np.float64), normals


def load_point_cloud(
    path: Path,
    mesh_sample_count: int = MESH_SAMPLE_COUNT,
    seed: int | None = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_INPUT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_INPUT_EXTENSIONS))
        raise ValueError(f"Unsupported input extension '{suffix}'. Supported: {supported}")

    cloud = load_trimesh_geometry(path)

    if suffix == ".stl":
        if not isinstance(cloud, trimesh.Trimesh):
            raise TypeError(f"STL input did not load as a mesh: {path}")
        points, normals = sample_mesh_surface(cloud, mesh_sample_count, seed=seed)
        return points, None, normals

    points = np.asarray(cloud.vertices, dtype=np.float64)
    if len(points) == 0:
        raise ValueError(f"Point cloud has no points: {path}")

    colors = getattr(cloud, "colors", None)
    if colors is None and hasattr(cloud, "visual"):
        colors = getattr(cloud.visual, "vertex_colors", None)

    if colors is not None and len(colors) == len(points):
        colors = np.asarray(colors, dtype=np.uint8)
    else:
        colors = None

    return points, colors, None


def estimate_point_spacing(points: np.ndarray) -> float:
    point_array = np.asarray(points, dtype=np.float64)
    if point_array.shape[0] < 2:
        return 0.0

    tree = cKDTree(point_array)
    nn_dists, _ = tree.query(point_array, k=2)
    nn_dists = np.asarray(nn_dists, dtype=np.float64)
    return float(np.median(nn_dists[:, 1]))


def compute_effective_threshold(points: np.ndarray) -> tuple[float, float]:
    point_spacing = estimate_point_spacing(points)
    effective_threshold = float(THRESHOLD) if THRESHOLD > 0 else float(THRESHOLD_SPACING_FACTOR) * point_spacing
    return effective_threshold, point_spacing


def combine_inlier_masks(inliers_masks: list[np.ndarray]) -> np.ndarray | None:
    if not inliers_masks:
        return None

    inlier_mask = np.asarray(inliers_masks[0], dtype=bool).copy()
    for mask in inliers_masks[1:]:
        inlier_mask |= np.asarray(mask, dtype=bool)
    return inlier_mask


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", nargs="?", default=PC_DIR / PC_NAME)
    parser.add_argument("--mesh-samples", type=int, default=MESH_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)
    pc_path = resolve_input_path(args.input_file)
    if not pc_path.exists():
        raise FileNotFoundError("Point cloud not found")

    points, point_colors, input_normals = load_point_cloud(
        pc_path,
        mesh_sample_count=args.mesh_samples,
        seed=args.seed,
    )
    print(f"Loaded {pc_path.name}: {len(points)} points")
    if pc_path.suffix.lower() == ".stl":
        print(f"Sampled STL surface | samples={args.mesh_samples}")
    if point_colors is not None:
        print("Loaded vertex colors; final GAIR view uses residual/inlier colors like main_pc_import.")

    effective_threshold, point_spacing = compute_effective_threshold(points)
    print(
        "Consensus | "
        f"min_threshold={THRESHOLD:.4f} "
        f"point_spacing={point_spacing:.4f} "
        f"effective_threshold={effective_threshold:.4f}"
    )

    if input_normals is None:
        print(f"Estimating normals with Open3D | k_neighbors={K_NEIGHBORS}")
        normals = estimate_normals_open3d_consistent(points, K_NEIGHBORS)
    else:
        print("Using normals sampled from input mesh faces")
        normals = input_normals
    if ALGORITHM_NAME == "gair":
        print("Running GAIR-RANSAC")
    else:
        print("Running GC-RANSAC")
    models, inliers_masks, total_best_mss_used, total_local_opts = gair_ransac(
        threshold=effective_threshold,
        point_cloud=points,
        max_models=MAX_MODELS,
        m_neighbors=M_NEIGHBORS,
        max_iterations=MAX_ITERATIONS,
        sample_size=SAMPLE_SIZE,
        min_inliers=MIN_INLIERS,
        inner_iterations=INNER_ITERATIONS,
        use_normal_coherence=True,
        normals=normals,
        random_seed=args.seed,
        min_coverage=MIN_COVERAGE,
        energy_strategy=FullGairEnergy() if ALGORITHM_NAME == "gair" else GcRansacEnergy(),
    )
    if not models:
        raise RuntimeError("gair_ransac did not return any model")

    print(f"GAIR-RANSAC found {len(models)} model(s) | local optimizations={total_local_opts}")
    inlier_mask = combine_inlier_masks(inliers_masks)
    if inlier_mask is not None:
        n_inliers = int(inlier_mask.sum())
        n_outliers = int(inlier_mask.shape[0] - n_inliers)
        print(
            f"Inliers: {n_inliers} | Outliers: {n_outliers} | "
            f"Total: {len(inlier_mask)} | Outlier ratio: {n_outliers / len(inlier_mask):.2%}"
        )

    palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
    list_mesh = [supmesh.superquadric_mesh(model) for model in models]
    mesh_colors = [palette[i % len(palette)] for i in range(len(list_mesh))]

    vis.show_mesh_and_points(
        list_mesh,
        pts=points,
        point_size=5,
        show_bounds=True,
        colors=mesh_colors,
        inlier_mask=inlier_mask,
        mss_used=total_best_mss_used,
        models=models,
        mesh_opacity=1.0,
        point_opacity=0.55,
        focus_on_points=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
