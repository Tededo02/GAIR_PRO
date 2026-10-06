from pathlib import Path

import numpy as np
import trimesh


TARGET_POINT_COUNT = 100_000
POINT_CLOUD_PATH = Path("test_objects/real_pc/et2_no_floor.ply")


def load_point_cloud(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    geometry = trimesh.load(str(path), process=False)
    if isinstance(geometry, trimesh.Scene):
        if not geometry.geometry:
            raise ValueError(f"Scene has no geometry: {path}")
        geometry = trimesh.util.concatenate(tuple(geometry.geometry.values()))

    points = np.asarray(geometry.vertices, dtype=np.float64)
    if points.size == 0:
        raise ValueError(f"Point cloud has no points: {path}")

    colors = getattr(geometry, "colors", None)
    if colors is None and hasattr(geometry, "visual"):
        colors = getattr(geometry.visual, "vertex_colors", None)

    if colors is not None and len(colors) == len(points):
        colors = np.asarray(colors, dtype=np.uint8)
    else:
        colors = None

    return points, colors


def sample_uniform_points(
    points: np.ndarray,
    colors: np.ndarray | None,
    target_count: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if target_count <= 0:
        raise ValueError(f"TARGET_POINT_COUNT must be positive, got {target_count}")
    if target_count > len(points):
        raise ValueError(f"Cannot sample {target_count} points from a cloud with {len(points)} points")

    rng = np.random.default_rng()
    indices = rng.choice(len(points), size=target_count, replace=False)
    sampled_points = points[indices]
    sampled_colors = None if colors is None else colors[indices]
    return sampled_points, sampled_colors


def output_path(input_path: Path, target_count: int) -> Path:
    return input_path.with_name(f"{input_path.stem}_resized_{target_count}{input_path.suffix}")


def main() -> int:
    input_path = POINT_CLOUD_PATH.expanduser()
    points, colors = load_point_cloud(input_path)
    sampled_points, sampled_colors = sample_uniform_points(points, colors, TARGET_POINT_COUNT)

    resized_cloud = trimesh.PointCloud(sampled_points, colors=sampled_colors)
    destination = output_path(input_path, TARGET_POINT_COUNT)
    resized_cloud.export(str(destination))

    print(f"Loaded: {input_path}")
    print(f"Original points: {len(points)}")
    print(f"Sampled points: {len(sampled_points)}")
    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
