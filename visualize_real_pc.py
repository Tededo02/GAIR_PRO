import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh


ROOT = Path(__file__).resolve().parent
REAL_PC_DIR = ROOT / "test_objects" / "real_pc"
SUPPORTED_EXTENSIONS = {".ply"}
PC_NAME = "car_pc_resized_100000.ply"
POINT_COLOR = "black"


def available_point_clouds() -> list[Path]:
    return sorted(
        path for path in REAL_PC_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def resolve_input_path(input_file: str | Path) -> Path:
    input_path = Path(input_file).expanduser()
    candidates = [input_path] if input_path.is_absolute() else [
        ROOT / input_path,
        REAL_PC_DIR / input_path,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"Point cloud not found: {input_file}")


def load_geometry(path: Path) -> trimesh.Trimesh | trimesh.PointCloud:
    geometry = trimesh.load(str(path), process=False)
    if isinstance(geometry, trimesh.Scene):
        if not geometry.geometry:
            raise ValueError(f"Scene has no geometry: {path}")
        geometry = trimesh.util.concatenate(tuple(geometry.geometry.values()))
    return geometry


def show_point_cloud(path: Path, point_size: float) -> None:
    geometry = load_geometry(path)
    points = np.asarray(geometry.vertices, dtype=np.float64)
    if points.size == 0:
        raise ValueError(f"Point cloud has no points: {path}")
    points = points.reshape(-1, 3)

    plotter = pv.Plotter()
    plotter.set_background("white")
    plotter.add_points(
        points,
        render_points_as_spheres=True,
        point_size=point_size,
        color=POINT_COLOR,
    )

    plotter.add_text(f"{path.name} | {len(points):,} points", font_size=10)
    plotter.add_axes()
    plotter.enable_eye_dome_lighting()
    plotter.reset_camera()
    plotter.show()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize point clouds from test_objects/real_pc with black points.",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help="File name in test_objects/real_pc or an absolute/relative path. Overrides PC_NAME.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Open every supported point cloud one at a time.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Rendered point diameter in pixels (default: 4).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.point_size <= 0:
        raise ValueError("--point-size must be positive")
    if args.input_file is not None and args.all:
        raise ValueError("Specify either input_file or --all, not both")

    if args.all:
        paths = available_point_clouds()
    elif args.input_file is not None:
        paths = [resolve_input_path(args.input_file)]
    else:
        paths = [resolve_input_path(PC_NAME)]

    if not paths:
        raise FileNotFoundError(f"No supported point clouds found in {REAL_PC_DIR}")

    for path in paths:
        show_point_cloud(path, args.point_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
