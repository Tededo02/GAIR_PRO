from pathlib import Path

import pyvista as pv
import numpy as np
from src.superquadrics.superquadric_residual import superquadric_radial_residual
from src.superquadrics import superquadric_mesh as supmesh
from src.gair_ransac.consensus import expanded_removal_mask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERA_POSITION = [
    (-4.5307183938878675, 1.7429659977931096, -0.2853180548569162),
    (0.0, 0.0, 0.0),
    (-0.1126909338726822, -0.13171346961495714, 0.9848615716662379),
]


def _focus_camera_on_points(plotter: pv.Plotter, points: np.ndarray) -> None:
    if points.size == 0:
        plotter.camera_position = DEFAULT_CAMERA_POSITION
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0

    default_position = np.asarray(DEFAULT_CAMERA_POSITION[0], dtype=np.float64)
    default_focal_point = np.asarray(DEFAULT_CAMERA_POSITION[1], dtype=np.float64)
    default_view_up = tuple(np.asarray(DEFAULT_CAMERA_POSITION[2], dtype=np.float64))

    view_direction = default_position - default_focal_point
    norm = np.linalg.norm(view_direction)
    if norm == 0.0:
        view_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        norm = 1.0

    plotter.camera_position = [
        tuple(center + (view_direction / norm)),
        tuple(center),
        default_view_up,
    ]
    plotter.reset_camera()
    plotter.camera.zoom(0.95)
    plotter.reset_camera_clipping_range()


def _add_meshes_to_plotter(
    plotter: pv.Plotter,
    meshes: list,
    colors: np.ndarray | None = None,
    opacity: float = 0.65,
) -> None:
    for i, mesh in enumerate(meshes):
        faces = np.hstack([
            np.full((len(mesh.faces), 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64),
        ]).ravel()
        poly = pv.PolyData(mesh.vertices, faces)
        mesh_color = None if colors is None else colors[i]
        plotter.add_mesh(poly, smooth_shading=True, opacity=opacity, color=mesh_color)


def save_point_cloud_inlier_view(
    points: np.ndarray,
    inlier_mask: np.ndarray,
    output_path: str | Path,
    point_size: int = 8,
) -> Path:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != points.shape[0]:
        raise ValueError(f"inlier_mask must have length {points.shape[0]}, got {mask.shape[0]}")

    image_path = Path(output_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    image_path.parent.mkdir(parents=True, exist_ok=True)

    display_point_size = max(point_size * 2, 12)
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    pl.set_background("white")

    if (~mask).any():
        pl.add_points(
            points[~mask],
            render_points_as_spheres=True,
            point_size=display_point_size/3,
            color="black",
            opacity=0.95,
        )
    if mask.any():
        pl.add_points(
            points[mask],
            render_points_as_spheres=True,
            point_size=display_point_size,
            color="red",
            opacity=1.0,
        )

    pl.enable_eye_dome_lighting()
    _focus_camera_on_points(pl, points)
    pl.screenshot(str(image_path))
    pl.close()
    return image_path


def save_point_cloud_inlier_model_view(
    points: np.ndarray,
    inlier_mask: np.ndarray,
    model,
    output_path: str | Path,
    point_size: int = 8,
    model_color: str = "#64b5f6",
    model_opacity: float = 0.35,
) -> Path:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != points.shape[0]:
        raise ValueError(f"inlier_mask must have length {points.shape[0]}, got {mask.shape[0]}")

    image_path = Path(output_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    image_path.parent.mkdir(parents=True, exist_ok=True)

    fitted_mesh = supmesh.superquadric_mesh(model)
    mesh_vertices = np.asarray(fitted_mesh.vertices, dtype=np.float64).reshape(-1, 3)
    frame_points = points if mesh_vertices.size == 0 else np.vstack((points, mesh_vertices))

    display_point_size = max(point_size * 2, 12)
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    pl.set_background("white")

    _add_meshes_to_plotter(pl, [fitted_mesh], colors=[model_color], opacity=model_opacity)

    if (~mask).any():
        pl.add_points(
            points[~mask],
            render_points_as_spheres=True,
            point_size=display_point_size / 3,
            color="black",
            opacity=0.95,
        )
    if mask.any():
        pl.add_points(
            points[mask],
            render_points_as_spheres=True,
            point_size=display_point_size,
            color="red",
            opacity=1.0,
        )

    pl.enable_eye_dome_lighting()
    _focus_camera_on_points(pl, frame_points)
    pl.screenshot(str(image_path))
    pl.close()
    return image_path


def show_point_cloud_only(points: np.ndarray, point_size: int = 8, camera_position: list | None = None) -> list:
    """Show the raw point cloud alone, all points black. Returns the camera position the
    window was closed at, so a following show_mesh_and_points call can pick up from there."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)

    pl = pv.Plotter()
    pl.set_background("white")
    pl.add_points(points, render_points_as_spheres=True, point_size=point_size, color="black")
    pl.enable_eye_dome_lighting()
    pl.camera_position = camera_position if camera_position is not None else DEFAULT_CAMERA_POSITION

    pl.show()
    return pl.camera_position


def show_mesh_and_points(meshes: list, pts: list =None, point_size=8, show_bounds=True,colors: np.ndarray = None, inlier_mask: np.ndarray = None,mss_used: np.ndarray = None,
        models=None,
        treshold=0.1,
        binary: bool = False,
        camera_position: list | None = None,
        mesh_opacity: float = 0.65,
        point_opacity: float = 1.0,
        focus_on_points: bool = False,
    ) -> list:

    # --- plotter ---
    pl = pv.Plotter()
    pl.set_background("white")  # sfondo chiaro


    # --- aggiungi tutte le mesh ---
    points = None if pts is None else np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    error_inlier = None
    remaining_mask = None
    cloud_point_size = max(point_size * 2, 12)

    if models and points is not None:
        error_inlier = np.full(points.shape[0], np.inf, dtype=np.float64)
        remaining_mask = np.ones(points.shape[0], dtype=bool)

        for model in models:
            temp_error_inlier = np.abs(superquadric_radial_residual(model, points))
            error_inlier = np.minimum(error_inlier, temp_error_inlier)

            if not remaining_mask.any():
                break

            current_indices = np.flatnonzero(remaining_mask)
            # Keep the original colormap semantics from commit `colormap`,
            # independent from newer consensus defaults used elsewhere.
            remove_mask = expanded_removal_mask(
                model,
                points[current_indices],
                treshold,
                factor=1.3,
            )
            remaining_mask[current_indices[remove_mask]] = False

    _add_meshes_to_plotter(pl, meshes, colors=colors, opacity=mesh_opacity)

    # --- points (if present) ---
    n_points_total = 0
    if points is not None:
        n_points_total = points.shape[0] # total number of points across all meshes
        has_residual_colormap = (not binary) and error_inlier is not None and np.isfinite(error_inlier).any()
        if has_residual_colormap:
            finite_error = error_inlier[np.isfinite(error_inlier)]
            color_max = float(np.percentile(finite_error, 90))
            if color_max <= 0.0:
                color_max = float(finite_error.max())
            if color_max <= 0.0:
                color_max = 1.0

            point_cloud = pv.PolyData(points)
            point_cloud["min_residual"] = np.nan_to_num(
                error_inlier,
                nan=color_max,
                posinf=color_max,
                neginf=0.0,
            )
            pl.add_mesh(
                point_cloud,
                scalars="min_residual",
                cmap="turbo",
                clim=(0.0, color_max),
                render_points_as_spheres=True,
                point_size=cloud_point_size,
                opacity=point_opacity,
                ambient=0.25,
                specular=0.15,
                nan_color="black",
                above_color="#7f0000",
                scalar_bar_args={
                    "title": "Min radial residual",
                    "color": "black",
                    "fmt": "%.3f",
                    "vertical": True,
                    "position_x": 0.82,
                    "position_y": 0.1,
                    "width": 0.08,
                    "height": 0.8,
                },
            )
        else:
            mask_is_valid = inlier_mask is not None and len(inlier_mask) == n_points_total
            if mask_is_valid:
                mask = np.asarray(inlier_mask, dtype=bool)
                if mask.any():
                    pl.add_points(
                        points[mask],
                        render_points_as_spheres=True,
                        point_size=cloud_point_size,
                        color="#00e676",
                        opacity=point_opacity,
                    )
                if (~mask).any():
                    pl.add_points(
                        points[~mask],
                        render_points_as_spheres=True,
                        point_size=cloud_point_size,
                        color="#ff1744",
                        opacity=min(point_opacity, 0.8),
                    )
            else:
                pl.add_points(
                    points,
                    render_points_as_spheres=True,
                    point_size=cloud_point_size,
                    color="#1565c0",
                    opacity=point_opacity,
                )
    pl.enable_eye_dome_lighting()

    if camera_position is not None:
        pl.camera_position = camera_position
    elif focus_on_points and points is not None:
        _focus_camera_on_points(pl, points)
    else:
        pl.camera_position = DEFAULT_CAMERA_POSITION

    pl.show()
    last_camera_position = pl.camera_position

    mesh_only_plotter = pv.Plotter()
    mesh_only_plotter.set_background("white")
    _add_meshes_to_plotter(mesh_only_plotter, meshes, colors=colors, opacity=mesh_opacity)
    mesh_only_plotter.enable_eye_dome_lighting()
    mesh_only_plotter.camera_position = last_camera_position
    mesh_only_plotter.show()
    return mesh_only_plotter.camera_position
