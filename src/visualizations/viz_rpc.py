from __future__ import annotations

import numpy as np
import pyvista as pv

from src.superquadrics.superquadric_residual import superquadric_radial_residual


DEFAULT_FRONT_REFERENCE = np.array([0.0, 0.0, 0.0], dtype=np.float64)
WORLD_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _as_points(points: np.ndarray | None) -> np.ndarray | None:
    if points is None:
        return None

    point_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if point_array.shape[0] == 0:
        return None
    return point_array


def _mesh_polydata(mesh) -> pv.PolyData:
    faces = np.hstack(
        [
            np.full((len(mesh.faces), 1), 3, dtype=np.int64),
            mesh.faces.astype(np.int64),
        ]
    ).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=np.float64), faces)


def _frame_points(points: np.ndarray | None, meshes: list) -> np.ndarray:
    chunks: list[np.ndarray] = []
    if points is not None:
        chunks.append(points)
    for mesh in meshes:
        vertices = np.asarray(mesh.vertices, dtype=np.float64).reshape(-1, 3)
        if vertices.size:
            chunks.append(vertices)

    if chunks:
        return np.vstack(chunks)
    return np.zeros((1, 3), dtype=np.float64)


def _normalise(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        return fallback / (np.linalg.norm(fallback) + 1e-12)
    return vector / norm


def _pca_front_camera(
    points: np.ndarray,
    frame: np.ndarray,
    front_reference: np.ndarray | None = DEFAULT_FRONT_REFERENCE,
    distance_scale: float = 1.9,
) -> list[tuple[float, float, float]]:
    center = points.mean(axis=0)
    centered = points - center

    if points.shape[0] >= 3 and np.linalg.matrix_rank(centered) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        view_direction = vh[-1]
        up_direction = vh[1]
    else:
        view_direction = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        up_direction = WORLD_UP.copy()

    if front_reference is not None:
        reference_direction = np.asarray(front_reference, dtype=np.float64).reshape(3) - center
        if np.linalg.norm(reference_direction) > 1e-12 and np.dot(view_direction, reference_direction) < 0.0:
            view_direction = -view_direction

    view_direction = _normalise(view_direction, np.array([0.0, -1.0, 0.0], dtype=np.float64))

    projected_world_up = WORLD_UP - view_direction * np.dot(WORLD_UP, view_direction)
    if np.linalg.norm(projected_world_up) > 1e-6:
        up_direction = projected_world_up
    else:
        up_direction = up_direction - view_direction * np.dot(up_direction, view_direction)
    up_direction = _normalise(up_direction, WORLD_UP)

    spans = np.ptp(frame, axis=0)
    extent = max(float(np.linalg.norm(spans)), 1e-6)
    camera_position = center + view_direction * extent * float(distance_scale)
    return [
        tuple(float(value) for value in camera_position),
        tuple(float(value) for value in center),
        tuple(float(value) for value in up_direction),
    ]


def _configure_plotter(plotter: pv.Plotter, background: str = "white") -> None:
    plotter.set_background(background)
    for call in (
        lambda: plotter.enable_anti_aliasing("ssaa"),
        lambda: plotter.enable_eye_dome_lighting(),
        lambda: plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.0),
    ):
        try:
            call()
        except Exception:
            pass


def _add_lights(plotter: pv.Plotter, camera_position: list[tuple[float, float, float]]) -> None:
    camera, focal, _ = camera_position
    camera = np.asarray(camera, dtype=np.float64)
    focal = np.asarray(focal, dtype=np.float64)
    view = _normalise(camera - focal, np.array([0.0, -1.0, 0.0], dtype=np.float64))
    side = _normalise(np.cross(WORLD_UP, view), np.array([1.0, 0.0, 0.0], dtype=np.float64))

    light_specs = [
        (camera, 0.85),
        (focal + side * 2.0 + WORLD_UP * 2.0 + view, 0.45),
        (focal - side * 2.0 + WORLD_UP * 1.0 - view, 0.25),
    ]
    for position, intensity in light_specs:
        try:
            plotter.add_light(
                pv.Light(
                    position=tuple(position),
                    focal_point=tuple(focal),
                    color="white",
                    intensity=float(intensity),
                )
            )
        except Exception:
            pass


def _apply_camera(
    plotter: pv.Plotter,
    camera_position: list[tuple[float, float, float]],
    zoom: float = 1.0,
) -> None:
    plotter.camera_position = camera_position
    plotter.reset_camera_clipping_range()
    if zoom != 1.0:
        plotter.camera.zoom(float(zoom))
        plotter.reset_camera_clipping_range()


def _add_meshes(
    plotter: pv.Plotter,
    meshes: list,
    colors: list[str] | np.ndarray | None = None,
    opacity: float = 0.72,
    show_edges: bool = False,
    wire_overlay: bool = False,
) -> None:
    has_colors = colors is not None and len(colors) > 0
    for i, mesh in enumerate(meshes):
        poly = _mesh_polydata(mesh)
        mesh_color = None if not has_colors else colors[i % len(colors)]
        plotter.add_mesh(
            poly,
            color=mesh_color,
            opacity=opacity,
            smooth_shading=True,
            ambient=0.18,
            diffuse=0.78,
            specular=0.38,
            specular_power=35,
            show_edges=show_edges,
            edge_color="#263238",
            line_width=0.35,
        )
        if wire_overlay:
            plotter.add_mesh(
                poly,
                style="wireframe",
                color="#111111",
                opacity=0.16,
                line_width=0.8,
            )


def _point_residuals(points: np.ndarray, models: list | None) -> np.ndarray | None:
    if not models:
        return None

    residuals = np.full(points.shape[0], np.inf, dtype=np.float64)
    for model in models:
        model_residuals = np.abs(superquadric_radial_residual(model, points))
        residuals = np.minimum(residuals, model_residuals)
    return residuals


def _add_points(
    plotter: pv.Plotter,
    points: np.ndarray,
    point_size: int,
    inlier_mask: np.ndarray | None = None,
    point_colors: np.ndarray | None = None,
    residuals: np.ndarray | None = None,
    residual_limit: float | None = None,
) -> None:
    display_point_size = max(point_size * 2, 10)

    if residuals is not None and np.isfinite(residuals).any():
        finite_residuals = residuals[np.isfinite(residuals)]
        color_max = float(np.percentile(finite_residuals, 90))
        if residual_limit is not None and residual_limit > 0.0:
            color_max = max(color_max, float(residual_limit) * 2.0)
        if color_max <= 0.0:
            color_max = max(float(finite_residuals.max()), 1.0)

        point_cloud = pv.PolyData(points)
        point_cloud["min_residual"] = np.nan_to_num(
            residuals,
            nan=color_max,
            posinf=color_max,
            neginf=0.0,
        )
        plotter.add_mesh(
            point_cloud,
            scalars="min_residual",
            cmap="turbo",
            clim=(0.0, color_max),
            render_points_as_spheres=True,
            point_size=display_point_size,
            ambient=0.3,
            specular=0.12,
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
        return

    if inlier_mask is not None and len(inlier_mask) == points.shape[0]:
        mask = np.asarray(inlier_mask, dtype=bool)
        if mask.any():
            plotter.add_points(
                points[mask],
                render_points_as_spheres=True,
                point_size=display_point_size,
                color="#00c853",
            )
        if (~mask).any():
            plotter.add_points(
                points[~mask],
                render_points_as_spheres=True,
                point_size=display_point_size,
                color="#d50000",
                opacity=0.78,
            )
        return

    if point_colors is not None and len(point_colors) == points.shape[0]:
        point_cloud = pv.PolyData(points)
        point_cloud["colors"] = np.asarray(point_colors, dtype=np.uint8)[:, :3]
        plotter.add_points(
            point_cloud,
            scalars="colors",
            rgb=True,
            render_points_as_spheres=True,
            point_size=display_point_size,
        )
        return

    plotter.add_points(
        points,
        render_points_as_spheres=True,
        point_size=display_point_size,
        color="#1565c0",
        opacity=0.92,
    )


def show_rpc_models(
    meshes: list,
    pts: np.ndarray | None = None,
    point_size: int = 8,
    colors: list[str] | np.ndarray | None = None,
    inlier_mask: np.ndarray | None = None,
    mss_used: np.ndarray | None = None,
    models: list | None = None,
    threshold: float = 0.1,
    point_colors: np.ndarray | None = None,
    front_reference: np.ndarray | None = DEFAULT_FRONT_REFERENCE,
) -> None:
    """
    Visualize real point-cloud GAIR results with a PCA front camera.

    The camera looks along the smallest-variance PCA axis of the point cloud.
    When possible, its sign is chosen toward front_reference, which is useful
    for RGB-D scans stored in camera coordinates where the sensor is near zero.
    """
    points = _as_points(pts)
    frame = _frame_points(points, meshes)
    camera_points = points if points is not None else frame
    camera_position = _pca_front_camera(camera_points, frame, front_reference=front_reference)

    residuals = _point_residuals(points, models) if points is not None else None

    plotter = pv.Plotter()
    _configure_plotter(plotter)
    _add_lights(plotter, camera_position)
    _add_meshes(plotter, meshes, colors=colors, opacity=0.58, show_edges=False)

    if points is not None:
        _add_points(
            plotter,
            points,
            point_size=point_size,
            inlier_mask=inlier_mask,
            point_colors=point_colors,
            residuals=residuals,
            residual_limit=threshold,
        )
    if mss_used is not None:
        mss_points = np.asarray(mss_used, dtype=np.float64).reshape(-1, 3)
        plotter.add_points(
            mss_points,
            render_points_as_spheres=True,
            point_size=max(point_size * 5, 18),
            color="violet",
        )

    _apply_camera(plotter, camera_position, zoom=1.03)
    plotter.show()

    mesh_plotter = pv.Plotter()
    _configure_plotter(mesh_plotter, background="#f7f8fa")
    _add_lights(mesh_plotter, camera_position)
    _add_meshes(
        mesh_plotter,
        meshes,
        colors=colors,
        opacity=0.92,
        show_edges=True,
        wire_overlay=True,
    )
    try:
        mesh_plotter.show_bounds(
            grid="back",
            location="outer",
            color="#6b7280",
            font_size=8,
            all_edges=False,
        )
    except Exception:
        pass
    _apply_camera(mesh_plotter, camera_position, zoom=1.08)
    mesh_plotter.show()
