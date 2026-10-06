# Run with: .venv/bin/python -m unittest discover -s checks -v

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from src.superquadrics.deformable_superquadric import DeformableSuperQuadricParams, inverse_bend
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.superquadric_mesh import apply_pose, superquadric_point
from src.superquadrics.superquadric_residual import (
    implicit_f_superquadric_residual, superquadric_first_order_residual,
    superquadric_normal_world, superquadric_radial_residual,
    superquadric_radial_residual_and_jacobian,
)
from src.gair_ransac.deformable_fitting import _pack, _unpack
from src.gair_ransac.inner_ransac import fit_superquadric_ls, inner_ransac
from src.gair_ransac.consensus import compute_consensus, expanded_removal_mask
from src.gair_ransac.interior_consensus import InteriorPenaltyContext, interior_strength
from src.gair_ransac.metal_consensus import create_metal_consensus_context
from src.gair_ransac.metal_superquadric import metal_available
from src.gair_ransac.gair_ransac import gair_ransac
from src.gair_ransac.ransac import ransac


def fixture_model(**kwargs):
    parameters = dict(
        a1=0.22, a2=0.16, a3=0.5, e1=0.8, e2=1.1,
        rot=[0.25, -0.3, 0.1], t=[0.6, 0.4, 0.5],
        taper=[0.35, -0.2], bending=[[0.35, 0.6], [0.25, -0.7], [0.7, 0.45]],
    )
    parameters.update(kwargs)
    return DeformableSuperQuadricParams(**parameters)


def surface(model, n_eta=12, n_omega=20):
    return apply_pose(superquadric_point(model, n_eta, n_omega), model)


class SuperFlexGeometryTests(unittest.TestCase):
    def test_deformation_round_trip_all_axes_and_negative_curvatures(self):
        model = fixture_model(bending=[[-0.35, 0.6], [0.25, -0.7], [-0.7, 0.45]])
        canonical = np.random.default_rng(19).uniform([-0.22, -0.16, -0.5], [0.22, 0.16, 0.5], (100, 3))
        np.testing.assert_allclose(model.inverse_deform(model.deform(canonical)), canonical, atol=2e-13)

    def test_zero_deformations_match_rigid_geometry_and_penetration(self):
        model = fixture_model(taper=[0, 0], bending=np.zeros((3, 2)))
        rigid = SuperQuadricParams(model.a1, model.a2, model.a3, model.e1, model.e2, model.rot, model.t)
        points = np.random.default_rng(3).uniform(0, 1, (100, 3))
        for function in (implicit_f_superquadric_residual, superquadric_radial_residual, superquadric_first_order_residual, superquadric_normal_world):
            np.testing.assert_allclose(function(model, points), function(rigid, points), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(surface(model), surface(rigid), atol=1e-14)
        np.testing.assert_allclose(interior_strength(model, points, 0.01), interior_strength(rigid, points, 0.01), atol=1e-12)

    def test_parametric_surface_satisfies_implicit_equation_and_consensus(self):
        model = fixture_model()
        points = surface(model)
        np.testing.assert_allclose(implicit_f_superquadric_residual(model, points), 0, atol=1e-10)
        np.testing.assert_allclose(superquadric_radial_residual(model, points), 0, atol=1e-10)
        self.assertTrue(compute_consensus(model, points, 1e-5).all())
        self.assertTrue(expanded_removal_mask(model, points, 1e-5).all())

    def test_normal_matches_numerical_world_gradient(self):
        model = fixture_model()
        points = surface(model)[24:34] + 0.002
        step = 1e-6
        gradient = np.column_stack([
            (implicit_f_superquadric_residual(model, points + step * axis) - implicit_f_superquadric_residual(model, points - step * axis)) / (2 * step)
            for axis in np.eye(3)
        ])
        gradient /= np.linalg.norm(gradient, axis=1, keepdims=True)
        np.testing.assert_allclose(superquadric_normal_world(model, points), gradient, atol=2e-7)
        self.assertTrue(compute_consensus(model, points, 0.01, normals=gradient).all())
        self.assertFalse(compute_consensus(model, points, 0.01, normals=-gradient).any())

    def test_large_gradients_preserve_normals_and_first_order_distances(self):
        model = fixture_model(a1=0.2, a2=0.2, a3=0.3, e1=0.06, e2=0.06,
                              taper=[0.9, 0.9], bending=np.zeros((3, 2)))
        local = np.array([[1.0, 0.0, -1.0], [1.0, 1.0, -1.0]])
        rotation = model.rotation_matrix()
        points = local @ rotation.T + model.t
        # The clamped inverse taper dominates the gradient in the x/y plane.
        expected = np.array([[1.0, 0.0, 0.0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0.0]]) @ rotation.T
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            normals = superquadric_normal_world(model, points)
            distances = superquadric_first_order_residual(model, points)
        np.testing.assert_allclose(normals, expected, atol=2e-14)
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=2e-14)
        np.testing.assert_allclose(distances, 0.03 * np.array([1.0, np.sqrt(2)]), rtol=2e-13)

    def test_inverse_bending_spatial_and_curvature_jacobians(self):
        points = np.random.default_rng(11).uniform(-0.3, 0.3, (10, 3))
        step = 1e-6
        for components in ([0.4, -0.3], [0, 0], [1e-7, -2e-7]):
            for axis in (0, 1, 2):
                _, spatial, parameter = inverse_bend(points, components, axis, True)
                numeric_spatial = np.stack([
                    (inverse_bend(points + step * direction, components, axis) - inverse_bend(points - step * direction, components, axis)) / (2 * step)
                    for direction in np.eye(3)
                ], axis=2)
                numeric_parameter = np.stack([
                    (inverse_bend(points, np.array(components) + step * direction, axis) - inverse_bend(points, np.array(components) - step * direction, axis)) / (2 * step)
                    for direction in np.eye(2)
                ], axis=2)
                np.testing.assert_allclose(spatial, numeric_spatial, atol=2e-8)
                np.testing.assert_allclose(parameter, numeric_parameter, atol=2e-8)

    def test_full_radial_jacobian_including_zero_curvature(self):
        points = np.random.default_rng(7).uniform([0.2, 0.1, 0.2], [0.9, 0.9, 0.9], (18, 3))
        step = 1e-6
        for model in (fixture_model(), fixture_model(bending=np.zeros((3, 2)))):
            parameters = _pack(model, 1.0)
            residual, analytic = superquadric_radial_residual_and_jacobian(model, points)
            numeric = np.column_stack([
                (superquadric_radial_residual(_unpack(parameters + step * direction, 1.0), points) - superquadric_radial_residual(_unpack(parameters - step * direction, 1.0), points)) / (2 * step)
                for direction in np.eye(19)
            ])
            np.testing.assert_allclose(residual, superquadric_radial_residual(model, points), atol=2e-12)
            np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-7)

    def test_interior_score_uses_deformed_volume(self):
        model = fixture_model()
        canonical = superquadric_point(model, 10, 20)[22:42]
        inner = apply_pose(0.6 * canonical, model)
        outside = apply_pose(1.15 * canonical, model)
        self.assertTrue((interior_strength(model, inner, 0.001) > 0.9).all())
        np.testing.assert_allclose(interior_strength(model, outside, 0.001), 0.0)
        context = InteriorPenaltyContext(np.vstack((inner, outside)), weight=25)
        score, mass = context.score(model, 20, 0.001)
        self.assertGreater(mass, 0)
        self.assertLess(score, 20)

    def test_input_validation_and_metal_routing(self):
        with self.assertRaises(ValueError):
            fixture_model(taper=[1, 0])
        with self.assertRaises(ValueError):
            fit_superquadric_ls(np.zeros((18, 3)), model_family="superflex")
        with self.assertRaises(ValueError):
            fit_superquadric_ls(np.zeros((20, 3)), model_family="unknown")
        with patch("src.gair_ransac.metal_superflex.metal_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "requires an accessible Metal GPU"):
                create_metal_consensus_context(np.zeros((20, 3)), model_family="superflex")
            with self.assertRaisesRegex(RuntimeError, "requires an accessible Metal GPU"):
                fit_superquadric_ls(np.zeros((20, 3)), model_family="superflex")


class SuperFlexFittingTests(unittest.TestCase):
    def test_default_family_preserves_rigid_cpu_fitting(self):
        model = SuperQuadricParams(0.2, 0.3, 0.5, 1, 1, [0.2, 0.3, -0.1], [0.5, 0.5, 0.5])
        points = surface(model)
        fitted = fit_superquadric_ls(points, backend="cpu")
        self.assertIs(type(fitted), SuperQuadricParams)
        self.assertEqual(fitted.parameter_count, 11)
        self.assertLess(float(np.max(np.abs(superquadric_radial_residual(fitted, points)))), 1e-5)

    def test_fit_taper_bending_and_combined(self):
        cases = [
            dict(taper=[0.5, -0.3], bending=np.zeros((3, 2))),
            dict(taper=[0.98, -0.95], bending=np.zeros((3, 2))),
            dict(taper=[0, 0], bending=[[0, 0], [0, 0], [1.0, 0.65]]),
            dict(),
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                points = surface(fixture_model(**kwargs), 10, 16)
                fitted = fit_superquadric_ls(points, model_family="superflex", backend="cpu", axis_penalty_weight=0)
                self.assertEqual(fitted.parameter_count, 19)
                error = np.abs(superquadric_radial_residual(fitted, points))
                self.assertLess(float(np.quantile(error, 0.95)), 0.003)

    @unittest.skipUnless(metal_available(), "An accessible Metal GPU is required")
    def test_inner_ransac_and_final_refit_preserve_deformations(self):
        points = surface(fixture_model(), 8, 12)
        result = inner_ransac(points, np.arange(len(points)), None, 0.015, n_iters=2, random_seed=4, model_family="superflex", axis_penalty_weight=0)
        self.assertEqual(result.best_model.parameter_count, 19)
        self.assertGreater(result.best_inlier_count, 0.9 * len(points))
        self.assertEqual(result.best_inliers_mask.shape, (len(points),))

    @unittest.skipUnless(metal_available(), "An accessible Metal GPU is required")
    def test_ransac_and_gair_extract_deformable_models(self):
        model = fixture_model()
        points = surface(model, 8, 12)
        normals = superquadric_normal_world(model, points)
        for fitter in (ransac, gair_ransac):
            kwargs = dict(threshold=0.02, max_models=1, max_iterations=2, inner_iterations=1, sample_size=40, min_inliers=40, random_seed=8, model_family="superflex")
            if fitter is gair_ransac:
                kwargs.update(normals=normals, interior_penalty_weight=0, axis_penalty_weight=0)
            result = fitter(points, **kwargs)
            self.assertEqual(len(result[0]), 1)
            self.assertEqual(result[0][0].parameter_count, 19)
            self.assertGreater(result[1][0].sum(), 0.8 * len(points))

    @unittest.skipUnless(metal_available(), "An accessible Metal GPU is required")
    def test_main_global_parameter_and_cli(self):
        import main_pc_import as main
        with patch.object(main, "MODEL_FAMILY", "superflex"):
            self.assertEqual(main.parse_args([]).model_family, "superflex")
            self.assertEqual(main.parse_args(["--model-family", "rigid"]).model_family, "rigid")
        model = fixture_model()
        points = surface(model, 8, 12)
        normals = superquadric_normal_world(model, points)
        with patch.object(main, "MODEL_FAMILY", "superflex"), patch.object(main, "ALGORITHM", "ls"), patch.object(main, "SAMPLED_POINT_COUNT", 30), patch.object(main, "resolve_input_path", return_value=main.PC_FILE), patch.object(main, "load_input_geometry", return_value=(points, normals)), patch.object(main.vis, "show_mesh_and_points") as show, redirect_stdout(io.StringIO()):
            main.create_and_estimate_supq()
        self.assertEqual(show.call_args.kwargs["models"][0].parameter_count, 19)

    @unittest.skipUnless(metal_available(), "An accessible Metal GPU is required")
    def test_main_algorithm_options_honor_global_family(self):
        import main_pc_import as main
        points = surface(fixture_model(), 8, 12)
        normals = superquadric_normal_world(fixture_model(), points)
        original_inner = main.inner_ransac
        original_ransac = main.ransac
        original_gair = main.gair_ransac

        def short_inner(*args, **kwargs):
            return original_inner(*args, **{**kwargs, "n_iters": 2})

        def short_ransac(*args, **kwargs):
            return original_ransac(*args, **{**kwargs, "max_iterations": 2, "inner_iterations": 1})

        def short_gair(*args, **kwargs):
            return original_gair(*args, **{**kwargs, "max_iterations": 2, "inner_iterations": 1})

        for algorithm in ("ls", "inner-ransac", "ransac", "gair-ransac", "gc-ransac"):
            with self.subTest(algorithm=algorithm), patch.object(main, "MODEL_FAMILY", "superflex"), patch.object(main, "ALGORITHM", algorithm), patch.object(main, "MAX_MODEL", 1), patch.object(main, "SAMPLED_POINT_COUNT", 30), patch.object(main, "resolve_input_path", return_value=main.PC_FILE), patch.object(main, "load_input_geometry", return_value=(points, normals)), patch.object(main, "inner_ransac", side_effect=short_inner), patch.object(main, "ransac", side_effect=short_ransac), patch.object(main, "gair_ransac", side_effect=short_gair), patch.object(main.vis, "show_mesh_and_points") as show, redirect_stdout(io.StringIO()):
                main.create_and_estimate_supq()
                self.assertEqual(show.call_args.kwargs["models"][0].parameter_count, 19)


if __name__ == "__main__":
    unittest.main()
