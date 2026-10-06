# Run GPU checks outside a sandbox that hides the Metal device.

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from test_superflex import fixture_model, surface
from src.gair_ransac.deformable_fitting import _pack, _unpack
from src.gair_ransac.inner_ransac import fit_superquadric_ls, _pca_initial_parameters
from src.gair_ransac.metal_superquadric import _metal_runtime, metal_available, fit_superquadric_metal_batch
from src.gair_ransac.metal_superflex import fit_superflex_hypothesis_batch, superflex_optimization_bounds
from src.gair_ransac.metal_consensus import create_metal_consensus_context
from src.gair_ransac.consensus import compute_consensus
from src.gair_ransac.interior_consensus import InteriorPenaltyContext
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.superquadric_residual import superquadric_radial_residual_and_jacobian, superquadric_radial_residual, superquadric_normal_world


@unittest.skipUnless(metal_available(), "An accessible Metal GPU is required")
class SuperFlexMetalTests(unittest.TestCase):
    def test_gpu_radial_and_all_19_derivatives_match_cpu(self):
        mx = _metal_runtime()
        directory = Path(__file__).resolve().parents[1] / "src/gair_ransac"
        kernel = mx.fast.metal_kernel(
            name="check_superflex_radial_jacobian",
            input_names=["points", "parameters"], output_names=["residual", "jacobian"],
            header=(directory / "superquadric_fit_helpers.metal").read_text() + "\n" + (directory / "superflex_geometry.metal").read_text(),
            source="""
                uint i = thread_position_in_grid.x;
                if (i < uint(points_shape[0])) {
                    float p[19], j[19];
                    for (int k = 0; k < 19; ++k) p[k] = parameters[k];
                    residual[i] = sf_radial<true>(float3(points[3*i], points[3*i+1], points[3*i+2]), p, 1e-12f, j);
                    for (int k = 0; k < 19; ++k) jacobian[19*i+k] = j[k];
                }
            """,
        )
        points = np.random.default_rng(7).uniform([0.2, 0.1, 0.2], [0.9, 0.9, 0.9], (40, 3)).astype(np.float32)
        cases = [fixture_model(), fixture_model(bending=np.zeros((3, 2))), fixture_model(bending=[[1e-4, 0.5], [2e-4, -0.2], [1e-4, 0.3]])]
        for model in cases:
            actual, jacobian = kernel(inputs=[mx.array(points), mx.array(_pack(model, 1).astype(np.float32))], grid=(64, 1, 1), threadgroup=(64, 1, 1), output_shapes=[(40,), (40, 19)], output_dtypes=[mx.float32, mx.float32], stream=mx.gpu)
            mx.eval(actual, jacobian)
            expected, expected_jacobian = superquadric_radial_residual_and_jacobian(model, points)
            np.testing.assert_allclose(np.array(actual), expected, rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(np.array(jacobian), expected_jacobian, rtol=3e-3, atol=3e-5)

    def test_metal_fits_all_deformation_variants_without_scipy(self):
        cases = [dict(taper=[0.5, -0.3], bending=np.zeros((3, 2))), dict(taper=[0, 0], bending=[[0, 0], [0, 0], [1, 0.65]]), dict()]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), patch("src.gair_ransac.inner_ransac.least_squares", side_effect=AssertionError("SciPy must not optimize a Metal fit")), patch("scipy.optimize.least_squares", side_effect=AssertionError("SciPy must not optimize a Metal fit")):
                points = surface(fixture_model(**kwargs), 10, 16)
                fitted = fit_superquadric_ls(points, model_family="superflex", backend="metal", axis_penalty_weight=0)
                error = np.abs(superquadric_radial_residual(fitted, points))
                self.assertEqual(fitted.parameter_count, 19)
                self.assertLess(float(np.quantile(error, 0.95)), 0.003)

    def test_consensus_normals_interior_and_subset_stay_on_gpu(self):
        model = fixture_model()
        surface_points = surface(model, 10, 16)
        points = np.vstack((surface_points, model.t + 0.6*(surface_points-model.t), surface_points + [0.5, 0.4, 0.5]))
        normals = superquadric_normal_world(model, points)
        normals[5:15] *= -1
        normals[25:27] = 0
        threshold = 0.012
        expected_mask = compute_consensus(model, points, threshold, normals=normals)
        interior = InteriorPenaltyContext(points, weight=25, normals=normals)
        expected_mass = interior.mass(model, threshold)
        context = create_metal_consensus_context(points, normals, interior_context=interior, model_family="superflex")
        with patch("src.gair_ransac.consensus.compute_consensus", side_effect=AssertionError("SuperFlex consensus must not invoke CPU boundary corrections")):
            result = context.evaluate([model], threshold)
            np.testing.assert_array_equal(result.mask(0), expected_mask)
            np.testing.assert_allclose(result.interior_masses, [expected_mass], rtol=3e-5, atol=3e-5)
            indices = np.arange(0, len(points), 3)
            view = context.for_subset(points[indices], normals[indices], indices)
            result = view.evaluate([model], threshold)
            np.testing.assert_array_equal(result.mask(0), expected_mask[indices])
            np.testing.assert_allclose(result.interior_masses, [expected_mass], rtol=3e-5, atol=3e-5)

    def test_batched_hypotheses_preserve_sample_order(self):
        rng = np.random.default_rng(23)
        model = fixture_model()
        points = surface(model, 12, 20)
        samples = np.stack([points[rng.choice(len(points), 40, replace=False)] for _ in range(4)])
        lower, upper, loss_scale, diagonal, support = superflex_optimization_bounds(points, 0)
        with patch("scipy.optimize.least_squares", side_effect=AssertionError("SciPy must not optimize GPU hypotheses")):
            result = fit_superflex_hypothesis_batch(samples, _pca_initial_parameters(samples), lower, upper, loss_scale, diagonal, support, 0)
        self.assertTrue(result.success.all(), result.diagnostics)
        for sample, parameters in zip(samples, result.parameters):
            fitted = _unpack(parameters, diagonal)
            self.assertLess(float(np.quantile(np.abs(superquadric_radial_residual(fitted, sample)), 0.95)), 0.004)

    def test_rigid_metal_solver_remains_compatible(self):
        model = SuperQuadricParams(0.2, 0.3, 0.5, 1, 1, [0.2, 0.3, -0.1], [0.5, 0.5, 0.5])
        points = surface(model)
        fitted = fit_superquadric_ls(points, backend="metal")
        self.assertEqual(fitted.parameter_count, 11)
        self.assertLess(float(np.max(np.abs(superquadric_radial_residual(fitted, points)))), 1e-4)


if __name__ == "__main__":
    unittest.main()
