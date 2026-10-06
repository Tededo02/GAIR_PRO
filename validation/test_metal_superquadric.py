import importlib
from pathlib import Path
import unittest
import time
from unittest.mock import patch

import numpy as np

from src.gair_ransac.inner_ransac import (
    _inner_candidate_models, _model_from_parameters, _optimization_bounds,
    _pca_initial_parameters, fit_superquadric_ls, inner_ransac,
)
from src.gair_ransac.metal_superquadric import (
    MetalBatchFitResult, fit_superquadric_metal, fit_superquadric_metal_batch, metal_available,
)
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.superquadric_residual import superquadric_radial_residual_and_jacobian


def surface_points(model, count=300, seed=4):
    rng = np.random.default_rng(seed)
    latitude = rng.uniform(-np.pi / 2, np.pi / 2, count)
    longitude = rng.uniform(-np.pi, np.pi, count)

    def signed_power(value, exponent):
        return np.sign(value) * np.abs(value) ** exponent

    cos_latitude = signed_power(np.cos(latitude), model.e1)
    canonical = np.column_stack((
        model.a1 * cos_latitude * signed_power(np.cos(longitude), model.e2),
        model.a2 * cos_latitude * signed_power(np.sin(longitude), model.e2),
        model.a3 * signed_power(np.sin(latitude), model.e1),
    ))
    return canonical @ model.rotation_matrix().T + model.t


def robust_cost(model, points):
    residual, _ = superquadric_radial_residual_and_jacobian(model, points)
    scale = max(1e-3, 0.02 * np.linalg.norm(np.ptp(points, axis=0)))
    return np.mean(residual**2 / (np.sqrt(1 + (residual / scale)**2) + 1))


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.points = surface_points(SuperQuadricParams(1.2, 0.8, 0.5, 1, 1), count=40)

    def test_cpu_backend_does_not_probe_metal(self):
        with patch("src.gair_ransac.inner_ransac.metal_available", side_effect=AssertionError):
            model = fit_superquadric_ls(self.points, backend="cpu")
        self.assertLess(robust_cost(model, self.points), 1e-8)

    def test_auto_uses_cpu_when_metal_is_unavailable(self):
        with patch("src.gair_ransac.inner_ransac.metal_available", return_value=False):
            model = fit_superquadric_ls(self.points)
        self.assertLess(robust_cost(model, self.points), 1e-8)

    def test_invalid_input_and_backend(self):
        for points in (np.zeros((11, 2)), np.full((11, 3), np.nan), np.zeros((10, 3))):
            with self.subTest(shape=points.shape):
                with self.assertRaises(ValueError):
                    fit_superquadric_ls(points)
        with self.assertRaises(ValueError):
            fit_superquadric_ls(self.points, backend="cuda")
        with self.assertRaises(ValueError):
            fit_superquadric_ls(self.points, bounds_reference_points=np.empty((0, 3)))

    def test_explicit_metal_fails_when_unavailable(self):
        with patch("src.gair_ransac.metal_superquadric._metal_runtime", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Metal fitting requires"):
                fit_superquadric_ls(self.points, backend="metal")

    def test_gpu_failure_does_not_retry_on_cpu(self):
        with patch("src.gair_ransac.inner_ransac.metal_available", return_value=True):
            with patch("src.gair_ransac.inner_ransac.fit_superquadric_metal", side_effect=RuntimeError("GPU failure")):
                with self.assertRaisesRegex(RuntimeError, "GPU failure"):
                    fit_superquadric_ls(self.points)

    def test_batch_boundaries_preserve_sample_order_and_iteration_count(self):
        support = surface_points(SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2), count=100)
        sampled_batches = []

        def fake_batch(points, initial, *args, **kwargs):
            sampled_batches.append(points.copy())
            return MetalBatchFitResult(initial, np.tile([1, 1, 0], (len(points), 1)))

        with patch("src.gair_ransac.inner_ransac.metal_available", return_value=True):
            with patch("src.gair_ransac.inner_ransac.fit_superquadric_metal_batch", side_effect=fake_batch):
                models = list(_inner_candidate_models(support, np.arange(100), support, 40, 83, np.random.default_rng(7), "radial", None))
        self.assertEqual(len(models), 83)
        self.assertEqual([len(batch) for batch in sampled_batches], [80, 3])
        rng = np.random.default_rng(7)
        expected = np.stack([support[rng.choice(np.arange(100), 40, replace=False)] for _ in range(83)])
        np.testing.assert_array_equal(np.concatenate(sampled_batches), expected)

    def test_cpu_inner_ransac_preserves_sequential_fitting(self):
        with patch("src.gair_ransac.inner_ransac.metal_available", return_value=False):
            with patch("src.gair_ransac.inner_ransac.fit_superquadric_metal_batch", side_effect=AssertionError):
                result = inner_ransac(self.points, np.arange(40), None, 0.03, n_iters=2, random_seed=7)
        self.assertEqual(result.best_inlier_count, 40)

    def test_expired_deadline_launches_no_fits(self):
        with patch("src.gair_ransac.inner_ransac.fit_superquadric_metal_batch", side_effect=AssertionError):
            result = inner_ransac(self.points, np.arange(40), None, 0.03, deadline=time.perf_counter() - 1)
        self.assertEqual(result.best_inlier_count, 0)

    def test_deadline_after_dispatch_keeps_a_completed_candidate(self):
        clock = [0.0]
        parameters = np.array([1.2, 0.8, 0.5, 1, 1, 0, 0, 0, 0, 0, 0])

        def fake_batch(points, *args, **kwargs):
            clock[0] = 2.0
            return MetalBatchFitResult(np.tile(parameters, (len(points), 1)), np.tile([1, 1, 0], (len(points), 1)))

        with patch("src.gair_ransac.inner_ransac.metal_available", return_value=True):
            with patch("src.gair_ransac.inner_ransac.time.perf_counter", side_effect=lambda: clock[0]):
                with patch("src.gair_ransac.inner_ransac.fit_superquadric_metal_batch", side_effect=fake_batch) as batches:
                    with patch("src.gair_ransac.inner_ransac.fit_superquadric_ls", side_effect=AssertionError("Refit after deadline")):
                        result = inner_ransac(self.points, np.arange(40), None, 0.03, deadline=1.0)
        self.assertEqual(batches.call_count, 1)
        self.assertEqual(result.best_inlier_count, 40)


@unittest.skipUnless(metal_available(), "An accessible Apple silicon Metal GPU and MLX are required")
class MetalFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mlx.core as mx
        cls.mx = mx
        directory = Path(importlib.import_module("src.gair_ransac.metal_superquadric").__file__).parent
        cls.evaluate = mx.fast.metal_kernel(
            name="test_superquadric_radial_jacobian",
            input_names=["points", "parameters"],
            output_names=["residuals", "jacobian"],
            header=(directory / "superquadric_fit_helpers.metal").read_text(),
            source="""
                uint i = thread_position_in_grid.x;
                float p[11], jac[11];
                for (int j = 0; j < 11; ++j) p[j] = parameters[j];
                float3 point(points[3*i], points[3*i+1], points[3*i+2]);
                residuals[i] = sq_radial<true>(point, p, 1e-12f, jac);
                for (int j = 0; j < 11; ++j) jacobian[11*i+j] = jac[j];
            """,
        )

    def test_radial_residual_and_analytic_jacobian_match_cpu(self):
        rng = np.random.default_rng(10)
        for e1, e2 in ((1, 1), (0.06, 0.1), (0.2, 2.8), (4.5, 4.5)):
            model = SuperQuadricParams(1.2, 0.8, 0.5, e1, e2, [0.4, -0.3, 0.2], [0.1, -0.2, 0.3])
            canonical = rng.uniform(-0.6, 0.6, (64, 3))
            points = (canonical @ model.rotation_matrix().T + model.t).astype(np.float32)
            parameters = np.array([model.a1, model.a2, model.a3, e1, e2, *model.rot, *model.t], dtype=np.float32)
            residuals, jacobian = self.evaluate(
                inputs=[self.mx.array(points), self.mx.array(parameters)],
                grid=(len(points), 1, 1), threadgroup=(32, 1, 1),
                output_shapes=[(len(points),), (len(points), 11)],
                output_dtypes=[self.mx.float32, self.mx.float32], stream=self.mx.gpu,
            )
            self.mx.eval(residuals, jacobian)
            reference_model = SuperQuadricParams(*parameters[:5], parameters[5:8], parameters[8:11])
            expected_residuals, expected_jacobian = superquadric_radial_residual_and_jacobian(reference_model, points)
            with self.subTest(e1=e1, e2=e2):
                np.testing.assert_allclose(np.asarray(residuals), expected_residuals, rtol=2e-4, atol=2e-5)
                np.testing.assert_allclose(np.asarray(jacobian), expected_jacobian, rtol=2e-3, atol=2e-4)

    def test_exact_axes_and_center_remain_finite(self):
        points = np.vstack((np.diag([0.3, 0.2, 0.1]), np.zeros((1, 3)))).astype(np.float32)
        for e1, e2 in ((0.06, 0.06), (0.2, 2.8), (4.5, 4.5)):
            parameters = np.array([1.2, 0.8, 0.5, e1, e2, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            residuals, jacobian = self.evaluate(
                inputs=[self.mx.array(points), self.mx.array(parameters)],
                grid=(len(points), 1, 1), threadgroup=(32, 1, 1),
                output_shapes=[(len(points),), (len(points), 11)],
                output_dtypes=[self.mx.float32, self.mx.float32], stream=self.mx.gpu,
            )
            self.mx.eval(residuals, jacobian)
            self.assertTrue(np.isfinite(np.asarray(residuals)).all())
            self.assertTrue(np.isfinite(np.asarray(jacobian)).all())

    def test_fit_quality_on_clean_noisy_and_small_samples(self):
        rng = np.random.default_rng(42)
        for e1, e2, count, noise in ((1, 1, 300, 0), (0.5, 0.7, 300, 0), (1.4, 1.8, 300, 0.01), (0.8, 1.2, 30, 0), (0.8, 1.2, 40, 0.005)):
            target = SuperQuadricParams(1.2, 0.8, 0.5, e1, e2, [0.4, -0.3, 0.2], [3, -2, 1])
            points = surface_points(target, count=count) + rng.normal(0, noise, (count, 3))
            with self.subTest(e1=e1, e2=e2, count=count, noise=noise):
                cpu = fit_superquadric_ls(points, backend="cpu")
                gpu = fit_superquadric_ls(points, backend="metal")
                diagonal = np.linalg.norm(np.ptp(points, axis=0))
                self.assertLessEqual(robust_cost(gpu, points), 1.25 * robust_cost(cpu, points) + diagonal**2 * 1e-6)

    def test_normalization_preserves_large_translation_and_small_scale(self):
        target = SuperQuadricParams(0.012, 0.008, 0.005, 0.7, 1.2, [0.4, -0.3, 0.2], [1e6, -2e6, 3e6])
        points = surface_points(target)
        model = fit_superquadric_ls(points, backend="metal")
        self.assertLess(robust_cost(model, points), 1e-10)

    def test_robust_fit_with_outliers(self):
        target = SuperQuadricParams(1.2, 0.8, 0.5, 0.5, 0.7, [0.4, -0.3, 0.2], [3, -2, 1])
        points = surface_points(target, count=300)
        rng = np.random.default_rng(5)
        points += rng.normal(0, 0.005, points.shape)
        points[:30] += rng.normal(0, 0.3, (30, 3))
        cpu = fit_superquadric_ls(points, backend="cpu")
        gpu = fit_superquadric_ls(points, backend="metal")
        self.assertLessEqual(robust_cost(gpu, points), 1.25 * robust_cost(cpu, points))
        residuals, _ = superquadric_radial_residual_and_jacobian(gpu, points[30:])
        self.assertLess(np.median(np.abs(residuals)), 0.01)

    def test_bounds_and_evaluation_limit(self):
        target = SuperQuadricParams(1.2, 0.8, 0.5, 0.7, 1.2)
        points = surface_points(target)
        initial = np.array([1, 1, 1, 1.5, 1.5, 0, 0, 0, 0, 0, 0], dtype=float)
        lower = np.array([0.1, 0.1, 0.1, 0.06, 0.06, -np.pi, -np.pi, -np.pi, -0.5, -0.5, -0.5])
        upper = np.array([1.0, 1.5, 1.5, 4.5, 4.5, np.pi, np.pi, np.pi, 0.5, 0.5, 0.5])
        fitted = fit_superquadric_metal(points, initial, lower, upper, 0.05, 3.0)
        self.assertTrue(np.all(fitted >= lower))
        self.assertTrue(np.all(fitted <= upper))
        with self.assertRaisesRegex(RuntimeError, "maximum function evaluations"):
            fit_superquadric_metal(points, initial, lower, upper, 0.05, 3.0, max_nfev=1)

    def test_auto_and_inner_ransac_never_call_scipy(self):
        target = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2)
        points = surface_points(target, count=80)
        with patch("src.gair_ransac.inner_ransac.least_squares", side_effect=AssertionError("CPU solver called")):
            fit_superquadric_ls(points)
            result = inner_ransac(points, np.arange(len(points)), None, 0.03, n_iters=3, random_seed=42)
        self.assertGreater(result.best_inlier_count, 60)

    def test_80_parallel_fits_match_individual_metal_fits(self):
        target = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2, [0.4, -0.3, 0.2], [3, -2, 1])
        points = surface_points(target, count=300)
        rng = np.random.default_rng(42)
        points += rng.normal(0, 0.005, points.shape)
        samples = np.stack([points[rng.choice(len(points), 40, replace=False)] for _ in range(80)])
        lower, upper, loss_scale, diagonal = _optimization_bounds(points)
        initial = np.clip(_pca_initial_parameters(samples), lower, upper)
        result = fit_superquadric_metal_batch(samples, initial, lower, upper, loss_scale, diagonal)
        self.assertTrue(result.success.all())
        for index, sample in enumerate(samples):
            with self.subTest(fit=index):
                single = fit_superquadric_metal(sample, initial[index], lower, upper, loss_scale, diagonal)
                np.testing.assert_allclose(result.parameters[index], single, rtol=2e-4, atol=2e-5)

    def test_two_simd_groups_include_all_40_points(self):
        target = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2)
        points = surface_points(target, count=40)
        points[32:] += np.array([0.4, 0.2, -0.3])
        lower, upper, loss_scale, diagonal = _optimization_bounds(points)
        initial = np.clip(_pca_initial_parameters(points[None]), lower, upper)
        result = fit_superquadric_metal_batch(points[None], initial, lower, upper, loss_scale, diagonal, max_nfev=1)
        residuals, _ = superquadric_radial_residual_and_jacobian(_model_from_parameters(initial[0]), points)
        expected_cost = np.mean(residuals**2 / (np.sqrt(1 + (residuals / loss_scale)**2) + 1)) / diagonal**2
        np.testing.assert_allclose(result.diagnostics[0, 2], expected_cost, rtol=2e-5, atol=1e-8)

    def test_one_failed_fit_does_not_discard_other_batch_entries(self):
        target = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2)
        samples = np.stack([surface_points(target, count=40, seed=seed) for seed in (4, 5, 6)])
        lower, upper, loss_scale, diagonal = _optimization_bounds(samples.reshape(-1, 3))
        initial = np.clip(_pca_initial_parameters(samples), lower, upper)
        result = fit_superquadric_metal_batch(samples, initial, lower, upper, loss_scale, diagonal, max_nfev=np.array([1000, 1, 1000]))
        np.testing.assert_array_equal(result.success, [True, False, True])

    def test_default_inner_ransac_submits_all_80_fits_once(self):
        target = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2)
        points = surface_points(target, count=160)
        with patch("src.gair_ransac.inner_ransac.fit_superquadric_metal_batch", wraps=fit_superquadric_metal_batch) as batches:
            result = inner_ransac(points, np.arange(160), None, 0.03, random_seed=42)
        self.assertEqual(batches.call_count, 1)
        self.assertEqual(batches.call_args.args[0].shape, (80, 40, 3))
        self.assertGreater(result.best_inlier_count, 150)


if __name__ == "__main__":
    unittest.main()
