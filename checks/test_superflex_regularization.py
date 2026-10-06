# Run with: .venv/bin/python -m unittest discover -s checks -v

from pathlib import Path
import unittest

import numpy as np

from test_superflex import fixture_model, surface
from src.gair_ransac.deformable_fitting import _pack, _unpack
from src.gair_ransac.inner_ransac import fit_superquadric_ls
from src.gair_ransac.metal_superquadric import _metal_runtime, fit_superquadric_metal_batch, metal_available
from src.gair_ransac.superflex_regularization import superflex_axis_penalty_residual_and_jacobian
from src.superquadrics.deformable_superquadric import DeformableSuperQuadricParams
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.superquadric_residual import superquadric_radial_residual


def ring_points(count=40):
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.column_stack((np.cos(angle), np.sin(angle), np.zeros(count)))


class SuperFlexRegularizationTests(unittest.TestCase):
    def test_analytic_derivatives_for_small_and_oversized_axes(self):
        rotation = SuperQuadricParams(1, 1, 1, 1, 1, [0.6, -0.2, 0.1]).rotation_matrix()
        box = rotation * np.array([1.0, 0.5, 0.2])
        for axes in ([0.1, 0.1, 0.1], [2.0, 1.7, 1.4]):
            parameters = np.r_[axes, 1, 1, -0.4, 0.3, -0.2, np.zeros(11)]
            _, analytic = superflex_axis_penalty_residual_and_jacobian(parameters, box, 0.13)
            step = 1e-6
            numeric = np.column_stack([
                (superflex_axis_penalty_residual_and_jacobian(parameters + step * direction, box, 0.13)[0]
                 - superflex_axis_penalty_residual_and_jacobian(parameters - step * direction, box, 0.13)[0]) / (2 * step)
                for direction in np.eye(19)
            ])
            np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=1e-9)

    def test_normalized_penalty_is_independent_of_world_units(self):
        model = DeformableSuperQuadricParams(1.3, 0.6, 0.2, 1, 1, [0.4, -0.3, 0.2])
        parameters = _pack(model, 1.0)
        box = np.diag([0.7, 0.3, 0.4])
        expected, expected_jacobian = superflex_axis_penalty_residual_and_jacobian(parameters, box, 0.13)
        for scale in (1e-4, 1e4):
            scaled = parameters.copy()
            scaled[:3] *= scale
            actual, jacobian = superflex_axis_penalty_residual_and_jacobian(scaled, box * scale, 0.13)
            np.testing.assert_allclose(actual, expected, rtol=1e-13)
            jacobian[:, :3] *= scale
            np.testing.assert_allclose(jacobian, expected_jacobian, rtol=1e-13, atol=1e-14)

    def test_added_size_penalty_is_invariant_to_rotation(self):
        box = np.diag([1.0, 0.5, 0.2])
        model = DeformableSuperQuadricParams(0.05, 0.1, 0.15, 1, 1)
        parameters = _pack(model, 1.0)
        expected, _ = superflex_axis_penalty_residual_and_jacobian(parameters, box, 0.13)
        self.assertTrue((expected > 0).all())
        for rotation in ([0.6, -0.2, 0.1], [-0.4, 0.3, -0.2], [2.0, 1.0, -1.2]):
            parameters[5:8] = rotation
            actual, jacobian = superflex_axis_penalty_residual_and_jacobian(parameters, box, 0.13)
            np.testing.assert_allclose(actual, expected, rtol=1e-14)
            np.testing.assert_array_equal(jacobian[:, 5:8], 0.0)

    def test_cpu_reference_prefers_compact_axes_and_keeps_the_deformed_surface(self):
        seed = fixture_model()
        points = surface(seed, 10, 16)
        unregularized = fit_superquadric_ls(points, initial_model=seed,
                                          backend="cpu", model_family="superflex", axis_penalty_weight=0)
        compact = fit_superquadric_ls(points, initial_model=seed,
                                     backend="cpu", model_family="superflex", axis_penalty_weight=0.1)
        size = lambda model: model.a1**2 + model.a2**2 + model.a3**2
        self.assertLess(size(compact), size(unregularized))
        self.assertLess(float(np.max(np.abs(superquadric_radial_residual(compact, points)))), 0.001)


@unittest.skipUnless(metal_available(), "An accessible Metal GPU is required")
class SuperFlexMetalRegularizationTests(unittest.TestCase):
    def test_gpu_penalty_and_all_derivatives_match_cpu(self):
        mx = _metal_runtime()
        directory = Path(__file__).resolve().parents[1] / "src/gair_ransac"
        evaluate = mx.fast.metal_kernel(
            name="check_superflex_axis_penalty",
            input_names=["parameters", "box"], output_names=["residual", "jacobian"],
            header=(directory / "superquadric_fit_helpers.metal").read_text() + "\n" + (directory / "superflex_geometry.metal").read_text(),
            source="""
                uint axis = thread_position_in_grid.x;
                if (axis < 3) {
                    float p[19], b[9], j[19];
                    for (int k = 0; k < 19; ++k) p[k] = parameters[k];
                    for (int k = 0; k < 9; ++k) b[k] = box[k];
                    residual[axis] = sq_model_axis_penalty<19, true>(axis, p, b, 0.13f, j);
                    for (int k = 0; k < 19; ++k) jacobian[19*axis+k] = j[k];
                }
            """,
        )
        rotation = SuperQuadricParams(1, 1, 1, 1, 1, [0.6, -0.2, 0.1]).rotation_matrix()
        box = (rotation * np.array([1.0, 0.5, 0.2])).astype(np.float32)
        for axes in ([0.1, 0.1, 0.1], [2.0, 1.7, 1.4]):
            parameters = np.r_[axes, 1, 1, -0.4, 0.3, -0.2, np.zeros(11)].astype(np.float32)
            actual, jacobian = evaluate(inputs=[mx.array(parameters), mx.array(box)],
                grid=(32, 1, 1), threadgroup=(32, 1, 1), output_shapes=[(3,), (3, 19)],
                output_dtypes=[mx.float32, mx.float32], stream=mx.gpu)
            mx.eval(actual, jacobian)
            expected, expected_jacobian = superflex_axis_penalty_residual_and_jacobian(parameters, box, 0.13)
            np.testing.assert_allclose(np.array(actual), expected, rtol=2e-5, atol=1e-7)
            np.testing.assert_allclose(np.array(jacobian), expected_jacobian, rtol=2e-5, atol=1e-7)

    def test_metal_shrinks_an_unsupported_axis_inside_the_box_only_for_superflex(self):
        points = ring_points()
        seed = _pack(DeformableSuperQuadricParams(1, 1, 0.9, 1, 1), 1.0)
        lower, upper = seed - 1e-6, seed + 1e-6
        lower[2], upper[2] = 0.2, 2.0
        box = np.diag([1.3, 1.3, 1.5])
        compact_axes = []
        for family, count in (("rigid", 11), ("superflex", 19)):
            for weight in (0.1, 0.5):
                result = fit_superquadric_metal_batch(points[None], seed[None, :count], lower[:count], upper[:count],
                    0.05, 3.0, axis_penalty_weight=weight, axis_support=box, model_family=family)
                self.assertTrue(result.success[0], result.diagnostics)
                if family == "rigid":
                    self.assertAlmostEqual(result.parameters[0, 2], 0.9, places=5)
                else:
                    compact_axes.append(result.parameters[0, 2])
                    compact = _unpack(result.parameters[0], 3.0)
                    self.assertLess(float(np.max(np.abs(superquadric_radial_residual(compact, points)))), 0.001)
        self.assertLess(compact_axes[0], seed[2])
        self.assertLess(compact_axes[1], compact_axes[0])
        self.assertLess(compact_axes[1], seed[2] / 2)

    def test_objective_is_consistent_across_sample_counts_and_unit_scales(self):
        seed = _pack(DeformableSuperQuadricParams(1, 1, 2, 1, 1), 1.0)
        box = np.diag([1.3, 1.3, 0.5])
        weight, loss_scale, diagonal = 0.1, 0.05, 3.0
        penalty, _ = superflex_axis_penalty_residual_and_jacobian(seed, box, loss_scale * np.sqrt(weight))
        expected = 0.5 * np.dot(penalty, penalty) / diagonal**2
        for count, units in ((40, 1.0), (80, 1.0), (40, 1e-3), (40, 1e3)):
            points = ring_points(count) * units
            parameters = seed.copy()
            parameters[:3] *= units
            result = fit_superquadric_metal_batch(points[None], parameters[None], parameters - 0.1 * units,
                parameters + 0.1 * units, loss_scale * units, diagonal * units, max_nfev=1,
                axis_penalty_weight=weight, axis_support=box * units, model_family="superflex")
            self.assertAlmostEqual(float(result.diagnostics[0, 2]), expected, delta=expected * 2e-5)


if __name__ == "__main__":
    unittest.main()
