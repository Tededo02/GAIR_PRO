import importlib
import unittest
from unittest.mock import patch

import numpy as np

from src.gair_ransac.consensus import (
    compute_consensus, distance_err, expanded_removal_mask, normal_alignment_score,
)
from src.gair_ransac.inner_ransac import inner_ransac
from src.gair_ransac.metal_consensus import (
    MetalConsensusContext, MetalConsensusResult, create_metal_consensus_context,
)
from src.gair_ransac.metal_superquadric import metal_available
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.superquadric_residual import superquadric_normal_world
from test_metal_superquadric import surface_points


class CpuConsensusTests(unittest.TestCase):
    def test_other_metrics_stop_between_cpu_candidate_evaluations(self):
        model = SuperQuadricParams(1, 1, 1, 1, 1)
        points = surface_points(model, count=40)
        clock = [0.0]

        def score(*args, **kwargs):
            clock[0] = 2.0
            return np.ones(len(points), dtype=bool)

        with patch("src.gair_ransac.inner_ransac._inner_candidate_model_batches", return_value=iter([[model] * 80])):
            with patch("src.gair_ransac.inner_ransac.time.perf_counter", side_effect=lambda: clock[0]):
                with patch("src.gair_ransac.inner_ransac.create_metal_consensus_context", return_value=None):
                    with patch("src.gair_ransac.inner_ransac.compute_consensus", side_effect=score) as calls:
                        result = inner_ransac(points, np.arange(40), None, 0.01, consensus_metric="first_order", deadline=1.0)
        self.assertEqual(calls.call_count, 1)
        self.assertEqual(result.best_inlier_count, 40)

    def test_cpu_checks_normals_only_after_the_residual_threshold(self):
        model = SuperQuadricParams(1, 1, 1, 1, 1)
        points = np.array([[1, 0, 0], [0, 1, 0], [10, 10, 10]], dtype=float)
        normals = np.array([[1, 0, 0], [0, -1, 0], [1, 0, 0]], dtype=float)
        with patch("src.gair_ransac.consensus.normal_alignment_score", wraps=normal_alignment_score) as alignment:
            result = compute_consensus(model, points, 0.01, normals=normals)
        np.testing.assert_array_equal(result, [True, False, False])
        self.assertEqual(alignment.call_args.args[1].shape, (2, 3))

    def test_cpu_skips_normal_evaluation_when_no_point_passes(self):
        model = SuperQuadricParams(1, 1, 1, 1, 1)
        points = np.array([[10, 10, 10]], dtype=float)
        with patch("src.gair_ransac.consensus.normal_alignment_score", side_effect=AssertionError):
            self.assertFalse(compute_consensus(model, points, 0.01, normals=np.ones_like(points)).any())

    def test_cpu_matches_the_original_full_normal_check_and_removal(self):
        model = SuperQuadricParams(1.2, 0.8, 0.5, 0.5, 2.8, [0.4, -0.3, 0.2], [3, -2, 1])
        points = surface_points(model, count=500)
        rng = np.random.default_rng(8)
        points += rng.normal(0, 0.02, points.shape)
        normals = rng.normal(size=points.shape)
        for metric in ("radial", "mix", "first_order"):
            expected = (distance_err(model, points, metric) < 0.01) & (normal_alignment_score(model, points, normals) >= 0.2)
            np.testing.assert_array_equal(compute_consensus(model, points, 0.01, metric, normals, 0.2), expected)
            removal = (distance_err(model, points, metric) <= 0.015) & (normal_alignment_score(model, points, normals) >= 0.2)
            np.testing.assert_array_equal(expanded_removal_mask(model, points, 0.01, 1.5, metric, normals, 0.2), removal)

    def test_context_factory_uses_cpu_without_metal_or_for_other_metrics(self):
        points = np.zeros((3, 3))
        with patch("src.gair_ransac.metal_consensus.metal_available", return_value=False):
            self.assertIsNone(create_metal_consensus_context(points))
        with patch("src.gair_ransac.metal_consensus.metal_available", side_effect=AssertionError):
            self.assertIsNone(create_metal_consensus_context(points, error_metric="first_order"))


@unittest.skipUnless(metal_available(), "An accessible Apple silicon Metal GPU and MLX are required")
class MetalConsensusTests(unittest.TestCase):
    def test_float32_inputs_reuse_their_context(self):
        model = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2)
        points = surface_points(model, count=40).astype(np.float32)
        normals = superquadric_normal_world(model, points).astype(np.float32)
        context = MetalConsensusContext(points, normals)
        expected = compute_consensus(model, points, 0.01, normals=normals)
        np.testing.assert_array_equal(compute_consensus(model, points, 0.01, normals=normals, metal_context=context), expected)

    def assert_consensus_matches(self, context, models, threshold, normal_cos_threshold=0.0):
        result = context.evaluate(models, threshold, normal_cos_threshold)
        expected = [compute_consensus(model, context.points, threshold, normals=context.normals, normal_cos_threshold=normal_cos_threshold) for model in models]
        np.testing.assert_array_equal(result.counts, [mask.sum() for mask in expected])
        for index in range(len(models)):
            np.testing.assert_array_equal(result.mask(index), expected[index])
        self.assertEqual(result.best_index, int(np.argmax([mask.sum() for mask in expected])))
        return result

    def test_80_models_normals_and_non_multiple_threadgroup_point_count(self):
        target = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2, [0.4, -0.3, 0.2], [3, -2, 1])
        points = surface_points(target, count=521)
        normals = superquadric_normal_world(target, points)
        rng = np.random.default_rng(2)
        normals[::7] *= -2
        normals[::11] = 0
        points += rng.normal(0, 0.02, points.shape)
        models = [SuperQuadricParams(target.a1, target.a2, target.a3, target.e1, target.e2, target.rot, target.t + rng.normal(0, 0.04, 3)) for _ in range(80)]
        context = MetalConsensusContext(points, normals)
        original_buffers = (context._gpu_points, context._gpu_normals)
        for cos_threshold in (0.0, 0.6):
            self.assert_consensus_matches(context, models, 0.01, cos_threshold)
        self.assertIs(context._gpu_points, original_buffers[0])
        self.assertIs(context._gpu_normals, original_buffers[1])

    def test_extreme_exponents_axes_center_and_distant_points(self):
        rng = np.random.default_rng(5)
        for e1, e2 in ((0.06, 0.06), (0.06, 4.5), (0.2, 2.8), (4.5, 4.5)):
            model = SuperQuadricParams(1.2, 0.8, 0.5, e1, e2, [0.4, -0.3, 0.2], [3, -2, 1])
            surface = surface_points(model, count=513)
            axes = np.vstack((np.diag([1.2, 0.8, 0.5]), -np.diag([1.2, 0.8, 0.5]), np.zeros((1, 3))))
            points = np.vstack((surface + rng.normal(0, 0.003, surface.shape), axes @ model.rotation_matrix().T + model.t, rng.uniform(-100, 100, (300, 3))))
            normals = rng.normal(size=points.shape)
            with self.subTest(e1=e1, e2=e2):
                self.assert_consensus_matches(MetalConsensusContext(points), [model], 0.01)
                self.assert_consensus_matches(MetalConsensusContext(points, normals), [model], 0.01)

    def test_distance_and_normal_threshold_boundaries_use_float64(self):
        model = SuperQuadricParams(1, 1, 1, 1, 1)
        threshold = 0.01
        radii = 1 + threshold + np.array([-1e-7, -1e-10, 0, 1e-10, 1e-7])
        points = np.column_stack((radii, np.zeros_like(radii), np.zeros_like(radii)))
        self.assert_consensus_matches(MetalConsensusContext(points), [model], threshold)
        points = np.tile([1.0, 0, 0], (7, 1))
        normals = np.column_stack(([-1e-7, -1e-12, 0, 1e-12, 1e-7, 1, -1], np.ones(7), np.zeros(7)))
        result = self.assert_consensus_matches(MetalConsensusContext(points, normals), [model], threshold)
        self.assertTrue(result._corrections)

    def test_large_translation_and_small_scale(self):
        model = SuperQuadricParams(0.012, 0.008, 0.005, 0.7, 1.2, [0.4, -0.3, 0.2], [1e6, -2e6, 3e6])
        points = surface_points(model, count=517)
        normals = superquadric_normal_world(model, points)
        points += np.random.default_rng(2).normal(0, 1e-4, points.shape)
        self.assert_consensus_matches(MetalConsensusContext(points, normals), [model], 1e-4)

    def test_empty_cloud_empty_batch_ties_and_invalid_context(self):
        model = SuperQuadricParams(1, 1, 1, 1, 1)
        empty = MetalConsensusContext(np.empty((0, 3)))
        result = empty.evaluate([model], 0.01)
        self.assertEqual(result.counts[0], 0)
        self.assertEqual(result.mask(0).shape, (0,))
        points = surface_points(model, count=40)
        context = MetalConsensusContext(points)
        self.assertEqual(context.evaluate([], 0.01).counts.size, 0)
        tied = self.assert_consensus_matches(context, [model, model], 0.01)
        self.assertEqual(tied.best_index, 0)
        with self.assertRaises(ValueError):
            compute_consensus(model, points.copy(), 0.01, metal_context=context)
        with self.assertRaises(ValueError):
            compute_consensus(model, points, 0.01, error_metric="first_order", metal_context=context)

    def test_inner_scores_one_80_model_batch_and_transfers_only_winner_mask(self):
        model = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2)
        points = surface_points(model, count=1000)
        normals = superquadric_normal_world(model, points)
        context = MetalConsensusContext(points, normals)
        with patch.object(context, "evaluate", wraps=context.evaluate) as batches:
            with patch.object(MetalConsensusResult, "mask", autospec=True, side_effect=MetalConsensusResult.mask) as masks:
                result = inner_ransac(points, np.arange(len(points)), None, 0.03, normals=normals, random_seed=42, consensus_context=context)
        self.assertEqual(len(batches.call_args_list[0].args[0]), 80)
        self.assertEqual(batches.call_count, 2)
        self.assertEqual(masks.call_count, 2)
        self.assertEqual(result.best_inlier_count, len(points))
        with patch("src.gair_ransac.inner_ransac.create_metal_consensus_context", return_value=None):
            cpu_result = inner_ransac(points, np.arange(len(points)), None, 0.03, normals=normals, random_seed=42)
        np.testing.assert_array_equal(result.best_inliers_mask, cpu_result.best_inliers_mask)

    def test_inner_actual_subset_and_more_than_80_hypotheses(self):
        model = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2)
        points = surface_points(model, count=300)
        normals = superquadric_normal_world(model, points)
        actual = np.arange(0, len(points), 2)
        result = inner_ransac(points, np.arange(len(points)), actual, 0.03, normals=normals, n_iters=83, random_seed=42)
        self.assertEqual(result.best_inlier_count, len(actual))
        self.assertEqual(result.best_inliers_mask.shape, (len(actual),))

    def test_gair_reuses_one_context_in_outer_inner_and_final_consensus(self):
        gr = importlib.import_module("src.gair_ransac.gair_ransac")
        model = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2)
        points = surface_points(model, count=300)
        normals = superquadric_normal_world(model, points)
        with patch.object(gr, "create_metal_consensus_context", wraps=create_metal_consensus_context) as contexts:
            models, masks, _, _ = gr.gair_ransac(points, normals, threshold=0.03, max_models=1, max_iterations=2, inner_iterations=3, min_inliers=20, random_seed=42)
        self.assertEqual(contexts.call_count, 1)
        self.assertEqual(len(models), 1)
        self.assertGreaterEqual(masks[0].sum(), 20)
        with patch.object(gr, "create_metal_consensus_context", return_value=None):
            with patch("src.gair_ransac.inner_ransac.create_metal_consensus_context", return_value=None):
                cpu_models, cpu_masks, _, _ = gr.gair_ransac(points, normals, threshold=0.03, max_models=1, max_iterations=2, inner_iterations=3, min_inliers=20, random_seed=42)
        self.assertEqual(len(cpu_models), len(models))
        np.testing.assert_array_equal(masks[0], cpu_masks[0])


if __name__ == "__main__":
    unittest.main()
