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
from src.gair_ransac.interior_consensus import (
    InteriorPenaltyContext, consensus_scores, interior_strength,
)


def enclosed_structure_scene():
    angles = np.linspace(0, 2*np.pi, 128, endpoint=False)
    ring = np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles)))
    angles = angles[:16]
    cap = np.column_stack((0.1*np.cos(angles), 0.1*np.sin(angles), np.full(16, 2*np.sqrt(0.99))))
    nested = SuperQuadricParams(0.05, 0.05, 0.05, 1, 1, t=[0, 0, 0.8])
    points = np.vstack((ring, cap, surface_points(nested, count=48)))
    large = SuperQuadricParams(1, 1, 2, 1, 1)
    small = SuperQuadricParams(1, 1, 0.2, 1, 1)
    return points, large, small, nested


class InteriorConsensusTests(unittest.TestCase):
    def test_strength_is_smooth_bounded_and_zero_in_the_surface_band(self):
        model = SuperQuadricParams(1, 1, 1, 1, 1)
        radii = np.array([1.1, 1, 0.995, 0.99, 0.99-1e-6, 0.98, 0.9, 0])
        points = np.column_stack((radii, np.zeros((len(radii), 2))))
        strength = interior_strength(model, points, 0.01)
        np.testing.assert_allclose(strength[:4], 0, atol=1e-25)
        self.assertGreater(strength[4], 0)
        self.assertLess(strength[4], 1e-7)
        self.assertAlmostEqual(strength[5], 0.5)
        self.assertTrue(np.all(np.diff(strength) >= -1e-25))
        self.assertLess(strength[-1], 1)
        self.assertGreater(strength[-1], 0.999)

    def test_coherent_cluster_has_more_mass_than_an_isolated_point(self):
        model = SuperQuadricParams(1, 1, 1, 1, 1)
        shell = surface_points(model, count=400)
        isolated = InteriorPenaltyContext(np.vstack((shell, [[0, 0, 0]])))
        cluster = np.random.default_rng(2).normal(0, 0.03, (48, 3))
        coherent = InteriorPenaltyContext(np.vstack((shell, cluster)))
        self.assertAlmostEqual(isolated.mass(model, 0.01), 0)
        self.assertGreater(coherent.mass(model, 0.01), 47)
        self.assertLessEqual(coherent.mass(model, 0.01), 48)

    def test_penalty_is_relative_and_has_no_count_cutoff(self):
        masses = np.array([0, 10, 100, 300, 1000], dtype=float)
        scores = consensus_scores(1000, masses, 25)
        np.testing.assert_allclose(scores, [1000, 997.5062344, 800, 307.6923077, 38.46153846])
        np.testing.assert_allclose(consensus_scores(10000, masses*10, 25), scores*10)
        np.testing.assert_array_equal(consensus_scores([10, 100], [1000, 1000], 0), [10, 100])

    def test_scale_rotation_translation_and_extreme_exponents(self):
        for e1, e2 in ((0.06, 0.06), (0.06, 4.5), (4.5, 0.06), (4.5, 4.5)):
            model = SuperQuadricParams(1.2, 0.8, 0.5, e1, e2)
            surface = surface_points(model, count=200)
            points = np.vstack((surface, surface*0.8, surface*0.1, [[0, 0, 0]]))
            strength = interior_strength(model, points, 0.01)
            transformed = SuperQuadricParams(0.012, 0.008, 0.005, e1, e2, [0.4, -0.3, 0.2], [1e6, -2e6, 3e6])
            moved = 0.01*points @ transformed.rotation_matrix().T + transformed.t
            # World-coordinate float64 quantization affects sharp shapes close to their canonical axes.
            np.testing.assert_allclose(interior_strength(transformed, moved, 1e-4), strength, atol=5e-4)
            self.assertTrue(np.isfinite(strength).all())
            self.assertTrue(np.all((strength >= 0) & (strength <= 1)))

    def test_cpu_inner_prefers_fewer_inliers_and_rejects_enclosing_refit(self):
        points, large, small, _ = enclosed_structure_scene()
        context = InteriorPenaltyContext(points)
        with patch("src.gair_ransac.inner_ransac._inner_candidate_model_batches", return_value=iter([[large, small]])):
            with patch("src.gair_ransac.inner_ransac.metal_available", return_value=False):
                with patch("src.gair_ransac.inner_ransac.fit_superquadric_ls", return_value=large):
                    result = inner_ransac(points, np.arange(len(points)), None, 0.01, interior_context=context, axis_penalty_weight=0)
        self.assertIs(result.best_model, small)
        self.assertEqual(result.best_inlier_count, 128)
        self.assertAlmostEqual(result.best_score, 128)
        self.assertGreater(compute_consensus(large, points, 0.01).sum(), result.best_inlier_count)

    def test_outer_selection_and_final_refit_use_score_and_zero_restores_counts(self):
        gr = importlib.import_module("src.gair_ransac.gair_ransac")
        points, large, small, _ = enclosed_structure_scene()
        for weight, expected in ((25, small), (0, large)):
            with self.subTest(weight=weight):
                with patch.object(gr, "create_metal_consensus_context", return_value=None):
                    with patch.object(gr, "fit_superquadric_ls", side_effect=[large, small, large]):
                        with patch.object(gr, "gair", return_value=np.zeros(len(points), dtype=bool)):
                            models, masks, _, _ = gr.gair_ransac(
                                points, threshold=0.01, max_iterations=2, sample_size=20,
                                min_inliers=20, random_seed=7, axis_penalty_weight=0, interior_penalty_weight=weight,
                            )
            self.assertIs(models[0], expected)
            self.assertEqual(masks[0].sum(), 128 if weight else 144)

    def test_local_optimization_accepts_smaller_support_when_score_improves(self):
        gr = importlib.import_module("src.gair_ransac.gair_ransac")
        points, large, small, _ = enclosed_structure_scene()
        mask = compute_consensus(small, points, 0.01)
        result = gr.InnerRansacResult(small, int(mask.sum()), mask)
        with patch.object(gr, "create_metal_consensus_context", return_value=None):
            with patch.object(gr, "fit_superquadric_ls", return_value=large):
                with patch.object(gr, "gair", return_value=np.ones(len(points), dtype=bool)):
                    with patch.object(gr, "inner_ransac", return_value=result) as refinements:
                        models, masks, _, _ = gr.gair_ransac(
                            points, threshold=0.01, max_iterations=1, sample_size=20,
                            min_inliers=20, random_seed=7, axis_penalty_weight=0,
                        )
        self.assertIs(models[0], small)
        self.assertEqual(masks[0].sum(), 128)
        self.assertEqual(refinements.call_count, 2)

    def test_already_extracted_structure_still_penalizes_later_models(self):
        gr = importlib.import_module("src.gair_ransac.gair_ransac")
        points, large, small, nested = enclosed_structure_scene()
        with patch.object(gr, "create_metal_consensus_context", return_value=None):
            with patch.object(gr, "fit_superquadric_ls", side_effect=[nested, nested, nested, large, small, large]):
                with patch.object(gr, "gair", return_value=np.zeros(len(points), dtype=bool)):
                    models, masks, _, _ = gr.gair_ransac(
                        points, threshold=0.01, max_iterations=2, max_models=2,
                        sample_size=20, min_inliers=20, random_seed=7, axis_penalty_weight=0,
                    )
        self.assertEqual(len(models), 2)
        self.assertIs(models[0], nested)
        self.assertIs(models[1], small)
        self.assertEqual(masks[0].sum(), 48)
        self.assertEqual(masks[1].sum(), 128)

    def test_invalid_weights_and_thresholds_fail_before_fitting(self):
        gr = importlib.import_module("src.gair_ransac.gair_ransac")
        points, _, _, _ = enclosed_structure_scene()
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "interior_penalty_weight"):
                gr.gair_ransac(points, interior_penalty_weight=weight)
        for threshold in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "threshold"):
                gr.gair_ransac(points, threshold=threshold)

    def test_inner_keeps_minimum_support_requirement_when_scoring(self):
        points, large, _, nested = enclosed_structure_scene()
        interior = InteriorPenaltyContext(points)
        with patch("src.gair_ransac.inner_ransac._inner_candidate_model_batches", return_value=iter([[large, nested]])):
            with patch("src.gair_ransac.inner_ransac.metal_available", return_value=False):
                with patch("src.gair_ransac.inner_ransac.fit_superquadric_ls", return_value=nested):
                    result = inner_ransac(points, np.arange(len(points)), None, 0.01, interior_context=interior,
                                          axis_penalty_weight=0, min_inliers=100)
        self.assertIs(result.best_model, large)
        self.assertEqual(result.best_inlier_count, 144)

    def test_ragged_graph_and_points_without_neighbors_have_finite_mass(self):
        model = SuperQuadricParams(1, 1, 1, 1, 1)
        points = np.array([[0, 0, 0], [0.1, 0, 0], [1, 0, 0]], dtype=float)
        interior = InteriorPenaltyContext(points, neighbors=[[1, 2], [0], []])
        strength = interior_strength(model, points, 0.01)
        expected = strength[0]*strength[1]/2 + strength[1]*strength[0]
        self.assertAlmostEqual(interior.mass(model, 0.01), expected)
        isolated = InteriorPenaltyContext(points, neighbors=[[], [], []])
        self.assertAlmostEqual(isolated.mass(model, 0.01), strength.sum())


@unittest.skipUnless(metal_available(), "An accessible Apple silicon Metal GPU and MLX are required")
class MetalInteriorConsensusTests(unittest.TestCase):
    def test_graph_with_no_neighbors_and_empty_inputs(self):
        model = SuperQuadricParams(1, 1, 1, 1, 1)
        points = np.array([[0, 0, 0], [0.1, 0, 0], [1, 0, 0]], dtype=float)
        interior = InteriorPenaltyContext(points, neighbors=[[], [], []])
        context = MetalConsensusContext(points, interior_context=interior)
        result = context.evaluate([model], 0.01)
        self.assertAlmostEqual(result.interior_masses[0], interior.mass(model, 0.01), places=5)
        empty = MetalConsensusContext(np.empty((0, 3)), interior_context=InteriorPenaltyContext(np.empty((0, 3))))
        result = empty.evaluate([model], 0.01)
        np.testing.assert_array_equal(result.scores, [0])
        np.testing.assert_array_equal(result.interior_masses, [0])
        self.assertEqual(result.mask(0).shape, (0,))
        self.assertEqual(context.evaluate([], 0.01).scores.shape, (0,))

    def test_inner_minimum_support_filters_batch_and_refit(self):
        points, large, _, nested = enclosed_structure_scene()
        interior = InteriorPenaltyContext(points)
        context = MetalConsensusContext(points, interior_context=interior)
        with patch("src.gair_ransac.inner_ransac._inner_candidate_model_batches", return_value=iter([[large, nested]])):
            with patch("src.gair_ransac.inner_ransac.fit_superquadric_ls", return_value=nested):
                result = inner_ransac(points, np.arange(len(points)), None, 0.01, consensus_context=context,
                                      axis_penalty_weight=0, min_inliers=100)
        self.assertIs(result.best_model, large)
        self.assertEqual(result.best_inlier_count, 144)

    def test_two_extractions_reuse_gpu_reference_and_penalize_removed_structure(self):
        gr = importlib.import_module("src.gair_ransac.gair_ransac")
        points, large, small, nested = enclosed_structure_scene()
        with patch.object(gr, "create_metal_consensus_context", wraps=create_metal_consensus_context) as contexts:
            with patch.object(gr, "fit_superquadric_ls", side_effect=[nested, nested, nested, large, small, large]):
                with patch.object(gr, "gair", return_value=np.zeros(len(points), dtype=bool)):
                    models, masks, _, _ = gr.gair_ransac(
                        points, threshold=0.01, max_iterations=2, max_models=2,
                        sample_size=20, min_inliers=20, random_seed=7, axis_penalty_weight=0,
                    )
        self.assertEqual(contexts.call_count, 1)
        self.assertEqual(len(models), 2)
        self.assertIs(models[0], nested)
        self.assertIs(models[1], small)
        self.assertEqual(masks[0].sum(), 48)
        self.assertEqual(masks[1].sum(), 128)

    def test_80_model_scores_and_winner_match_cpu(self):
        points, large, small, _ = enclosed_structure_scene()
        interior = InteriorPenaltyContext(points)
        context = MetalConsensusContext(points, interior_context=interior)
        models = [large, small]*40
        result = context.evaluate(models, 0.01)
        expected_counts = [compute_consensus(model, points, 0.01).sum() for model in models]
        expected_mass = [interior.mass(model, 0.01) for model in models]
        np.testing.assert_array_equal(result.counts, expected_counts)
        np.testing.assert_allclose(result.interior_masses, expected_mass, rtol=2e-5, atol=1e-5)
        np.testing.assert_allclose(result.scores, consensus_scores(expected_counts, expected_mass, 25), rtol=2e-5)
        self.assertEqual(result.best_index, 1)
        np.testing.assert_array_equal(result.mask(result.best_index), compute_consensus(small, points, 0.01))

    def test_subset_reuses_original_gpu_buffers_and_keeps_removed_points_in_mass(self):
        points, large, small, _ = enclosed_structure_scene()
        interior = InteriorPenaltyContext(points)
        full = MetalConsensusContext(points, interior_context=interior)
        indices = np.arange(143, -1, -1)
        active = points[indices]
        context = full.for_subset(active, None, indices)
        context.check_inputs(active, None)
        self.assertIs(context._gpu_points, full._gpu_points)
        self.assertIs(context._gpu_neighbors, full._gpu_neighbors)
        result = context.evaluate([large, small], 0.01)
        self.assertGreater(result.interior_masses[0], 47)
        self.assertEqual(result.best_index, 1)
        for i, model in enumerate([large, small]):
            np.testing.assert_array_equal(result.mask(i), compute_consensus(model, active, 0.01))

    def test_subset_boundary_corrections_map_back_to_active_indices(self):
        model = SuperQuadricParams(1, 1, 1, 1, 1)
        radii = 1.01 + np.array([-1e-10, 1e-10, 0, -1e-7])
        points = np.vstack((np.column_stack((radii, np.zeros((4, 2)))), [[0, 0, 0], [0.1, 0, 0]]))
        interior = InteriorPenaltyContext(points)
        full = MetalConsensusContext(points, interior_context=interior)
        indices = np.array([3, 1, 0])
        active = points[indices]
        context = full.for_subset(active, None, indices)
        result = context.evaluate([model], 0.01)
        self.assertTrue(result._corrections)
        np.testing.assert_array_equal(result.mask(0), compute_consensus(model, active, 0.01))
        self.assertEqual(result.counts[0], result.mask(0).sum())

    def test_extreme_exponents_center_translation_and_scale_match_cpu(self):
        for e1, e2 in ((0.06, 0.06), (0.06, 4.5), (4.5, 0.06), (4.5, 4.5)):
            for factor, translation in ((1.0, [0, 0, 0]), (0.01, [1e6, -2e6, 3e6])):
                with self.subTest(e1=e1, e2=e2, factor=factor):
                    model = SuperQuadricParams(1.2*factor, 0.8*factor, 0.5*factor, e1, e2, [0.4, -0.3, 0.2], translation)
                    surface = surface_points(model, count=261)
                    points = np.vstack((surface, model.t+0.8*(surface-model.t), model.t+0.1*(surface-model.t), model.t[None]))
                    interior = InteriorPenaltyContext(points)
                    result = MetalConsensusContext(points, interior_context=interior).evaluate([model], 0.01*factor)
                    np.testing.assert_allclose(result.interior_masses[0], interior.mass(model, 0.01*factor), rtol=5e-5, atol=1e-4)
                    np.testing.assert_array_equal(result.mask(0), compute_consensus(model, points, 0.01*factor))

    def test_inner_penalty_uses_batch_scores_and_downloads_only_winner(self):
        points, large, small, _ = enclosed_structure_scene()
        interior = InteriorPenaltyContext(points)
        context = MetalConsensusContext(points, interior_context=interior)
        with patch("src.gair_ransac.inner_ransac._inner_candidate_model_batches", return_value=iter([[large, small]*40])):
            with patch("src.gair_ransac.inner_ransac.fit_superquadric_ls", return_value=large):
                with patch.object(MetalConsensusResult, "mask", autospec=True, side_effect=MetalConsensusResult.mask) as masks:
                    result = inner_ransac(points, np.arange(len(points)), None, 0.01, consensus_context=context, axis_penalty_weight=0)
        self.assertIs(result.best_model, small)
        self.assertEqual(result.best_inlier_count, 128)
        self.assertEqual(masks.call_count, 2)
        self.assertEqual(masks.call_args_list[0].args[1], 1)

    def test_zero_weight_retains_legacy_kernel_and_exact_selection(self):
        points, large, small, _ = enclosed_structure_scene()
        context = MetalConsensusContext(points, interior_context=InteriorPenaltyContext(points, weight=0))
        result = context.evaluate([large, small], 0.01)
        self.assertIsNone(context.interior_context)
        self.assertIsNone(result.interior_masses)
        np.testing.assert_array_equal(result.scores, result.counts)
        self.assertEqual(result.best_index, 0)


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
