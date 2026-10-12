# Verify independent RNG streams, complete batch use, deadlines and per-sample GPU bounds.
import importlib
from time import perf_counter
import unittest
from unittest.mock import patch

import numpy as np

from test_superflex import fixture_model, surface
from src.gair_ransac.deformable_fitting import _pack, fit_deformable_hypotheses_batch
from src.gair_ransac.inner_ransac import fit_superquadric_ls
from src.gair_ransac.metal_superquadric import metal_available
from src.gair_ransac.mss import adaptive_local_fps_mss, build_adaptive_local_fps_sampler_context
from src.superquadrics.superquadric_residual import superquadric_normal_world

outer = importlib.import_module("src.gair_ransac.gair_ransac")


class OuterBatchOrderingTests(unittest.TestCase):
    def setUp(self):
        self.points = np.random.default_rng(31).normal(size=(100, 3))
        self.normals = np.tile([0.0, 0.0, 1.0], (100, 1))
        self.context = build_adaptive_local_fps_sampler_context(self.points, self.normals)

    def stream(self, rng, iterations=11, batch_size=8, deadline=None):
        return outer._batched_superflex_outer_hypotheses(
            self.points, self.normals, self.context, 30, iterations, rng,
            batch_size, 0.4, deadline,
        )

    @staticmethod
    def local_search_rng(rng, iteration):
        if iteration in (0, 3, 8):
            for _ in range(iteration % 3 + 1):
                rng.integers(0, np.iinfo(np.int32).max)

    def test_local_searches_do_not_resample_or_invalidate_prepared_hypotheses(self):
        reference_rng = np.random.default_rng(12345777)
        expected = []
        for iteration in range(11):
            expected.append(adaptive_local_fps_mss(
                self.points, normals=self.normals, sample_size=30, initial_k=300,
                candidate_multiplier=12, rng=reference_rng, sampler_context=self.context,
            ))

        def fit(samples, weight):
            return [None if float(sample.sum()) < 0 else float(sample.sum()) for sample in samples]

        for batch_size in (2, 8, 16):
            with self.subTest(batch_size=batch_size), patch(
                "src.gair_ransac.deformable_fitting.fit_deformable_hypotheses_batch", side_effect=fit,
            ) as fitter, patch.object(outer, "adaptive_local_fps_mss", wraps=adaptive_local_fps_mss) as sampler:
                rng = np.random.default_rng(12345777)
                local_rng = np.random.default_rng(987)
                actual = []
                for iteration, (sample, model) in enumerate(self.stream(rng, batch_size=batch_size)):
                    actual.append(sample)
                    self.assertEqual(model, None if float(sample.sum()) < 0 else float(sample.sum()))
                    self.local_search_rng(local_rng, iteration)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(rng.bit_generator.state, reference_rng.bit_generator.state)
                self.assertEqual(sampler.call_count, 11)
                self.assertEqual([len(call.args[0]) for call in fitter.call_args_list],
                                 [min(batch_size, 11 - start) for start in range(0, 11, batch_size)])

    def test_local_search_seeds_are_unique_even_if_the_generator_repeats_a_value(self):
        from unittest.mock import Mock

        rng = Mock()
        rng.integers.side_effect = [11, 11, 12, 12, 13]
        used = set()
        self.assertEqual([outer._next_local_seed(rng, used) for _ in range(3)], [11, 12, 13])
        self.assertEqual(used, {11, 12, 13})

    def test_no_sampling_after_an_expired_deadline(self):
        rng = np.random.default_rng(4)
        expected_state = rng.bit_generator.state
        with patch.object(outer, "adaptive_local_fps_mss") as sampler:
            self.assertEqual(list(self.stream(rng, deadline=perf_counter() - 1)), [])
        sampler.assert_not_called()
        self.assertEqual(rng.bit_generator.state, expected_state)

    def test_only_first_candidate_is_scored_when_gpu_dispatch_exceeds_deadline(self):
        clock = [0.0]

        def fit(samples, weight):
            clock[0] = 2.0
            return [None] * len(samples)

        rng = np.random.default_rng(5)
        reference_rng = np.random.default_rng(5)
        expected = [adaptive_local_fps_mss(
            self.points, normals=self.normals, sample_size=30, initial_k=300,
            candidate_multiplier=12, rng=reference_rng, sampler_context=self.context,
        ) for _ in range(8)]
        with patch.object(outer.time, "perf_counter", side_effect=lambda: clock[0]), patch(
            "src.gair_ransac.deformable_fitting.fit_deformable_hypotheses_batch", side_effect=fit,
        ):
            actual = list(self.stream(rng, deadline=1.0))
        self.assertEqual(len(actual), 1)
        np.testing.assert_array_equal(actual[0][0], expected[0])
        self.assertEqual(rng.bit_generator.state, reference_rng.bit_generator.state)


@unittest.skipUnless(metal_available(), "An accessible Metal GPU is required")
class OuterBatchMetalTests(unittest.TestCase):
    def test_independent_bounds_scales_and_failures_match_individual_fits(self):
        rng = np.random.default_rng(44)
        points = surface(fixture_model(), 12, 20)
        samples = np.stack([
            points[rng.choice(len(points), 30, replace=False)] * (0.5 + 0.2 * index)
            + [index * 2.0, -index * 0.7, index * 0.3]
            for index in range(16)
        ])
        samples[3, 0, 0] = np.nan
        expected = []
        for sample in samples:
            try:
                expected.append(fit_superquadric_ls(sample, model_family="superflex", axis_penalty_weight=0.4))
            except (ValueError, RuntimeError):
                expected.append(None)
        with patch("scipy.optimize.least_squares", side_effect=AssertionError("GPU fitting must not invoke SciPy")):
            actual = fit_deformable_hypotheses_batch(samples, 0.4)
        self.assertEqual(len(actual), len(expected))
        self.assertIsNone(actual[3])
        self.assertTrue(any(model is not None for model in actual))
        for index, (model, reference) in enumerate(zip(actual, expected)):
            with self.subTest(index=index):
                self.assertEqual(model is None, reference is None)
                if model is not None:
                    np.testing.assert_array_equal(_pack(model, 1.0), _pack(reference, 1.0))

    def test_complete_gair_matches_serial_models_masks_and_winning_samples(self):
        model = fixture_model(taper=[0.2, -0.1], bending=[[0.2, 0.4], [0.1, -0.2], [0.25, 0.6]])
        points = surface(model, 10, 16)
        normals = superquadric_normal_world(model, points)
        kwargs = dict(normals=normals, threshold=0.025, max_models=1, max_iterations=6,
                      sample_size=30, min_inliers=40, inner_iterations=2, random_seed=8,
                      model_family="superflex", axis_penalty_weight=0.4, interior_penalty_weight=25)
        expected = outer.gair_ransac(points, outer_batch_size=1, **kwargs)
        self.assertGreater(len(expected[0]), 0)
        for batch_size in (8, 16):
            with self.subTest(batch_size=batch_size):
                actual = outer.gair_ransac(points, outer_batch_size=batch_size, **kwargs)
                self.assertEqual(actual[3], expected[3])
                self.assertEqual(len(actual[0]), len(expected[0]))
                np.testing.assert_array_equal(actual[2], expected[2])
                for model, reference, mask, reference_mask in zip(actual[0], expected[0], actual[1], expected[1]):
                    np.testing.assert_array_equal(_pack(model, 1.0), _pack(reference, 1.0))
                    np.testing.assert_array_equal(mask, reference_mask)


if __name__ == "__main__":
    unittest.main()
