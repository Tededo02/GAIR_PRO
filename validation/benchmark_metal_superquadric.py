import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

# Allow running this benchmark directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gair_ransac.inner_ransac import (
    _optimization_bounds, _pca_initial_parameters, fit_superquadric_ls, inner_ransac,
)
from src.gair_ransac.metal_superquadric import fit_superquadric_metal, fit_superquadric_metal_batch, metal_available
from src.superquadrics.superquadric_param import SuperQuadricParams
from test_metal_superquadric import robust_cost, surface_points


def median_ms(operation, repeats=5):
    operation()
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        operation()
        timings.append(1000 * (perf_counter() - start))
    return float(np.median(timings))


def benchmark_batch(target):
    points = surface_points(target, count=3000)
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.005, points.shape)
    samples = np.stack([points[rng.choice(len(points), 40, replace=False)] for _ in range(80)])
    lower, upper, loss_scale, diagonal = _optimization_bounds(points)
    initial = np.clip(_pca_initial_parameters(samples), lower, upper)

    def individual_fits():
        return np.stack([
            fit_superquadric_metal(sample, parameters, lower, upper, loss_scale, diagonal)
            for sample, parameters in zip(samples, initial)
        ])

    def batched_fits():
        result = fit_superquadric_metal_batch(samples, initial, lower, upper, loss_scale, diagonal)
        if not result.success.all():
            raise RuntimeError("A benchmark fit failed to converge")
        return result.parameters

    np.testing.assert_allclose(batched_fits(), individual_fits(), rtol=2e-4, atol=2e-5)
    individual_ms = median_ms(individual_fits)
    batched_ms = median_ms(batched_fits)
    print(json.dumps({
        "benchmark": "80 fits of 40 points, prepared samples and initialization",
        "individual_ms": individual_ms, "batch_ms": batched_ms,
        "speedup": individual_ms / batched_ms,
    }), flush=True)

    def run_inner():
        return inner_ransac(points, np.arange(len(points)), None, 0.03, random_seed=42)

    with patch("src.gair_ransac.inner_ransac.INNER_RANSAC_BATCH_SIZE", 1):
        sequential_result = run_inner()
        sequential_inner_ms = median_ms(run_inner)
    batched_result = run_inner()
    batch_inner_ms = median_ms(run_inner)
    np.testing.assert_array_equal(batched_result.best_inliers_mask, sequential_result.best_inliers_mask)
    print(json.dumps({
        "benchmark": "inner_ransac including sampling, PCA, consensus and final refit",
        "support_points": len(points), "hypotheses": 80, "sample_points": 40,
        "sequential_ms": sequential_inner_ms, "batch_ms": batch_inner_ms,
        "speedup": sequential_inner_ms / batch_inner_ms,
        "inliers": batched_result.best_inlier_count,
    }), flush=True)


def main():
    if not metal_available():
        raise RuntimeError("An accessible Apple silicon Metal GPU and MLX are required")
    import mlx.core as mx
    print(json.dumps({"device": mx.device_info()}, default=str), flush=True)
    target = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2, [0.4, -0.3, 0.2], [3, -2, 1])
    for count in (40, 300, 3000):
        points = surface_points(target, count=count)
        points += np.random.default_rng(42).normal(0, 0.005, points.shape)
        result = {"points": count}
        for backend in ("cpu", "metal"):
            fit_superquadric_ls(points, backend=backend)
            timings = []
            for _ in range(5):
                start = perf_counter()
                model = fit_superquadric_ls(points, backend=backend)
                timings.append(1000 * (perf_counter() - start))
            result[backend + "_ms"] = float(np.median(timings))
            result[backend + "_cost"] = robust_cost(model, points)
        result["speedup"] = result["cpu_ms"] / result["metal_ms"]
        print(json.dumps(result), flush=True)
    benchmark_batch(target)


if __name__ == "__main__":
    main()
