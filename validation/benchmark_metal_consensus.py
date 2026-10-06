import argparse
import importlib
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np

# Allow running this benchmark directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gair_ransac.consensus import compute_consensus, distance_err, normal_alignment_score
from src.gair_ransac.metal_consensus import MetalConsensusContext
from src.gair_ransac.metal_superquadric import metal_available
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.superquadrics.superquadric_residual import superquadric_normal_world
from test_metal_superquadric import surface_points


def original_cpu_consensus(model, points, threshold, error_metric="radial", normals=None, normal_cos_threshold=None, metal_context=None):
    mask = distance_err(model, points, error_metric) < threshold
    if normals is not None:
        mask &= normal_alignment_score(model, points, normals) >= (0.0 if normal_cos_threshold is None else normal_cos_threshold)
    return mask


def median_ms(operation, repeats=5):
    operation()
    timings = []
    for _ in range(repeats):
        start = perf_counter()
        operation()
        timings.append(1000 * (perf_counter() - start))
    return float(np.median(timings))


def benchmark_consensus():
    target = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2, [0.4, -0.3, 0.2], [3, -2, 1])
    points = surface_points(target, count=55966)
    normals = superquadric_normal_world(target, points)
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.02, points.shape)
    points[::3] += rng.normal(0, 0.5, points[::3].shape)
    models = [SuperQuadricParams(target.a1, target.a2, target.a3, target.e1, target.e2, target.rot, target.t + rng.normal(0, 0.05, 3)) for _ in range(80)]
    context = MetalConsensusContext(points, normals)

    def cpu_batch(function):
        masks = [function(model, points, 0.01, normals=normals) for model in models]
        counts = np.array([mask.sum() for mask in masks])
        return counts, masks[int(np.argmax(counts))]

    def gpu_batch():
        result = context.evaluate(models, 0.01)
        return result.counts, result.mask(result.best_index)

    reference = cpu_batch(original_cpu_consensus)
    for result in (cpu_batch(compute_consensus), gpu_batch()):
        np.testing.assert_array_equal(result[0], reference[0])
        np.testing.assert_array_equal(result[1], reference[1])
    cpu_ms = median_ms(lambda: cpu_batch(original_cpu_consensus))
    optimized_cpu_ms = median_ms(lambda: cpu_batch(compute_consensus))
    gpu_ms = median_ms(gpu_batch)
    print(json.dumps({
        "benchmark": "80-model radial and normal consensus, including winner mask transfer",
        "points": len(points), "models": len(models),
        "original_cpu_ms": cpu_ms, "filtered_cpu_ms": optimized_cpu_ms,
        "metal_ms": gpu_ms, "speedup": cpu_ms / gpu_ms,
        "counts_and_winner_mask_identical": True,
    }), flush=True)


def benchmark_main_scan():
    import main_scan_pc as main
    gr = importlib.import_module("src.gair_ransac.gair_ransac")
    inner = importlib.import_module("src.gair_ransac.inner_ransac")
    path = main.resolve_input_path(main.PC_DIR / main.PC_NAME)
    start = perf_counter()
    points, _, normals = main.load_point_cloud(path, main.MESH_SAMPLE_COUNT, main.RANDOM_SEED)
    threshold, _ = main.compute_effective_threshold(points)
    if normals is None:
        normals = main.estimate_normals_open3d_consistent(points, main.K_NEIGHBORS)
    preparation_seconds = perf_counter() - start

    def run():
        return gr.gair_ransac(
            points, normals=normals, threshold=threshold, max_models=main.MAX_MODELS,
            m_neighbors=main.M_NEIGHBORS, max_iterations=main.MAX_ITERATIONS,
            sample_size=main.SAMPLE_SIZE, min_inliers=main.MIN_INLIERS,
            inner_iterations=main.INNER_ITERATIONS, use_normal_coherence=True,
            random_seed=main.RANDOM_SEED, min_coverage=main.MIN_COVERAGE,
            energy_strategy=main.FullGairEnergy() if main.ALGORITHM_NAME == "gair" else main.GcRansacEnergy(),
        )

    # The reference keeps the same Metal fitter, sampling, seeds, and GAIR energy.
    print("Running main_scan_pc algorithm with original CPU consensus", flush=True)
    with patch.object(gr, "create_metal_consensus_context", return_value=None):
        with patch.object(inner, "create_metal_consensus_context", return_value=None):
            with patch.object(gr, "compute_consensus", side_effect=original_cpu_consensus):
                with patch.object(inner, "compute_consensus", side_effect=original_cpu_consensus):
                    start = perf_counter()
                    cpu = run()
                    cpu_seconds = perf_counter() - start
    print(f"CPU reference finished in {cpu_seconds:.3f}s; running Metal consensus", flush=True)
    start = perf_counter()
    gpu = run()
    gpu_seconds = perf_counter() - start
    if len(cpu[0]) != len(gpu[0]):
        raise AssertionError("CPU and Metal extracted different numbers of models")
    for cpu_mask, gpu_mask in zip(cpu[1], gpu[1]):
        np.testing.assert_array_equal(gpu_mask, cpu_mask)
    np.testing.assert_array_equal(gpu[2], cpu[2])
    if gpu[3] != cpu[3]:
        raise AssertionError("CPU and Metal performed different local optimizations")
    print(json.dumps({
        "benchmark": "main_scan_pc GAIR algorithm, visualization excluded",
        "input": str(path), "points": len(points), "models": len(gpu[0]),
        "local_optimizations": gpu[3], "preparation_seconds": preparation_seconds,
        "original_cpu_consensus_seconds": cpu_seconds, "metal_consensus_seconds": gpu_seconds,
        "algorithm_speedup": cpu_seconds / gpu_seconds,
        "all_inlier_masks_and_selected_samples_identical": True,
    }), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-scan", action="store_true", help="Also compare the complete algorithm with main_scan_pc settings")
    args = parser.parse_args()
    if not metal_available():
        raise RuntimeError("An accessible Apple silicon Metal GPU and MLX are required")
    benchmark_consensus()
    if args.main_scan:
        benchmark_main_scan()


if __name__ == "__main__":
    main()
