import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from point_cloud_utils import chamfer_distance
from scipy.spatial import cKDTree
from src.gair_ransac.energy_strategies import FullGairEnergy, GcRansacEnergy
from src.gair_ransac.ransacov import ransacov
from src.superquadrics.superquadric_residual import superquadric_radial_residual
from src.gair_ransac.gair_ransac import gair_ransac
from src.gair_ransac.ransac import ransac
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics import superquadric_sampling as samp
from src.visualizations import plot as vis

# ── config ────────────────────────────────────────────────────────────────────
VISUALIZE    = False   # set to False to skip visualization and run headlessly
PC_DIR       = ROOT / "data" / "point_clouds"
THRESHOLD    = 0.9
GRAPH_RADIUS = 0.1
MAX_MODELS   = 10
MAX_ITER     = 20
INNER_ITER   = 25
N_TRIALS     = 5
K            = 1
EVAL_SEED    = 42     # fixed so Chamfer is comparable across trials
OUT_DIR      = ROOT / "data" / "results"
N_OUTLIERS   = 0    # constant uniform outliers injected per trial (0 = none)
NOISE        = 0.0    # Gaussian noise std on positions and normals (0.0 = none)
MAX_LOAD_PTS = 8000   # random subsample cap after background removal (None = no cap)
MAX_COVER_ITER = 1000 # combinatorial cap for exhaustive set-cover search
COVER_METHODS  = ("ransacov",)  # set-cover methods to run; add "exhaustive" back in to re-enable it
PC_CONFIGS_PATH = Path(__file__).parent / "data" / "pc_configs.json"
# ──────────────────────────────────────────────────────────────────────────────


def _load_pc_config(pc_stem: str) -> dict:
    if PC_CONFIGS_PATH.exists():
        with open(PC_CONFIGS_PATH) as f:
            all_configs = json.load(f)
        cfg = dict(all_configs.get("default", {}))
        cfg.update(all_configs.get(pc_stem, {}))
        return cfg
    return {}


def _subsample(points: np.ndarray, max_pts: int | None):
    if max_pts is None or len(points) <= max_pts:
        return points, np.arange(len(points))
    print(f"  capping {len(points)} -> {max_pts} points (random subsample)")
    idx = np.random.default_rng().choice(len(points), max_pts, replace=False)
    idx.sort()
    return points[idx], idx


def load_point_cloud(path: Path):
    scene = trimesh.load(str(path))
    if isinstance(scene, trimesh.Scene):
        mesh_raw = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        mesh_raw = scene

    points = np.asarray(mesh_raw.vertices, dtype=np.float64)

    if isinstance(mesh_raw, trimesh.PointCloud):
        keep = np.ones(len(points), dtype=bool)
        points, sub_idx = _subsample(points, MAX_LOAD_PTS)

        normals_path = path.parent / f"normals_{path.name}"
        if normals_path.exists():
            all_normals = np.asarray(trimesh.load(str(normals_path)).vertices, dtype=np.float64)
            normals = all_normals[keep][sub_idx]
        else:
            import point_cloud_utils as pcu
            _, normals = pcu.estimate_point_cloud_normals_knn(points.astype(np.float32), num_neighbors=10)
            normals = normals.astype(np.float64)
    else:
        normals = np.asarray(mesh_raw.vertex_normals, dtype=np.float64)
        points, sub_idx = _subsample(points, MAX_LOAD_PTS)
        normals = normals[sub_idx]

    if points.shape[0] == 0:
        raise ValueError(f"Point cloud is empty after loading {path.name} (all points removed by background filter or subsample)")

    bb_min, bb_max = points.min(axis=0), points.max(axis=0)
    bb_size = bb_max - bb_min
    scale = float(np.max(bb_size))
    if scale <= 0.0:
        scale = 1.0
    points = (points - bb_min) / scale
    print(f"  bounding box after normalization: min={points.min(axis=0)}  max={points.max(axis=0)}")

    return points, normals


def corrupt_point_cloud(points: np.ndarray, normals: np.ndarray, rng: np.random.Generator):
    if NOISE > 0.0:
        points  = points  + rng.normal(0, NOISE, points.shape)
        normals = normals + rng.normal(0, NOISE, normals.shape)
        norms   = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(norms > 0, norms, 1.0)

    if N_OUTLIERS > 0:
        lo, hi = points.min(axis=0), points.max(axis=0)
        outlier_pts = rng.uniform(lo, hi, (N_OUTLIERS, 3))
        outlier_nrm = rng.normal(0, 1, (N_OUTLIERS, 3))
        outlier_nrm /= np.linalg.norm(outlier_nrm, axis=1, keepdims=True)
        points  = np.vstack([points,  outlier_pts])
        normals = np.vstack([normals, outlier_nrm])

    return points, normals


def score_combo(combo_indices, candidates, points):
    meshes = [supmesh.superquadric_mesh(candidates[i]) for i in combo_indices]
    sampled, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
    return chamfer_distance(points, np.vstack(sampled))


def exhaustive_best_cover(candidates, points, k_max):
    """Return (selected_indices, n_evals) via capped exhaustive Chamfer search."""
    n = len(candidates)
    overall_best_score = np.inf
    overall_best_combo = None
    overall_best_k     = None
    total_evals        = 0

    for k in range(1, k_max + 1):
        n_combos   = math.comb(n, k)
        print(f"  k={k}  ({n_combos} combinations, cap={MAX_COVER_ITER})")
        best_score = np.inf
        best_combo = None
        for i, combo in enumerate(combinations(range(n), k)):
            if i >= MAX_COVER_ITER:
                break
            score = score_combo(combo, candidates, points)
            total_evals += 1
            if score < best_score:
                best_score = score
                best_combo = combo
        print(f"  best combo={best_combo}  cd={best_score:.4f}")
        if best_score < overall_best_score:
            overall_best_score = best_score
            overall_best_combo = best_combo
            overall_best_k     = k

    print(f"\n  => best K={overall_best_k}  combo={overall_best_combo}  cd={overall_best_score:.4f}")
    return list(overall_best_combo), total_evals


def run_one(points, normals, clean_points, n_clean, algorithm: str, seed: int, pc_cfg: dict | None = None):
    if pc_cfg is None:
        pc_cfg = {}
    t0        = time.perf_counter()
    threshold = pc_cfg.get("threshold", THRESHOLD)
    mss_used  = None

    if algorithm in ("vanilla", "lo-ransac"):
        models, inliers_masks, n_local_opts = ransac(
            points,
            threshold=threshold,
            max_models=MAX_MODELS,
            max_iterations=MAX_ITER,
            inner_iterations=INNER_ITER,
            random_seed=seed,
            local_optimization=(algorithm == "lo-ransac"),
        )
    elif algorithm in ("gair-ransac", "gc-ransac"):
        models, inliers_masks, mss_used, n_local_opts = gair_ransac(
            points,
            normals if algorithm == "gair-ransac" else None,
            threshold=threshold,
            max_models=MAX_MODELS,
            max_iterations=MAX_ITER,
            inner_iterations=INNER_ITER,
            min_coverage=0.4,
            random_seed=seed,
            sample_size=pc_cfg.get("mss_sample_size", 20),
            m_neighbors=pc_cfg.get("m_neighbors", 6),
            min_inliers=20,
            energy_strategy=(
                FullGairEnergy()
                if algorithm == "gair-ransac"
                else GcRansacEnergy()
            ),
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    runtime = time.perf_counter() - t0

    if not models:
        return None

    meshes = [supmesh.superquadric_mesh(m) for m in models]
    sampled_est, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
    est_pts = np.vstack(sampled_est)

    cd       = chamfer_distance(clean_points, est_pts)
    tree_est = cKDTree(est_pts)
    tree_inp = cKDTree(clean_points)
    d_inp_to_est = tree_est.query(clean_points, k=1)[0]
    d_est_to_inp = tree_inp.query(est_pts,      k=1)[0]
    cd_coverage  = float(d_inp_to_est.mean())
    cd_accuracy  = float(d_est_to_inp.mean())
    hd           = max(d_inp_to_est.max(), d_est_to_inp.max())

    palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
    colors  = [palette[i % len(palette)] for i in range(len(meshes))]

    inlier_mask = None
    if inliers_masks:
        inlier_mask = inliers_masks[0].copy()
        for mask in inliers_masks[1:]:
            inlier_mask |= mask

    if inlier_mask is not None:
        true_inliers_correct  = int(inlier_mask[:n_clean].sum())
        true_outliers_correct = int((~inlier_mask[n_clean:]).sum())
        correctly_classified  = true_inliers_correct + true_outliers_correct
        classification_rate   = correctly_classified / len(points)
    else:
        correctly_classified = 0
        classification_rate  = 0.0

    print(f"    correctly_classified={correctly_classified}/{len(points)}  ({classification_rate*100:.1f}%)")
    if VISUALIZE and algorithm in ("gair-ransac", "gc-ransac"):
        vis.show_mesh_and_points(meshes, pts=points, point_size=5, colors=colors, inlier_mask=inlier_mask, mss_used=mss_used, models=models)

    n_inliers = int(inlier_mask[:n_clean].sum()) if inlier_mask is not None else 0
    return {
        "chamfer": cd, "cd_coverage": cd_coverage, "cd_accuracy": cd_accuracy,
        "hausdorff": hd, "classification_rate": classification_rate,
        "runtime_s": runtime, "n_models": len(models),
        "n_inliers": n_inliers, "n_local_opts": n_local_opts, "models": models,
    }


def _run_one_worker(args):
    """Top-level wrapper so ProcessPoolExecutor can pickle it."""
    trial, algo, seed, points, normals, clean_points, n_clean, pc_cfg = args
    result = run_one(points, normals, clean_points, n_clean, algo, seed, pc_cfg)
    return trial, algo, seed, result


def run_for_pc(pc_file: Path, rng: np.random.Generator):
    print(f"\n{'='*60}")
    print(f"Point cloud: {pc_file.name}")
    print(f"{'='*60}")

    points_clean, normals_clean = load_point_cloud(pc_file)
    n_clean      = len(points_clean)
    clean_points = points_clean.copy()
    pc_cfg       = _load_pc_config(pc_file.stem)
    print(f"  {n_clean} points loaded  |  config: {pc_cfg}")

    trial_seeds   = [int(rng.integers(0, 2**31)) for _ in range(N_TRIALS)]
    outlier_fracs = [0.01]  # TEMP: testing only, restore to [round(x * 0.05, 2) for x in range(9)] for full sweep
    algorithms    = ["vanilla", "lo-ransac", "gc-ransac", "gair-ransac"]

    pc_out_dir = OUT_DIR / pc_file.stem
    pc_out_dir.mkdir(parents=True, exist_ok=True)

    for out_frac in outlier_fracs:
        n_outliers = int(n_clean * out_frac)
        print(f"\n{'#'*60}")
        print(f"Outlier fraction: {out_frac:.2f}  ({n_outliers} outliers injected)")
        print(f"{'#'*60}")

        points, normals = corrupt_point_cloud(points_clean.copy(), normals_clean.copy(), rng)
        if n_outliers > 0:
            lo, hi = points_clean.min(axis=0), points_clean.max(axis=0)
            outlier_pts = rng.uniform(lo, hi, (n_outliers, 3))
            outlier_nrm = rng.normal(0, 1, (n_outliers, 3))
            outlier_nrm /= np.linalg.norm(outlier_nrm, axis=1, keepdims=True)
            points  = np.vstack([points,  outlier_pts])
            normals = np.vstack([normals, outlier_nrm])

        frac_out_dir = pc_out_dir / f"outliers_{out_frac:.2f}"
        frac_out_dir.mkdir(parents=True, exist_ok=True)

        all_results   = []
        all_candidates = {algo: [] for algo in algorithms}

        tasks = [
            (trial, algo, trial_seeds[trial], points, normals, clean_points, n_clean, pc_cfg)
            for trial in range(N_TRIALS)
            for algo in algorithms
        ]
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(_run_one_worker, t): t for t in tasks}
            for future in as_completed(futures):
                trial, algo, seed, result = future.result()
                if result is None:
                    print(f"  trial {trial} {algo}: no model found, skipping")
                    continue
                print(
                    f"  trial {trial} {algo}  seed={seed}"
                    f"  CD={result['chamfer']:.4f}"
                    f"  CD_COV={result['cd_coverage']:.4f}"
                    f"  CD_ACC={result['cd_accuracy']:.4f}"
                    f"  HD={result['hausdorff']:.4f}"
                    f"  CLF={result['classification_rate']:.3f}"
                    f"  RT={result['runtime_s']:.1f}s"
                    f"  models={result['n_models']}"
                )
                all_results.append({
                    "algo": algo, "trial": trial, "seed": seed, "outlier_frac": out_frac,
                    **{k: v for k, v in result.items() if k != "models"},
                })
                all_candidates[algo].extend(result["models"])

        # per-algorithm summary
        print("\n=== summary ===")
        for algo in algorithms:
            rows = [r for r in all_results if r["algo"] == algo]
            if not rows:
                continue
            for metric in ("chamfer", "cd_coverage", "cd_accuracy", "hausdorff", "classification_rate", "runtime_s"):
                vals = [r[metric] for r in rows]
                print(f"  {algo:12s}  {metric}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

        csv_path = frac_out_dir / "results_sequential.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "algo", "trial", "seed", "outlier_frac",
                "chamfer", "cd_coverage", "cd_accuracy", "hausdorff",
                "classification_rate", "runtime_s", "n_models", "n_inliers", "n_local_opts",
            ])
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {csv_path}")

        # ── set-cover ─────────────────────────────────────────────────────────
        cover_threshold = pc_cfg.get("threshold", THRESHOLD)
        cover_results   = []

        for algo in algorithms:
            candidates = all_candidates[algo]
            if not candidates:
                continue

            for method in COVER_METHODS:
                print(f"\n=== set-cover [{method}]  {algo} — {len(candidates)} candidates ===")

                if method == "exhaustive":
                    selected_idx, n_evals = exhaustive_best_cover(candidates, clean_points, k_max=K)
                    if selected_idx:
                        inlier_mask = np.zeros(len(clean_points), dtype=bool)
                        for idx in selected_idx:
                            res = np.abs(superquadric_radial_residual(candidates[idx], clean_points))
                            inlier_mask |= (res < cover_threshold)
                        n_covered = int(inlier_mask.sum())
                    else:
                        n_covered = 0
                else:
                    selected_idx, n_covered = ransacov(
                        candidates, clean_points, k=K,
                        threshold=cover_threshold,
                        residual_fn=lambda m, pts: np.abs(superquadric_radial_residual(m, pts)),
                    )
                    n_evals = len(candidates)

                if not selected_idx:
                    print("  no models selected")
                    continue

                selected_models = [candidates[i] for i in selected_idx]
                meshes = [supmesh.superquadric_mesh(m) for m in selected_models]
                sampled_est, _ = samp.sampling_sq_random(meshes, n_points=4000, seed=EVAL_SEED)
                est_pts = np.vstack(sampled_est)

                cd       = chamfer_distance(clean_points, est_pts)
                tree_est = cKDTree(est_pts)
                tree_inp = cKDTree(clean_points)
                d_inp_to_est = tree_est.query(clean_points, k=1)[0]
                d_est_to_inp = tree_inp.query(est_pts,      k=1)[0]
                cd_coverage  = float(d_inp_to_est.mean())
                cd_accuracy  = float(d_est_to_inp.mean())
                hd           = max(d_inp_to_est.max(), d_est_to_inp.max())

                full_inlier_mask = np.zeros(len(points), dtype=bool)
                for m in selected_models:
                    res = np.abs(superquadric_radial_residual(m, points))
                    full_inlier_mask |= (res < cover_threshold)
                true_inliers_correct  = int(full_inlier_mask[:n_clean].sum())
                true_outliers_correct = int((~full_inlier_mask[n_clean:]).sum())
                classification_rate   = (true_inliers_correct + true_outliers_correct) / len(points)

                print(
                    f"  {algo} [{method}] | k={len(selected_models)}"
                    f"  n_covered={n_covered}  n_evals={n_evals}"
                    f"  CD={cd:.4f}  COV={cd_coverage:.4f}  ACC={cd_accuracy:.4f}  HD={hd:.4f}"
                    f"  CLF={classification_rate:.3f}"
                )
                cover_results.append({
                    "algo": algo, "method": method, "outlier_frac": out_frac,
                    "k": len(selected_models), "n_covered": n_covered, "n_evals": n_evals,
                    "chamfer": round(cd, 6), "cd_coverage": round(cd_coverage, 6),
                    "cd_accuracy": round(cd_accuracy, 6), "hausdorff": round(hd, 6),
                    "classification_rate": round(classification_rate, 6),
                    "n_candidates": len(candidates),
                })

                if VISUALIZE:
                    palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
                    colors  = [palette[i % len(palette)] for i in range(len(meshes))]
                    vis.show_mesh_and_points(meshes, pts=clean_points, point_size=5, colors=colors, models=selected_models)

        csv_cover = frac_out_dir / "results_setcover.csv"
        with open(csv_cover, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "algo", "method", "outlier_frac", "k", "n_covered", "n_evals",
                "chamfer", "cd_coverage", "cd_accuracy", "hausdorff", "classification_rate",
                "n_candidates",
            ])
            writer.writeheader()
            writer.writerows(cover_results)
        print(f"Set-cover results saved to {csv_cover}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pc_files = sorted(
        f for f in PC_DIR.iterdir()
        if f.is_file() and not f.name.startswith("normals_") and not f.name.startswith(".")
    )
    if not pc_files:
        print(f"No files found in {PC_DIR}")
        return
    print(f"Found {len(pc_files)} point cloud(s): {[f.name for f in pc_files]}")
    rng = np.random.default_rng()
    for pc_file in pc_files:
        run_for_pc(pc_file, rng)


if __name__ == "__main__":
    main()
