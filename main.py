import csv
import math
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from point_cloud_utils import chamfer_distance
from src.gair_ransac.energy_strategies import FullGairEnergy, GcRansacEnergy
from src.gair_ransac.gair_ransac import gair_ransac
from src.superquadrics import superquadric_mesh as supmesh
from src.superquadrics import superquadric_sampling as samp
from src.superquadrics.superquadric_param import SuperQuadricParams
from src.visualizations import visualization as vis

# ── config ────────────────────────────────────────────────────────────────────
VISUALIZE    = False
NOISE_STD    = 0.4
THRESHOLD    = 2.5 * NOISE_STD
N_POINTS      = 30000    # surface points
N_OUTLIERS   = 0     # random outliers injected
N_EVAL       = 4000     # clean reference points for metric evaluation
MAX_ITER     = 10
INNER_ITER   = 40
N_TRIALS     = 10
K            = 4
EVAL_SEED    = 42
OUT_DIR      = ROOT / "experiments" / "artifacts" / "main_experiment"

GT_PARAMS = [
    SuperQuadricParams(9.0, 9.0, 9.0, 3.5, 2.09, [2.0, 2.0, 1.0], [ 5.0,  5.0,  5.0]),
    SuperQuadricParams(3.0, 3.0, 3.0, 0.5, 0.90, [2.0, 2.0, 1.0], [-5.0, -5.0, -5.0]),
    SuperQuadricParams(4.0, 4.0, 4.0, 0.8, 1.10, [1.7, 2.1, 0.9], [13.0,  5.0,  5.0]),
    SuperQuadricParams(2.5, 2.5, 2.5, 0.7, 1.20, [2.1, 1.9, 0.8], [-10.5,-5.0, -5.0]),
]
# ──────────────────────────────────────────────────────────────────────────────


def generate_point_cloud():
    meshes = [supmesh.superquadric_mesh(p) for p in GT_PARAMS]

    noisy_pts_list, noisy_nrm_list = samp.sampling_sq_noisy(
        meshes, n_points=N_POINTS, noise_std=NOISE_STD, clip_k=3.0, seed=42
    )
    outlier_pts, outlier_nrm = samp.sampling_outliers(
        meshes, n_out=N_OUTLIERS, margin=0.10, mode="uniform", seed=42
    )
    clean_pts_list, _ = samp.sampling_sq_random(meshes, n_points=N_EVAL, seed=EVAL_SEED)

    points  = np.vstack([*noisy_pts_list,  outlier_pts])
    normals = np.vstack([*noisy_nrm_list,  outlier_nrm])
    clean_points = np.vstack(clean_pts_list)
    n_clean = points.shape[0] - N_OUTLIERS   # first n_clean rows are true inliers

    print(f"  {n_clean} surface points + {N_OUTLIERS} outliers = {points.shape[0]} total")
    return points, normals, clean_points, n_clean


def run_one(points, normals, clean_points, n_clean, algorithm: str, seed: int):
    t0 = time.perf_counter()
    models, inliers_masks, _, _ = gair_ransac(
        points,
        normals if algorithm == "gair-ransac" else None,
        threshold=THRESHOLD,
        max_models=len(GT_PARAMS),
        max_iterations=MAX_ITER,
        inner_iterations=INNER_ITER,
        min_coverage=0.4,
        sample_size=25,
        m_neighbors=6,
        random_seed=seed,
        use_normal_coherence=(algorithm == "gair-ransac"),
        energy_strategy=(
            FullGairEnergy()
            if algorithm == "gair-ransac"
            else GcRansacEnergy()
        ),
    )
    runtime = time.perf_counter() - t0

    if not models:
        return None

    meshes = [supmesh.superquadric_mesh(m) for m in models]
    sampled_est, _ = samp.sampling_sq_random(meshes, n_points=N_EVAL, seed=EVAL_SEED)
    est_pts = np.vstack(sampled_est)

    cd = chamfer_distance(clean_points, est_pts)
    tree_est = cKDTree(est_pts)
    tree_inp = cKDTree(clean_points)
    d_inp_to_est = tree_est.query(clean_points, k=1)[0]
    d_est_to_inp = tree_inp.query(est_pts,      k=1)[0]
    cd_coverage = float(d_inp_to_est.mean())
    cd_accuracy = float(d_est_to_inp.mean())
    hd = max(d_inp_to_est.max(), d_est_to_inp.max())

    inlier_mask = None
    if inliers_masks:
        inlier_mask = inliers_masks[0].copy()
        for mask in inliers_masks[1:]:
            inlier_mask |= mask

    if inlier_mask is not None:
        n_inliers_correct  = int(inlier_mask[:n_clean].sum())
        n_outliers_correct = int((~inlier_mask[n_clean:]).sum())
        correctly_classified = n_inliers_correct + n_outliers_correct
        classification_rate  = correctly_classified / len(points)
    else:
        correctly_classified = 0
        classification_rate  = 0.0
    print(f"    correctly_classified={correctly_classified}/{len(points)}  ({classification_rate*100:.1f}%)")

    if VISUALIZE:
        palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
        colors = [palette[i % len(palette)] for i in range(len(meshes))]
        vis.show_mesh_and_points(meshes, pts=points, point_size=5, colors=colors, inlier_mask=inlier_mask, models=models)

    return {
        "chamfer": cd, "cd_coverage": cd_coverage, "cd_accuracy": cd_accuracy,
        "hausdorff": hd, "classification_rate": classification_rate,
        "runtime_s": runtime, "n_models": len(models), "models": models,
    }


def score_combo(combo_indices, candidates, clean_points):
    meshes = [supmesh.superquadric_mesh(candidates[i]) for i in combo_indices]
    sampled, _ = samp.sampling_sq_random(meshes, n_points=N_EVAL, seed=EVAL_SEED)
    return chamfer_distance(clean_points, np.vstack(sampled))


def exhaustive_best_cover(candidates, clean_points, k_max):
    n = len(candidates)
    overall_best_score = np.inf
    overall_best_combo = None
    overall_best_k     = None
    for k in range(1, k_max + 1):
        print(f"  k={k}  ({math.comb(n, k)} combinations)")
        best_score = np.inf
        best_combo = None
        for combo in combinations(range(n), k):
            score = score_combo(combo, candidates, clean_points)
            if score < best_score:
                best_score = score
                best_combo = combo
        print(f"  best combo={best_combo}  cd={best_score:.4f}")
        if best_score < overall_best_score:
            overall_best_score = best_score
            overall_best_combo = best_combo
            overall_best_k     = k
    print(f"\n  => best K={overall_best_k}  combo={overall_best_combo}  cd={overall_best_score:.4f}")
    return list(overall_best_combo)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Generating point cloud ...")
    points, normals, clean_points, n_clean = generate_point_cloud()

    rng = np.random.default_rng()
    algorithms = ["gair-ransac", "gc-ransac"]
    all_results   = []
    all_candidates = {algo: [] for algo in algorithms}

    for algo in algorithms:
        print(f"\n=== {algo} ===")
        for trial in range(N_TRIALS):
            seed   = int(rng.integers(0, 2**31))
            result = run_one(points, normals, clean_points, n_clean, algo, seed)
            if result is None:
                print(f"  trial {trial}: no model found, skipping")
                continue
            print(
                f"  trial {trial} | CD={result['chamfer']:.4f}  "
                f"CD_COV={result['cd_coverage']:.4f}  CD_ACC={result['cd_accuracy']:.4f}  "
                f"HD={result['hausdorff']:.4f}  CLF={result['classification_rate']:.3f}  "
                f"RT={result['runtime_s']:.1f}s  models={result['n_models']}"
            )
            all_results.append({"algo": algo, "trial": trial, "seed": seed,
                                 **{k: v for k, v in result.items() if k != "models"}})
            all_candidates[algo].extend(result["models"])

    # summary
    print("\n=== summary ===")
    for algo in algorithms:
        rows = [r for r in all_results if r["algo"] == algo]
        if not rows:
            continue
        for metric in ("chamfer", "cd_coverage", "cd_accuracy", "hausdorff", "classification_rate", "runtime_s"):
            vals = [r[metric] for r in rows]
            print(f"  {algo:12s}  {metric}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    csv_path = OUT_DIR / "results_sequential.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "trial", "seed", "chamfer", "cd_coverage",
                                                "cd_accuracy", "hausdorff", "classification_rate",
                                                "runtime_s", "n_models"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {csv_path}")

    # ── set-cover ─────────────────────────────────────────────────────────────
    print(f"\n=== set-cover (k_max={K}) ===")
    cover_results = []
    for algo in algorithms:
        candidates = all_candidates[algo]
        if not candidates:
            print(f"  {algo}: no candidates, skipping")
            continue
        print(f"\n  {algo} — {len(candidates)} candidates")
        selected_idx    = exhaustive_best_cover(candidates, clean_points, k_max=K)
        selected_models = [candidates[i] for i in selected_idx]

        meshes = [supmesh.superquadric_mesh(m) for m in selected_models]
        sampled_est, _ = samp.sampling_sq_random(meshes, n_points=N_EVAL, seed=EVAL_SEED)
        est_pts = np.vstack(sampled_est)

        cd = chamfer_distance(clean_points, est_pts)
        tree_est = cKDTree(est_pts)
        tree_inp = cKDTree(clean_points)
        d_inp_to_est = tree_est.query(clean_points, k=1)[0]
        d_est_to_inp = tree_inp.query(est_pts,      k=1)[0]
        cd_coverage  = float(d_inp_to_est.mean())
        cd_accuracy  = float(d_est_to_inp.mean())
        hd = max(d_inp_to_est.max(), d_est_to_inp.max())

        print(f"  {algo} | k={len(selected_models)}  CD={cd:.4f}  CD_COV={cd_coverage:.4f}  CD_ACC={cd_accuracy:.4f}  HD={hd:.4f}")
        cover_results.append({"algo": algo, "k": len(selected_models),
                               "chamfer": round(cd, 6), "cd_coverage": round(cd_coverage, 6),
                               "cd_accuracy": round(cd_accuracy, 6), "hausdorff": round(hd, 6),
                               "n_candidates": len(candidates)})

        if VISUALIZE:
            palette = ["lightgreen", "orange", "violet", "cyan", "yellow", "red", "lime", "pink", "gold", "turquoise"]
            colors = [palette[i % len(palette)] for i in range(len(meshes))]
            vis.show_mesh_and_points(meshes, pts=points, point_size=5, colors=colors, models=selected_models)

    csv_cover = OUT_DIR / "results_setcover.csv"
    with open(csv_cover, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["algo", "k", "chamfer", "cd_coverage",
                                                "cd_accuracy", "hausdorff", "n_candidates"])
        writer.writeheader()
        writer.writerows(cover_results)
    print(f"Set-cover results saved to {csv_cover}")


if __name__ == "__main__":
    main()
