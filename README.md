# 3D superquadric decomposition

`fit_superquadric_ls` automatically uses a custom Metal kernel on Apple silicon
when MLX and the GPU are available. Install the project dependencies with `uv sync`.
MLX is included only on macOS arm64; other platforms use SciPy.

The Metal kernel runs the entire bounded least-squares optimization on the GPU:
radial residuals, analytic derivatives, `soft_l1` weighting, damped Gauss-Newton
steps, the 11-parameter linear solve, and convergence checks. PCA initialization
and bound preparation remain in NumPy. Kernel compilation is cached per process.
Points and parameters are normalized before conversion to float32 to preserve
precision for small shapes at large world coordinates. This solver minimizes
the same objective as SciPy's TRF solver, but its iterations and stopping criteria
differ, so fitted parameters can differ.

Fitting adds a quadratic penalty only for axes that exceed the support of the
reference points. A PCA bounding box supplies the directional half-span `s_j`;
its half-spans include 10% slack plus twice the robust loss scale, and respect
the existing minimum axis length. The box is projected into the current model
rotation, so supported elongated shapes and axis permutations remain valid.
Inner hypotheses share the box of the full refined set rather than penalizing
axes against each 40-point sample.

The objective per fitted point is:

```text
mean(soft_l1(radial_residual))
    + 0.5 * axis_penalty_weight * robust_loss_scale**2
      * sum(max(a_j / s_j - 1, 0)**2)
```

The axis term stays quadratic instead of being downweighted as an outlier.
CPU and Metal use the same analytic derivatives, including rotation derivatives,
and the weight is independent of sample count and Metal length normalization.
`fit_superquadric_ls`, `inner_ransac`, and `gair_ransac` default to
`axis_penalty_weight=0.1`. Setting it to zero restores the original fitting and
disables the compact-axis tie preference. Direct low-level Metal fitting defaults to zero;
positive weights require an `axis_support` box.

GAIR-RANSAC penalizes models that enclose other structures. For each original
cloud point, radial penetration `d` gives the continuous interior strength
`w = x**2 / (x**2 + threshold**2)`, where `x = max(d - threshold, 0)`.
Points in the existing surface tolerance band contribute zero; deep interior
points approach one. The unclamped implicit shape preserves deep penetration,
and a conservative inscribed radius handles the undefined ray at the center.

The existing kNN graph supplies spatial coherence: the effective interior mass
`M` is the sum of each strength times the mean strength of its neighbors.
An isolated interior point surrounded by surface points contributes zero, while
an enclosed, densely sampled structure contributes approximately its point count.
There is no additional cutoff between "few" and "many" internal points.

Candidate selection maximizes `Q = I / (1 + interior_penalty_weight * (M / max(I, 1))**2)`,
where `I` is the surface inlier count on the remaining cloud. The default weight
is `25`: interior masses of 1%, 10%, and 30% of the support retain approximately
99.75%, 80%, and 30.77% of the original score. A model can win with fewer surface
inliers when it avoids enclosing a substantial structure. Counts and masks remain
actual inliers, and the minimum-inlier requirement still applies.

The same score governs outer hypotheses, inner batch winners, local acceptance,
and final refits. Interior scoring always uses the original cloud, including
points removed by earlier extractions, and ignores normal alignment. The
nonlinear fitter continues to optimize only its support. Equal scores and inlier
counts prefer the smaller sum of squared semi-axes when the axis penalty is enabled;
a relative score tolerance of `1e-6` stabilizes float32 ties. Standalone
`inner_ransac` retains count-based selection unless given an `InteriorPenaltyContext`
or a Metal consensus context carrying one.

```sh
uv run main_scan_pc.py --interior-penalty-weight 25
uv run main_scan_pc.py --interior-penalty-weight 0
uv run main_scan_pc.py --interior-penalty-weight 0 --axis-penalty-weight 0
```

```python
from src.gair_ransac.inner_ransac import fit_superquadric_ls

model = fit_superquadric_ls(points)                  # Select Metal automatically.
model = fit_superquadric_ls(points, backend="metal") # Require Metal.
model = fit_superquadric_ls(points, backend="cpu")   # Use SciPy for comparisons.
```

Explicit `backend="metal"` raises an error if Metal is unavailable. GPU failures
raise an error rather than silently retrying the optimization on the CPU.
Existing RANSAC callers use automatic selection without API changes.

`inner_ransac` uses 40-point samples and defaults to 80 hypotheses. On Metal,
the 80 independent optimizations run in one dispatch, with a separate 64-thread
group per fit (two SIMD groups). Both SIMD groups contribute to the cost,
gradient, and Hessian. Smaller samples of up to 32 points use 32 threads.
Larger iteration counts are split into batches of up to 80; explicit
`n_iters`/`inner_iterations` settings are honored. The RANSAC and GAIR-RANSAC
defaults are also 80 inner iterations.

PCA initialization is vectorized across samples, and support bounds are computed
once per inner search. Failed fits are skipped individually, and candidate order
is preserved; with a positive axis penalty, tied scores and counts prefer smaller axes.

Radial consensus also runs in a batched Metal kernel: the point transformation,
radial residual, normal alignment, and inlier counting run in parallel over
points and hypotheses. A 256-thread group processes each point tile for each
candidate; this is independent of the 64-thread fitting groups. GAIR-RANSAC and
RANSAC reuse one GPU-resident cloud and normalized normals per model extraction,
including outer candidates, inner searches, and final refits. With interior scoring,
GAIR-RANSAC uploads the original cloud, normals, and neighbor graph once and uses
active-point masks across extractions. The consensus kernel also emits interior
strengths; a second kernel gathers neighbors and reduces the coherent mass.
Only counts, masses, scores, and the winning mask are returned for each inner batch, together with sparse indices
for ambiguous points near residual/normal thresholds. Those points are rechecked
in float64 on the CPU to avoid changing consensus decisions through float32
rounding. GPU masks and interior strengths for other candidates stay on the device.
Setting `interior_penalty_weight=0` skips interior computation and the neighbor kernel.

Without Metal, or for other residual metrics, consensus uses NumPy and evaluates
model normals only for points that pass the residual threshold. Deadlines are
checked before each batch; in-flight GPU dispatches complete before returning.
If fitting overruns a deadline, the first completed candidate is still scored,
and the final refit is skipped.

Run correctness checks with `.venv/bin/python -m unittest discover -s validation -v`.
GPU checks are skipped when Metal is unavailable. Run the warm-up-aware timing
comparison with `.venv/bin/python validation/benchmark_metal_superquadric.py`.
The benchmark also compares 80 individual 40-point fits against an 80-fit batch
and measures the complete inner-RANSAC path, including consensus and refinement.

Run `.venv/bin/python validation/benchmark_axis_regularization.py` to compare
penalty weights `0` and `main_scan_pc.AXIS_PENALTY_WEIGHT` on the actual scan,
with shared normal estimation and no visualization. The comparison reports
covered points, axes, surface support, model count, and runtime.

Compare the original CPU consensus, the filtered CPU implementation, and Metal
with `.venv/bin/python validation/benchmark_metal_consensus.py`. Add `--main-scan`
to compare the complete GAIR algorithm using the current `main_scan_pc.py` input
and settings, checking all extracted inlier masks and selected samples. The
benchmark excludes visualization and reports preprocessing separately.

Use `.venv/bin/python validation/benchmark_metal_consensus.py --interior --main-scan`
to measure the added GPU consensus cost and compare disabled/enabled interior
scoring with the current scan settings. Both paths are warmed up, share normal
estimation, and run twice. The output reports runtime, coverage, local searches,
and effective enclosed mass. `--input test_objects/real/etp_no_floor.ply` selects
a different scan. Different model selections can change the number of local
searches, so total runtime is not determined by kernel overhead alone.

## Original MSS acceleration

MSS retains its original random seeds, all 40 GAIR trials, adaptive k-NN pool
expansion, fallback behavior, FPS start point, score, and parameters. Only
neighbor filtering and FPS arithmetic use cached serial Numba kernels. Filtering
stops once the original truncated pool is known; FPS updates squared distances
without allocating temporary coordinate arrays at every step. Roundoff guards
route near-threshold filters and ambiguous FPS maxima through the original NumPy
operations to preserve decisions and tie breaking. No CPU parallelism or
`fastmath` is enabled. First use includes JIT compilation or cache loading.

Run exact pool, sample, and RNG regression checks with
`.venv/bin/python -m unittest validation.test_mss -v`.
Run `.venv/bin/python validation/benchmark_mss.py --main-scan` to compare the
frozen original sampler against the optimized sampler on `main_scan_pc.py`'s
real input. The comparison checks every MSS call, final inlier masks, fitted
parameters, model count, local optimization count, and RNG state through the
complete GAIR pipeline. Normal estimation is shared, visualization is excluded,
and MSS kernel warm-up is excluded from sampling timings. Use `--input path.ply`
to select another scan. Validation helpers remain in the Git-ignored
`validation` directory.
