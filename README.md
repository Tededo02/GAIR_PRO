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
once per inner search. Failed fits are skipped individually, while candidate
order, consensus selection, and final refinement are preserved.

Radial consensus also runs in a batched Metal kernel: the point transformation,
radial residual, normal alignment, and inlier counting run in parallel over
points and hypotheses. A 256-thread group processes each point tile for each
candidate; this is independent of the 64-thread fitting groups. GAIR-RANSAC and
RANSAC reuse one GPU-resident cloud and normalized normals per model extraction,
including outer candidates, inner searches, and final refits. Only counts and
the winning mask are returned for each inner batch, together with sparse indices
for ambiguous points near residual/normal thresholds. Those points are rechecked
in float64 on the CPU to avoid changing consensus decisions through float32
rounding. GPU masks for other candidates stay on the device.

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

Compare the original CPU consensus, the filtered CPU implementation, and Metal
with `.venv/bin/python validation/benchmark_metal_consensus.py`. Add `--main-scan`
to compare the complete GAIR algorithm using the current `main_scan_pc.py` input
and settings, checking all extracted inlier masks and selected samples. The
benchmark excludes visualization and reports preprocessing separately.
