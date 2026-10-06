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

Run correctness checks with `.venv/bin/python -m unittest discover -s validation -v`.
GPU checks are skipped when Metal is unavailable. Run the warm-up-aware timing
comparison with `.venv/bin/python validation/benchmark_metal_superquadric.py`.
