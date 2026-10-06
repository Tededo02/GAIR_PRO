import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

# Allow running this benchmark directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gair_ransac.inner_ransac import fit_superquadric_ls
from src.gair_ransac.metal_superquadric import metal_available
from src.superquadrics.superquadric_param import SuperQuadricParams
from test_metal_superquadric import robust_cost, surface_points


def main():
    if not metal_available():
        raise RuntimeError("An accessible Apple silicon Metal GPU and MLX are required")
    import mlx.core as mx
    print(json.dumps({"device": mx.device_info()}, default=str), flush=True)
    target = SuperQuadricParams(1.2, 0.8, 0.5, 0.8, 1.2, [0.4, -0.3, 0.2], [3, -2, 1])
    for count in (30, 300, 3000):
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


if __name__ == "__main__":
    main()
