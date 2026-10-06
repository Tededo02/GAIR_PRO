# Edit these inclusive (minimum, maximum) pairs to control parameters 12 through 19.
# D is the bounding-box diagonal of the reference points used for the fit.
# Bending uses dimensionless Cartesian components to stay smooth at zero curvature.

import numpy as np


SUPERFLEX_EXTRA_PARAMETER_BOUNDS = (
    (-0.25, 0.25),  # 12: taper_x
    (-0.25, 0.25),  # 13: taper_y
    (-0.30, 0.30),  # 14: D * k_x * cos(alpha_x)
    (-0.30, 0.30),  # 15: D * k_x * sin(alpha_x)
    (-0.30, 0.30),  # 16: D * k_y * cos(alpha_y)
    (-0.30, 0.30),  # 17: D * k_y * sin(alpha_y)
    (-0.30, 0.30),  # 18: D * k_z * cos(alpha_z)
    (-0.30, 0.30),  # 19: D * k_z * sin(alpha_z)
)


def extend_superflex_bounds(lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extra = np.asarray(SUPERFLEX_EXTRA_PARAMETER_BOUNDS, dtype=np.float64)
    if extra.shape != (8, 2) or not np.isfinite(extra).all():
        raise ValueError("SUPERFLEX_EXTRA_PARAMETER_BOUNDS must contain eight finite (minimum, maximum) pairs")
    if np.any(extra[:, 0] >= extra[:, 1]):
        raise ValueError("Each SuperFlex minimum must be strictly less than its maximum")
    if np.any(np.abs(extra[:2]) >= 1.0):
        raise ValueError("SuperFlex taper bounds must be strictly between -1 and 1")
    return np.r_[lower, extra[:, 0]], np.r_[upper, extra[:, 1]]
