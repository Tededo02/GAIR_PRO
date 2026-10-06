from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def rotz(yaw: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [[cy, -sy, 0.0],
         [sy,  cy, 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )


def roty(pitch: float) -> np.ndarray:
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array(
        [[ cp, 0.0, sp],
         [0.0, 1.0, 0.0],
         [-sp, 0.0, cp]],
        dtype=float,
    )


def rotx(roll: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array(
        [[1.0, 0.0, 0.0],
         [0.0,  cr, -sr],
         [0.0,  sr,  cr]],
        dtype=float,
    )




@dataclass
class SuperQuadricParams:
    a1: float
    a2: float
    a3: float
    e1: float
    e2: float
    rot: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))  # (yaw, pitch, roll) in radians
    t: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))    # translation (3,)

    def __post_init__(self) -> None:
        self.a1 = float(self.a1)
        self.a2 = float(self.a2)
        self.a3 = float(self.a3)
        self.e1 = float(self.e1)
        self.e2 = float(self.e2)

        self.rot = np.asarray(self.rot, dtype=float).reshape(3).copy()
        self.t = np.asarray(self.t, dtype=float).reshape(3).copy()

        self.validate()

    def validate(self) -> None:
        if not np.isfinite([self.a1, self.a2, self.a3, self.e1, self.e2]).all():
            raise ValueError("a, b, c, e1, e2 must be finite numbers")

        if self.a1 <= 0.0 or self.a2 <= 0.0 or self.a3 <= 0.0:
            raise ValueError("a, b, c must be > 0")

        if self.e1 <= 0.0 or self.e2 <= 0.0:
            raise ValueError("e1, e2 must be > 0")

        if self.rot.shape != (3,) or not np.isfinite(self.rot).all():
            raise ValueError("rot must be a finite array with shape (3,) = (yaw, pitch, roll)")

        if self.t.shape != (3,) or not np.isfinite(self.t).all():
            raise ValueError("t must be a finite array with shape (3,)")

    def rotation_matrix(self) -> np.ndarray:
        yaw, pitch, roll = self.rot
        # Angles are in radians
        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        return rotz(yaw) @ roty(pitch) @ rotx(roll)