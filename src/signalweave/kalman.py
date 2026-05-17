"""Minimal Kalman filter for 1D state estimation."""
from __future__ import annotations
import numpy as np


class Kalman1D:
    """Scalar Kalman filter with constant-velocity option."""

    def __init__(self, process_var: float = 1e-4, measure_var: float = 1e-2,
                 init_state: float = 0.0, init_cov: float = 1.0):
        self.x = float(init_state)
        self.P = float(init_cov)
        self.Q = float(process_var)
        self.R = float(measure_var)

    def update(self, z: float) -> float:
        # Predict
        self.P += self.Q
        # Update
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= (1.0 - K)
        return self.x

    def filter(self, signal: np.ndarray) -> np.ndarray:
        out = np.empty_like(signal, dtype=float)
        for i, z in enumerate(signal):
            out[i] = self.update(float(z))
        return out
