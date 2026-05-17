"""Constant-velocity Kalman filter (position + velocity state)."""
from __future__ import annotations
import numpy as np


class KalmanCV:
    """2-state Kalman filter: [position, velocity]."""

    def __init__(self, dt: float = 1.0, process_var: float = 1e-3, measure_var: float = 1e-2):
        self.dt = dt
        self.x = np.zeros(2)
        self.P = np.eye(2)
        self.F = np.array([[1.0, dt], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = process_var * np.array([[dt**4 / 4, dt**3 / 2],
                                          [dt**3 / 2, dt**2]])
        self.R = np.array([[measure_var]])

    def step(self, z: float) -> np.ndarray:
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Update
        y = np.array([z]) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ y).ravel()
        self.P = (np.eye(2) - K @ self.H) @ self.P
        return self.x.copy()

    def run(self, measurements: np.ndarray) -> np.ndarray:
        out = np.empty((len(measurements), 2))
        for i, z in enumerate(measurements):
            out[i] = self.step(float(z))
        return out
