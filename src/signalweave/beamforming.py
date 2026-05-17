"""Delay-and-sum beamforming for uniform linear arrays."""
from __future__ import annotations
import numpy as np


def steering_vector(n_mics: int, angle_deg: float, spacing: float,
                    freq: float, c: float = 343.0) -> np.ndarray:
    """Narrowband steering vector for ULA at given angle (broadside=0)."""
    theta = np.deg2rad(angle_deg)
    k = 2 * np.pi * freq / c
    n = np.arange(n_mics)
    return np.exp(-1j * k * spacing * n * np.sin(theta))


def delay_and_sum(signals: np.ndarray, angle_deg: float, spacing: float,
                  sr: float, c: float = 343.0) -> np.ndarray:
    """Time-domain delay-and-sum across channels (signals shape: [n_mics, n_samples])."""
    n_mics, n_samples = signals.shape
    theta = np.deg2rad(angle_deg)
    delays_samples = (np.arange(n_mics) * spacing * np.sin(theta) / c) * sr
    out = np.zeros(n_samples)
    for m in range(n_mics):
        d = int(round(delays_samples[m]))
        if d >= 0:
            out[d:] += signals[m, : n_samples - d]
        else:
            out[:d] += signals[m, -d:]
    return out / n_mics
