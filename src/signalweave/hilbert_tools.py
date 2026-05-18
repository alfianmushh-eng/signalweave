"""Hilbert-transform-based analytic signal utilities."""
from __future__ import annotations
import numpy as np
from scipy.signal import hilbert


def analytic_signal(x: np.ndarray) -> np.ndarray:
    return hilbert(x)


def instantaneous_phase(x: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(hilbert(x)))


def instantaneous_frequency(x: np.ndarray, sr: float) -> np.ndarray:
    phase = instantaneous_phase(x)
    return np.diff(phase) * sr / (2.0 * np.pi)


def envelope(x: np.ndarray) -> np.ndarray:
    return np.abs(hilbert(x))
