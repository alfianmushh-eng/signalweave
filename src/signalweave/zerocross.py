"""Zero-crossing analysis utilities."""
from __future__ import annotations
import numpy as np


def zero_crossings(x: np.ndarray) -> np.ndarray:
    """Indices where signal changes sign."""
    s = np.sign(x)
    return np.where(np.diff(s) != 0)[0]


def zero_crossing_rate(x: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    """ZCR per frame."""
    n_frames = max(1, 1 + (len(x) - frame_size) // hop)
    out = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        frame = x[start:start + frame_size]
        out[i] = float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))
    return out


def fundamental_period_zc(x: np.ndarray, sr: float) -> float:
    """Estimate fundamental period from average distance between zero crossings."""
    zc = zero_crossings(x)
    if len(zc) < 2:
        return 0.0
    return float(2.0 * np.mean(np.diff(zc)) / sr)
