Time-domain feature extraction.
from __future__ import annotations
import numpy as np

def peak_to_peak(data: np.ndarray) -> float:
    return float(data.max() - data.min())

def energy_ratio(data: np.ndarray, *, start: float = 0.0, end: float = 0.1) -> float:
    n = len(data)
    seg_start, seg_end = int(start * n), int(end * n)
    seg = data[seg_start:seg_end]
    total = data
    return float(np.sum(seg**2) / (np.sum(total**2) + 1e-10))

def zero_crossing_density(data: np.ndarray) -> float:
    signs = np.sign(data)
    zc = ((signs[:-1] != signs[1:]) & (signs[:-1] != 0)).sum()
    return float(zc) / max(len(data) - 1, 1)
