Signal normalization utilities.
from __future__ import annotations
import numpy as np

def peak_normalize(data: np.ndarray) -> np.ndarray:
    return data / (np.abs(data).max() + 1e-10)

def rms_normalize(data: np.ndarray, *, target: float = 1.0) -> np.ndarray:
    rms = np.sqrt(np.mean(data**2))
    return data * target / (rms + 1e-10)

def dc_remove(data: np.ndarray) -> np.ndarray:
    return data - data.mean()
