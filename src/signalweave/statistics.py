Signal statistics and descriptors.
from __future__ import annotations
import numpy as np
from scipy import stats as _st

def crest_factor(data: np.ndarray) -> float:
    return float(np.abs(data).max() / (np.sqrt(np.mean(data**2)) + 1e-10))

def form_factor(data: np.ndarray) -> float:
    rms = np.sqrt(np.mean(data**2))
    return float(np.mean(np.abs(data)) / (rms + 1e-10))

def pulse_indicator(data: np.ndarray) -> float:
    return float(np.abs(data).max() / (np.mean(np.abs(data)) + 1e-10))

def skewness_kurtosis(data: np.ndarray) -> dict[str, float]:
    return {"skewness": float(_st.skew(data)), "kurtosis": float(_st.kurtosis(data))}
