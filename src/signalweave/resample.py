Resampling utilities.
from __future__ import annotations
import numpy as np
from scipy import signal as _signal

def resample(data: np.ndarray, *, orig_rate: float, target_rate: float) -> np.ndarray:
    if orig_rate == target_rate: return data
    num = int(len(data) * target_rate / orig_rate)
    return _signal.resample(data, num)

def decimate(data: np.ndarray, *, factor: int) -> np.ndarray:
    return _signal.decimate(data, factor)

def interpolate(data: np.ndarray, *, factor: int) -> np.ndarray:
    return _signal.resample(data, len(data) * factor)
