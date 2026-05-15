Envelope detection.
from __future__ import annotations
import numpy as np
from scipy import signal as _signal

def hilbert_envelope(data: np.ndarray) -> np.ndarray:
    return np.abs(_signal.hilbert(data))

def peak_envelope(data: np.ndarray, *, window: int = 100) -> np.ndarray:
    out = np.zeros_like(data)
    for i in range(0, len(data), window):
        seg = data[i:i + window]
        if len(seg) == 0: continue
        out[i:i + window] = np.max(np.abs(seg))
    return out

def rms_envelope(data: np.ndarray, *, window: int = 100) -> np.ndarray:
    out = np.zeros_like(data)
    for i in range(0, len(data), window):
        seg = data[i:i + window]
        if len(seg) == 0: continue
        out[i:i + window] = np.sqrt(np.mean(seg ** 2))
    return out
