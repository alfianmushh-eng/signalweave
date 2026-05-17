Filter design and analysis utilities.
from __future__ import annotations
import numpy as np
from scipy import signal as _signal

def freq_response(b: np.ndarray, a: np.ndarray, *, sample_rate: float = 1000) -> tuple[np.ndarray, np.ndarray]:
    w, h = _signal.freqz(b, a, worN=4096)
    freq = w * sample_rate / (2 * np.pi)
    return freq, h

def sos_freq_response(sos: np.ndarray, *, sample_rate: float = 1000) -> tuple[np.ndarray, np.ndarray]:
    w, h = _signal.sosfreqz(sos, worN=4096)
    freq = w * sample_rate / (2 * np.pi)
    return freq, h

def group_delay(b: np.ndarray, a: np.ndarray, *, sample_rate: float = 1000) -> tuple[np.ndarray, np.ndarray]:
    w, gd = _signal.group_delay((b, a), w=4096)
    freq = w * sample_rate / (2 * np.pi)
    return freq, gd
