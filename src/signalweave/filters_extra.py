Additional filter implementations.
from __future__ import annotations
import numpy as np
from scipy import signal as _signal, ndimage

def fir_lowpass(data: np.ndarray, *, cutoff: float, sample_rate: float, numtaps: int = 101) -> np.ndarray:
    taps = _signal.firwin(numtaps, cutoff, fs=sample_rate)
    return _signal.convolve(data, taps, mode="same")

def median_filter(data: np.ndarray, *, size: int = 5) -> np.ndarray:
    return ndimage.median_filter(data, size)

def savgol(data: np.ndarray, *, window: int = 11, order: int = 2) -> np.ndarray:
    return _signal.savgol_filter(data, window, order)

def butter_bandstop(data: np.ndarray, *, lo: float, hi: float, sample_rate: float, order: int = 4) -> np.ndarray:
    sos = _signal.iirfilter(order, [lo, hi], fs=sample_rate, btype="bandstop", ftype="butter")
    return _signal.sosfiltfilt(sos, data)
