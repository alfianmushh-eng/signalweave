Phase vocoder and phase manipulation.
from __future__ import annotations
import numpy as np
from scipy import signal as _signal

def unwrap_phase(data: np.ndarray) -> np.ndarray:
    return _signal.detrend(np.unwrap(np.angle(data)))

def instant_frequency(data: np.ndarray) -> np.ndarray:
    analytic = _signal.hilbert(data)
    phase = np.unwrap(np.angle(analytic))
    return np.diff(phase, prepend=0)

def phase_vocoder_stretch(data: np.ndarray, *, rate: float, sample_rate: float, nperseg: int = 256) -> np.ndarray:
    from librosa.effects import time_stretch
    return time_stretch(y=data, rate=rate)
