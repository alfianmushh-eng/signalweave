Impulse response utilities.
from __future__ import annotations
import numpy as np

def generate_ir(*, sample_rate: float, rt60: float = 0.5, length: float | None = None) -> np.ndarray:
    n = int((length or rt60 * 3) * sample_rate)
    t = np.arange(n) / sample_rate
    ir = np.random.randn(n) * np.exp(-3 * np.log(10) * t / rt60)
    return ir / (np.abs(ir).max() + 1e-10)

def apply_ir(data: np.ndarray, ir: np.ndarray) -> np.ndarray:
    from scipy import signal as _signal
    return _signal.convolve(data, ir, mode="same")[:len(data)]
