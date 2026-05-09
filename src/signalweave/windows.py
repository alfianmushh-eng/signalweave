"""Window functions for short-time analysis."""
from __future__ import annotations

import numpy as np

_WINDOWS = {"rectangular", "hann", "hamming", "blackman", "bartlett"}


def window(name: str, n: int) -> np.ndarray:
    """Return a length-``n`` window of the requested type."""
    name = name.lower()
    if name == "rectangular":
        return np.ones(n)
    if name == "hann":
        return np.hanning(n)
    if name == "hamming":
        return np.hamming(n)
    if name == "blackman":
        return np.blackman(n)
    if name == "bartlett":
        return np.bartlett(n)
    raise ValueError(f"unknown window: {name!r} (available: {sorted(_WINDOWS)})")


def frame(signal: np.ndarray, *, size: int, hop: int) -> np.ndarray:
    """Split a 1-D signal into overlapping frames.

    Returns a 2-D array of shape ``(n_frames, size)``. Trailing samples that do
    not fill a full frame are discarded.
    """
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    if size <= 0 or hop <= 0:
        raise ValueError("size and hop must be positive")
    n = len(signal)
    if n < size:
        return np.empty((0, size), dtype=signal.dtype)
    n_frames = 1 + (n - size) // hop
    out = np.empty((n_frames, size), dtype=signal.dtype)
    for i in range(n_frames):
        start = i * hop
        out[i] = signal[start:start + size]
    return out
