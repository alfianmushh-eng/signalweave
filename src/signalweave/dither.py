Dithering and noise shaping utilities.
from __future__ import annotations
import numpy as np

def dither(data: np.ndarray, *, bits: int = 16, rng: np.random.Generator | None = None) -> np.ndarray:
    r = rng or np.random.default_rng()
    scale = 2 ** (bits - 1)
    noise = r.uniform(-0.5 / scale, 0.5 / scale, size=data.shape)
    return data + noise

def quantize(data: np.ndarray, *, bits: int = 16) -> np.ndarray:
    scale = 2 ** (bits - 1)
    return np.clip(np.round(data * scale) / scale, -1.0, 1.0)
