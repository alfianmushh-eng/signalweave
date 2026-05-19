Dynamic range compression.
from __future__ import annotations
import numpy as np

def compress(data: np.ndarray, *, threshold: float = 0.3, ratio: float = 4.0, attack: float = 0.01, release: float = 0.1, sample_rate: float = 1000) -> np.ndarray:
    n = len(data)
    gain = np.ones(n)
    envelope = np.abs(data)
    env_smooth = np.zeros_like(data)
    alpha_a = np.exp(-1.0 / (attack * sample_rate + 1))
    alpha_r = np.exp(-1.0 / (release * sample_rate + 1))
    for i in range(1, n):
        alpha = alpha_a if envelope[i] > env_smooth[i - 1] else alpha_r
        env_smooth[i] = alpha * env_smooth[i - 1] + (1 - alpha) * envelope[i]
    above = env_smooth > threshold
    gain[above] = threshold + (env_smooth[above] - threshold) / ratio
    gain[above] = gain[above] / (env_smooth[above] + 1e-10)
    gain[~above] = 1.0
    return data * gain
