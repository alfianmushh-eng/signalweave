Advanced spectral and temporal features.
from __future__ import annotations
import numpy as np

def spectral_centroid(f: np.ndarray, S: np.ndarray) -> np.ndarray:
    return np.sum(f[:, None] * S, axis=0) / (np.sum(S, axis=0) + 1e-10)

def spectral_bandwidth(f: np.ndarray, S: np.ndarray) -> np.ndarray:
    centroid = spectral_centroid(f, S)
    return np.sqrt(np.sum((f[:, None] - centroid) ** 2 * S, axis=0) / (np.sum(S, axis=0) + 1e-10))

def spectral_rolloff(f: np.ndarray, S: np.ndarray, *, roll_percent: float = 0.85) -> np.ndarray:
    cum = np.cumsum(S, axis=0)
    total = cum[-1, :]
    idx = np.argmax(cum >= roll_percent * total[None, :], axis=0)
    return f[idx]

def spectral_flatness(S: np.ndarray) -> np.ndarray:
    geom = np.exp(np.mean(np.log(S + 1e-10), axis=0))
    arith = np.mean(S, axis=0)
    return geom / (arith + 1e-10)
