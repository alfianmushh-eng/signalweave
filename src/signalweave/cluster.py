"""Lightweight signal-feature clustering wrappers."""
from __future__ import annotations
import numpy as np


def kmeans(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain Lloyd k-means. Returns (labels, centroids)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=k, replace=False)
    C = X[idx].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        new = np.argmin(d, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            mask = labels == j
            if mask.any():
                C[j] = X[mask].mean(axis=0)
    return labels, C


def inertia(X: np.ndarray, labels: np.ndarray, C: np.ndarray) -> float:
    total = 0.0
    for j in range(C.shape[0]):
        mask = labels == j
        if mask.any():
            total += float(((X[mask] - C[j]) ** 2).sum())
    return total
