Adaptive filters (LMS/NLMS).
from __future__ import annotations
import numpy as np

def lms_filter(data: np.ndarray, desired: np.ndarray, *, mu: float = 0.01, order: int = 32) -> tuple[np.ndarray, np.ndarray]:
    n = len(data)
    w = np.zeros(order)
    e = np.zeros(n)
    y = np.zeros(n)
    for i in range(order, n):
        x = data[i - order:i][::-1]
        y[i] = np.dot(w, x)
        e[i] = desired[i] - y[i]
        w += mu * e[i] * x / (np.dot(x, x) + 1e-10)
    return y, e

def nlms_filter(data: np.ndarray, desired: np.ndarray, *, mu: float = 0.1, order: int = 32, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    n = len(data)
    w = np.zeros(order)
    e = np.zeros(n)
    y = np.zeros(n)
    for i in range(order, n):
        x = data[i - order:i][::-1]
        y[i] = np.dot(w, x)
        e[i] = desired[i] - y[i]
        w += mu * e[i] * x / (np.dot(x, x) + eps)
    return y, e
