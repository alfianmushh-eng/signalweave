Feature extraction over frame or window.
from __future__ import annotations

from typing import Callable

import numpy as np


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def peak_to_peak(x: np.ndarray) -> float:
    return float(x.max() - x.min())


def zero_crossing_rate(x: np.ndarray) -> float:
    signs = np.sign(x)
    return float((signs[:-1] != signs[1:]).sum()) / max(len(x) - 1, 1)


def energy(x: np.ndarray) -> float:
    return float(np.sum(x**2))


_WINDOW_FEATURES: dict[str, Callable[[np.ndarray], float]] = {
    "rms": rms,
    "peak_to_peak": peak_to_peak,
    "zcr": zero_crossing_rate,
    "energy": energy,
    "mean": lambda x: float(np.mean(x)),
    "std": lambda x: float(np.std(x)),
    "min": lambda x: float(np.min(x)),
    "max": lambda x: float(np.max(x)),
}


def extract_window_features(data: np.ndarray, *, names: list[str] | None = None) -> dict[str, float]:
    if names is None:
        names = list(_WINDOW_FEATURES)
    out: dict[str, float] = {}
    for name in names:
        fn = _WINDOW_FEATURES.get(name)
        if fn is None:
            raise ValueError(f"unknown feature: {name!r} (available: {sorted(_WINDOW_FEATURES)})")
        out[name] = fn(data)
    return out


def sliding_features(
    data: np.ndarray,
    *,
    size: int,
    hop: int,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    from signalweave import windows as _w

    frames = _w.frame(data, size=size, hop=hop)
    if not frames.shape[0]:
        return np.empty((0, 0)), []
    keys: list[str] | None = None
    rows: list[list[float]] = []
    for f in frames:
        feat = extract_window_features(f, names=feature_names)
        if keys is None:
            keys = list(feat)
        rows.append([feat[k] for k in keys])
    return np.array(rows), list(keys or [])
