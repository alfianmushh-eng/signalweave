"""Peak detection with prominence/distance constraints."""
from __future__ import annotations
import numpy as np


def find_peaks(x: np.ndarray, height: float | None = None, distance: int = 1,
               prominence: float | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    candidates = np.where((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:]))[0] + 1
    if height is not None:
        candidates = candidates[x[candidates] >= height]
    if prominence is not None:
        keep = []
        for p in candidates:
            left = x[max(0, p - 50):p].min(initial=x[p])
            right = x[p + 1:min(len(x), p + 50)].min(initial=x[p])
            base = max(left, right)
            if x[p] - base >= prominence:
                keep.append(p)
        candidates = np.array(keep, dtype=int)
    if distance > 1 and len(candidates) > 1:
        order = np.argsort(-x[candidates])
        keep_mask = np.ones(len(candidates), dtype=bool)
        for idx in order:
            if not keep_mask[idx]:
                continue
            for j in range(len(candidates)):
                if j == idx or not keep_mask[j]:
                    continue
                if abs(candidates[j] - candidates[idx]) < distance:
                    keep_mask[j] = False
        candidates = candidates[keep_mask]
        candidates.sort()
    return candidates


def peak_widths(x: np.ndarray, peaks: np.ndarray, rel_height: float = 0.5) -> np.ndarray:
    widths = np.zeros(len(peaks))
    for i, p in enumerate(peaks):
        h = x[p] - rel_height * (x[p] - x.min())
        l = p
        while l > 0 and x[l] > h:
            l -= 1
        r = p
        while r < len(x) - 1 and x[r] > h:
            r += 1
        widths[i] = r - l
    return widths
