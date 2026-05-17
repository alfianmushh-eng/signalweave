Signal segmentation utilities.
from __future__ import annotations
import numpy as np

def silence_detect(data: np.ndarray, *, threshold: float = 0.01, min_silence: int = 100) -> list[tuple[int, int]]:
    above = np.abs(data) > threshold
    changes = np.diff(np.concatenate([[0], above.astype(int), [0]]))
    starts = np.where(changes == -1)[0]
    ends = np.where(changes == 1)[0]
    segments = [(s, e) for s, e in zip(starts, ends) if e - s >= min_silence]
    return segments

def split_on_silence(data: np.ndarray, *, threshold: float = 0.01, min_silence: int = 100, min_segment: int = 50) -> list[np.ndarray]:
    silent = silence_detect(data, threshold=threshold, min_silence=min_silence)
    segments = []
    prev = 0
    for s, _ in silent:
        if s - prev >= min_segment:
            segments.append(data[prev:s])
        prev = s
    if len(data) - prev >= min_segment:
        segments.append(data[prev:])
    return segments
