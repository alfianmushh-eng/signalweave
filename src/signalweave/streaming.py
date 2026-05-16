Online/streaming signal processing.
from __future__ import annotations
import numpy as np

class RingBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size)
        self.pos = 0
        self._full = False

    def write(self, data: np.ndarray) -> None:
        n = len(data)
        if n >= self.size:
            self.buffer = data[-self:]
            self.pos = 0
            self._full = True
            return
        end = self.pos + n
        if end <= self.size:
            self.buffer[self.pos:end] = data
        else:
            first = self.size - self.pos
            self.buffer[self.pos:] = data[:first]
            self.buffer[:n - first] = data[first:]
        self.pos = end % self.size
        if self.pos != 0 or not self._full:
            if self.pos == 0 and self._full:
                pass
            if not self._full and end >= self.size:
                self._full = True

    def read(self, *, newest: int | None = None) -> np.ndarray:
        n = newest or self.size
        if n > self.size: n = self.size
        if not self._full:
            return self.buffer[:self.pos][-n:] if n > 0 else np.array([])
        idx = (self.pos - n) % self.size
        if idx < self.pos:
            return self.buffer[idx:self.pos]
        return np.concatenate([self.buffer[idx:], self.buffer[:self.pos]])

    @property
    def is_full(self) -> bool:
        return self._full

class OnlineRMS:
    def __init__(self, window: int = 100):
        self.buf = RingBuffer(window)
        self.sq_sum = 0.0
        self.n = 0

    def update(self, x: float) -> float:
        if self.n < self.buf.size:
            self.sq_sum += x * x
            self.n += 1
        else:
            oldest = self.buf.buffer[(self.buf.pos - self.n) % self.buf.size] if self.n > 0 else 0
        self.buf.write(np.array([x]))
        return float(np.sqrt(self.sq_sum / self.n))
