"""Common test signal generators."""
from __future__ import annotations

import numpy as np


def sine(*, frequency: float, sample_rate: float, duration: float, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
    """Generate a sine wave."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


def square(*, frequency: float, sample_rate: float, duration: float, amplitude: float = 1.0, duty: float = 0.5) -> np.ndarray:
    """Generate a square wave with configurable duty cycle (0 < duty < 1)."""
    if not 0 < duty < 1:
        raise ValueError("duty must be in (0, 1)")
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    phase = (t * frequency) % 1.0
    return np.where(phase < duty, amplitude, -amplitude).astype(np.float64)


def chirp(*, f0: float, f1: float, sample_rate: float, duration: float) -> np.ndarray:
    """Linear chirp from f0 Hz to f1 Hz over ``duration`` seconds."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    k = (f1 - f0) / duration
    return np.sin(2 * np.pi * (f0 * t + 0.5 * k * t * t))


def white_noise(*, sample_rate: float, duration: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Zero-mean unit-variance white noise."""
    r = rng or np.random.default_rng()
    return r.standard_normal(int(sample_rate * duration))
