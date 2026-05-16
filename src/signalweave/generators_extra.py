Additional signal generators.
from __future__ import annotations
import numpy as np

def sawtooth(*, frequency: float, sample_rate: float, duration: float, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    phase = (t * frequency) % 1.0
    return amplitude * (2 * phase - 1)

def triangle(*, frequency: float, sample_rate: float, duration: float, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    phase = (t * frequency) % 1.0
    return amplitude * (4 * np.abs(phase - 0.5) - 1)

def pulse_train(*, frequency: float, sample_rate: float, duration: float, pulse_width: float = 0.01, amplitude: float = 1.0) -> np.ndarray:
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    out = np.zeros(n)
    for peak in np.arange(0, duration, 1.0 / frequency):
        idx = int(peak * sample_rate)
        width = int(pulse_width * sample_rate)
        out[idx:idx + width] = amplitude
    return out[:n]

def am_modulate(carrier_freq: float, mod_freq: float, *, sample_rate: float, duration: float, mod_index: float = 0.5) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    carrier = np.sin(2 * np.pi * carrier_freq * t)
    modulator = 1 + mod_index * np.sin(2 * np.pi * mod_freq * t)
    return carrier * modulator

def fm_modulate(carrier_freq: float, mod_freq: float, *, sample_rate: float, duration: float, dev: float = 50.0) -> np.ndarray:
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return np.sin(2 * np.pi * carrier_freq * t + dev / mod_freq * np.sin(2 * np.pi * mod_freq * t))
