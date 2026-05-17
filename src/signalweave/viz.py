Signal visualization helpers.
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_waveform(data: np.ndarray, *, sample_rate: float | None = None, figsize=(10, 3)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    t = np.arange(len(data)) / sample_rate if sample_rate else np.arange(len(data))
    ax.plot(t, data, linewidth=0.5)
    ax.set_xlabel("Time (s)" if sample_rate else "Sample"); ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    plt.tight_layout(); return fig

def plot_spectrogram(f: np.ndarray, t: np.ndarray, S: np.ndarray, *, figsize=(10, 5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(t, f, 10 * np.log10(S + 1e-10), shading="gouraud", cmap="inferno")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")
    plt.tight_layout(); return fig

def plot_filter_response(freq: np.ndarray, h: np.ndarray, *, figsize=(8, 5)) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    ax1.semilogx(freq, 20 * np.log10(np.abs(h) + 1e-10))
    ax1.set_ylabel("Magnitude (dB)"); ax1.set_title("Frequency Response")
    ax2.semilogx(freq, np.angle(h, deg=True))
    ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Phase (deg)")
    plt.tight_layout(); return fig
