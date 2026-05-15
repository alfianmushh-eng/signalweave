Advanced spectral analysis: STFT, mel, MFCC, cepstrum.
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy import signal as _signal

@dataclass
class StftResult:
    f: np.ndarray; t: np.ndarray; Zxx: np.ndarray

def stft(data: np.ndarray, *, sample_rate: float, nperseg: int = 256, noverlap: int | None = None) -> StftResult:
    f, t, Zxx = _signal.stft(data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap or nperseg // 2)
    return StftResult(f=f, t=t, Zxx=Zxx)

def mel_spectrogram(data: np.ndarray, *, sample_rate: float, n_mels: int = 128, nperseg: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    from librosa.feature import melspectrogram as librosa_mel
    spec = librosa_mel(y=data, sr=sample_rate, n_mels=n_mels, n_fft=nperseg, hop_length=nperseg // 2)
    return spec, np.linspace(0, sample_rate / 2, spec.shape[0])

def mfcc(data: np.ndarray, *, sample_rate: float, n_mfcc: int = 13, n_mels: int = 128) -> np.ndarray:
    from librosa.feature import mfcc as librosa_mfcc
    return librosa_mfcc(y=data, sr=sample_rate, n_mfcc=n_mfcc, n_mels=n_mels)

def cepstrum(data: np.ndarray) -> np.ndarray:
    return np.fft.irfft(np.log(np.abs(np.fft.rfft(data)) + 1e-10))
