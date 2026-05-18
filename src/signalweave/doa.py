"""Direction-of-arrival estimation via MUSIC and conventional beamforming."""
from __future__ import annotations
import numpy as np
from .beamforming import steering_vector


def covariance(signals: np.ndarray) -> np.ndarray:
    """Sample covariance matrix across snapshots."""
    return (signals @ signals.conj().T) / signals.shape[1]


def music_spectrum(signals: np.ndarray, n_sources: int, spacing: float,
                   freq: float, angles_deg: np.ndarray, c: float = 343.0) -> np.ndarray:
    """MUSIC pseudo-spectrum over candidate angles."""
    R = covariance(signals)
    eigvals, eigvecs = np.linalg.eigh(R)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    En = eigvecs[:, n_sources:]
    n_mics = signals.shape[0]
    spec = np.empty(len(angles_deg))
    for i, ang in enumerate(angles_deg):
        a = steering_vector(n_mics, ang, spacing, freq, c)
        proj = En.conj().T @ a
        spec[i] = 1.0 / float(np.real(proj.conj() @ proj) + 1e-12)
    return spec


def conventional_doa(signals: np.ndarray, spacing: float, freq: float,
                     angles_deg: np.ndarray, c: float = 343.0) -> np.ndarray:
    R = covariance(signals)
    n_mics = signals.shape[0]
    spec = np.empty(len(angles_deg))
    for i, ang in enumerate(angles_deg):
        a = steering_vector(n_mics, ang, spacing, freq, c)
        spec[i] = float(np.real(a.conj() @ R @ a))
    return spec
