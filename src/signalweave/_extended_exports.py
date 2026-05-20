"""Re-exports for extended DSP modules added in v0.3.0."""
from .kalman import Kalman1D
from .kalman_cv import KalmanCV
from .beamforming import delay_and_sum, steering_vector
from .doa import music_spectrum, conventional_doa
from .pitch import autocorr_pitch, yin_lite
from .cepstrum import real_cepstrum, complex_cepstrum, cepstral_pitch
from .hilbert_tools import analytic_signal, envelope, instantaneous_frequency, instantaneous_phase
from .peaks import find_peaks, peak_widths
from .zerocross import zero_crossings, zero_crossing_rate, fundamental_period_zc
from .wavelet import haar_dwt, haar_idwt
from .wiener import wiener_filter, spectral_subtraction
from .czt import chirp_z, zoom_fft
from .cluster import kmeans, inertia

__all__ = [
    "Kalman1D", "KalmanCV",
    "delay_and_sum", "steering_vector",
    "music_spectrum", "conventional_doa",
    "autocorr_pitch", "yin_lite",
    "real_cepstrum", "complex_cepstrum", "cepstral_pitch",
    "analytic_signal", "envelope", "instantaneous_frequency", "instantaneous_phase",
    "find_peaks", "peak_widths",
    "zero_crossings", "zero_crossing_rate", "fundamental_period_zc",
    "haar_dwt", "haar_idwt",
    "wiener_filter", "spectral_subtraction",
    "chirp_z", "zoom_fft",
    "kmeans", "inertia",
]
