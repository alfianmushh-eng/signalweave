# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-05-24

### Added
- `generators` module: sine, square, chirp, white noise signal synthesis
- `windows` module: Hann/Hamming/Blackman/Bartlett windows and overlapping framing
- `filtering` module: Butterworth IIR lowpass/highpass/bandpass/notch, moving average
- `spectral` module: spectrogram, periodogram PSD, and peak finding
- `features` module: RMS, ZCR, peak-to-peak, energy, mean/std/min/max, sliding-window feature frames
- `signalweave` CLI with `gen`, `frame`, `features`, and `spectrogram` subcommands
- `examples/ecg_like.py` synthetic ECG signal walkthrough
