# Changelog

## [0.2.0] - 2026-05-29

### Added
- Extra generators: sawtooth, triangle, pulse train, AM, FM
- Extra filters: FIR, median, Savitzky-Golay, bandstop
- Filter design: frequency response, group delay analysis
- Advanced spectral: STFT, mel-spectrogram, MFCC, cepstrum
- Advanced features: spectral centroid, bandwidth, rolloff, flatness
- Envelope detection: Hilbert, peak, RMS
- Cross-correlation, auto-correlation, convolution
- Resampling: sample rate conversion, decimate, interpolate
- Streaming: RingBuffer, OnlineRMS
- Audio I/O: WAV read/write, load with resampling
- Signal normalization: peak, RMS, DC removal
- Silence detection and segmentation
- Onset/transient detection via spectral flux
- Harmonic analysis: HPS, fundamental frequency
- Phase vocoder utilities
- Colored noise: pink, brown, blue
- Dithering and quantization
- Signal statistics: crest factor, form factor, skewness
- Time-domain features
- Visualization: waveform, spectrogram, filter response
- Extended CLI: correlate, envelope, info commands
- Integration test suite
- Pipeline demo example

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
