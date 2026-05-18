Demo of complete signal processing pipeline.
from signalweave import generators, filtering, spectral, viz, features, audio_io

def main():
    print("Generating chirp signal...")
    sig = generators.chirp(f0=100, f1=1000, sample_rate=4000, duration=2.0)
    print(f"Filtering (lowpass 800 Hz)...")
    filtered = filtering.lowpass(sig, cutoff=800, sample_rate=4000)
    print(f"Extracting features...")
    mat, keys = features.sliding_features(filtered, size=256, hop=128)
    print(f"Feature matrix: {mat.shape}")
    print(f"Saving spectrogram...")
    r = spectral.spectrogram(filtered, sample_rate=4000, nperseg=256)
    import numpy as np
    np.savez("demo_spectrogram.npz", f=r.f, t=r.t, S=r.S)
    print("Done. Output: demo_spectrogram.npz")
if __name__ == "__main__":
    main()
