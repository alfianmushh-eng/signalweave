"""Example: Delay-and-sum beamforming on a 4-mic ULA."""
import numpy as np
from signalweave.beamforming import delay_and_sum
from signalweave.doa import music_spectrum


def main() -> None:
    sr = 16000
    c = 343.0
    spacing = 0.05
    n_mics = 4
    f = 1000.0
    duration = 0.05
    angle_true = 30.0
    rng = np.random.default_rng(0)

    n_samples = int(sr * duration)
    t = np.arange(n_samples) / sr
    src = np.sin(2 * np.pi * f * t)
    delays = np.arange(n_mics) * spacing * np.sin(np.deg2rad(angle_true)) / c
    signals = np.array([np.interp(t - d, t, src, left=0, right=0) for d in delays])
    signals += 0.05 * rng.normal(size=signals.shape)

    enhanced = delay_and_sum(signals, angle_true, spacing, sr)
    print(f"Beamformer output RMS: {np.sqrt(np.mean(enhanced ** 2)):.3f}")

    angles = np.linspace(-90, 90, 181)
    spec = music_spectrum(signals.astype(complex), n_sources=1,
                          spacing=spacing, freq=f, angles_deg=angles)
    est = angles[int(np.argmax(spec))]
    print(f"True angle: {angle_true} deg | MUSIC estimate: {est:.1f} deg")


if __name__ == "__main__":
    main()
