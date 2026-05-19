"""Example: Kalman filter tracking a noisy ramp."""
import numpy as np
from signalweave.kalman_cv import KalmanCV


def main() -> None:
    rng = np.random.default_rng(42)
    t = np.arange(200)
    truth_pos = 0.05 * t
    measurements = truth_pos + rng.normal(0, 0.5, len(t))
    kf = KalmanCV(dt=1.0, process_var=1e-3, measure_var=0.25)
    estimates = kf.run(measurements)
    err = np.mean((estimates[:, 0] - truth_pos) ** 2)
    print(f"Tracking MSE: {err:.4f}")


if __name__ == "__main__":
    main()
