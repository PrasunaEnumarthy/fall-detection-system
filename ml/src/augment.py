# augment.py
# Six-way data augmentation for IMU fall windows.
# Applied to fall windows during training to improve generalisation and handle class imbalance.

import numpy as np
from scipy.interpolate import interp1d


def gaussian_noise(window: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Gaussian noise at a specified Signal-to-Noise Ratio.

    Higher SNR = less noise.  Tested values: 10dB (heavy), 20dB, 30dB (light).
    This simulates sensor noise variation across different IMU devices.

    Parameters
    ----------
    window : np.ndarray
        Shape (200, 6) float32 — one normalised IMU window.
    snr_db : float
        Target signal-to-noise ratio in decibels.

    Returns
    -------
    np.ndarray
        Shape (200, 6) float32 with Gaussian noise added.
    """
    # Convert SNR from dB to linear power ratio
    snr_linear = 10.0 ** (snr_db / 10.0)

    # Signal power: mean of squared values across the window
    signal_power = np.mean(window ** 2)

    # Noise power required to achieve the desired SNR
    noise_power = signal_power / snr_linear
    noise_std   = np.sqrt(noise_power)

    rng   = np.random.default_rng()  # fresh seed each call for diversity
    noise = rng.normal(0.0, noise_std, size=window.shape).astype(np.float32)

    return (window + noise).astype(np.float32)


def time_warp(window: np.ndarray, factor: float) -> np.ndarray:
    """
    Stretch or compress the temporal axis using interpolation.

    factor < 1 compresses the signal (fast motion), factor > 1 stretches it (slow motion).
    The output is always the same number of timesteps as the input.

    This simulates different fall speeds and movement tempos across subjects.

    Parameters
    ----------
    window : np.ndarray
        Shape (200, 6) float32.
    factor : float
        Warp factor: 0.85 = compress (faster), 1.15 = stretch (slower).

    Returns
    -------
    np.ndarray
        Shape (200, 6) float32.
    """
    n = window.shape[0]
    n_warped = max(2, int(n * factor))

    t_orig   = np.linspace(0, 1, n)
    t_warped = np.linspace(0, 1, n_warped)

    # Resample the original n-point signal to n_warped points
    warped = np.zeros((n_warped, window.shape[1]), dtype=np.float32)
    for ch in range(window.shape[1]):
        f = interp1d(t_orig, window[:, ch], kind="linear",
                     bounds_error=False, fill_value=(window[0, ch], window[-1, ch]))
        warped[:, ch] = f(t_warped).astype(np.float32)

    # Crop (factor > 1: only the first n samples of the stretched signal)
    if n_warped >= n:
        return warped[:n, :]

    # Pad (factor < 1: motion completed early, repeat last frame to fill window)
    pad = np.tile(warped[-1:, :], (n - n_warped, 1))
    return np.vstack([warped, pad]).astype(np.float32)


def axis_flip(window: np.ndarray) -> np.ndarray:
    """
    Flip the accelerometer X-axis to simulate a fall in the opposite lateral direction.

    Only acc_x (column 0) is negated; the other 5 channels are unchanged.
    This is physically meaningful because left/right lateral falls are mirror images.

    Parameters
    ----------
    window : np.ndarray
        Shape (200, 6) float32.

    Returns
    -------
    np.ndarray
        Shape (200, 6) float32 with column 0 negated.
    """
    flipped = window.copy()
    flipped[:, 0] *= -1.0   # negate acc_x only
    return flipped.astype(np.float32)


def augment_window(window: np.ndarray) -> list[np.ndarray]:
    """
    Apply all six augmentation strategies to one fall window.

    Augmentations:
      1. Gaussian noise at SNR 30dB  (light noise)
      2. Gaussian noise at SNR 20dB  (moderate noise)
      3. Gaussian noise at SNR 10dB  (heavy noise)
      4. Time warp at factor 0.85    (compressed / faster fall)
      5. Time warp at factor 1.15    (stretched / slower fall)
      6. Axis flip on acc_x          (lateral mirror)

    Parameters
    ----------
    window : np.ndarray
        Shape (200, 6) float32 — a single fall window (already normalised).

    Returns
    -------
    list[np.ndarray]
        Exactly 6 augmented windows, each shape (200, 6) float32.
    """
    return [
        gaussian_noise(window, snr_db=30),   # aug 1 — light noise
        gaussian_noise(window, snr_db=20),   # aug 2 — moderate noise
        gaussian_noise(window, snr_db=10),   # aug 3 — heavy noise
        time_warp(window, factor=0.85),      # aug 4 — faster fall
        time_warp(window, factor=1.15),      # aug 5 — slower fall
        axis_flip(window),                   # aug 6 — lateral mirror
    ]


# ---------------------------------------------------------------------------
# Standalone demo — run:  python src/augment.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== augment.py demo ===")

    rng    = np.random.default_rng(42)
    window = rng.standard_normal((200, 6)).astype(np.float32)
    print(f"Original window: shape={window.shape}  mean={window.mean():.4f}")

    augmented = augment_window(window)
    print(f"\nGenerated {len(augmented)} augmented windows:")

    names = [
        "noise SNR=30dB", "noise SNR=20dB", "noise SNR=10dB",
        "time_warp x0.85", "time_warp x1.15", "axis_flip",
    ]
    for name, aug in zip(names, augmented):
        diff = np.abs(aug - window).mean()
        print(f"  {name:<18}  shape={aug.shape}  mean_abs_diff={diff:.4f}")

    print("\nAll shapes correct:", all(a.shape == (200, 6) for a in augmented))
    print("All dtypes float32:", all(a.dtype == np.float32 for a in augmented))
