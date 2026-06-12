# windower.py
# Splits a continuous IMU clip into fixed-length overlapping windows.
# Also provides per-window z-score normalisation for stable model inputs.

import numpy as np


def slice_windows(
    clip: np.ndarray,
    window_size: int = 200,
    step_size: int = 100,
) -> list[np.ndarray]:
    """
    Slide a fixed-size window over a clip and return all complete windows.

    With window_size=200 and step_size=100 at 100Hz:
      - Each window = 2 seconds of sensor data
      - Windows overlap by 50% (new window starts every 1 second)

    Parameters
    ----------
    clip : np.ndarray
        Shape (n_samples, 6) float32. Clip already at 100Hz after resampling.
    window_size : int
        Number of timesteps per window (default 200 = 2 seconds at 100Hz).
    step_size : int
        Number of timesteps to advance between windows (default 100 = 1 second).

    Returns
    -------
    list[np.ndarray]
        Each element has shape (window_size, 6) = (200, 6).
        Windows shorter than window_size (at the clip end) are discarded.
    """
    windows = []
    n_samples = clip.shape[0]
    start = 0

    while start + window_size <= n_samples:
        window = clip[start : start + window_size]  # shape (200, 6)
        windows.append(window.astype(np.float32))
        start += step_size

    return windows


def zscore_normalise(window: np.ndarray) -> np.ndarray:
    """
    Normalise a single window using per-channel z-score standardisation.

    Each of the 6 channels is centred to mean=0 and scaled to std=1.
    A small epsilon (1e-8) prevents division by zero for flat channels.

    This step ensures that sensor biases and scale differences between
    subjects and devices do not affect model inputs.

    Parameters
    ----------
    window : np.ndarray
        Shape (200, 6) float32 — a single IMU window.

    Returns
    -------
    np.ndarray
        Shape (200, 6) float32 with each channel standardised independently.
    """
    epsilon = 1e-8  # prevents divide-by-zero on constant channels

    # Compute per-channel statistics across the time axis (axis=0)
    mean = window.mean(axis=0)   # shape (6,)
    std = window.std(axis=0)     # shape (6,)

    normalised = (window - mean) / (std + epsilon)
    return normalised.astype(np.float32)


# ---------------------------------------------------------------------------
# Standalone demo — run:  python src/windower.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== windower.py demo ===")

    # Simulate a 6-second clip at 100Hz (600 samples, 6 channels)
    rng = np.random.default_rng(7)
    clip = rng.standard_normal((600, 6)).astype(np.float32)
    print(f"Clip shape: {clip.shape}")

    windows = slice_windows(clip, window_size=200, step_size=100)
    print(f"Windows extracted: {len(windows)}")
    print(f"Each window shape: {windows[0].shape}")

    # Demonstrate normalisation on the first window
    norm_w = zscore_normalise(windows[0])
    print(f"\nNormalised window[0]:")
    print(f"  Channel means : {norm_w.mean(axis=0).round(4)}")  # should be ~0
    print(f"  Channel stds  : {norm_w.std(axis=0).round(4)}")   # should be ~1
