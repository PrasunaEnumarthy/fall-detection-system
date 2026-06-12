# preprocessor.py
# Applies a Butterworth low-pass filter to each IMU channel independently.
# Removes high-frequency noise above human movement frequencies (cutoff 20Hz).

import numpy as np
from scipy.signal import butter, filtfilt


def butterworth_filter(
    clip: np.ndarray,
    cutoff: float = 20.0,
    fs: float = 100.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a 4th-order Butterworth low-pass filter to all 6 IMU channels.

    Uses filtfilt (zero-phase forward-backward filtering) to avoid any phase
    shift in the signal — important for preserving the exact timing of impact
    peaks in fall recordings.

    Human body movements relevant to fall detection lie below ~20Hz; anything
    above that is sensor noise and electrical interference.

    Parameters
    ----------
    clip : np.ndarray
        Shape (n_samples, 6) float32. All 6 channels are filtered independently.
    cutoff : float
        Low-pass cutoff frequency in Hz (default 20Hz).
    fs : float
        Sampling frequency of the clip in Hz (default 100Hz after resampling).
    order : int
        Filter order (default 4 — good balance of roll-off steepness vs. ringing).

    Returns
    -------
    np.ndarray
        Shape (n_samples, 6) float32 — same shape as input, noise removed.
    """
    # Nyquist frequency is half the sampling rate
    nyquist = 0.5 * fs

    # Normalise cutoff to [0, 1] as required by scipy.signal.butter
    normal_cutoff = cutoff / nyquist

    # Design the filter coefficients (b = numerator, a = denominator)
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    # Apply the filter to each of the 6 channels independently
    filtered = np.zeros_like(clip, dtype=np.float32)
    for ch in range(clip.shape[1]):
        filtered[:, ch] = filtfilt(b, a, clip[:, ch]).astype(np.float32)

    return filtered


# ---------------------------------------------------------------------------
# Standalone demo — run:  python src/preprocessor.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")  # headless — no display needed for the demo
    import matplotlib.pyplot as plt

    print("=== preprocessor.py demo ===")

    # Simulate a noisy 3-second IMU clip at 100Hz
    rng = np.random.default_rng(0)
    t = np.linspace(0, 3, 300, dtype=np.float32)

    # Low-frequency signal (5Hz) + high-frequency noise (50Hz)
    signal = np.sin(2 * np.pi * 5 * t)
    noise = 0.5 * np.sin(2 * np.pi * 50 * t)
    noisy_clip = np.stack([signal + noise] * 6, axis=1)
    print(f"Input  shape: {noisy_clip.shape}  dtype: {noisy_clip.dtype}")

    filtered = butterworth_filter(noisy_clip, cutoff=20, fs=100, order=4)
    print(f"Output shape: {filtered.shape}  dtype: {filtered.dtype}")

    # Quick visual check — saved to file so it works in headless environments
    plt.figure(figsize=(10, 3))
    plt.plot(t, noisy_clip[:, 0], alpha=0.5, label="noisy")
    plt.plot(t, filtered[:, 0], linewidth=2, label="filtered (20Hz LP)")
    plt.legend()
    plt.title("Butterworth low-pass filter — channel 0")
    plt.tight_layout()
    plt.savefig("preprocessor_demo.png", dpi=100)
    print("Demo plot saved to preprocessor_demo.png")
