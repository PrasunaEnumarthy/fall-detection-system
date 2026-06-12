# resampler.py
# Downsamples SisFall clips from 200Hz to 100Hz using Fourier-based resampling.
# Keeps all 6 channels intact and enforces float32 throughout.

import numpy as np
from scipy.signal import resample


def resample_clip(
    clip: np.ndarray,
    orig_hz: int = 200,
    target_hz: int = 100,
) -> np.ndarray:
    """
    Downsample a multi-channel IMU clip using the Fourier method.

    scipy.signal.resample applies a Fourier-domain resample (sinc interpolation),
    which is preferred over decimation because it avoids aliasing artifacts
    without needing a separate anti-alias filter step.

    Parameters
    ----------
    clip : np.ndarray
        Shape (n_samples, 6) float32 input at orig_hz.
    orig_hz : int
        Original sampling rate in Hz (default 200 for SisFall).
    target_hz : int
        Desired output rate in Hz (default 100).

    Returns
    -------
    np.ndarray
        Shape (n_samples * target_hz // orig_hz, 6) float32.
    """
    # Calculate how many samples the output should have
    n_input = clip.shape[0]
    n_output = int(n_input * target_hz / orig_hz)

    # scipy.resample works along axis=0 by default, resampling each column independently
    resampled = resample(clip, num=n_output, axis=0)

    # Cast back to float32 — scipy may upcast to float64 internally
    return resampled.astype(np.float32)


def validate_resample(original: np.ndarray, resampled: np.ndarray) -> bool:
    """
    Validate that a resample operation produced a well-formed result.

    Checks:
      1. Output length is exactly half the input length (for 200→100Hz).
      2. No NaN values were introduced by the resampling.

    Parameters
    ----------
    original : np.ndarray
        The original clip before resampling.
    resampled : np.ndarray
        The output from resample_clip().

    Returns
    -------
    bool
        True if both checks pass, False otherwise.
    """
    expected_len = original.shape[0] // 2

    # Check 1: shape must be (n/2, channels)
    if resampled.shape[0] != expected_len:
        print(
            f"[WARN] validate_resample: expected {expected_len} samples, "
            f"got {resampled.shape[0]}"
        )
        return False

    # Check 2: no NaN allowed
    if not np.all(np.isfinite(resampled)):
        print("[WARN] validate_resample: resampled clip contains NaN/Inf")
        return False

    return True


# ---------------------------------------------------------------------------
# Standalone demo — run:  python src/resampler.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== resampler.py demo ===")

    # Simulate a 4-second SisFall clip at 200Hz (800 samples, 6 channels)
    rng = np.random.default_rng(42)
    dummy_clip = rng.standard_normal((800, 6)).astype(np.float32)
    print(f"Original shape : {dummy_clip.shape}  dtype: {dummy_clip.dtype}")

    resampled = resample_clip(dummy_clip, orig_hz=200, target_hz=100)
    print(f"Resampled shape: {resampled.shape}  dtype: {resampled.dtype}")

    ok = validate_resample(dummy_clip, resampled)
    print(f"Validation     : {'PASS' if ok else 'FAIL'}")
