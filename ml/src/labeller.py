# labeller.py
# Converts SisFall activity codes into structured labels used for training.
# Provides fall type, pre-activity context, and post-fall state estimation.

import numpy as np

# ---------------------------------------------------------------------------
# Activity code → fall type mapping (F01–F15)
# Groups fall codes into three biomechanical categories
# ---------------------------------------------------------------------------
FALL_TYPE_MAP: dict[str, str] = {
    "F01": "slip",  "F02": "slip",  "F03": "slip",  "F04": "slip",  "F05": "slip",
    "F06": "trip",  "F07": "trip",  "F08": "trip",  "F09": "trip",  "F10": "trip",
    "F11": "faint", "F12": "faint", "F13": "faint", "F14": "faint", "F15": "faint",
}

# ---------------------------------------------------------------------------
# ADL activity code → human-readable activity name (D01–D19)
# ---------------------------------------------------------------------------
PRE_ACTIVITY_MAP: dict[str, str] = {
    "D01": "walking",  "D02": "walking",
    "D03": "walking",  "D04": "walking",
    "D05": "standing", "D06": "standing",
    "D07": "sitting",  "D08": "sitting",
    "D09": "bending",  "D10": "bending",
    "D11": "bending",  "D12": "standing",
    "D13": "standing", "D14": "sitting",
    "D15": "walking",  "D16": "walking",
    "D17": "sitting",  "D18": "standing",
    "D19": "walking",
}

# ---------------------------------------------------------------------------
# What was the person doing just before each fall type?
# Used to assign pre-activity to fall clips (they have no D-code)
# ---------------------------------------------------------------------------
FALL_TO_PRE_ACTIVITY: dict[str, str] = {
    "slip":  "walking",   # slips happen while walking on slippery surfaces
    "trip":  "walking",   # trips happen while walking and hitting an obstacle
    "faint": "standing",  # faints happen from a standing/stationary position
}

# Thresholds for post-fall state classification based on signal stillness
_STILLNESS_UNCONSCIOUS = 0.85
_STILLNESS_STUNNED     = 0.55

# Number of tail samples used to estimate post-fall stillness
_POST_FALL_TAIL = 30  # 0.3 seconds at 100Hz


def get_fall_label(activity_code: str) -> int:
    """
    Return binary fall label.

    Parameters
    ----------
    activity_code : str
        SisFall activity code e.g. 'F01', 'D05'.

    Returns
    -------
    int
        1 if this is a fall clip, 0 if this is a daily-activity clip.
    """
    return 1 if activity_code.upper().startswith("F") else 0


def get_fall_type(activity_code: str) -> str:
    """
    Return the fall type string for a given activity code.

    Parameters
    ----------
    activity_code : str
        SisFall code — must start with 'F' for a meaningful return value.

    Returns
    -------
    str
        'slip', 'trip', or 'faint' for fall codes.
        'none' for ADL codes (not a fall).
    """
    return FALL_TYPE_MAP.get(activity_code.upper(), "none")


def get_pre_activity(activity_code: str) -> str:
    """
    Return the activity being performed before (or during) the clip.

    For fall clips — the preceding activity is inferred from the fall type
    (e.g. slips and trips happen while walking).
    For ADL clips — the activity is read directly from PRE_ACTIVITY_MAP.

    Parameters
    ----------
    activity_code : str
        SisFall activity code e.g. 'F06' or 'D01'.

    Returns
    -------
    str
        One of: 'walking', 'standing', 'bending', 'sitting', or 'unknown'.
    """
    code = activity_code.upper()

    if code.startswith("F"):
        # Look up what precedes this fall type
        fall_type = get_fall_type(code)
        return FALL_TO_PRE_ACTIVITY.get(fall_type, "unknown")

    # ADL clip — direct lookup
    return PRE_ACTIVITY_MAP.get(code, "unknown")


def compute_post_state(window: np.ndarray, is_fall: bool = True) -> str:
    """
    Estimate the physical state of the person immediately after a fall.

    Method:
      1. Compute resultant accelerometer magnitude at each timestep.
      2. Measure variance of the magnitude over the last 30 samples (0.3s).
      3. Convert variance to a 'stillness' score via  1 / (1 + variance).
         High stillness → person is motionless → likely unconscious or stunned.

    Parameters
    ----------
    window : np.ndarray
        Shape (200, 6) float32 — one normalised IMU window.
        Columns 0-2 are acc_x, acc_y, acc_z.
    is_fall : bool
        If False (ADL window), always returns 'moving' without computation.

    Returns
    -------
    str
        'unconscious' | 'stunned' | 'moving'
    """
    # ADL windows never classify as unconscious or stunned
    if not is_fall:
        return "moving"

    # Extract accelerometer channels (columns 0, 1, 2)
    acc = window[:, :3]  # shape (200, 3)

    # Resultant magnitude: sqrt(x² + y² + z²) at each timestep
    acc_magnitude = np.sqrt(np.sum(acc ** 2, axis=1))  # shape (200,)

    # Look at the tail of the window — the period just after impact
    tail = acc_magnitude[-_POST_FALL_TAIL:]  # last 0.3 seconds

    # Variance measures how much movement is happening
    variance = float(np.var(tail))

    # Stillness score: high = barely moving, low = lots of movement
    stillness = 1.0 / (1.0 + variance)

    if stillness > _STILLNESS_UNCONSCIOUS:
        return "unconscious"
    elif stillness > _STILLNESS_STUNNED:
        return "stunned"
    else:
        return "moving"


# ---------------------------------------------------------------------------
# Standalone demo — run:  python src/labeller.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== labeller.py demo ===")

    test_codes = ["F01", "F06", "F11", "D01", "D05", "D09", "UNKNOWN"]

    for code in test_codes:
        label = get_fall_label(code)
        ftype = get_fall_type(code)
        pre   = get_pre_activity(code)
        print(f"  {code:<10} | fall={label} | type={ftype:<6} | pre_activity={pre}")

    # Demonstrate post-state detection
    print("\nPost-state examples:")

    # Simulate a 'still' window (small noise after impact)
    rng = np.random.default_rng(99)
    still_window = rng.normal(0, 0.01, (200, 6)).astype(np.float32)
    print(f"  Near-static window → {compute_post_state(still_window, is_fall=True)}")

    # Simulate an active window (large movement throughout)
    active_window = rng.normal(0, 5.0, (200, 6)).astype(np.float32)
    print(f"  Active window      → {compute_post_state(active_window, is_fall=True)}")

    # ADL window always returns 'moving'
    adl_window = rng.normal(0, 1.0, (200, 6)).astype(np.float32)
    print(f"  ADL window         → {compute_post_state(adl_window, is_fall=False)}")
