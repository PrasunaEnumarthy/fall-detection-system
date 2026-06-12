# infer.py
# Real-time inference using ONNX Runtime + HTTP alert dispatch to the backend.
# Simulation mode: replays fall windows from dataset.csv to test the full pipeline.

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import os
import onnxruntime as ort

# ------------------------------------------------------------------ #
# Load environment variables from ml/.env                            #
# ------------------------------------------------------------------ #
# Walk up from src/ to find .env in ml/
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=str(_ENV_PATH))

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:3001")

# ------------------------------------------------------------------ #
# Label decode tables — index matches model output head ordering      #
# ------------------------------------------------------------------ #
FALL_TYPE_IDX: dict[int, str] = {
    0: "slip",
    1: "trip",
    2: "faint",
}

PRE_ACTIVITY_IDX: dict[int, str] = {
    0: "walking",
    1: "standing",
    2: "bending",
    3: "sitting",
}

# ------------------------------------------------------------------ #
# ONNX model path                                                     #
# ------------------------------------------------------------------ #
ONNX_PATH   = Path(__file__).parent.parent / "models" / "model.onnx"
DATASET_CSV = Path(__file__).parent.parent / "data"   / "dataset.csv"

# Stillness thresholds (same as labeller.py)
_STILLNESS_UNCONSCIOUS = 0.85
_STILLNESS_STUNNED     = 0.55

# Seconds to sleep between simulated alerts (avoids flooding the backend)
ALERT_SLEEP_SECONDS = 2


def _load_onnx_session(onnx_path: str = None) -> ort.InferenceSession:
    """
    Load the ONNX Runtime inference session.

    Parameters
    ----------
    onnx_path : str, optional
        Path to model.onnx. Defaults to ml/models/model.onnx.

    Returns
    -------
    ort.InferenceSession
    """
    path = Path(onnx_path) if onnx_path else ONNX_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at {path}\n"
            "Run:  python src/export_onnx.py  first."
        )

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print(f"[INFER] Loaded ONNX model from {path.name}")
    return session


# Module-level session — loaded once, reused for every window
_session: ort.InferenceSession | None = None


def _get_session() -> ort.InferenceSession:
    """Lazy-load and cache the ONNX session at module level."""
    global _session
    if _session is None:
        _session = _load_onnx_session()
    return _session


def run_inference(window: np.ndarray) -> dict | None:
    """
    Run fall detection inference on a single 2-second IMU window.

    Parameters
    ----------
    window : np.ndarray
        Shape (200, 6) float32 — already z-score normalised.

    Returns
    -------
    dict or None
        None if fall probability < 0.5 (no fall detected).
        Otherwise: {'fall_type': str, 'pre_activity': str, 'confidence': float}
    """
    session = _get_session()

    # Reshape to (1, 200, 6) — ONNX input expects a batch dimension
    x = window.reshape(1, 200, 6).astype(np.float32)

    # Run all three output heads in one call
    ort_outputs = session.run(None, {"imu_window": x})
    fall_raw, fall_type_logits, pre_activity_logits = ort_outputs

    # fall_prob is already a sigmoid probability (from the fall head)
    fall_prob = float(fall_raw[0, 0])

    # No fall detected — skip silently
    if fall_prob < 0.5:
        return None

    # Determine fall type from argmax of logits
    # Index 0 = 'none', so real types start at index 1
    ft_idx    = int(np.argmax(fall_type_logits[0]))
    fall_type = FALL_TYPE_IDX.get(ft_idx, "slip")

    # Determine pre-activity from argmax
    pre_idx      = int(np.argmax(pre_activity_logits[0]))
    pre_activity = PRE_ACTIVITY_IDX.get(pre_idx, "walking")

    confidence = round(fall_prob, 4)

    return {
        "fall_type":    fall_type,
        "pre_activity": pre_activity,
        "confidence":   confidence,
    }


def compute_post_state(signal_after_fall: np.ndarray) -> str:
    """
    Estimate the physical state of the person using the signal immediately after a fall.

    Uses the variance of the resultant acceleration magnitude to measure
    how much movement is still happening after the impact.

    Parameters
    ----------
    signal_after_fall : np.ndarray
        Shape (300, 6) float32 — 3 seconds at 100Hz immediately after the fall event.
        Columns 0-2 are acc_x, acc_y, acc_z.

    Returns
    -------
    str
        'unconscious' | 'stunned' | 'moving'
    """
    # Resultant accelerometer magnitude at each timestep
    acc            = signal_after_fall[:, :3]   # (300, 3)
    acc_magnitude  = np.sqrt(np.sum(acc ** 2, axis=1))  # (300,)

    # Variance of the full 3-second window
    variance  = float(np.var(acc_magnitude))
    stillness = 1.0 / (1.0 + variance)

    if stillness > _STILLNESS_UNCONSCIOUS:
        return "unconscious"
    elif stillness > _STILLNESS_STUNNED:
        return "stunned"
    else:
        return "moving"


def send_alert(result: dict, post_state: str) -> None:
    """
    POST a fall alert to the backend API.

    The backend (Team B) expects:
      { fall_type, pre_activity, post_state, confidence }

    Handles connection errors gracefully — inference continues even
    when the backend is unreachable (e.g. during standalone testing).

    Parameters
    ----------
    result : dict
        Output from run_inference(): keys fall_type, pre_activity, confidence.
    post_state : str
        State estimate from compute_post_state().
    """
    payload = {
        "fall_type":    result["fall_type"],
        "pre_activity": result["pre_activity"],
        "post_state":   post_state,
        "confidence":   result["confidence"],
    }

    url = f"{BACKEND_URL}/api/alert"

    try:
        response = requests.post(url, json=payload, timeout=5)

        if response.status_code in (200, 201):
            print(
                f"  [SENT]  fall_type={result['fall_type']:<8} | "
                f"pre={result['pre_activity']:<10} | "
                f"post={post_state:<12} | "
                f"conf={result['confidence']:.4f}"
            )
        else:
            print(
                f"  [ERROR] HTTP {response.status_code} from {url}: "
                f"{response.text[:200]}"
            )

    except requests.exceptions.ConnectionError:
        # Backend is offline — print and continue, do not crash
        print(
            f"  [ERROR] Backend unreachable at {url} "
            f"(fall_type={result['fall_type']}, conf={result['confidence']:.4f})"
        )
    except requests.exceptions.Timeout:
        print(f"  [ERROR] Request to {url} timed out after 5s")
    except Exception as exc:
        print(f"  [ERROR] Unexpected error sending alert: {exc}")


def run_simulation(dataset_csv: str = None) -> None:
    """
    Simulation mode: replay all fall windows from dataset.csv through the full pipeline.

    For each fall window:
      1. Reconstruct the (200,6) array from the flat f_0…f_1199 columns.
      2. Run inference — skip if no fall detected.
      3. Estimate post-fall state using the same window (approximate).
      4. Send the alert to the backend.
      5. Sleep 2 seconds to simulate real-time cadence.

    Parameters
    ----------
    dataset_csv : str, optional
        Path to the flat feature CSV. Defaults to ml/data/dataset.csv.
    """
    csv_path = Path(dataset_csv) if dataset_csv else DATASET_CSV

    if not csv_path.exists():
        print(f"[INFER] Dataset CSV not found: {csv_path}")
        print("        Run python src/dataset_builder.py first.")
        return

    print(f"\n[INFER] Loading dataset from {csv_path.name} ...")
    df = pd.read_csv(str(csv_path))

    # Keep only fall windows — ADL windows are not used in simulation
    fall_df = df[df["fall_label"] == 1].reset_index(drop=True)
    print(f"[INFER] Fall windows available: {len(fall_df):,}")
    print(f"[INFER] Backend URL: {BACKEND_URL}")
    print(f"[INFER] Starting simulation — {ALERT_SLEEP_SECONDS}s pause between alerts\n")

    alerts_sent  = 0
    alerts_skipped = 0

    for row_idx in range(len(fall_df)):
        row = fall_df.iloc[row_idx]

        # Reconstruct (200, 6) window from the 1200 flat feature values
        flat   = row[[f"f_{i}" for i in range(1200)]].values.astype(np.float32)
        window = flat.reshape(200, 6)

        # Run fall detection
        result = run_inference(window)

        if result is None:
            # Model didn't fire on this window (fall_prob < 0.5)
            alerts_skipped += 1
            continue

        # Estimate post-fall state using the same window as a proxy
        # In production this would be 300 samples AFTER the fall event;
        # here we use the available window as an approximation.
        post_signal = np.tile(window, (2, 1))[:300]   # pad to 300 samples
        post_state  = compute_post_state(post_signal)

        send_alert(result, post_state)
        alerts_sent += 1

        # Simulate real-time gap between fall events
        time.sleep(ALERT_SLEEP_SECONDS)

    print(f"\n[INFER] Simulation complete.")
    print(f"  Total fall windows : {len(fall_df):,}")
    print(f"  Alerts sent        : {alerts_sent:,}")
    print(f"  Skipped (conf<0.5) : {alerts_skipped:,}")


# ---------------------------------------------------------------------------
# Standalone entry point — run:  python src/infer.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== infer.py — fall detection simulation ===")
    print(f"Backend URL : {BACKEND_URL}")
    run_simulation()
