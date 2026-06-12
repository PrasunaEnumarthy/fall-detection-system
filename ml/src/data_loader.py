# data_loader.py
# Loads raw SisFall .txt clips from disk and returns validated numpy arrays.
# Also provides a batch loader that scans the full dataset directory tree.
#
# SisFall raw format (9 columns, comma-separated, each row ends with ';'):
#   col 0-2  ADXL345  → acc_x,  acc_y,  acc_z      ← KEEP
#   col 3-5  MMA8451  → (second accelerometer)      ← DISCARD
#   col 6-8  ITG3200  → gyro_x, gyro_y, gyro_z     ← KEEP
# Output is always (n_samples, 6): acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z

from pathlib import Path
import numpy as np

# Minimum number of samples a clip must have to be usable (2 seconds at 200Hz)
MIN_SAMPLES = 400
# Raw SisFall files have 9 columns; we select 6 after discarding MMA8451
RAW_COLUMNS    = 9
OUTPUT_COLUMNS = 6

# Column indices to keep from the 9-column raw file
_KEEP_COLS = [0, 1, 2, 6, 7, 8]  # ADXL345 + ITG3200


def load_clip(filepath: str) -> np.ndarray | None:
    """
    Load a single SisFall .txt recording file.

    Raw format: 9 comma-separated values per line, row ends with ';'.
    We keep ADXL345 (cols 0-2) as acc and ITG3200 (cols 6-8) as gyro,
    discarding the MMA8451 second accelerometer (cols 3-5).

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the .txt clip file.

    Returns
    -------
    np.ndarray or None
        Shape (n_samples, 6) float32 with columns
        [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z], or None if invalid.
    """
    path = Path(filepath)

    # --- guard: file must exist ---
    if not path.exists():
        print(f"[SKIP] File not found: {path.name}")
        return None

    rows = []
    try:
        with open(str(path), "r") as f:
            for line in f:
                # Strip whitespace and the trailing semicolon that SisFall appends
                line = line.strip().rstrip(";")
                if not line:
                    continue  # skip blank lines

                values = [float(x) for x in line.split(",")]

                # Every valid row must have exactly 9 sensor values
                if len(values) != RAW_COLUMNS:
                    print(
                        f"[SKIP] {path.name}: row with {len(values)} values "
                        f"(expected {RAW_COLUMNS}) — skipping file"
                    )
                    return None

                # Select ADXL345 (0,1,2) + ITG3200 (6,7,8), drop MMA8451 (3,4,5)
                rows.append([values[i] for i in _KEEP_COLS])

    except Exception as exc:
        print(f"[SKIP] Cannot read {path.name}: {exc}")
        return None

    if not rows:
        print(f"[SKIP] {path.name}: file produced no valid rows")
        return None

    data = np.array(rows, dtype=np.float32)  # shape (n_samples, 6)

    # --- guard: must have at least 2 seconds of data ---
    if data.shape[0] < MIN_SAMPLES:
        print(f"[SKIP] {path.name}: only {data.shape[0]} samples (need ≥{MIN_SAMPLES})")
        return None

    # --- guard: no NaN or Inf values allowed ---
    if not np.all(np.isfinite(data)):
        print(f"[SKIP] {path.name}: contains NaN or Inf values")
        return None

    return data


def _parse_filename(path: Path) -> tuple[str, str]:
    """
    Extract activity_code and subject from a SisFall filename.

    SisFall naming convention: <ActivityCode>_<Subject>_<RepNumber>.txt
    Example: F01_SA01_R01.txt  →  activity_code='F01', subject='SA01'

    Parameters
    ----------
    path : Path
        The .txt file path.

    Returns
    -------
    tuple[str, str]
        (activity_code, subject). Returns ('UNKNOWN', 'UNKNOWN') on parse error.
    """
    stem = path.stem  # e.g. "F01_SA01_R01"
    parts = stem.split("_")

    if len(parts) >= 2:
        return parts[0], parts[1]

    # Fallback — log and continue rather than crash
    print(f"[WARN] Cannot parse filename: {path.name}")
    return "UNKNOWN", "UNKNOWN"


def scan_clips(sisfall_root: str) -> list[dict]:
    """
    Scan the SisFall directory and return file metadata WITHOUT loading data.

    This is the memory-efficient entry point for the dataset builder — it
    returns only paths and labels so the builder can load one clip at a time
    and immediately discard it, keeping peak RAM at ~one-clip size instead
    of loading all 4500 clips simultaneously (~2 GB).

    Parameters
    ----------
    sisfall_root : str
        Path to the SisFall root directory.

    Returns
    -------
    list[dict]
        Each dict has keys: path, activity_code, is_fall, subject.
        No 'data' key — data must be loaded on demand with load_clip().
    """
    root = Path(sisfall_root)

    if not root.exists():
        print(f"[ERROR] SisFall root does not exist: {root}")
        return []

    all_txt_files = sorted(root.rglob("*.txt"))
    print(f"[INFO] Found {len(all_txt_files)} .txt files under {root}")

    meta = []
    for filepath in all_txt_files:
        activity_code, subject = _parse_filename(filepath)
        if activity_code == "UNKNOWN":
            continue  # skip non-recording files like Readme.txt
        is_fall = activity_code.upper().startswith("F")
        meta.append({
            "path":          str(filepath.resolve()),
            "activity_code": activity_code,
            "is_fall":       is_fall,
            "subject":       subject,
        })

    fall_count = sum(1 for m in meta if m["is_fall"])
    print(f"[INFO] Indexed {len(meta)} clips — Falls: {fall_count} | ADL: {len(meta)-fall_count}")
    return meta


def load_all_clips(sisfall_root: str) -> list[dict]:
    """
    Recursively scan a SisFall root directory and load every valid clip.

    Parameters
    ----------
    sisfall_root : str
        Path to the SisFall dataset root (contains SA01, SA02, ..., SE01, ... subfolders).

    Returns
    -------
    list[dict]
        Each dict has keys:
          - path         : str  — absolute path to the source file
          - data         : np.ndarray  shape (n, 6) float32
          - activity_code: str  — e.g. 'F01', 'D05'
          - is_fall      : bool — True if activity_code starts with 'F'
          - subject      : str  — e.g. 'SA01'
    """
    root = Path(sisfall_root)

    if not root.exists():
        print(f"[ERROR] SisFall root does not exist: {root}")
        return []

    # Collect all .txt files anywhere under the root
    all_txt_files = sorted(root.rglob("*.txt"))
    total = len(all_txt_files)
    print(f"[INFO] Found {total} .txt files under {root}")

    clips = []
    loaded = skipped = fall_count = adl_count = 0

    for idx, filepath in enumerate(all_txt_files, start=1):
        # Progress indicator every 50 files so long runs stay visible
        if idx % 50 == 0 or idx == total:
            print(f"  Loading clip {idx}/{total}...")

        data = load_clip(str(filepath))

        if data is None:
            skipped += 1
            continue

        activity_code, subject = _parse_filename(filepath)
        is_fall = activity_code.upper().startswith("F")

        clips.append({
            "path": str(filepath.resolve()),
            "data": data,
            "activity_code": activity_code,
            "is_fall": is_fall,
            "subject": subject,
        })
        loaded += 1

        if is_fall:
            fall_count += 1
        else:
            adl_count += 1

    # Summary line — useful for a quick sanity check after loading
    print(
        f"\n[SUMMARY] Loaded: {loaded} | Skipped: {skipped} | "
        f"Falls: {fall_count} | ADL: {adl_count}"
    )
    return clips


# ---------------------------------------------------------------------------
# Standalone demo — run:  python src/data_loader.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Default path resolved relative to this script's location (ml/src/ → ml/data/)
    _default = Path(__file__).parent.parent / "data" / "SisFall_dataset"
    sisfall_path = sys.argv[1] if len(sys.argv) > 1 else str(_default)

    print(f"=== data_loader.py demo — scanning {sisfall_path} ===")
    clips = load_all_clips(sisfall_path)

    if clips:
        first = clips[0]
        print(f"\nFirst clip : {Path(first['path']).name}")
        print(f"  Activity : {first['activity_code']}  |  Subject: {first['subject']}")
        print(f"  Is fall  : {first['is_fall']}")
        print(f"  Shape    : {first['data'].shape}  dtype: {first['data'].dtype}")
        print(f"  Row[0]   : {first['data'][0]}")
        print(f"  Columns  : acc_x  acc_y  acc_z  gyro_x  gyro_y  gyro_z")
    else:
        print("No clips loaded — check that the SisFall data is present.")
