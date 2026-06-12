# dataset_builder.py
# Orchestrates the full data pipeline: load → resample → filter → window → normalise → label.
# Outputs a flat CSV where each row is one 2-second window ready for model training.

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add src/ to path so we can import sibling modules when run as __main__
sys.path.insert(0, str(Path(__file__).parent))

from data_loader   import scan_clips, load_clip
from resampler     import resample_clip, validate_resample
from preprocessor  import butterworth_filter
from windower      import slice_windows, zscore_normalise
from labeller      import (
    get_fall_label,
    get_fall_type,
    get_pre_activity,
    compute_post_state,
)

# Number of feature columns = window_size * n_channels = 200 * 6 = 1200
WINDOW_SIZE = 200
N_CHANNELS  = 6
N_FEATURES  = WINDOW_SIZE * N_CHANNELS  # 1200


def build_dataset(sisfall_root: str, output_csv: str) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline and save a flat feature CSV.

    Pipeline per clip:
      1.  load_clip()          — raw float32 array (200Hz)
      2.  resample_clip()      — downsample 200Hz → 100Hz
      3.  butterworth_filter() — 20Hz low-pass, remove noise
      4.  slice_windows()      — 2-second windows, 50% overlap
      5.  zscore_normalise()   — per-channel standardisation
      6.  label extraction     — fall_label, fall_type, pre_activity, post_state
      7.  flatten window       — (200,6) → 1200 float32 values, f_0 … f_1199

    Output CSV columns:
      f_0, f_1, ..., f_1199  — flattened window features
      fall_label             — 0 or 1
      fall_type              — 'none' | 'slip' | 'trip' | 'faint'
      pre_activity           — 'walking' | 'standing' | 'bending' | 'sitting'
      post_state             — 'moving' | 'stunned' | 'unconscious'
      source_file            — original .txt filename (for traceability)

    Parameters
    ----------
    sisfall_root : str
        Path to the SisFall root directory containing SA/SE subject subfolders.
    output_csv : str
        Destination path for the output CSV file.

    Returns
    -------
    pd.DataFrame
        The same data that was written to output_csv.
    """
    print(f"\n{'='*60}")
    print(" DATASET BUILDER — starting full pipeline")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # Step 1 — Index clip files (metadata only, no data loaded yet)       #
    # scan_clips() is used instead of load_all_clips() so that we never   #
    # hold more than one clip in RAM at a time (~2 GB saved).             #
    # ------------------------------------------------------------------ #
    clip_meta = scan_clips(sisfall_root)

    if not clip_meta:
        print("[ERROR] No clips found. Check that SisFall data exists at:")
        print(f"        {Path(sisfall_root).resolve()}")
        return pd.DataFrame()

    total_clips = len(clip_meta)
    print(f"\n[BUILD] Processing {total_clips} clips through the pipeline...\n")

    skipped_clips  = 0
    total_windows  = 0
    fall_windows   = 0
    adl_windows    = 0

    # Column header for the CSV — written once before the loop
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_cols = [f"f_{i}" for i in range(N_FEATURES)]
    all_cols     = feature_cols + ["fall_label", "fall_type", "pre_activity", "post_state", "source_file"]

    # Write header row immediately so the file exists even if we crash mid-run
    with open(str(output_path), "w") as f:
        f.write(",".join(all_cols) + "\n")

    # Buffer rows and flush to CSV every FLUSH_EVERY clips to keep RAM low.
    # Each clip produces ~16 windows × 1200 floats × 4 bytes ≈ 75 KB — a
    # buffer of 200 clips (~15 MB) gives fast I/O without hoarding memory.
    FLUSH_EVERY = 200
    row_buffer: list[dict] = []

    def _flush(buf: list[dict]) -> None:
        """Append buffered rows to the CSV and clear the buffer."""
        if not buf:
            return
        chunk = pd.DataFrame(buf)
        chunk.to_csv(str(output_path), index=False, mode="a", header=False)
        buf.clear()

    for clip_idx, clip_dict in enumerate(clip_meta, start=1):
        # Progress report every 100 clips so long runs stay visible
        if clip_idx % 100 == 0 or clip_idx == total_clips:
            print(f"  Processing clip {clip_idx}/{total_clips}  "
                  f"(windows so far: {total_windows:,})")

        activity_code = clip_dict["activity_code"]
        is_fall       = clip_dict["is_fall"]
        source_file   = Path(clip_dict["path"]).name

        # ---- Load this clip's data — discarded after this iteration ----
        raw_data = load_clip(clip_dict["path"])
        if raw_data is None:
            skipped_clips += 1
            continue

        # -------------------------------------------------------------- #
        # Step 2 — Resample 200Hz → 100Hz                                 #
        # -------------------------------------------------------------- #
        try:
            resampled = resample_clip(raw_data, orig_hz=200, target_hz=100)
        except Exception as exc:
            print(f"  [SKIP] Resample failed for {source_file}: {exc}")
            skipped_clips += 1
            continue

        if not validate_resample(raw_data, resampled):
            print(f"  [SKIP] Resample validation failed for {source_file}")
            skipped_clips += 1
            continue

        del raw_data  # free immediately — no longer needed

        # -------------------------------------------------------------- #
        # Step 3 — Low-pass filter (Butterworth 20Hz)                     #
        # -------------------------------------------------------------- #
        try:
            filtered = butterworth_filter(resampled, cutoff=20, fs=100, order=4)
        except Exception as exc:
            print(f"  [SKIP] Filter failed for {source_file}: {exc}")
            skipped_clips += 1
            continue

        del resampled

        # -------------------------------------------------------------- #
        # Step 4 — Sliding window segmentation                            #
        # -------------------------------------------------------------- #
        windows = slice_windows(filtered, window_size=WINDOW_SIZE, step_size=100)
        del filtered

        if not windows:
            print(f"  [SKIP] No windows extracted from {source_file}")
            skipped_clips += 1
            continue

        # -------------------------------------------------------------- #
        # Step 5–7 — Normalise, label, and flatten each window            #
        # -------------------------------------------------------------- #
        fall_label   = get_fall_label(activity_code)
        fall_type    = get_fall_type(activity_code)
        pre_activity = get_pre_activity(activity_code)

        for window in windows:
            norm_window = zscore_normalise(window)
            post_state  = compute_post_state(norm_window, is_fall=is_fall)
            flat        = norm_window.flatten().astype(np.float32)

            row = {f"f_{i}": flat[i] for i in range(N_FEATURES)}
            row["fall_label"]   = fall_label
            row["fall_type"]    = fall_type
            row["pre_activity"] = pre_activity
            row["post_state"]   = post_state
            row["source_file"]  = source_file
            row_buffer.append(row)

            total_windows += 1
            if is_fall:
                fall_windows += 1
            else:
                adl_windows += 1

        # Flush buffer every FLUSH_EVERY clips to keep memory bounded
        if clip_idx % FLUSH_EVERY == 0:
            _flush(row_buffer)

    # Final flush for any remaining rows
    _flush(row_buffer)

    # ------------------------------------------------------------------ #
    # Summary                                                             #
    # ------------------------------------------------------------------ #
    if total_windows == 0:
        print("[ERROR] No windows were generated. The output CSV is empty.")
        return pd.DataFrame()

    balance_ratio = fall_windows / adl_windows if adl_windows > 0 else float("inf")

    print(f"\n{'='*60}")
    print(" DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Total windows  : {total_windows:>8,}")
    print(f"  Fall windows   : {fall_windows:>8,}  ({100*fall_windows/total_windows:.1f}%)")
    print(f"  ADL windows    : {adl_windows:>8,}  ({100*adl_windows/total_windows:.1f}%)")
    print(f"  Balance ratio  : {balance_ratio:.3f}  (falls / ADL)")
    print(f"  Clips skipped  : {skipped_clips}")
    print(f"  Features/row   : {N_FEATURES}")
    print(f"\n[SAVED] Dataset written to: {output_path.resolve()}")

    # Return a lightweight summary DataFrame (labels only, no features)
    # — avoids re-loading 800 MB just for the return value
    df_summary = pd.read_csv(str(output_path), usecols=["fall_label","fall_type","pre_activity","post_state","source_file"])
    return df_summary


# ---------------------------------------------------------------------------
# Standalone entry point — run:  python src/dataset_builder.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Default paths resolved relative to this script's location (ml/src/ → ml/data/)
    _base        = Path(__file__).parent.parent / "data"
    sisfall_root = str(_base / "SisFall_dataset")
    output_csv   = str(_base / "dataset.csv")

    # Allow override via command-line args
    if len(sys.argv) >= 3:
        sisfall_root = sys.argv[1]
        output_csv   = sys.argv[2]
    elif len(sys.argv) == 2:
        sisfall_root = sys.argv[1]

    df = build_dataset(sisfall_root, output_csv)

    if not df.empty:
        print(f"\nDataFrame shape: {df.shape}")
        print("First row labels:")
        print(df[["fall_label", "fall_type", "pre_activity", "post_state", "source_file"]].head(3))
