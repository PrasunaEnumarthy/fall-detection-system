"""
test_adl_fp.py - Find real ADL windows that the model misclassifies as falls,
then run them through the adaptive confirmation window to see how many get suppressed.

This tests the realistic false-positive case: genuine non-fall activity that
briefly triggered the detector. Uses real data from dataset_test.csv only.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

time.sleep = lambda s: None   # skip waits during this analysis

sys.path.insert(0, str(Path(__file__).parent))
import infer

ROOT    = Path(__file__).parent.parent
SEP     = "-" * 64
N_SHOW  = 10   # how many borderline windows to show in detail


def tail_stillness(window: np.ndarray) -> float:
    acc_mag = np.sqrt(np.sum(window[:, :3] ** 2, axis=1))
    return 1.0 / (1.0 + float(np.var(acc_mag[150:])))


def main() -> None:
    csv_path = ROOT / "data" / "dataset_test.csv"
    if not csv_path.exists():
        csv_path = ROOT / "data" / "dataset.csv"

    print(f"\nLoading {csv_path.name} (first 8 000 rows)...")
    df      = pd.read_csv(str(csv_path), nrows=8000)
    adl_df  = df[df["fall_label"] == 0].reset_index(drop=True)
    print(f"ADL windows in sample : {len(adl_df)}")
    print("Scanning for model false positives (fall_prob >= FALL_THRESHOLD)...\n")

    false_positives = []
    for i in range(len(adl_df)):
        flat   = adl_df.iloc[i][["f_" + str(j) for j in range(1200)]].values.astype("float32")
        window = flat.reshape(200, 6)
        result = infer.run_inference(window)
        if result is not None:
            false_positives.append({
                "idx":    i,
                "window": window,
                "conf":   result["confidence"],
                "tail_s": tail_stillness(window),
            })

    total_fp = len(false_positives)
    print(f"ADL windows that fired the model : {total_fp} / {len(adl_df)}")
    if total_fp == 0:
        print("No false positives found in this sample.")
        return

    false_positives.sort(key=lambda r: r["conf"])
    confs = [r["conf"] for r in false_positives]
    print(f"Confidence range : {confs[0]:.4f} -- {confs[-1]:.4f}")
    print(f"(FALL_THRESHOLD  = {infer.FALL_THRESHOLD:.2f})\n")

    # Run the N_SHOW most borderline windows (lowest confidence, closest to threshold)
    # through the full confirmation pipeline.
    borderline = false_positives[:N_SHOW]

    print(SEP)
    print(f"Confirmation window results -- {N_SHOW} most borderline ADL false positives")
    print(f"(sorted by confidence ascending; closest to threshold first)")
    print(SEP)
    print(f"  {'#':>3}  {'conf':>6}  {'tail_s':>8}  {'decision':<12}  note")
    print(f"  {'-'*3}  {'-'*6}  {'-'*8}  {'-'*12}  ----")

    n_suppressed = 0
    for rank, row in enumerate(borderline):
        confirmed, ms = infer._adaptive_confirm(row["window"], row["conf"])
        ts = row["tail_s"]

        if confirmed:
            decision = "CONFIRMED"
            # Explain why it wasn't caught
            gate_note = "(primary+secondary both passed -- simulation reuses same window)"
        else:
            decision = "SUPPRESSED"
            n_suppressed += 1
            if ts < infer.CONFIRM_TAIL_STILLNESS_MIN:
                gate_note = f"(secondary gate: tail_s={ts:.4f} < {infer.CONFIRM_TAIL_STILLNESS_MIN})"
            else:
                gate_note = "(primary gate: model no longer fired)"

        print(f"  {rank+1:>3}  {row['conf']:.4f}  {ts:>8.4f}  {decision:<12}  {gate_note}")

    print()
    print(f"Suppressed : {n_suppressed} / {len(borderline)}")
    print(f"Confirmed  : {len(borderline) - n_suppressed} / {len(borderline)}")

    # Summary across ALL false positives (not just the borderline N_SHOW)
    print()
    print(SEP)
    print("Full false-positive scan -- all ADL windows that fired the model")
    print(SEP)
    all_suppressed = 0
    secondary_gate_catches = 0
    primary_gate_catches   = 0
    for row in false_positives:
        confirmed, _ = infer._adaptive_confirm(row["window"], row["conf"])
        if not confirmed:
            all_suppressed += 1
            if row["tail_s"] < infer.CONFIRM_TAIL_STILLNESS_MIN:
                secondary_gate_catches += 1
            else:
                primary_gate_catches += 1

    print(f"  Total ADL false positives     : {total_fp}")
    print(f"  Suppressed by confirmation    : {all_suppressed}  ({100*all_suppressed/total_fp:.1f}%)")
    print(f"    - via secondary gate (tail) : {secondary_gate_catches}")
    print(f"    - via primary gate (model)  : {primary_gate_catches}")
    print(f"  Still confirmed (not caught)  : {total_fp - all_suppressed}  ({100*(total_fp-all_suppressed)/total_fp:.1f}%)")
    print()
    print("NOTE: In simulation mode the primary gate re-runs the model on the")
    print("identical window, so it never rejects what it already accepted.")
    print("Only the secondary (tail-variance) gate can suppress in simulation.")
    print("In production, the primary gate would use a fresh sensor window")
    print("collected during the wait, giving it a genuine chance to reject")
    print("continued non-fall activity.")


if __name__ == "__main__":
    main()
