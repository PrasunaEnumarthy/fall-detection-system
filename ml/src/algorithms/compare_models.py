#!/usr/bin/env python3
# compare_models.py
# Load all three results/<algo>_metrics.json files, run any missing ones first,
# then print and save a side-by-side comparison table.
#
# Usage
# ─────
#   # Compare all three (runs any missing models first):
#   python ml/src/algorithms/compare_models.py
#
#   # Force-retrain every model before comparing:
#   python ml/src/algorithms/compare_models.py --force
#
# Output
# ──────
#   Console  — formatted table (aligned columns)
#   results/comparison_table.md  — GitHub-flavoured Markdown table
#   results/comparison_table.csv — comma-separated values

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_ALGO_DIR   = Path(__file__).parent          # ml/src/algorithms/
_SRC_DIR    = _ALGO_DIR.parent               # ml/src/
RESULTS_DIR = _ALGO_DIR / "results"

ALGOS = ["lstm", "gru", "transformer"]
METRICS_FILES = {algo: RESULTS_DIR / f"{algo}_metrics.json" for algo in ALGOS}

TABLE_CSV = RESULTS_DIR / "comparison_table.csv"
TABLE_MD  = RESULTS_DIR / "comparison_table.md"

# Columns shown in the comparison table (in order)
_DISPLAY_COLS = [
    ("Model",           "model"),
    ("Accuracy",        "accuracy"),
    ("Precision",       "precision"),
    ("Recall",          "recall"),
    ("F1-Score",        "f1"),
    ("# Parameters",    "n_parameters"),
    ("Train Time (s)",  "training_time_s"),
    ("TN",              "_cm_tn"),
    ("FP",              "_cm_fp"),
    ("FN",              "_cm_fn"),
    ("TP",              "_cm_tp"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ensure_metrics(algo: str, force: bool) -> None:
    """
    Check whether <algo>_metrics.json exists.  If not (or --force), call
    run_algo.py for that algo as a subprocess so the comparison script never
    silently returns partial data.
    """
    path = METRICS_FILES[algo]
    if path.exists() and not force:
        print(f"[compare] {algo}: metrics found at {path.name}")
        return

    verb = "Force-retraining" if (path.exists() and force) else "Missing metrics — training"
    print(f"[compare] {verb} '{algo}' model ...")
    cmd = [
        sys.executable,
        str(_ALGO_DIR / "run_algo.py"),
        "--algo", algo,
    ]
    if force:
        cmd.append("--force")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"run_algo.py failed for algo='{algo}' (exit code {result.returncode}).\n"
            "Check the output above for details."
        )


def _load_metrics(algo: str) -> dict:
    path = METRICS_FILES[algo]
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics JSON not found at {path} after running the training script. "
            "This should not happen — check run_algo.py output for errors."
        )
    with open(str(path)) as f:
        return json.load(f)


def _flatten_cm(metrics: dict) -> dict:
    """
    Unpack the 2×2 confusion matrix into TN / FP / FN / TP fields.
    Handles both 1×1 (all-same-class) and 2×2 shapes defensively.
    """
    cm = metrics.get("confusion_matrix", [])
    if len(cm) == 2 and len(cm[0]) == 2:
        tn, fp = cm[0]
        fn, tp = cm[1]
    elif len(cm) == 1 and len(cm[0]) == 1:
        # Edge case: only one class present in test set
        tn, fp, fn, tp = cm[0][0], 0, 0, 0
    else:
        tn = fp = fn = tp = "N/A"
    out = dict(metrics)
    out["_cm_tn"] = tn
    out["_cm_fp"] = fp
    out["_cm_fn"] = fn
    out["_cm_tp"] = tp
    return out


def _fmt(val) -> str:
    """Format a value for display."""
    if isinstance(val, float):
        return f"{val:.6f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def _build_rows(all_metrics: dict[str, dict]) -> list[dict]:
    """Build a list of flat row-dicts for the comparison table."""
    rows = []
    for algo in ALGOS:
        m    = _flatten_cm(all_metrics[algo])
        row  = {}
        for header, key in _DISPLAY_COLS:
            row[header] = _fmt(m.get(key, "N/A"))
        rows.append(row)
    return rows


def _print_table(rows: list[dict]) -> None:
    """Print an aligned console table."""
    headers = [h for h, _ in _DISPLAY_COLS]
    # Determine column widths
    widths = {h: len(h) for h in headers}
    for row in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(row.get(h, ""))))

    sep  = "  ".join("-" * widths[h] for h in headers)
    head = "  ".join(h.ljust(widths[h]) for h in headers)
    print("\n" + "=" * len(head))
    print("ABLATION STUDY — MODEL COMPARISON")
    print("=" * len(head))
    print(head)
    print(sep)
    for row in rows:
        print("  ".join(str(row.get(h, "")).ljust(widths[h]) for h in headers))
    print("=" * len(head) + "\n")


def _save_csv(rows: list[dict]) -> None:
    headers = [h for h, _ in _DISPLAY_COLS]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(TABLE_CSV), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[compare] CSV  saved → {TABLE_CSV}")


def _save_md(rows: list[dict]) -> None:
    headers = [h for h, _ in _DISPLAY_COLS]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(TABLE_MD), "w", encoding="utf-8") as f:
        # Header row
        f.write("| " + " | ".join(headers) + " |\n")
        # Separator
        f.write("| " + " | ".join("---" for _ in headers) + " |\n")
        # Data rows
        for row in rows:
            f.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")
    print(f"[compare] MD   saved → {TABLE_MD}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare CNN-LSTM, CNN-GRU, and CNN-Transformer on the SisFall test split.\n"
            "Automatically trains any model whose metrics JSON is missing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train all models even if their metrics JSON files already exist.",
    )
    args = parser.parse_args()

    # ── Ensure every model's metrics exist ────────────────────────────────────
    for algo in ALGOS:
        _ensure_metrics(algo, force=args.force)

    # ── Load all three metrics ─────────────────────────────────────────────────
    all_metrics: dict[str, dict] = {}
    for algo in ALGOS:
        print(f"[compare] Loading {algo} metrics ...")
        all_metrics[algo] = _load_metrics(algo)

    # ── Build & display table ──────────────────────────────────────────────────
    rows = _build_rows(all_metrics)
    _print_table(rows)
    _save_csv(rows)
    _save_md(rows)

    print("[compare] Done.")


if __name__ == "__main__":
    main()
