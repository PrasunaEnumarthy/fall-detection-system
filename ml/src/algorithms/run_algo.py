#!/usr/bin/env python3
# run_algo.py
# CLI entrypoint for the ablation study.
#
# Usage
# ─────
#   python ml/src/algorithms/run_algo.py --algo lstm
#   python ml/src/algorithms/run_algo.py --algo gru
#   python ml/src/algorithms/run_algo.py --algo transformer
#
# What each choice does
# ─────────────────────
#   lstm        — imports the EXISTING train.py (read-only) and calls its
#                 train() function, then evaluates on the same held-out test
#                 split to produce a metrics JSON that is comparable with the
#                 GRU / Transformer runs.
#
#   gru         — trains cnn_gru_model.py and saves results/gru_metrics.json
#
#   transformer — trains cnn_transformer_model.py and saves
#                 results/transformer_metrics.json
#
# All three use the same dataset.csv, the same 70/15/15 stratified split
# (random_state=42), and the same data augmentation as the original train.py.
# The LSTM results are re-derived from scratch so no stale checkpoint is used.
#
# Output
# ──────
#   ml/src/algorithms/results/<algo>_metrics.json

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ── resolve paths ──────────────────────────────────────────────────────────────
_ALGO_DIR = Path(__file__).parent          # ml/src/algorithms/
_SRC_DIR  = _ALGO_DIR.parent              # ml/src/
_ML_DIR   = _SRC_DIR.parent               # ml/

RESULTS_DIR = _ALGO_DIR / "results"

# ml/src/ must be on the path so that train.py's relative imports (model, augment) work
sys.path.insert(0, str(_SRC_DIR))


# ═══════════════════════════════════════════════════════════════════════════════
# LSTM runner — wraps the existing train.py without modifying it
# ═══════════════════════════════════════════════════════════════════════════════

def _run_lstm() -> dict:
    """
    Run the existing CNN-LSTM model (train.py) and produce a metrics JSON
    compatible with the comparison table.

    Strategy
    ────────
    We import train.py's internal helpers to replicate the exact training
    pipeline, then do an additional test-set evaluation pass.  train.py is
    never modified; we only read its exported symbols.
    """
    import csv as _csv

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix,
    )
    from torch.utils.data import DataLoader

    # Import read-only symbols from train.py
    import train as train_module
    from model import FallDetectorCNN_LSTM

    METRICS_PATH = RESULTS_DIR / "lstm_metrics.json"

    print("[LSTM] Starting CNN-LSTM training (via train.py pipeline) ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LSTM] Device: {device}")

    # ── Data (identical to train.py) ──────────────────────────────────────────
    X, labels = train_module.load_data()
    X_train, X_val, X_test, l_train, l_val, l_test, _idx_test = (
        train_module.split_data(X, labels)
    )

    print("[LSTM] Augmenting fall windows ...")
    X_train, l_train = train_module.augment_training_set(X_train, l_train)

    train_ds = train_module.FallWindowDataset(X_train, l_train)
    val_ds   = train_module.FallWindowDataset(X_val,   l_val)
    train_loader = DataLoader(
        train_ds, batch_size=train_module.BATCH_SIZE, shuffle=True,  num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_module.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model     = FallDetectorCNN_LSTM().to(device)
    bce       = nn.BCELoss()
    ce        = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_module.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        patience=train_module.LR_PATIENCE, factor=0.5,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    best_state: dict | None = None
    print(f"\n[LSTM] Training for {train_module.N_EPOCHS} epochs ...\n")
    t_start = time.time()

    for epoch in range(1, train_module.N_EPOCHS + 1):
        tr_loss = train_module.train_epoch(model, train_loader, optimizer, bce, ce, device)
        val_loss, val_acc = train_module.eval_epoch(model, val_loader, bce, ce, device)
        scheduler.step(val_loss)
        print(
            f"  Epoch {epoch:>2}/{train_module.N_EPOCHS}  "
            f"train_loss={tr_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    [BEST] val_loss={val_loss:.4f}")

    training_time = time.time() - t_start
    print(f"\n[LSTM] Training done in {training_time:.1f}s")

    # ── Restore best checkpoint and evaluate on test set ──────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    test_ds     = train_module.FallWindowDataset(X_test, l_test)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for windows, targets in test_loader:
            windows = windows.to(device)
            out     = model(windows)
            preds   = (out["fall"].squeeze(1) >= 0.5).long().cpu().numpy()
            tgts    = targets["fall"].long().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(tgts.tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    cm     = confusion_matrix(y_true, y_pred).tolist()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    metrics = {
        "model":            "CNN-LSTM",
        "accuracy":         round(float(accuracy_score(y_true, y_pred)), 6),
        "precision":        round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":           round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1":               round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "confusion_matrix": cm,
        "n_parameters":     n_params,
        "training_time_s":  round(training_time, 2),
        "n_epochs":         train_module.N_EPOCHS,
        "best_val_loss":    round(best_val_loss, 6),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(METRICS_PATH), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[LSTM] Metrics saved to {METRICS_PATH}")
    print(
        f"[LSTM] Test accuracy={metrics['accuracy']:.4f}  "
        f"F1={metrics['f1']:.4f}  params={n_params:,}"
    )
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# GRU runner
# ═══════════════════════════════════════════════════════════════════════════════

def _run_gru() -> dict:
    """Delegate to cnn_gru_model.train_and_evaluate()."""
    # Local import so this file can be imported without the module being present
    # (only the selected algo's module is required at runtime)
    sys.path.insert(0, str(_ALGO_DIR))
    from cnn_gru_model import train_and_evaluate
    return train_and_evaluate()


# ═══════════════════════════════════════════════════════════════════════════════
# Transformer runner
# ═══════════════════════════════════════════════════════════════════════════════

def _run_transformer() -> dict:
    """Delegate to cnn_transformer_model.train_and_evaluate()."""
    sys.path.insert(0, str(_ALGO_DIR))
    from cnn_transformer_model import train_and_evaluate
    return train_and_evaluate()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

_RUNNERS = {
    "lstm":        _run_lstm,
    "gru":         _run_gru,
    "transformer": _run_transformer,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single fall-detection model variant and save its metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  python ml/src/algorithms/run_algo.py --algo lstm
  python ml/src/algorithms/run_algo.py --algo gru
  python ml/src/algorithms/run_algo.py --algo transformer

Output
  ml/src/algorithms/results/<algo>_metrics.json
        """,
    )
    parser.add_argument(
        "--algo",
        choices=list(_RUNNERS.keys()),
        required=True,
        help="Model variant to train and evaluate.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-train even if a metrics JSON already exists for this algo.",
    )
    args = parser.parse_args()

    metrics_path = RESULTS_DIR / f"{args.algo}_metrics.json"

    if metrics_path.exists() and not args.force:
        print(
            f"[run_algo] Metrics already exist at {metrics_path}\n"
            "           Pass --force to re-train."
        )
        with open(str(metrics_path)) as f:
            metrics = json.load(f)
        print(f"[run_algo] Loaded existing metrics: {metrics}")
        return

    print(f"[run_algo] Running algo='{args.algo}' ...")
    runner  = _RUNNERS[args.algo]
    metrics = runner()
    print(f"\n[run_algo] Done. Results → {metrics_path}")


if __name__ == "__main__":
    main()
