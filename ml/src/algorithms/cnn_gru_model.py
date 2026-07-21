# cnn_gru_model.py
# CNN-GRU fall-detection model — ablation variant swapping LSTM → GRU.
#
# Architecture
# ─────────────
# Identical CNN feature extractor to FallDetectorCNN_LSTM (model.py):
#   Block 1 : Conv1d(6→64,  k=5, p=2) → BN → ReLU → MaxPool(2)  → (batch, 64,  100)
#   Block 2 : Conv1d(64→128, k=5, p=2) → BN → ReLU → MaxPool(2) → (batch, 128,  50)
#
# Temporal stage (changed):
#   2-layer GRU, input_size=128, hidden_size=128, dropout=0.3, batch_first=True
#   → take the last timestep output → (batch, 128)
#
# WHY same hidden_size/num_layers?
#   Keeping both at 128 / 2 makes the GRU and LSTM parameter counts comparable
#   (GRU has 3 gates vs LSTM's 4, so it will have slightly fewer parameters —
#   that difference is real and is what we want to measure in the ablation).
#
# Output heads (identical to baseline):
#   head_fall         : Linear(128, 1) + Sigmoid
#   head_fall_type    : Linear(128, 3)
#   head_pre_activity : Linear(128, 4)
#
# Data pipeline:
#   Reuses data loading / splitting / augmentation logic from train.py verbatim
#   (copied, not imported, so this file is independently runnable without
#    touching train.py).  Random seed and split fractions are identical.

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import Dataset, DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
# This file lives at  ml/src/algorithms/cnn_gru_model.py
# ml/ root is three levels up.
_ALGO_DIR   = Path(__file__).parent                         # ml/src/algorithms/
_SRC_DIR    = _ALGO_DIR.parent                              # ml/src/
_ML_DIR     = _SRC_DIR.parent                               # ml/
DATASET_CSV = _ML_DIR / "data" / "dataset.csv"
RESULTS_DIR = _ALGO_DIR / "results"
METRICS_PATH = RESULTS_DIR / "gru_metrics.json"

# Add ml/src/ to sys.path so we can import augment.py (read-only, not modified)
sys.path.insert(0, str(_SRC_DIR))
from augment import augment_window  # noqa: E402  (import after sys.path tweak)

# ── Hyper-parameters (mirror train.py exactly) ────────────────────────────────
BATCH_SIZE  = 32
N_EPOCHS    = 30
LR          = 1e-3
LR_PATIENCE = 5
W_FALL_TYPE    = 0.8
W_PRE_ACTIVITY = 0.8
TRAIN_FRAC  = 0.70
# VAL_FRAC = 0.15, TEST_FRAC = 0.15 (remainder after two splits)

# Label encoding maps (identical to train.py)
FALL_TYPE_ENC: dict[str, int] = {
    "none": 0, "slip": 0, "trip": 1, "faint": 2,
}
PRE_ACTIVITY_ENC: dict[str, int] = {
    "walking": 0, "standing": 1, "bending": 2, "sitting": 3,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class FallDetectorCNN_GRU(nn.Module):
    """
    CNN-GRU fall detection model.

    Identical to FallDetectorCNN_LSTM (ml/src/model.py) except the LSTM is
    replaced with a GRU of the same hidden_size (128) and num_layers (2).

    Input  : (batch, 200, 6)
    Output : dict with keys 'fall' (batch,1), 'fall_type' (batch,3),
             'pre_activity' (batch,4)
    """

    def __init__(self) -> None:
        super().__init__()

        # ── CNN Block 1 ──────────────────────────────────────────────────────
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # 200 → 100
        )

        # ── CNN Block 2 ──────────────────────────────────────────────────────
        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # 100 → 50
        )

        # ── GRU (replaces LSTM) ───────────────────────────────────────────────
        # dropout=0.3 is only active between layers (num_layers >= 2), matching
        # the LSTM configuration in model.py exactly.
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        # ── Output heads (identical to baseline) ─────────────────────────────
        self.head_fall = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.head_fall_type    = nn.Linear(128, 3)
        self.head_pre_activity = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # CNN expects (batch, channels, time)
        x = x.permute(0, 2, 1)          # (batch, 6, 200)
        x = self.cnn_block1(x)           # (batch, 64, 100)
        x = self.cnn_block2(x)           # (batch, 128, 50)

        # GRU expects (batch, time, features)
        x = x.permute(0, 2, 1)          # (batch, 50, 128)
        gru_out, _ = self.gru(x)        # (batch, 50, 128)

        # Take the last timestep — encodes the full sequence context
        x = gru_out[:, -1, :]           # (batch, 128)

        return {
            "fall":         self.head_fall(x),           # (batch, 1)
            "fall_type":    self.head_fall_type(x),      # (batch, 3)
            "pre_activity": self.head_pre_activity(x),   # (batch, 4)
        }


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ═══════════════════════════════════════════════════════════════════════════════
# Data pipeline  (mirrors train.py — identical split logic, same random_state=42)
# ═══════════════════════════════════════════════════════════════════════════════

class FallWindowDataset(Dataset):
    """PyTorch Dataset wrapping the flat feature CSV (1200 cols → (200,6) window)."""

    def __init__(self, features: np.ndarray, labels: dict) -> None:
        self.features     = torch.tensor(features, dtype=torch.float32)
        self.fall         = torch.tensor(labels["fall"],         dtype=torch.float32)
        self.fall_type    = torch.tensor(labels["fall_type"],    dtype=torch.long)
        self.pre_activity = torch.tensor(labels["pre_activity"], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        window = self.features[idx].reshape(200, 6)
        return window, {
            "fall":         self.fall[idx],
            "fall_type":    self.fall_type[idx],
            "pre_activity": self.pre_activity[idx],
        }


def _load_data() -> tuple[np.ndarray, dict]:
    """Load and integer-encode the dataset CSV."""
    print(f"[GRU] Loading dataset from {DATASET_CSV} ...")
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset CSV not found at {DATASET_CSV}\n"
            "Run:  python src/dataset_builder.py  first."
        )
    df = pd.read_csv(str(DATASET_CSV))
    print(f"[GRU] Loaded {len(df):,} rows")

    feature_cols = [f"f_{i}" for i in range(1200)]
    X = df[feature_cols].values.astype(np.float32)

    labels = {
        "fall":         df["fall_label"].values.astype(np.int32),
        "fall_type":    df["fall_type"].map(FALL_TYPE_ENC).fillna(0).astype(np.int64).values,
        "pre_activity": df["pre_activity"].map(PRE_ACTIVITY_ENC).fillna(0).astype(np.int64).values,
    }
    return X, labels


def _split_data(X: np.ndarray, labels: dict):
    """
    Stratified 70/15/15 train/val/test split — random_state=42, identical to train.py.
    Returns (X_train, X_val, X_test, l_train, l_val, l_test).
    """
    n       = len(X)
    indices = np.arange(n)
    strat   = labels["fall"]

    idx_train, idx_tmp = train_test_split(
        indices, test_size=(1 - TRAIN_FRAC), stratify=strat, random_state=42
    )
    strat_tmp = strat[idx_tmp]
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.5, stratify=strat_tmp, random_state=42
    )

    def _sub(idx):
        return {k: v[idx] for k, v in labels.items()}

    print(
        f"[GRU] Split — train: {len(idx_train):,}  "
        f"val: {len(idx_val):,}  test: {len(idx_test):,}"
    )
    return (
        X[idx_train], X[idx_val], X[idx_test],
        _sub(idx_train), _sub(idx_val), _sub(idx_test),
    )


def _augment_training_set(X_train: np.ndarray, labels_train: dict):
    """Apply 6-way augmentation to fall windows (mirrors train.py)."""
    fall_idx = np.where(labels_train["fall"] == 1)[0]
    aug_X: list[np.ndarray] = []
    aug_labels: dict = {k: [] for k in labels_train}

    for idx in fall_idx:
        window = X_train[idx].reshape(200, 6)
        for aug_win in augment_window(window):
            aug_X.append(aug_win.flatten())
            for k in labels_train:
                aug_labels[k].append(labels_train[k][idx])

    aug_X_np      = np.array(aug_X, dtype=np.float32)
    aug_labels_np = {k: np.array(v, dtype=labels_train[k].dtype) for k, v in aug_labels.items()}
    X_aug      = np.vstack([X_train, aug_X_np])
    labels_aug = {k: np.concatenate([labels_train[k], aug_labels_np[k]]) for k in labels_train}

    n_fall = len(fall_idx)
    n_adl  = int((labels_train["fall"] == 0).sum())
    print(f"[GRU][AUG] Before: {len(X_train):,}  (fall: {n_fall:,}  ADL: {n_adl:,})")
    print(f"[GRU][AUG] After : {len(X_aug):,}  (fall: {n_fall * 7:,}  ADL: {n_adl:,})")
    return X_aug, labels_aug


# ═══════════════════════════════════════════════════════════════════════════════
# Loss / training / evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_loss(outputs, targets, bce, ce) -> torch.Tensor:
    """Multi-task loss: fall (BCE) + fall_type (CE, fall-only) + pre_activity (CE)."""
    fall_pred   = outputs["fall"].squeeze(1)
    fall_target = targets["fall"].float()
    loss_fall   = bce(fall_pred, fall_target)

    loss_pre = ce(outputs["pre_activity"], targets["pre_activity"])

    fall_mask = targets["fall"].bool()
    if fall_mask.sum() > 0:
        loss_ft = ce(outputs["fall_type"][fall_mask], targets["fall_type"][fall_mask])
    else:
        loss_ft = torch.tensor(0.0, device=fall_pred.device)

    return loss_fall + W_FALL_TYPE * loss_ft + W_PRE_ACTIVITY * loss_pre


def _train_epoch(model, loader, optimizer, bce, ce, device) -> float:
    model.train()
    total = 0.0
    for windows, targets in loader:
        windows = windows.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        optimizer.zero_grad()
        loss = _compute_loss(model(windows), targets, bce, ce)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def _eval_epoch(model, loader, bce, ce, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0; correct = 0; n = 0
    for windows, targets in loader:
        windows = windows.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        out = model(windows)
        total_loss += _compute_loss(out, targets, bce, ce).item()
        preds   = (out["fall"].squeeze(1) >= 0.5).long()
        correct += (preds == targets["fall"]).sum().item()
        n       += len(windows)
    return total_loss / len(loader), correct / n if n > 0 else 0.0


@torch.no_grad()
def _evaluate_test(model, X_test: np.ndarray, labels_test: dict, device) -> dict:
    """Run model on the held-out test set and return metric dict."""
    model.eval()
    ds     = FallWindowDataset(X_test, labels_test)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)

    all_preds, all_targets = [], []
    for windows, targets in loader:
        windows = windows.to(device)
        out     = model(windows)
        preds   = (out["fall"].squeeze(1) >= 0.5).long().cpu().numpy()
        tgts    = targets["fall"].long().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(tgts.tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    cm     = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall":    round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "confusion_matrix": cm,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate() -> dict:
    """
    Train the CNN-GRU model on SisFall (same split as train.py) and evaluate
    on the held-out test set.

    Returns
    -------
    dict
        Metrics dict saved to results/gru_metrics.json.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[GRU] Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    X, labels = _load_data()
    X_train, X_val, X_test, l_train, l_val, l_test = _split_data(X, labels)

    print("[GRU] Augmenting fall windows ...")
    X_train, l_train = _augment_training_set(X_train, l_train)

    train_ds = FallWindowDataset(X_train, l_train)
    val_ds   = FallWindowDataset(X_val,   l_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model     = FallDetectorCNN_GRU().to(device)
    bce       = nn.BCELoss()
    ce        = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=LR_PATIENCE, factor=0.5
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    print(f"\n[GRU] Training for {N_EPOCHS} epochs ...\n")
    t_start = time.time()

    for epoch in range(1, N_EPOCHS + 1):
        tr_loss          = _train_epoch(model, train_loader, optimizer, bce, ce, device)
        val_loss, val_acc = _eval_epoch(model, val_loader, bce, ce, device)
        scheduler.step(val_loss)
        print(
            f"  Epoch {epoch:>2}/{N_EPOCHS}  "
            f"train_loss={tr_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    [BEST] val_loss={val_loss:.4f}")

    training_time = time.time() - t_start
    print(f"\n[GRU] Training done in {training_time:.1f}s")

    # ── Restore best checkpoint and evaluate ──────────────────────────────────
    model.load_state_dict(best_state)
    test_metrics = _evaluate_test(model, X_test, l_test, device)

    n_params = count_parameters(model)
    metrics = {
        "model":          "CNN-GRU",
        "accuracy":       test_metrics["accuracy"],
        "precision":      test_metrics["precision"],
        "recall":         test_metrics["recall"],
        "f1":             test_metrics["f1"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "n_parameters":   n_params,
        "training_time_s": round(training_time, 2),
        "n_epochs":       N_EPOCHS,
        "best_val_loss":  round(best_val_loss, 6),
    }

    # ── Save metrics ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(str(METRICS_PATH), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[GRU] Metrics saved to {METRICS_PATH}")
    print(f"[GRU] Test accuracy={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  params={n_params:,}")

    return metrics


# ─── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_evaluate()
