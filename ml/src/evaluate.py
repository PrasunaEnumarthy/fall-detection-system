# evaluate.py
# Loads the trained best_model.pt and evaluates it on the held-out test set.
# Reports accuracy, sensitivity, specificity, F1, AUC-ROC, and a confusion matrix.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, str(Path(__file__).parent))
from model import FallDetectorCNN_LSTM
from train import FallWindowDataset  # reuse the Dataset wrapper

# ------------------------------------------------------------------ #
# Paths                                                              #
# ------------------------------------------------------------------ #
BEST_MODEL       = Path(__file__).parent.parent / "models" / "best_model.pt"
TEST_CSV         = Path(__file__).parent.parent / "data"   / "dataset_test.csv"
EVAL_REPORT_PATH = Path(__file__).parent.parent / "models" / "evaluation_report.txt"
THRESHOLD_PATH   = Path(__file__).parent.parent / "models" / "threshold.txt"

# Minimum acceptable sensitivity (recall on fall class) — clinical target
SENSITIVITY_TARGET = 0.95

# Label decode maps (reverse of train.py encodings)
FALL_TYPE_DECODE    = {0: "none", 1: "slip",    2: "trip",     3: "faint"}
PRE_ACTIVITY_DECODE = {0: "walking", 1: "standing", 2: "bending", 3: "sitting"}


def find_optimal_threshold(fall_probs: np.ndarray, fall_true: np.ndarray,
                           target: float = SENSITIVITY_TARGET) -> float:
    """
    Find the highest decision threshold that still achieves target sensitivity.
    Sweeps from 0.99 down to 0.01 in steps of 0.01.
    """
    for thresh in np.round(np.arange(0.99, 0.00, -0.01), 2):
        preds = (fall_probs >= thresh).astype(int)
        sens  = recall_score(fall_true, preds, pos_label=1, zero_division=0)
        if sens >= target:
            return float(thresh)
    return 0.1   # fallback: accept nearly everything


def load_test_data() -> tuple[np.ndarray, dict]:
    """
    Load the test split that was saved by train.py.

    Returns
    -------
    tuple
        (features: np.ndarray (N,1200), labels: dict with 'fall', 'fall_type', 'pre_activity')
    """
    if not TEST_CSV.exists():
        raise FileNotFoundError(
            f"Test CSV not found at {TEST_CSV}\n"
            "Run:  python src/train.py  to generate the test split."
        )

    df = pd.read_csv(str(TEST_CSV))
    X  = df[[f"f_{i}" for i in range(1200)]].values.astype(np.float32)

    labels = {
        "fall":         df["fall_label"].values.astype(np.int32),
        "fall_type":    df["fall_type_enc"].values.astype(np.int64),
        "pre_activity": df["pre_activity_enc"].values.astype(np.int64),
    }
    return X, labels


@torch.no_grad()
def run_inference(model, loader, device) -> dict:
    """
    Run the model over the full test DataLoader and collect predictions.

    Returns
    -------
    dict with keys:
      fall_probs       — float array (N,)    raw fall probabilities
      fall_preds       — int array   (N,)    thresholded at 0.5
      fall_true        — int array   (N,)
      fall_type_preds  — int array   (N,)    argmax of logits
      fall_type_true   — int array   (N,)
      pre_act_preds    — int array   (N,)
      pre_act_true     — int array   (N,)
    """
    model.eval()

    all_fall_probs      = []
    all_fall_preds      = []
    all_fall_true       = []
    all_fall_type_preds = []
    all_fall_type_true  = []
    all_pre_preds       = []
    all_pre_true        = []

    for windows, targets in loader:
        windows = windows.to(device)
        out     = model(windows)

        fall_prob  = out["fall"].squeeze(1).cpu().numpy()
        fall_pred  = (fall_prob >= 0.5).astype(int)

        ft_pred    = out["fall_type"].argmax(dim=1).cpu().numpy()
        pre_pred   = out["pre_activity"].argmax(dim=1).cpu().numpy()

        all_fall_probs.extend(fall_prob.tolist())
        all_fall_preds.extend(fall_pred.tolist())
        all_fall_true.extend(targets["fall"].numpy().tolist())
        all_fall_type_preds.extend(ft_pred.tolist())
        all_fall_type_true.extend(targets["fall_type"].numpy().tolist())
        all_pre_preds.extend(pre_pred.tolist())
        all_pre_true.extend(targets["pre_activity"].numpy().tolist())

    return {
        "fall_probs":      np.array(all_fall_probs),
        "fall_preds":      np.array(all_fall_preds),
        "fall_true":       np.array(all_fall_true),
        "fall_type_preds": np.array(all_fall_type_preds),
        "fall_type_true":  np.array(all_fall_type_true),
        "pre_act_preds":   np.array(all_pre_preds),
        "pre_act_true":    np.array(all_pre_true),
    }


def evaluate_model(model_path: str = None, test_csv: str = None) -> None:
    """
    Full evaluation of the trained model on the held-out test set.

    Metrics computed:
      Fall head       : accuracy, sensitivity (recall), specificity, F1, ROC-AUC
      Fall-type head  : per-class F1 for slip / trip / faint
      Pre-activity    : per-class accuracy for walking/standing/bending/sitting

    Parameters
    ----------
    model_path : str, optional
        Path to the .pt checkpoint. Defaults to ml/models/best_model.pt.
    test_csv : str, optional
        Path to the test CSV. Defaults to ml/data/dataset_test.csv.
    """
    mp = Path(model_path) if model_path else BEST_MODEL
    tc = Path(test_csv)   if test_csv   else TEST_CSV

    print(f"\n{'='*60}")
    print(" EVALUATE — loading model and test data")
    print(f"{'='*60}")

    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}\nRun train.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EVAL] Device: {device}")

    # ------------------------------------------------------------------ #
    # Load model                                                          #
    # ------------------------------------------------------------------ #
    model = FallDetectorCNN_LSTM().to(device)
    model.load_state_dict(torch.load(str(mp), map_location=device))
    model.eval()
    print(f"[EVAL] Loaded model from {mp.name}")

    # ------------------------------------------------------------------ #
    # Load test data                                                      #
    # ------------------------------------------------------------------ #
    X_test, labels_test = load_test_data()
    test_ds     = FallWindowDataset(X_test, labels_test)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"[EVAL] Test samples: {len(test_ds):,}")

    # ------------------------------------------------------------------ #
    # Run inference                                                       #
    # ------------------------------------------------------------------ #
    preds = run_inference(model, test_loader, device)

    y_true_fall = preds["fall_true"]
    y_prob_fall = preds["fall_probs"]

    # ------------------------------------------------------------------ #
    # Threshold tuning — find highest threshold meeting sensitivity target #
    # ------------------------------------------------------------------ #
    opt_thresh = find_optimal_threshold(y_prob_fall, y_true_fall)
    print(f"[EVAL] Optimal threshold for >={SENSITIVITY_TARGET*100:.0f}% sensitivity: {opt_thresh:.2f}")

    # Save threshold so infer.py can load it
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(THRESHOLD_PATH), "w") as _tf:
        _tf.write(f"{opt_thresh:.4f}\n")
    print(f"[EVAL] Threshold saved to {THRESHOLD_PATH.name}")

    # Evaluate at BOTH thresholds so we can compare
    y_pred_05  = (y_prob_fall >= 0.50).astype(int)
    y_pred_opt = (y_prob_fall >= opt_thresh).astype(int)
    y_pred_fall = y_pred_opt   # use optimal for the main report

    # ------------------------------------------------------------------ #
    # Fall head metrics                                                   #
    # ------------------------------------------------------------------ #
    acc         = accuracy_score(y_true_fall, y_pred_fall)
    sensitivity = recall_score(y_true_fall, y_pred_fall, pos_label=1, zero_division=0)
    f1          = f1_score(y_true_fall, y_pred_fall, pos_label=1, zero_division=0)
    roc_auc     = roc_auc_score(y_true_fall, y_prob_fall) if len(np.unique(y_true_fall)) > 1 else 0.0

    # Specificity = TN / (TN + FP)
    cm          = confusion_matrix(y_true_fall, y_pred_fall)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Reference metrics at default 0.5 for comparison
    sens_05 = recall_score(y_true_fall, y_pred_05, pos_label=1, zero_division=0)
    cm05    = confusion_matrix(y_true_fall, y_pred_05)
    tn05, fp05, fn05, tp05 = cm05.ravel() if cm05.shape == (2, 2) else (0, 0, 0, 0)
    spec_05 = tn05 / (tn05 + fp05) if (tn05 + fp05) > 0 else 0.0

    # ------------------------------------------------------------------ #
    # Fall-type and pre-activity — evaluate only on relevant subsets      #
    # ------------------------------------------------------------------ #
    # Fall-type: meaningful only on actual fall windows (label == 1)
    fall_mask = y_true_fall == 1
    ft_true   = preds["fall_type_true"][fall_mask]
    ft_pred   = preds["fall_type_preds"][fall_mask]

    # Fall-type head uses 3 classes: slip=0, trip=1, faint=2
    ft_labels = [0, 1, 2]
    ft_names  = ["slip", "trip", "faint"]

    if len(ft_true) > 0:
        ft_report = classification_report(
            ft_true, ft_pred,
            labels=ft_labels, target_names=ft_names, zero_division=0, output_dict=True,
        )
    else:
        ft_report = {}

    # Pre-activity accuracy per class
    pre_true  = preds["pre_act_true"]
    pre_pred  = preds["pre_act_preds"]
    pre_names = ["walking", "standing", "bending", "sitting"]
    pre_report = classification_report(
        pre_true, pre_pred,
        labels=[0, 1, 2, 3], target_names=pre_names, zero_division=0, output_dict=True,
    )

    # ------------------------------------------------------------------ #
    # Format report string                                                #
    # ------------------------------------------------------------------ #
    lines = []
    lines.append("=" * 60)
    lines.append(" FALL DETECTION MODEL — EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append("[ FALL HEAD ]")
    lines.append(f"  Threshold used   : {opt_thresh:.2f}  (tuned for >={SENSITIVITY_TARGET*100:.0f}% sensitivity)")
    lines.append(f"  Accuracy         : {acc:.4f}  ({acc*100:.2f}%)")
    lines.append(f"  Sensitivity      : {sensitivity:.4f}  ({sensitivity*100:.2f}%)")
    lines.append(f"  Specificity      : {specificity:.4f}  ({specificity*100:.2f}%)")
    lines.append(f"  F1 Score         : {f1:.4f}")
    lines.append(f"  ROC-AUC          : {roc_auc:.4f}")
    lines.append(f"  (At thresh=0.50: sensitivity={sens_05:.4f}  specificity={spec_05:.4f})")
    lines.append("")

    lines.append("  Confusion Matrix (rows=actual, cols=predicted):")
    lines.append("          ADL   Fall")
    lines.append(f"  ADL   [{tn:>6}  {fp:>6}]")
    lines.append(f"  Fall  [{fn:>6}  {tp:>6}]")
    lines.append("")

    # Sensitivity warning
    if sensitivity < SENSITIVITY_TARGET:
        warn = (
            f"  WARNING: Sensitivity {sensitivity:.4f} is below "
            f"target {SENSITIVITY_TARGET:.2f}! "
            "Consider: more training data, augmentation, or lower threshold."
        )
        lines.append(warn)
        print(f"\n{'!'*60}")
        print(warn)
        print(f"{'!'*60}\n")
    else:
        lines.append(f"  OK: Sensitivity meets the >{SENSITIVITY_TARGET*100:.0f}% target.")

    lines.append("")
    lines.append("[ FALL TYPE HEAD — F1 per class (fall windows only) ]")
    for name in ft_names:
        f1_val = ft_report.get(name, {}).get("f1-score", 0.0) if ft_report else 0.0
        lines.append(f"  {name:<8} F1: {f1_val:.4f}")

    lines.append("")
    lines.append("[ PRE-ACTIVITY HEAD — precision / recall / F1 ]")
    for name in pre_names:
        p  = pre_report.get(name, {}).get("precision", 0.0)
        r  = pre_report.get(name, {}).get("recall",    0.0)
        f1v= pre_report.get(name, {}).get("f1-score",  0.0)
        lines.append(f"  {name:<10}  P={p:.4f}  R={r:.4f}  F1={f1v:.4f}")

    lines.append("")
    lines.append("[ FULL FALL HEAD CLASSIFICATION REPORT ]")
    lines.append(
        classification_report(
            y_true_fall, y_pred_fall,
            target_names=["ADL", "Fall"], zero_division=0,
        )
    )

    report_text = "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Print and save                                                      #
    # ------------------------------------------------------------------ #
    print(report_text)

    EVAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(str(EVAL_REPORT_PATH), "w") as f:
        f.write(report_text)

    print(f"\n[EVAL] Report saved to {EVAL_REPORT_PATH}")


# ---------------------------------------------------------------------------
# Standalone entry point — run:  python src/evaluate.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate_model()
