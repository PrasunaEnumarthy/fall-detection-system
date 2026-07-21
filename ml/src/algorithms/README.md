# Ablation Study — CNN-LSTM vs CNN-GRU vs CNN-Transformer

This directory contains three fall-detection model variants that share the same
CNN feature extractor and SisFall data pipeline, differing **only** in their
temporal-modelling stage. The purpose is an ablation study to measure the
contribution of that stage in isolation.

All commands below are run **from the `ml/` directory**.

---

## Quick-start: run all three and compare in one command

```bash
# From the ml/ directory:
python src/algorithms/compare_models.py
```

This will:
1. Check which model metrics JSONs are missing in `src/algorithms/results/`.
2. Train any missing models automatically.
3. Print a comparison table to the console.
4. Save `src/algorithms/results/comparison_table.csv` and
   `src/algorithms/results/comparison_table.md`.

To force-retrain **all** models even if results already exist:

```bash
python src/algorithms/compare_models.py --force
```

---

## Running each model individually

### 1 — CNN-LSTM (existing baseline, run through the new interface)

```bash
python src/algorithms/run_algo.py --algo lstm
```

**What it does:** Imports the unmodified `train.py` pipeline, trains
`FallDetectorCNN_LSTM`, evaluates on the held-out 15 % test split, and saves
metrics to `src/algorithms/results/lstm_metrics.json`.

**Output:** `src/algorithms/results/lstm_metrics.json`

---

### 2 — CNN-GRU

```bash
python src/algorithms/run_algo.py --algo gru
```

**What it does:** Trains `cnn_gru_model.py` on the same data split and saves
metrics to `src/algorithms/results/gru_metrics.json`.

**Output:** `src/algorithms/results/gru_metrics.json`

---

### 3 — CNN-Transformer

```bash
python src/algorithms/run_algo.py --algo transformer
```

**What it does:** Trains `cnn_transformer_model.py` on the same data split and
saves metrics to `src/algorithms/results/transformer_metrics.json`.

**Output:** `src/algorithms/results/transformer_metrics.json`

---

## Force-retraining a single model

Pass `--force` to overwrite an existing metrics file:

```bash
python src/algorithms/run_algo.py --algo gru --force
```

---

## One-line model descriptions (for mentor review)

| Model | File | What changes vs. baseline |
|-------|------|--------------------------|
| **CNN-LSTM** | `ml/src/model.py` (existing, unmodified) | Baseline — 2-layer LSTM (hidden=128) as the temporal stage |
| **CNN-GRU** | `cnn_gru_model.py` | LSTM replaced with a GRU of identical depth and hidden size; ~25 % fewer temporal parameters because GRU has 3 gates vs LSTM's 4 |
| **CNN-Transformer** | `cnn_transformer_model.py` | LSTM replaced with sinusoidal positional encoding + 2-layer TransformerEncoder (d_model=128, 4 heads, FFN=256, dropout=0.2) + mean-pooling; no recurrence |

---

## Architecture details

### CNN stage (shared, identical across all three models)

```
Input  (batch, 200, 6)
  → permute  → (batch, 6, 200)
  → Conv1d(6→64,  k=5, p=2) → BN → ReLU → MaxPool(2)  → (batch, 64, 100)
  → Conv1d(64→128, k=5, p=2) → BN → ReLU → MaxPool(2) → (batch, 128, 50)
  → permute  → (batch, 50, 128)
```

### Temporal stage (swapped per variant)

| Variant | Temporal stage | Pooling |
|---------|---------------|---------|
| LSTM | `nn.LSTM(128, 128, num_layers=2, dropout=0.3)` | Last timestep |
| GRU  | `nn.GRU(128, 128, num_layers=2, dropout=0.3)` | Last timestep |
| Transformer | Sinusoidal PE → `TransformerEncoder(2 layers, nhead=4, FFN=256, dropout=0.2)` | **Mean-pool** over 50 steps |

**Why mean-pooling for the Transformer?**
Mean-pooling is parameter-free and aggregates context from every timestep,
which is preferable for fall detection where the discriminative signal (the
impact) may occur anywhere in the 2-second window. It performs comparably to
CLS-token approaches on fixed-length sequences of this size.

**Why d_model=128?**
Equal to the CNN output channels — no projection layer is needed, keeping the
CNN–Transformer boundary clean.

### Output heads (shared, identical across all three models)

```
head_fall         : Linear(128, 1)  + Sigmoid  → fall probability ∈ [0, 1]
head_fall_type    : Linear(128, 3)             → slip / trip / faint  (logits)
head_pre_activity : Linear(128, 4)             → walking / standing / bending / sitting (logits)
```

---

## Data pipeline (identical for all three)

- **Source:** `ml/data/dataset.csv` (built by `ml/src/dataset_builder.py`)
- **Preprocessing:** handled by the existing `data_loader.py` (SisFall raw → 6-channel IMU)
- **Windowing:** 200-sample (2 s) windows stored as flat 1200-element feature rows in `dataset.csv`
- **Split:** stratified 70 / 15 / 15 train / val / test (`random_state=42`, same as `train.py`)
- **Augmentation:** 6-way augmentation on fall windows in the training set (using `augment.py`)

---

## Metrics saved per model

Each `results/<algo>_metrics.json` contains:

| Field | Description |
|-------|-------------|
| `accuracy` | Binary fall-detection accuracy on the test split |
| `precision` | Precision (positive class = fall) |
| `recall` | Recall (sensitivity) |
| `f1` | F1-score |
| `confusion_matrix` | 2 × 2 list `[[TN, FP], [FN, TP]]` |
| `n_parameters` | Total trainable parameter count |
| `training_time_s` | Wall-clock training time in seconds |
| `n_epochs` | Number of training epochs |
| `best_val_loss` | Best validation loss achieved |

---

## Expected output locations

| Command | Results written to |
|---------|--------------------|
| `run_algo.py --algo lstm` | `src/algorithms/results/lstm_metrics.json` |
| `run_algo.py --algo gru` | `src/algorithms/results/gru_metrics.json` |
| `run_algo.py --algo transformer` | `src/algorithms/results/transformer_metrics.json` |
| `compare_models.py` | `src/algorithms/results/comparison_table.csv` + `comparison_table.md` |

---

## Files in this directory

```
ml/src/algorithms/
├── __init__.py               — sub-package marker
├── cnn_gru_model.py          — CNN-GRU model + train/eval pipeline (standalone)
├── cnn_transformer_model.py  — CNN-Transformer model + train/eval pipeline (standalone)
├── run_algo.py               — CLI: --algo {lstm,gru,transformer}
├── compare_models.py         — loads all three JSONs, prints + saves comparison table
├── results/                  — populated at runtime
│   ├── lstm_metrics.json
│   ├── gru_metrics.json
│   ├── transformer_metrics.json
│   ├── comparison_table.csv
│   └── comparison_table.md
└── README.md                 — this file
```

**Hard constraints honoured:**
- Zero changes to any file outside `ml/src/algorithms/` (train.py, infer.py,
  data_loader.py, backend/, frontend/ are untouched).
- Each model file (`cnn_gru_model.py`, `cnn_transformer_model.py`) is
  independently runnable — deleting one does not break the other.
- All metrics come from actually running the model on the SisFall test split;
  no placeholders or hardcoded numbers.
