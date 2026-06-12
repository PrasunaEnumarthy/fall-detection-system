# export_onnx.py
# Exports the trained PyTorch model to ONNX format for cross-platform deployment.
# Validates the export by comparing PyTorch and ONNX outputs on 10 test samples.

import sys
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent))
from model import FallDetectorCNN_LSTM

# ------------------------------------------------------------------ #
# Paths                                                              #
# ------------------------------------------------------------------ #
BEST_MODEL  = Path(__file__).parent.parent / "models" / "best_model.pt"
ONNX_PATH   = Path(__file__).parent.parent / "models" / "model.onnx"

# Maximum allowed absolute difference between PyTorch and ONNX outputs
MAX_ABS_ERROR = 1e-5

# Number of validation samples to compare
N_VALIDATION_SAMPLES = 10


def export_to_onnx(model_path: str = None, onnx_path: str = None) -> str:
    """
    Export a trained FallDetectorCNN_LSTM to ONNX format.

    The export uses a dummy batch of size 1 as the tracing input.
    Dynamic axes are configured so the batch dimension accepts any size
    at inference time.

    Parameters
    ----------
    model_path : str, optional
        Path to the .pt checkpoint. Defaults to ml/models/best_model.pt.
    onnx_path : str, optional
        Output path for the .onnx file. Defaults to ml/models/model.onnx.

    Returns
    -------
    str
        Absolute path to the exported ONNX file.
    """
    mp  = Path(model_path) if model_path else BEST_MODEL
    op  = Path(onnx_path)  if onnx_path  else ONNX_PATH

    print(f"\n{'='*60}")
    print(" EXPORT — loading model")
    print(f"{'='*60}")

    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}\nRun train.py first.")

    device = torch.device("cpu")   # ONNX export must happen on CPU
    model  = FallDetectorCNN_LSTM().to(device)
    model.load_state_dict(torch.load(str(mp), map_location=device))
    model.eval()
    print(f"[EXPORT] Loaded checkpoint: {mp.name}")

    # Dummy input matching the expected inference shape (batch=1, 200 timesteps, 6 channels)
    dummy_input = torch.randn(1, 200, 6, dtype=torch.float32)

    # Wrap forward() so torch.onnx.export can trace it.
    # The model returns a dict; we convert to a tuple of tensors for ONNX compatibility.
    class _OnnxWrapper(torch.nn.Module):
        """Thin wrapper that returns a tuple instead of a dict — required by torch.onnx.export."""
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            out = self.inner(x)
            return out["fall"], out["fall_type"], out["pre_activity"]

    wrapped = _OnnxWrapper(model)
    wrapped.eval()

    op.parent.mkdir(parents=True, exist_ok=True)

    print(f"[EXPORT] Exporting to {op} ...")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy_input,
            str(op),
            opset_version=17,
            input_names=["imu_window"],
            output_names=["fall_prob", "fall_type_logits", "pre_activity_logits"],
            dynamic_axes={
                "imu_window":            {0: "batch"},
                "fall_prob":             {0: "batch"},
                "fall_type_logits":      {0: "batch"},
                "pre_activity_logits":   {0: "batch"},
            },
            do_constant_folding=True,
            dynamo=False,   # use TorchScript-based legacy exporter (stable in 2.x)
        )

    print(f"[EXPORT] ONNX file written to: {op.resolve()}")

    # ------------------------------------------------------------------ #
    # Check the ONNX graph is structurally valid                          #
    # ------------------------------------------------------------------ #
    onnx_model = onnx.load(str(op))
    onnx.checker.check_model(onnx_model)
    print("[EXPORT] ONNX model structure check: PASS")

    return str(op.resolve())


def validate_onnx(model_path: str = None, onnx_path: str = None) -> bool:
    """
    Compare PyTorch and ONNX Runtime outputs on random inputs.

    Asserts that the maximum absolute difference across all outputs is < MAX_ABS_ERROR.
    A mismatch here indicates the export introduced numerical differences (should not happen
    with a simple opset-17 export but worth verifying).

    Parameters
    ----------
    model_path : str, optional
        Path to the .pt checkpoint.
    onnx_path : str, optional
        Path to the .onnx file.

    Returns
    -------
    bool
        True if validation passes for all N_VALIDATION_SAMPLES samples.
    """
    mp = Path(model_path) if model_path else BEST_MODEL
    op = Path(onnx_path)  if onnx_path  else ONNX_PATH

    print(f"\n[VALIDATE] Comparing PyTorch vs ONNX on {N_VALIDATION_SAMPLES} random samples ...")

    # Load PyTorch model
    device = torch.device("cpu")
    pt_model = FallDetectorCNN_LSTM().to(device)
    pt_model.load_state_dict(torch.load(str(mp), map_location=device))
    pt_model.eval()

    # Load ONNX Runtime session
    ort_session = ort.InferenceSession(str(op), providers=["CPUExecutionProvider"])

    all_passed = True
    rng = np.random.default_rng(12345)

    for sample_idx in range(N_VALIDATION_SAMPLES):
        # Random (1, 200, 6) window
        x_np = rng.standard_normal((1, 200, 6)).astype(np.float32)
        x_pt = torch.tensor(x_np)

        # PyTorch output
        with torch.no_grad():
            pt_out  = pt_model(x_pt)
            pt_fall = pt_out["fall"].numpy()
            pt_ft   = pt_out["fall_type"].numpy()
            pt_pre  = pt_out["pre_activity"].numpy()

        # ONNX Runtime output
        ort_inputs  = {"imu_window": x_np}
        ort_outputs = ort_session.run(None, ort_inputs)
        ort_fall, ort_ft, ort_pre = ort_outputs

        # Compute maximum absolute difference for each head
        err_fall = np.abs(pt_fall - ort_fall).max()
        err_ft   = np.abs(pt_ft   - ort_ft).max()
        err_pre  = np.abs(pt_pre  - ort_pre).max()
        max_err  = max(err_fall, err_ft, err_pre)

        if max_err >= MAX_ABS_ERROR:
            print(
                f"  [FAIL] Sample {sample_idx}: max_abs_error={max_err:.2e} "
                f"(threshold={MAX_ABS_ERROR:.0e})"
            )
            all_passed = False
        else:
            print(f"  Sample {sample_idx:02d}: max_abs_error={max_err:.2e}  PASS")

    if all_passed:
        print("\n[VALIDATE] ONNX export validated successfully")
    else:
        print("\n[VALIDATE] VALIDATION FAILED — some samples exceeded error threshold")

    return all_passed


# ---------------------------------------------------------------------------
# Standalone entry point — run:  python src/export_onnx.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    onnx_file = export_to_onnx()
    validate_onnx(onnx_path=onnx_file)
