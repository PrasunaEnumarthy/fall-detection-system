# model.py
# CNN-LSTM fall detection model in PyTorch.
# Three output heads: binary fall detection, fall type (3-class), pre-activity (4-class).

import torch
import torch.nn as nn


class FallDetectorCNN_LSTM(nn.Module):
    """
    Two-stage CNN-LSTM architecture for IMU-based fall detection.

    Architecture overview
    ---------------------
    Input  : (batch, 200, 6)  — 2 seconds at 100Hz, 6 IMU channels

    CNN stage (extracts local temporal patterns):
      Block 1: Conv1d(6→64,  k=5) → BN → ReLU → MaxPool(2)   → (batch, 64, 100)
      Block 2: Conv1d(64→128, k=5) → BN → ReLU → MaxPool(2)   → (batch, 128, 50)

    LSTM stage (models temporal dependencies across the sequence):
      2-layer LSTM(128→128)  → last timestep  → (batch, 128)

    Output heads (three separate linear projections):
      head_fall         : Linear(128, 1)  + Sigmoid  → fall probability
      head_fall_type    : Linear(128, 3)             → slip / trip / faint  logits
      head_pre_activity : Linear(128, 4)             → walking / standing / bending / sitting logits
    """

    def __init__(self) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # CNN Block 1: captures short-duration impact signatures             #
        # kernel_size=5, padding=2 keeps temporal length unchanged before pool#
        # ------------------------------------------------------------------ #
        self.cnn_block1 = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # 200 → 100
        )

        # ------------------------------------------------------------------ #
        # CNN Block 2: captures slightly longer motion patterns              #
        # ------------------------------------------------------------------ #
        self.cnn_block2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # 100 → 50
        )

        # ------------------------------------------------------------------ #
        # LSTM: models how the movement pattern evolves over the full window  #
        # dropout=0.3 only active between stacked layers (num_layers ≥ 2)    #
        # ------------------------------------------------------------------ #
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        # ------------------------------------------------------------------ #
        # Output heads                                                        #
        # ------------------------------------------------------------------ #
        # Binary fall detection (sigmoid converts to probability in [0,1])
        self.head_fall = nn.Sequential(
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Fall type: slip (0), trip (1), faint (2) — logits, no softmax here
        # CrossEntropyLoss expects raw logits
        self.head_fall_type = nn.Linear(128, 3)

        # Pre-activity: walking (0), standing (1), bending (2), sitting (3)
        self.head_pre_activity = nn.Linear(128, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 200, 6).

        Returns
        -------
        dict with keys:
          'fall'         — (batch, 1)  fall probability after sigmoid
          'fall_type'    — (batch, 3)  raw logits for slip/trip/faint
          'pre_activity' — (batch, 4)  raw logits for activity class
        """
        # CNN expects (batch, channels, time) — permute from (batch, time, channels)
        x = x.permute(0, 2, 1)          # (batch, 6, 200)

        x = self.cnn_block1(x)           # (batch, 64, 100)
        x = self.cnn_block2(x)           # (batch, 128, 50)

        # LSTM expects (batch, time, features) — permute back
        x = x.permute(0, 2, 1)          # (batch, 50, 128)

        lstm_out, _ = self.lstm(x)       # (batch, 50, 128)

        # Take only the last timestep — it encodes the full sequence context
        x = lstm_out[:, -1, :]           # (batch, 128)

        return {
            "fall":          self.head_fall(x),           # (batch, 1)
            "fall_type":     self.head_fall_type(x),      # (batch, 3)
            "pre_activity":  self.head_pre_activity(x),   # (batch, 4)
        }


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Standalone demo — run:  python src/model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== model.py demo ===")

    model = FallDetectorCNN_LSTM()
    model.eval()

    # Simulate a batch of 4 windows
    dummy_input = torch.randn(4, 200, 6)
    print(f"Input shape : {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"\nOutput shapes:")
    print(f"  fall          : {output['fall'].shape}          — values in [0,1]")
    print(f"  fall_type     : {output['fall_type'].shape}     — raw logits")
    print(f"  pre_activity  : {output['pre_activity'].shape}  — raw logits")

    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Verify fall head values are valid probabilities
    assert output["fall"].min() >= 0.0 and output["fall"].max() <= 1.0, \
        "fall head output out of [0,1] range!"
    print("Fall probability range check: PASS")
