#!/usr/bin/env python3
"""
train_paired.py

Implements the “paired” PPG2BP‐Net from Figure 4 of 
https://www.nature.com/articles/s41598-023-35492-y

Each training sample consists of:
  - ppg_c:  500‐sample calibration PPG waveform
  - sbp_c:   scalar (calibration SBP)
  - dbp_c:   scalar (calibration DBP)
  - ppg_t:  500‐sample target PPG waveform  (we want to predict its BP)
  - sbp_t:   scalar (ground‐truth SBP for target)
  - dbp_t:   scalar (ground‐truth DBP for target)

Architecture overview:
  1) Two 1D‐CNN “towers” (shared weights):
       CNN(ppg_c)  → 64‐dim feature vector f_c
       CNN(ppg_t)  → 64‐dim feature vector f_t
  2) Calibration MLP (blue block):
       MLP([sbp_c, dbp_c]) → 16‐dim feature vector f_cal
  3) Compute abs_diff = |f_c − f_t|  (64‐dim)
  4) Concatenate [f_t, f_c, abs_diff, f_cal] → 208‐dim 
  5) Final FCL (purple block):
       → [sbp_t_pred, dbp_t_pred] (2 scalars)

Usage example:
    python train_paired.py \
      --processed   data/processed \
      --splits      data/splits    \
      --save        checkpoints     \
      --logdir      runs/exp1       \
      --batch_size  32              \
      --lr          1e-3            \
      --epochs      20              \
      --num_workers 2
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Import the “paired” DataLoader factory from src/ppg2bp_dataset.py
from src.ppg2bp_dataset import make_dataloaders

# ────────────────────────────────────────────────────────────────────────────────
# Step 1: Define the “paired” network as in Figure 4
# ────────────────────────────────────────────────────────────────────────────────
class PairedPPG2BPNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ------ 1D‐CNN Towers (shared) [red blocks] ------
        # We will build a small three‐layer 1D‐CNN. Each layer does:
        #   Conv1d → BatchNorm1d → ReLU
        # At the end, we do AdaptiveAvgPool1d(1) to collapse time and produce 64 features.
        #
        # Input: [B, 1, 500]  (channel=1, length=500)
        # Output: [B, 64, 1]  → flatten → [B, 64]
        self.cnn_tower = nn.Sequential(
            # Layer 1:  1 → 16 channels, kernel_size=5, padding=2 (to keep length=500)
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # Layer 2: 16 → 32 channels
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Layer 3: 32 → 64 channels
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # Finally collapse the time dimension (500 → 1)
            nn.AdaptiveAvgPool1d(1)  
            # Output shape = [B, 64, 1]
        )

        # ------ Calibration MLP (blue block) ------
        # Input: [sbp_c, dbp_c]  shape [B, 2]
        # Output: 16‐dim vector [B, 16]
        self.calib_mlp = nn.Sequential(
            nn.Linear(2, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        # ------ Final Fully‐Connected Layers (purple block) ------
        # We will concatenate:
        #   f_t:      64‐dim (CNN on target PPG)
        #   f_c:      64‐dim (CNN on calib  PPG)
        #   abs_diff: 64‐dim (|f_t − f_c|)
        #   f_cal:    16‐dim (MLP on [sbp_c, dbp_c])
        # Total = 64 + 64 + 64 + 16 = 208 dimensions.
        # The final MLP reduces 208 → 128 → 2 (SBP_t_pred, DBP_t_pred).
        self.final_fcl = nn.Sequential(
            nn.Linear(64 + 64 + 64 + 16, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 2)  # output = [sbp_t_pred, dbp_t_pred]
        )

    def forward(self, ppg_t, ppg_c, sbp_c_scalar, dbp_c_scalar):
        """
        Forward pass:
          - ppg_t:       [B, 500]   (target PPG waveform)
          - ppg_c:       [B, 500]   (calibration PPG waveform)
          - sbp_c_scalar: [B,   1]  (calibration SBP)
          - dbp_c_scalar: [B,   1]  (calibration DBP)

        Returns:
          - out:       [B, 2]  (predicted [sbp_t, dbp_t])
        """
        B = ppg_t.size(0)

        # Convert each [B, 500] → [B, 1, 500] so Conv1d can consume it:
        x_t = ppg_t.unsqueeze(1)  # [B, 1, 500]
        x_c = ppg_c.unsqueeze(1)  # [B, 1, 500]

        # Pass through the shared CNN tower:
        f_t = self.cnn_tower(x_t)  # [B, 64, 1]
        f_c = self.cnn_tower(x_c)  # [B, 64, 1]

        # Flatten the last dimension → [B, 64]:
        f_t = f_t.view(B, 64)
        f_c = f_c.view(B, 64)

        # Compute element‐wise absolute difference [B, 64]:
        abs_diff = torch.abs(f_t - f_c)

        # Build the calibration input [B, 2] = concat([sbp_c, dbp_c]), each is [B,1]:
        calib_input = torch.cat([sbp_c_scalar, dbp_c_scalar], dim=1)  # [B, 2]
        f_calib = self.calib_mlp(calib_input)                         # [B, 16]

        # Concatenate all four feature vectors → [B, 64+64+64+16 = 208]:
        combined = torch.cat([f_t, f_c, abs_diff, f_calib], dim=1)     # [B, 208]

        # Final fully connected to predict [sbp_t, dbp_t]:
        out = self.final_fcl(combined)  # [B, 2]
        return out


# ────────────────────────────────────────────────────────────────────────────────
# Step 2: Define training/validation loops
# ────────────────────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in loader:
        # Pull out exactly the six fields per sample:
        ppg_t = batch["ppg_t"].to(device)   # [B, 500]
        ppg_c = batch["ppg_c"].to(device)   # [B, 500]
        sbp_c = batch["sbp_c"].to(device)   # [B,   1]
        dbp_c = batch["dbp_c"].to(device)   # [B,   1]
        sbp_t = batch["sbp_t"].to(device)   # [B,   1]
        dbp_t = batch["dbp_t"].to(device)   # [B,   1]

        # Forward pass → [B, 2]
        pred = model(ppg_t, ppg_c, sbp_c, dbp_c)

        # Build target tensor [B, 2] from sbp_t, dbp_t:
        target = torch.cat([sbp_t, dbp_t], dim=1).to(device)  # [B, 2]

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * ppg_t.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            ppg_t = batch["ppg_t"].to(device)
            ppg_c = batch["ppg_c"].to(device)
            sbp_c = batch["sbp_c"].to(device)
            dbp_c = batch["dbp_c"].to(device)
            sbp_t = batch["sbp_t"].to(device)
            dbp_t = batch["dbp_t"].to(device)

            pred = model(ppg_t, ppg_c, sbp_c, dbp_c)
            target = torch.cat([sbp_t, dbp_t], dim=1).to(device)

            loss = loss_fn(pred, target)
            total_loss += loss.item() * ppg_t.size(0)

    return total_loss / len(loader.dataset)


# ────────────────────────────────────────────────────────────────────────────────
# Step 3: Main entry‐point
# ────────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed",   default="data/processed",
                        help="Folder containing .parquet segments (output of preprocess).")
    parser.add_argument("--splits",      default="data/splits",
                        help="Folder containing split CSVs (train.csv, val.csv, test.csv).")
    parser.add_argument("--save",        default="checkpoints",
                        help="Directory to save best‐model checkpoint.")
    parser.add_argument("--logdir",      default="runs",
                        help="TensorBoard log directory.")
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--num_workers", type=int,   default=2)
    args = parser.parse_args()

    # Device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create DataLoaders using PairedSegmentDataset:
    train_loader, val_loader, test_loader = make_dataloaders(
        processed_root=args.processed,
        splits_root=args.splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        paired=True   # <=== ensure PairedSegmentDataset is used
    )

    # Instantiate the “paired” network
    model     = PairedPPG2BPNet().to(device)
    loss_fn   = nn.MSELoss()                     # training with mean‐squared‐error
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # TensorBoard
    writer = SummaryWriter(log_dir=args.logdir)
    os.makedirs(args.save, exist_ok=True)

    best_val_loss = float("inf")
    best_ckpt_path = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss   = validate(model, val_loader,   loss_fn, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        print(f"Epoch {epoch:2d}  Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        # Save best‐performing model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.save, "best_model_paired.pt")
            torch.save(model.state_dict(), ckpt_path)
            best_ckpt_path = ckpt_path
            print(f"  → saved new best model → {ckpt_path}")

    writer.close()
    print("Training complete. Best validation loss:", best_val_loss)
    print("Best checkpoint:", best_ckpt_path)


if __name__ == "__main__":
    main()
