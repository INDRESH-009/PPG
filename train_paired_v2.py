#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.ppg2bp_dataset import make_dataloaders

# ── Paired MLP Definition ─────────────────────────────────────────────────────
class PairedPPG2BPNet(nn.Module):
    """
    A minimal “concatenate‐calib+target” MLP.  We take two 500‐sample PPG
    signals (calibration PPG + target PPG), concatenate them into a 1000‐dim
    vector, and then run through a few Linear→ReLU→Dropout layers to predict
    two outputs: (SBP_target, DBP_target).
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)   # final outputs: [SBP_target, DBP_target]
        )

    def forward(self, ppg_c, ppg_t):
        # ppg_c, ppg_t each have shape [B, 500]
        x = torch.cat([ppg_c, ppg_t], dim=1)  # [B, 1000]
        return self.net(x)                    # [B, 2] = (pred_SBP_t, pred_DBP_t)


# ── Training & Validation Routines ───────────────────────────────────────────
def prepare_target(batch, device):
    """
    From PairedSegmentDataset, batch["sbp_t"] and batch["dbp_t"] are each [B,1].
    We want to produce a [B,2] tensor: [SBP_target, DBP_target].  
    """
    sbp_t = batch["sbp_t"]  # [B,1]
    dbp_t = batch["dbp_t"]  # [B,1]
    # They are already shape [B,1], so just cat and move to device:
    y = torch.cat([sbp_t, dbp_t], dim=1).to(device)  # [B,2]
    return y

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        # Each of these is already a Tensor on CPU; we push ppg_c and ppg_t to device.
        ppg_c = batch["ppg_c"].to(device)  # [B,500]
        ppg_t = batch["ppg_t"].to(device)  # [B,500]
        y_true = prepare_target(batch, device)  # [B,2]

        # Forward pass
        y_pred = model(ppg_c, ppg_t)  # [B,2]

        loss = loss_fn(y_pred, y_true)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * ppg_c.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            ppg_c = batch["ppg_c"].to(device)
            ppg_t = batch["ppg_t"].to(device)
            y_true = prepare_target(batch, device)

            y_pred = model(ppg_c, ppg_t)
            loss = loss_fn(y_pred, y_true)
            total_loss += loss.item() * ppg_c.size(0)
    return total_loss / len(loader.dataset)


# ── Main Entrypoint ───────────────────────────────────────────────────────────
def main(args):
    # Use CUDA if available; otherwise MPS (macOS) or CPU
    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))

    # 1) Build Paired DataLoaders
    train_loader, val_loader, test_loader = make_dataloaders(
        processed_root=args.processed,
        splits_root=args.splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        paired=True
    )

    # 2) Instantiate model, loss, optimizer
    model = PairedPPG2BPNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3) TensorBoard writer
    writer = SummaryWriter(log_dir=args.logdir)
    os.makedirs(args.save, exist_ok=True)

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)

        # Log to console + TensorBoard
        print(f"Epoch {epoch:2d}  Train: {train_loss:.4f}  Val: {val_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        # Save the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(args.save, "best_model_paired.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → saved new best model → {ckpt_path}")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed", default="data/processed",
        help="Folder where .parquet segments live (output of preprocess step)"
    )
    parser.add_argument(
        "--splits", default="data/splits",
        help="Folder where train.csv, val.csv, test.csv live"
    )
    parser.add_argument(
        "--save", default="checkpoints",
        help="Directory in which to save best_model_paired.pt"
    )
    parser.add_argument(
        "--logdir", default="runs",
        help="TensorBoard log directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for paired dataloader"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--num_workers", type=int, default=2,
        help="Number of DataLoader worker processes"
    )
    args = parser.parse_args()
    main(args)
