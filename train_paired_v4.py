#!/usr/bin/env python3
"""
train_paired_v2.py

A Paired‐Segment model that takes:
  • 500‐sample “calibration” PPG
  • calibration SBP and DBP (two scalars)
  • 500‐sample “target” PPG

and predicts (SBP_target, DBP_target).  Uses a small 1D‐CNN front end + FC head,
plus AdamW, weight‐decay, LR scheduling, and mixed precision (if on CUDA).

Usage (example):
    python train_paired_v2.py \
      --processed    data/processed \
      --splits       data/splits \
      --save         checkpoints \
      --logdir       runs/paired_v2 \
      --batch_size   32 \
      --lr            1e-3 \
      --weight_decay  1e-5 \
      --lr_factor     0.5 \
      --lr_patience   3 \
      --epochs        20 \
      --num_workers   4
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from src.ppg2bp_dataset import make_dataloaders  # must return Paired DataLoaders



# ────────────────────────────────────────────────────────────────────────────────
# 1) Model Definition: small 1D‐CNN for PPG → feature vector, then FC including
#    the two calibration scalars (SBP_c, DBP_c)
# ────────────────────────────────────────────────────────────────────────────────

class PairedPPG2BPNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        # ── 1D‐CNN front end: each PPG segment is length=500
        #    We convolve with kernel=7,5,3 + pooling so that each 500→25 → flatten → feature dim=64*25=1600
        self.cnn = nn.Sequential(
            # Input: [B, 1, 500]
            nn.Conv1d(1, 16, kernel_size=7, padding=3),  # [B,16,500]
            nn.ReLU(),
            nn.MaxPool1d(2),                             # [B,16,250]
            nn.Conv1d(16, 32, kernel_size=5, padding=2), # [B,32,250]
            nn.ReLU(),
            nn.MaxPool1d(2),                             # [B,32,125]
            nn.Conv1d(32, 64, kernel_size=3, padding=1), # [B,64,125]
            nn.ReLU(),
            nn.MaxPool1d(5),                             # [B,64,25]
            nn.Flatten()                                 # [B,64*25] = [B,1600]
        )

        # ── Fully‐connected head: (1600 from calib PPG) + (1600 from target PPG) + 2 scalars → 512 → 128 → 2
        self.fc = nn.Sequential(
            nn.Linear(1600 * 2 + 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # output: [SBP_target, DBP_target]
        )

    def forward(self, ppg_c, ppg_t, sbp_c, dbp_c):
        # ppg_c, ppg_t: [B,500] floats; sbp_c, dbp_c: [B] or [B,1] floats
        # 1) Per‐segment normalization (zero‐mean/ unit‐variance)
        mean_c = ppg_c.mean(dim=1, keepdim=True)             # [B,1]
        std_c  = ppg_c.std(dim=1, keepdim=True) + 1e-6        # [B,1]
        ppg_c_norm = (ppg_c - mean_c) / std_c                 # [B,500]

        mean_t = ppg_t.mean(dim=1, keepdim=True)
        std_t  = ppg_t.std(dim=1, keepdim=True) + 1e-6
        ppg_t_norm = (ppg_t - mean_t) / std_t                 # [B,500]

        # 2) CNN expects shape [B, 1, 500]
        x_c = ppg_c_norm.unsqueeze(1)  # [B,1,500]
        x_t = ppg_t_norm.unsqueeze(1)  # [B,1,500]

        f_c = self.cnn(x_c)            # [B,1600]
        f_t = self.cnn(x_t)            # [B,1600]

        # 3) calibration SBP/DBP: ensure shape [B,1] and concatenate
        if sbp_c.dim() == 1:
            sbp_c = sbp_c.unsqueeze(1)  # [B,1]
        if dbp_c.dim() == 1:
            dbp_c = dbp_c.unsqueeze(1)  # [B,1]
        cals = torch.cat([sbp_c, dbp_c], dim=1)  # [B,2]

        # 4) Final feature vector: [B, 1600 + 1600 + 2 = 3202]
        x = torch.cat([f_c, f_t, cals], dim=1)    # [B,3202]
        return self.fc(x)                         # [B,2]  → (SBP_t, DBP_t)



# ────────────────────────────────────────────────────────────────────────────────
# 2) Helper: pack target SBP/DBP from batch into a [B,2] tensor on device
# ────────────────────────────────────────────────────────────────────────────────

def prepare_target(batch, device):
    """
    Given batch["sbp_target"], batch["dbp_target"], each [B] or [B,1],
    return a [B,2] tensor on `device` = [[sbp_target, dbp_target], ...].
    """
    sbp_t = batch["sbp_t"]
    dbp_t = batch["dbp_t"]
    if sbp_t.dim() == 1:
        sbp_t = sbp_t.unsqueeze(1)   # [B,1]
    if dbp_t.dim() == 1:
        dbp_t = dbp_t.unsqueeze(1)   # [B,1]
    return torch.cat([sbp_t, dbp_t], dim=1).to(device)  # [B,2]



# ────────────────────────────────────────────────────────────────────────────────
# 3) Training & Validation loops (w/ AMP and scheduler)
# ────────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        # 1) Pull everything onto device
        ppg_c = batch["ppg_c"].to(device)    # [B,500]
        ppg_t = batch["ppg_t"].to(device)   # [B,500]
        sbp_c = batch["sbp_c"].to(device)    # [B]
        dbp_c = batch["dbp_c"].to(device)    # [B]
        y_true = prepare_target(batch, device)   # [B,2]

        optimizer.zero_grad()
        with autocast():  # mixed precision context
            preds = model(ppg_c, ppg_t, sbp_c, dbp_c)  # [B,2]
            loss  = loss_fn(preds, y_true)              # MSE

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * ppg_c.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            ppg_c = batch["ppg_c"].to(device)
            ppg_t = batch["ppg_t"].to(device)
            sbp_c = batch["sbp_c"].to(device)
            dbp_c = batch["dbp_c"].to(device)
            y_true = prepare_target(batch, device)

            preds = model(ppg_c, ppg_t, sbp_c, dbp_c)
            loss  = loss_fn(preds, y_true)
            total_loss += loss.item() * ppg_c.size(0)

    return total_loss / len(loader.dataset)



# ────────────────────────────────────────────────────────────────────────────────
# 4) Main training routine
# ────────────────────────────────────────────────────────────────────────────────

def main(args):
    # 4.1) Device setup + AMP scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()

    # 4.2) DataLoaders (Paired: returns ppg_calib, ppg_target, sbp_calib, dbp_calib, sbp_target, dbp_target)
    train_loader, val_loader, _ = make_dataloaders(
        processed_root=args.processed,
        splits_root=args.splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        paired=True  # Use paired=True to load paired data
    )

    # 4.3) Model, loss, optimizer (AdamW w/ L2)
    model = PairedPPG2BPNetV2().to(device)
    loss_fn = nn.L1Loss()  
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)

    # 4.4) LR Scheduler: reduce LR on plateau (monitor validation MSE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=1e-6
    )

    # 4.5) TensorBoard writer
    writer = SummaryWriter(log_dir=args.logdir)
    os.makedirs(args.save, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device)
        val_loss   = validate(model, val_loader,   loss_fn, device)

        # 4.6) Step LR scheduler on val_loss
        scheduler.step(val_loss)

        # 4.7) Log metrics
        writer.add_scalar("Loss/train",   train_loss, epoch)
        writer.add_scalar("Loss/val",     val_loss,   epoch)
        writer.add_scalar("LR/current", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch:2d}  Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 4.8) Save best model
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.save, "best_model_paired_v2.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → saved new best model → {ckpt_path}")

    writer.close()


# ────────────────────────────────────────────────────────────────────────────────
# 5) CLI arguments
# ────────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed",    default="data/processed",
                   help="Folder with .parquet segments (paired version)")
    p.add_argument("--splits",       default="data/splits",
                   help="Folder with train/val/test CSVs (paired format)")
    p.add_argument("--save",         default="checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--logdir",       default="runs/paired_v2",
                   help="TensorBoard log directory")
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-3,
                   help="Initial learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-5,
                   help="L2 weight decay (AdamW)")
    p.add_argument("--lr_factor",    type=float, default=0.5,
                   help="Factor to multiply LR when val plateaus")
    p.add_argument("--lr_patience",  type=int,   default=3,
                   help="Epochs with no val improvement before LR is reduced")
    p.add_argument("--epochs",       type=int,   default=20)
    p.add_argument("--num_workers",  type=int,   default=2,
                   help="Number of DataLoader worker processes")
    args = p.parse_args()

    main(args)
