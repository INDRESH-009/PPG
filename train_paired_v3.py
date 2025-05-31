# train_paired.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from src.ppg2bp_dataset import make_dataloaders  # assumes PairedSegmentDataset is in there

# ── Model Definition: normalize each 500‐sample PPG before concatenation ─────────────────────
class PairedPPG2BPNet(nn.Module):
    def __init__(self):
        super().__init__()
        # After normalization, we will have 500 (calib) + 500 (target) = 1000 inputs
        self.net = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)  # predict [SBP, DBP]
        )

    def forward(self, ppg_c, ppg_t):
        # ppg_c, ppg_t: each [B, 500]
        # 1) Compute per‐sample mean & std
        mean_c = ppg_c.mean(dim=1, keepdim=True)        # [B,1]
        std_c  = ppg_c.std(dim=1, keepdim=True) + 1e-6  # [B,1], avoid zero‐div
        ppg_c_norm = (ppg_c - mean_c) / std_c           # [B,500]

        mean_t = ppg_t.mean(dim=1, keepdim=True)
        std_t  = ppg_t.std(dim=1, keepdim=True) + 1e-6
        ppg_t_norm = (ppg_t - mean_t) / std_t           # [B,500]

        # 2) Concatenate normalized calib + normalized target
        x = torch.cat([ppg_c_norm, ppg_t_norm], dim=1)   # [B,1000]
        return self.net(x)                              # [B,2]


# ── Helper: collate SBP/DBP from batch dict → [B,2] ─────────────────────────────────────────
def prepare_target(batch, device):
    """
    Given batch["sbp_calib"], batch["dbp_calib"], but we only train on target SBP/DBP.
    We want [B,2] = [sbp_target, dbp_target].
    """
    sbp_t = batch["sbp_t"]
    dbp_t = batch["dbp_t"]
    # Ensure shape [B,1]
    if sbp_t.dim() == 1:
        sbp_t = sbp_t.unsqueeze(1)
    if dbp_t.dim() == 1:
        dbp_t = dbp_t.unsqueeze(1)
    return torch.cat([sbp_t, dbp_t], dim=1).to(device)


# ── Training & Validation Routines ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        ppg_c = batch["ppg_c"].to(device)   # [B,500]
        ppg_t = batch["ppg_t"].to(device)  # [B,500]
        y_true = prepare_target(batch, device)  # [B,2]

        optimizer.zero_grad()
        with autocast():  # Automatic Mixed‐Precision
            preds = model(ppg_c, ppg_t)        # [B,2]
            loss = loss_fn(preds, y_true)

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
            y_true = prepare_target(batch, device)

            preds = model(ppg_c, ppg_t)
            loss = loss_fn(preds, y_true)
            total_loss += loss.item() * ppg_c.size(0)

    return total_loss / len(loader.dataset)


# ── Main Routine ───────────────────────────────────────────────────────────────────────────────
def main(args):
    # 1) Device & AMP scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler()

    # 2) DataLoaders (Paired data)
    train_loader, val_loader, _ = make_dataloaders(
        processed_root=args.processed,
        splits_root=args.splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        paired=True
    )

    # 3) Model, loss, optimizer (with weight_decay)
    model = PairedPPG2BPNet().to(device)
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr,
                            weight_decay=args.weight_decay)

    # 4) Scheduler: Reduce LR on plateau (monitor validation loss)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=1e-6
    )

    # 5) TensorBoard
    writer = SummaryWriter(log_dir=args.logdir)
    os.makedirs(args.save, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler, device)
        val_loss   = validate(model, val_loader,   loss_fn, device)

        # 6) Scheduler step (plateau on val_loss)
        scheduler.step(val_loss)

        # 7) Log to TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("LR/current", optimizer.param_groups[0]["lr"], epoch)

        print(f"Epoch {epoch:2d}  Train: {train_loss:.4f}  Val: {val_loss:.4f}  "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 8) Save best
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.save, "best_model_paired.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → saved new best model → {ckpt_path}")

    writer.close()


# ── CLI ────────────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed",   default="data/processed",
                   help="Folder with .parquet segments (paired version)")
    p.add_argument("--splits",      default="data/splits",
                   help="Folder with train/val/test CSVs (paired format)")
    p.add_argument("--save",        default="checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--logdir",      default="runs/paired_exp",
                   help="TensorBoard log directory")
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3,
                   help="Initial learning rate")
    p.add_argument("--weight_decay",type=float, default=1e-5,
                   help="L2 weight decay (AdamW)")
    p.add_argument("--lr_factor",   type=float, default=0.5,
                   help="Factor by which to reduce LR on plateau")
    p.add_argument("--lr_patience", type=int,   default=3,
                   help="Number of epochs with no improvement before reducing LR")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--num_workers", type=int,   default=2,
                   help="Number of DataLoader worker processes")
    args = p.parse_args()

    main(args)
