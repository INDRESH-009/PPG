#!/usr/bin/env python3
import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.ppg2bp_dataset import make_dataloaders


# ── Model Definition ───────────────────────────────────────────────────────────
class PPG2BPNet(nn.Module):
    """
    “Comparative paired” PPG2BP-Net as in Figure 4 of Sci. Rep. 13, 35492 (2023).

    - Shared 1D-CNN encoder for both “target PPG” and “calibration PPG”.
    - Two separate small MLPs to embed scalar calibration SBP and DBP.
    - Final fully connected block to fuse CNN features + calibration MLP.
    - Output is [SBP_pred, DBP_pred].
    """

    def __init__(self):
        super().__init__()

        # 1) Shared 1D-CNN Encoder (input: [B, 1, 500] → output: [B, 128]).
        #    We assume raw PPG is already shaped [B, 500] and we'll unsqueeze to [B,1,500].
        self.encoder = nn.Sequential(
            # Conv Layer 1: 1 → 16
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # Conv Layer 2: 16 → 32
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Conv Layer 3: 32 → 64
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Conv Layer 4: 64 → 128
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Conv Layer 5: 128 → 256
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Global average pooling over the time dimension (500 → 1).
            #    Use kernel_size=500 so that each channel is averaged to a single number.
            nn.AvgPool1d(kernel_size=500),

            # [B, 256, 1] → flatten → [B, 256].
            nn.Flatten(),

            # FC 256 → 128, then BN → ReLU to obtain a 128-dim feature.
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Dropout, just as in the paper.
            nn.Dropout(p=0.2)
        )


        # 2) Two small MLPs for calibration SBP and DBP (each scalar → 32-dim).
        #    We’ll build a reused submodule “SmallCalibMLP” to avoid code duplication.
        class SmallCalibMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Linear(32, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU()
                )

            def forward(self, x_scalar):
                # x_scalar is shape [B, 1]
                return self.net(x_scalar)

        self.sbp_mlp = SmallCalibMLP()
        self.dbp_mlp = SmallCalibMLP()

        # 3) Final fully-connected “fusion” block.
        #    It will take the concatenation of:
        #       • target_CNN_feature [B,128]
        #       • calib_CNN_feature   [B,128]
        #       • calib_numeric_embed [B, 64]  (i.e. concat of SBP_embed and DBP_embed)
        #
        #    → total dimension = 128 + 128 + 64 = 320 
        #
        #    Then we pass through two FC layers to obtain final [B,2].
        self.fuse_fc = nn.Sequential(
            nn.Linear(320, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2)   # output: [SBP_pred, DBP_pred]
        )


    def forward(self, batch):
        """
        Expects `batch` to be a dictionary containing:
          - "ppg_target":       PPG tensor [B, 500]
          - "ppg_calib":        PPG tensor [B, 500]
          - "sbp_calib": scalar [B] or [B,1]
          - "dbp_calib": scalar [B] or [B,1]
          
        Returns:
          - preds [B,2]: [SBP_pred, DBP_pred]
        """

        # 1) Encode “target” PPG
        #    Input shape: [B, 500] → reshape to [B,1,500], feed to encoder → [B,128]
        x_t = batch["ppg_target"].unsqueeze(1)        # [B, 1, 500]
        feat_t = self.encoder(x_t)                    # [B, 128]

        # 2) Encode “calibration” PPG (same shared encoder)
        x_c = batch["ppg_calib"].unsqueeze(1)         # [B, 1, 500]
        feat_c = self.encoder(x_c)                    # [B, 128]

        # 3) Embed numeric SBP_calib and DBP_calib via small MLPs
        #    Convert them to shape [B,1] if needed:
        sbp = batch["sbp_calib"]
        dbp = batch["dbp_calib"]
        if sbp.dim() == 1:
            sbp = sbp.unsqueeze(1)   # [B, 1]
        if dbp.dim() == 1:
            dbp = dbp.unsqueeze(1)   # [B, 1]

        sbp_embed = self.sbp_mlp(sbp)  # [B, 32]
        dbp_embed = self.dbp_mlp(dbp)  # [B, 32]

        # Concatenate those two → [B, 64]
        calib_embed = torch.cat([sbp_embed, dbp_embed], dim=1)  # [B, 64]

        # 4) Concatenate [feat_t (128), feat_c (128), calib_embed (64)] → [B, 320]
        fused = torch.cat([feat_t, feat_c, calib_embed], dim=1)  # [B, 320]

        # 5) Final FCL to produce [B,2]
        out = self.fuse_fc(fused)  # [B, 2]

        return out



# ── Training & Validation Routines ─────────────────────────────────────────────
def prepare_target(batch, device):
    """
    In our new pipeline, targets are still SBP and DBP of the *target* segment.
    But because the model also takes in "calibration" SBP and DBP as inputs,
    we do not use those for training the loss. Instead, the final output is
    compared to the *true* SBP/DBP of the target segment. 
    So `batch["sbp_target"]`, `batch["dbp_target"]` must exist in your data loader.
    """
    sbp_t = batch["sbp_target"]
    dbp_t = batch["dbp_target"]
    if sbp_t.dim() == 1:
        sbp_t = sbp_t.unsqueeze(1)
    if dbp_t.dim() == 1:
        dbp_t = dbp_t.unsqueeze(1)
    return torch.cat([sbp_t, dbp_t], dim=1).to(device)  # [B,2]


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        # Each batch dictionary must now contain four keys:
        #   batch["ppg_target"], batch["ppg_calib"], batch["sbp_calib"], batch["dbp_calib"],
        #   and also batch["sbp_target"], batch["dbp_target"].
        #
        x_out = model(batch)                # [B, 2]
        y_true = prepare_target(batch, device)  # [B, 2]
        loss = loss_fn(x_out, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x_out.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            x_out = model(batch)
            y_true = prepare_target(batch, device)
            loss = loss_fn(x_out, y_true)
            total_loss += loss.item() * x_out.size(0)
    return total_loss / len(loader.dataset)


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Your make_dataloaders(...) must now construct batches that include
    #   "ppg_target" (500-sample PPG for target)        → torch.FloatTensor [B,500]
    #   "ppg_calib"  (500-sample PPG for calibration)   → torch.FloatTensor [B,500]
    #   "sbp_calib"  (scalar SBP of calib segment)      → torch.FloatTensor [B] or [B,1]
    #   "dbp_calib"  (scalar DBP of calib segment)      → torch.FloatTensor [B] or [B,1]
    #   "sbp_target" (scalar SBP of target segment)     → torch.FloatTensor [B] or [B,1]
    #   "dbp_target" (scalar DBP of target segment)     → torch.FloatTensor [B] or [B,1]
    #
    # For now, we assume your data pipeline yields exactly those keys.
    # (We will show you how to adjust `ppg2bp_dataset.py` at the end.)

    train_loader, val_loader, _ = make_dataloaders(
        processed_root=args.processed,
        splits_root=args.splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    model     = PPG2BPNet().to(device)
    loss_fn   = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=args.logdir)

    best_val = float('inf')
    os.makedirs(args.save, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss   = validate(model, val_loader,     loss_fn, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        print(f"Epoch {epoch:2d}  Train: {train_loss:.4f}  Val: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.save, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → saved new best model to {ckpt_path}")

    writer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed",  default="data/processed",
                   help="Folder with .parquet segments")
    p.add_argument("--splits",     default="data/splits",
                   help="Folder with train/val/test CSVs")
    p.add_argument("--save",       default="checkpoints",
                   help="Directory to save model checkpoints")
    p.add_argument("--logdir",     default="runs",
                   help="TensorBoard log directory")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--num_workers",type=int, default=2,
                   help="Number of DataLoader worker processes")
    args = p.parse_args()

    main(args)
