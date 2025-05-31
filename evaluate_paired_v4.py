

#!/usr/bin/env python3
"""
evaluate_paired_v2.py

Loads the best PairedPPG2BPNetV2 checkpoint and reports Test MSE/MAE.
"""


import torch
import torch.nn as nn

from src.ppg2bp_dataset import make_dataloaders
from train_paired_v4 import PairedPPG2BPNetV2, prepare_target

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) DataLoader (test only)
    _, _, test_loader = make_dataloaders(
        processed_root="data/processed",
        splits_root="data/splits",
        batch_size=32,
        num_workers=2,
        paired=True  # Use paired=True to load paired data
    )

    # 2) Load the best saved checkpoint
    model = PairedPPG2BPNetV2().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model_paired_v2.pt", map_location=device))
    model.eval()

    mse_fn = nn.MSELoss(reduction="sum")
    mae_fn = nn.L1Loss(reduction="sum")

    total_mse = 0.0
    total_mae = 0.0
    total_n   = 0

    with torch.no_grad():
        for batch in test_loader:
            ppg_c = batch["ppg_c"].to(device)
            ppg_t = batch["ppg_t"].to(device)
            sbp_c = batch["sbp_c"].to(device)
            dbp_c = batch["dbp_c"].to(device)
            y_true = prepare_target(batch, device)    # [B,2]

            y_pred = model(ppg_c, ppg_t, sbp_c, dbp_c)  # [B,2]

            bsz = ppg_c.size(0)
            total_mse += mse_fn(y_pred, y_true).item()
            total_mae += mae_fn(y_pred, y_true).item()
            total_n   += bsz

    mean_mse = total_mse / total_n
    mean_mae = total_mae / total_n
    print(f"Test   | MSE: {mean_mse:.4f}  MAE: {mean_mae:.4f}")

if __name__ == "__main__":
    main()
