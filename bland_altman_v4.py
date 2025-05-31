#!/usr/bin/env python3
"""
bland_altman_paired_v2.py

Compute Bland–Altman for the new PairedPPG2BPNetV2 (SBP and DBP).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.ppg2bp_dataset import make_dataloaders
from train_paired_v4 import PairedPPG2BPNetV2, prepare_target

def gather_predictions(checkpoint="checkpoints/best_model_paired_v2.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = make_dataloaders(
        processed_root="data/processed",
        splits_root="data/splits",
        batch_size=32,
        num_workers=0,
        paired=True  # Use paired=True to load paired data
    )

    model = PairedPPG2BPNetV2().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    all_trues = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            ppg_c = batch["ppg_c"].to(device)
            ppg_t = batch["ppg_t"].to(device)
            sbp_c = batch["sbp_c"].to(device)
            dbp_c = batch["dbp_c"].to(device)

            y = prepare_target(batch, device)          # [B,2]
            p = model(ppg_c, ppg_t, sbp_c, dbp_c)       # [B,2]
            all_trues.append(y.cpu().numpy())
            all_preds.append(p.cpu().numpy())

    trues = np.vstack(all_trues)   # [N,2]
    preds = np.vstack(all_preds)   # [N,2]
    return trues, preds

def bland_altman(true_vals, pred_vals, title):
    mean_vals = (true_vals + pred_vals) / 2
    diffs     = pred_vals - true_vals
    bias      = np.mean(diffs)
    sd_diff   = np.std(diffs, ddof=1)

    plt.figure(figsize=(6,6))
    plt.scatter(mean_vals, diffs, alpha=0.3, s=5)
    plt.axhline(bias,      color='gray', linestyle='--',
                label=f"Bias = {bias:.2f}")
    plt.axhline(bias + 1.96*sd_diff, color='red', linestyle=':',
                label=f"+1.96σ = {bias + 1.96*sd_diff:.2f}")
    plt.axhline(bias - 1.96*sd_diff, color='red', linestyle=':',
                label=f"-1.96σ = {bias - 1.96*sd_diff:.2f}")
    plt.xlabel("Mean of True & Predicted")
    plt.ylabel("Predicted − True")
    plt.title(f"Bland–Altman Plot ({title})")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    trues, preds = gather_predictions()
    bland_altman(trues[:, 0], preds[:, 0], title="SBP")
    bland_altman(trues[:, 1], preds[:, 1], title="DBP")

if __name__ == "__main__":
    main()
