#!/usr/bin/env python3
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.ppg2bp_dataset import make_dataloaders
from train import PPG2BPNet  # Import your model definition

def prepare_target(batch, device):
    """
    Same logic as in evaluate.py: turn sbp/dbp into shape [B,1],
    then concatenate → [B,2].
    """
    sbp = batch["sbp"].to(device)
    dbp = batch["dbp"].to(device)

    if sbp.dim() == 1:
        sbp = sbp.unsqueeze(1)
    if dbp.dim() == 1:
        dbp = dbp.unsqueeze(1)

    return torch.cat([sbp, dbp], dim=1)  # [B,2]

def gather_predictions(checkpoint="checkpoints/best_model.pt"):
    """
    Runs the model over the entire test set, returns two (N×2) NumPy arrays:
      - trues:  ground‐truth [SBP, DBP]
      - preds: predicted [SBP, DBP]
    """
    # Choose device (GPU > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device for Bland–Altman gathering: {device}")

    # Build only the test DataLoader (num_workers=0 for simplicity)
    _, _, test_loader = make_dataloaders(
        processed_root="data/processed",
        splits_root="data/splits",
        batch_size=64,
        num_workers=0
    )

    # Load checkpoint
    model = PPG2BPNet().to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_trues = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["ppg"].to(device)            # [B,500]
            y = prepare_target(batch, device)      # [B,2]
            p = model(x)                           # [B,2]

            all_trues.append(y.cpu().numpy())
            all_preds.append(p.cpu().numpy())

    trues = np.vstack(all_trues)    # shape [N,2]
    preds = np.vstack(all_preds)    # shape [N,2]
    return trues, preds

def bland_altman(true_vals, pred_vals, title):
    """
    Draw a Bland–Altman plot for one channel (either SBP or DBP).
    - x-axis: mean of (true + pred)
    - y-axis: (pred − true)
    - horizontal lines for bias ± 1.96σ
    """
    diffs     = pred_vals - true_vals
    mean_vals = (pred_vals + true_vals) / 2.0

    bias    = np.mean(diffs)
    sd_diff = np.std(diffs, ddof=1)

    plt.figure(figsize=(6,6))
    plt.scatter(mean_vals, diffs, alpha=0.3, s=5)
    plt.axhline(bias, color='gray', linestyle='--', label=f"Bias = {bias:.2f}")
    plt.axhline(bias + 1.96*sd_diff, color='red', linestyle=':', label=f"+1.96σ = {(bias + 1.96*sd_diff):.2f}")
    plt.axhline(bias - 1.96*sd_diff, color='red', linestyle=':', label=f"-1.96σ = {(bias - 1.96*sd_diff):.2f}")
    plt.xlabel("Mean of True & Predicted")
    plt.ylabel("Predicted − True")
    plt.title(f"Bland–Altman Plot ({title})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

def main():
    trues, preds = gather_predictions(checkpoint="checkpoints/best_model.pt")

    # Column 0 = SBP, Column 1 = DBP
    bland_altman(trues[:, 0], preds[:, 0], title="SBP")
    bland_altman(trues[:, 1], preds[:, 1], title="DBP")

if __name__ == "__main__":
    main()
