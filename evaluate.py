#!/usr/bin/env python3
import torch
import torch.nn as nn
from src.ppg2bp_dataset import make_dataloaders
from train import PPG2BPNet    # Import your model definition

def prepare_target(batch, device):
    """
    Given a batch dict with keys "sbp" and "dbp" (either shape [B] or [B,1]),
    convert each to shape [B,1] and then concatenate → [B,2].
    """
    sbp = batch["sbp"].to(device)
    dbp = batch["dbp"].to(device)

    # If sbp is 1D ([B]), turn into [B,1]
    if sbp.dim() == 1:
        sbp = sbp.unsqueeze(1)
    if dbp.dim() == 1:
        dbp = dbp.unsqueeze(1)

    return torch.cat([sbp, dbp], dim=1)  # [B,2]


def main():
    # Paths / hyperparameters
    processed_root = "data/processed"
    splits_root    = "data/splits"
    checkpoint     = "checkpoints/best_model.pt"
    batch_size     = 32
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    # Build DataLoaders (note: no pin_memory argument here)
    train_loader, val_loader, test_loader = make_dataloaders(
        processed_root=processed_root,
        splits_root=splits_root,
        batch_size=batch_size,
        num_workers=2
    )

    # Load model
    model = PPG2BPNet().to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Define loss functions (sum‐over‐batch to accumulate properly)
    mse_fn = nn.MSELoss(reduction="sum")
    mae_fn = nn.L1Loss(reduction="sum")

    total_mse = 0.0
    total_mae = 0.0
    total_n   = 0

    # Iterate through test set
    with torch.no_grad():
        for batch in test_loader:
            x = batch["ppg"].to(device)             # shape [B,500]
            y = prepare_target(batch, device)       # shape [B,2]
            pred = model(x)                         # shape [B,2]

            bsize = x.size(0)
            total_mse += mse_fn(pred, y).item()
            total_mae += mae_fn(pred, y).item()
            total_n   += bsize

    mse_val = total_mse / total_n
    mae_val = total_mae / total_n
    print(f"Test   | MSE: {mse_val:.4f}  MAE: {mae_val:.4f}")


if __name__ == "__main__":
    main()
