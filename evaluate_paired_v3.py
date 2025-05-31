# evaluate_paired.py
import torch
import torch.nn as nn
from src.ppg2bp_dataset import make_dataloaders
from train_paired_v3 import PairedPPG2BPNet, prepare_target

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test DataLoader
    _, _, test_loader = make_dataloaders(
        processed_root="data/processed",
        splits_root="data/splits",
        batch_size=32,
        num_workers=2,
        paired=True  # Use paired=True to load paired data
    )

    # Load best model
    model = PairedPPG2BPNet().to(device)
    model.load_state_dict(torch.load("checkpoints/best_model_paired.pt", map_location=device))
    model.eval()

    mse_fn = nn.MSELoss(reduction="sum")
    mae_fn = nn.L1Loss(reduction="sum")

    total_mse = 0.0
    total_mae = 0.0
    n_total = 0

    with torch.no_grad():
        for batch in test_loader:
            ppg_c = batch["ppg_c"].to(device)
            ppg_t = batch["ppg_t"].to(device)
            y_true = prepare_target(batch, device)   # [B,2]
            y_pred = model(ppg_c, ppg_t)            # [B,2]

            bsz = ppg_c.size(0)
            total_mse += mse_fn(y_pred, y_true).item()
            total_mae += mae_fn(y_pred, y_true).item()
            n_total += bsz

    print(f"Test   | MSE: {total_mse / n_total:.2f}  MAE: {total_mae / n_total:.2f}")

if __name__ == "__main__":
    main()

