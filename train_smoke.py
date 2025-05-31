# train_smoke.py
import torch ,torch.nn as nn, torch.optim as optim
from src.ppg2bp_dataset import make_dataloaders

# hyperparams
BATCH = 8; LR = 1e-3

train_loader, val_loader, test_loader = make_dataloaders(
    processed_root="data/processed",
    splits_root="data/splits",
    batch_size=BATCH
)

# a minimal linear model
model = nn.Sequential(
    nn.Linear(500, 64),
    nn.ReLU(),
    nn.Linear(64, 2)
)
opt   = optim.Adam(model.parameters(), lr=LR)
lossf = nn.MSELoss()

# take one batch
batch = next(iter(train_loader))
# separate calib vs target
ppg = batch["ppg"]      # [B,500]
bp  = torch.cat([batch["sbp"], batch["dbp"]], dim=1)  # [B,2]

# forward + loss
pred = model(ppg)
loss = lossf(pred, bp)
loss.backward()
print("Smoke test OK, loss:", loss.item())
