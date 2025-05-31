#!/usr/bin/env python3
# filepath: intermina.py

from src.ppg2bp_dataset import make_dataloaders

# Create DataLoaders with paired=True and a small batch size
train_loader, val_loader, test_loader = make_dataloaders(
    processed_root="data/processed",
    splits_root="data/splits",
    batch_size=4,
    num_workers=0,
    paired=True
)

# Get a sample batch from the train_loader
batch = next(iter(train_loader))

print("Batch keys:", batch.keys())
print("ppg_t shape:", batch["ppg_t"].shape)
print("sbp_t shape:", batch["sbp_t"].shape)
print("ppg_c shape:", batch["ppg_c"].shape)
print("sbp_c shape:", batch["sbp_c"].shape)

# Drop into an interactive shell with the globals available.
import code
code.interact(local=globals())