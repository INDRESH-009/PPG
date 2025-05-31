# src/ppg2bp_dataset.py

import csv
import pathlib
import random

import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq


class PairedSegmentDataset(Dataset):
    """
    Each __getitem__ returns a 'calibration' segment (the row marked is_calib==1)
    and one random 'target' segment (is_calib==0) from the same subject.  Both are
    10‐second PPG + SBP/DBP.

    The manifest CSV columns must be: file,row_idx,subject_id,is_calib,sbp,dbp
      - 'file'      : name of the parquet file (e.g. "12345.parquet")
      - 'row_idx'   : row index of that segment within the parquet
      - 'subject_id': string/ID of the subject
      - 'is_calib'  : '1' if this row is designated calibration (else '0')
      - 'sbp'       : float SBP for that segment
      - 'dbp'       : float DBP for that segment
    """

    def __init__(self, manifest_csv: str, processed_root: str):
        self.processed_root = pathlib.Path(processed_root)
        self.entries = []

        # 1) Load all rows from the manifest CSV
        with open(manifest_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.entries.append({
                    "file":      row["file"],
                    "row_idx":   int(row["row_idx"]),
                    "subject":   row["subject_id"],
                    "is_calib":  bool(int(row["is_calib"])),
                    "sbp":       float(row["sbp"]),
                    "dbp":       float(row["dbp"])
                })

        # 2) Group entries by subject_id
        self.by_subject = {}
        for entry in self.entries:
            sid = entry["subject"]
            self.by_subject.setdefault(sid, []).append(entry)

        # 3) For each subject, pick exactly one calibration row
        self.calib_index = {}
        for sid, rows in self.by_subject.items():
            calib_rows = [r for r in rows if r["is_calib"]]
            if len(calib_rows) == 0:
                raise ValueError(f"No calibration segment found for subject {sid}")
            # If more than one row is marked is_calib, just pick the first.
            self.calib_index[sid] = calib_rows[0]

        # 4) Build a list of unique subject IDs
        self.subject_list = list(self.by_subject.keys())

    def __len__(self):
        # We define length = number of distinct subjects
        return len(self.subject_list)

    def __getitem__(self, idx):
        """
        Returns a dictionary with exactly these keys:
          - "ppg_t"  : torch.FloatTensor [500]   (target PPG waveform)
          - "sbp_t"  : torch.FloatTensor [1]     (target SBP scalar)
          - "dbp_t"  : torch.FloatTensor [1]     (target DBP scalar)
          - "ppg_c"  : torch.FloatTensor [500]   (calibration PPG waveform)
          - "sbp_c"  : torch.FloatTensor [1]     (calibration SBP scalar)
          - "dbp_c"  : torch.FloatTensor [1]     (calibration DBP scalar)
        """
        subject_id = self.subject_list[idx]
        rows = self.by_subject[subject_id]

        # 1) Calibration entry (we stored it above)
        calib_entry = self.calib_index[subject_id]

        # 2) Pick a random target row from the same subject,
        #    excluding the calibration row if possible.
        candidate_targets = [r for r in rows if r["row_idx"] != calib_entry["row_idx"]]
        if len(candidate_targets) == 0:
            # If there is no "other" row, fallback to using the calibration row itself.
            target_entry = calib_entry
        else:
            target_entry = random.choice(candidate_targets)

        # 3) Load the calibration waveform + numeric from the parquet file
        pq_path_c = self.processed_root / calib_entry["file"]
        table_c = pq.read_table(pq_path_c, columns=["ppg", "sbp", "dbp", "is_calib"])
        row_c = table_c.slice(calib_entry["row_idx"], 1).to_pydict()
        ppg_calib = torch.tensor(row_c["ppg"][0], dtype=torch.float32)  # [500]
        sbp_calib = torch.tensor([calib_entry["sbp"]], dtype=torch.float32)  # [1]
        dbp_calib = torch.tensor([calib_entry["dbp"]], dtype=torch.float32)  # [1]

        # 4) Load the target waveform + numeric
        pq_path_t = self.processed_root / target_entry["file"]
        table_t = pq.read_table(pq_path_t, columns=["ppg", "sbp", "dbp", "is_calib"])
        row_t = table_t.slice(target_entry["row_idx"], 1).to_pydict()
        ppg_target = torch.tensor(row_t["ppg"][0], dtype=torch.float32)  # [500]
        sbp_target = torch.tensor([target_entry["sbp"]], dtype=torch.float32)  # [1]
        dbp_target = torch.tensor([target_entry["dbp"]], dtype=torch.float32)  # [1]

        return {
            "ppg_t":   ppg_target,   # [500]
            "sbp_t":   sbp_target,   # [1]
            "dbp_t":   dbp_target,   # [1]
            "ppg_c":   ppg_calib,    # [500]
            "sbp_c":   sbp_calib,    # [1]
            "dbp_c":   dbp_calib     # [1]
        }


def make_dataloaders(
    processed_root: str,
    splits_root: str,
    batch_size: int = 8,
    num_workers: int = 0,
    paired: bool = False
):
    """
    Returns (train_loader, val_loader, test_loader).  If `paired=True`, uses PairedSegmentDataset;
    otherwise it will raise NotImplementedError (for now).

    Each CSV must have columns:
        file,row_idx,subject_id,is_calib,sbp,dbp

    Example usage for paired:
        train_loader, val_loader, test_loader = make_dataloaders(
            processed_root="data/processed",
            splits_root="data/splits",
            batch_size=32,
            num_workers=2,
            paired=True
        )
    """
    if paired:
        train_ds = PairedSegmentDataset(f"{splits_root}/train.csv", processed_root)
        val_ds   = PairedSegmentDataset(f"{splits_root}/val.csv",   processed_root)
        test_ds  = PairedSegmentDataset(f"{splits_root}/test.csv",  processed_root)
    else:
        raise NotImplementedError("Only paired=True is currently supported for PairedSegmentDataset")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
