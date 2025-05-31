#!/usr/bin/env python3
"""
Create train/val/test manifests with columns:
    file,row_idx,subject_id,is_calib,sbp,dbp

Each subject’s Parquet (data/processed/<subject_id>.parquet) was written
by preprocess.py and contains rows (segments) with columns:
    ppg (list of 500 floats), sbp (float), dbp (float), is_calib (bool)

We now ensure that each subject has exactly one calibration segment.
If preprocess.py dropped the “first” segment, we will pick the smallest
row_idx as the “forced” calibration.

Usage:
    python src/scripts/split_dataset.py \
        --in        data/processed \
        --out       data/splits \
        --train_frac 0.6 \
        --val_frac   0.2 \
        --seed       42
"""

import argparse, pathlib, csv, random
import pyarrow.parquet as pq

def main(processed_root: pathlib.Path,
         splits_root:    pathlib.Path,
         train_frac:     float,
         val_frac:       float,
         seed:           int):

    processed_root = pathlib.Path(processed_root)
    splits_root    = pathlib.Path(splits_root)
    splits_root.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Read every Parquet and collect “entries” per subject ───────────── #
    # We'll build a dict: subject_id → list_of_entries.
    #
    # Each “entry” is a dict:
    #   {
    #     "file":       <parquet filename>,
    #     "row_idx":    <int>,
    #     "subject_id": <string>,
    #     "is_calib":   <0 or 1>,
    #     "sbp":        <float>,
    #     "dbp":        <float>
    #   }
    subject_to_entries = {}
    subject_ids = []

    for parquet_path in processed_root.glob("*.parquet"):
        subject_id = parquet_path.stem
        subject_ids.append(subject_id)

        # Read only columns we need:
        table = pq.read_table(
            str(parquet_path),
            columns=["sbp", "dbp", "is_calib"]
        )
        num_rows = table.num_rows
        sbp_array      = table.column("sbp")
        dbp_array      = table.column("dbp")
        is_calib_array = table.column("is_calib")

        entries = []
        for row_idx in range(num_rows):
            sbp_val     = float(sbp_array[row_idx].as_py())
            dbp_val     = float(dbp_array[row_idx].as_py())
            is_calib_val= bool(is_calib_array[row_idx].as_py())
            entries.append({
                "file":       parquet_path.name,      # e.g. “140.parquet”
                "row_idx":    row_idx,
                "subject_id": subject_id,
                "is_calib":   1 if is_calib_val else 0,
                "sbp":        sbp_val,
                "dbp":        dbp_val
            })

        subject_to_entries[subject_id] = entries

    # ── Step 2: Ensure exactly one calibration row per subject ─────────────────── #
    # If a subject has no entry with is_calib=1, pick the smallest row_idx and set is_calib=1.
    for sid, entries in subject_to_entries.items():
        # Count how many already have is_calib=1
        n_calibs = sum(e["is_calib"] for e in entries)
        if n_calibs == 0:
            # Force the row with the SMALLEST row_idx to be calibration
            # (they are already in ascending row_idx order because we appended in ascending order)
            entries[0]["is_calib"] = 1
        elif n_calibs > 1:
            # In theory preprocess.py never marks more than one,
            # but if it did, we ensure only the FIRST stays is_calib=1, zero‐out the rest.
            first_found = False
            for e in entries:
                if e["is_calib"] == 1:
                    if not first_found:
                        first_found = True
                    else:
                        e["is_calib"] = 0

    # ── Step 3: Flatten out all entries into a single list, then split by subject ─ #
    all_entries = []
    for sid in subject_ids:
        all_entries.extend(subject_to_entries[sid])

    # Shuffle subjects and split them
    random.seed(seed)
    random.shuffle(subject_ids)

    n_subj   = len(subject_ids)
    n_train  = int(train_frac * n_subj)
    n_val    = int(val_frac   * n_subj)
    train_subjs = set(subject_ids[:n_train])
    val_subjs   = set(subject_ids[n_train:n_train + n_val])
    test_subjs  = set(subject_ids[n_train + n_val:])

    # ── Step 4: Write CSV splits ────────────────────────────────────────────────── #
    def write_split(split_subjs, out_csv_path):
        with open(str(out_csv_path), "w", newline="") as fp:
            writer = csv.DictWriter(fp,
                fieldnames=["file","row_idx","subject_id","is_calib","sbp","dbp"]
            )
            writer.writeheader()
            for entry in all_entries:
                if entry["subject_id"] in split_subjs:
                    writer.writerow(entry)

    write_split(train_subjs, splits_root / "train.csv")
    write_split(val_subjs,   splits_root / "val.csv")
    write_split(test_subjs,  splits_root / "test.csv")

    print(f"Wrote {splits_root/'train.csv'}: {len(train_subjs)} subjects")
    print(f"Wrote {splits_root/'val.csv'}:   {len(val_subjs)} subjects")
    print(f"Wrote {splits_root/'test.csv'}:  {len(test_subjs)} subjects")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in",         dest="processed_root", required=True,
                        help="Folder of .parquet segments (e.g. data/processed)")
    parser.add_argument("--out",        dest="splits_root",    required=True,
                        help="Output folder for train/val/test CSVs (e.g. data/splits)")
    parser.add_argument("--train_frac", type=float, default=0.6,
                        help="Fraction of subjects to use for training")
    parser.add_argument("--val_frac",   type=float, default=0.2,
                        help="Fraction of subjects to use for validation")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    main(
        pathlib.Path(args.processed_root),
        pathlib.Path(args.splits_root),
        args.train_frac,
        args.val_frac,
        args.seed
    )
