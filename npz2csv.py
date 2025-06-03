#!/usr/bin/env python3
"""
npz_to_csv.py

A simple script to load all arrays from a .npz file and save each one as a separate .csv.

Usage:
    python npz_to_csv.py --input path/to/data.npz --output-dir path/to/csv_folder

Each array inside the .npz will be written to:
    <output_dir>/<array_name>.csv

Only 1D or 2D arrays are supported.  Higher‐dimensional arrays are skipped with a warning.
"""

import numpy as np
import argparse
import os
import sys

def npz_to_csv(npz_path, output_dir):
    # Ensure input file exists
    if not os.path.isfile(npz_path):
        print(f"Error: input file '{npz_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the .npz archive
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"Error: failed to load '{npz_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # Iterate over each array stored in the .npz
    for name in data.files:
        arr = data[name]
        csv_filename = os.path.join(output_dir, f"{name}.csv")

        # Only allow 1D or 2D arrays
        if arr.ndim == 1:
            # Save a 1D array as a single-column CSV
            try:
                np.savetxt(csv_filename, arr, delimiter=",", fmt="%s")
                print(f"Wrote 1D array '{name}' → {csv_filename}")
            except Exception as e:
                print(f"Warning: could not save array '{name}' to CSV: {e}", file=sys.stderr)

        elif arr.ndim == 2:
            # Save a 2D array as a normal CSV
            try:
                np.savetxt(csv_filename, arr, delimiter=",", fmt="%s")
                print(f"Wrote 2D array '{name}' (shape {arr.shape}) → {csv_filename}")
            except Exception as e:
                print(f"Warning: could not save array '{name}' to CSV: {e}", file=sys.stderr)

        else:
            # Skip higher-dimensional arrays
            print(f"Skipping '{name}' (ndim={arr.ndim}); only 1D or 2D arrays are supported.", file=sys.stderr)

    data.close()


def main():
    parser = argparse.ArgumentParser(
        description="Convert all arrays in a .npz archive into separate .csv files."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input .npz file (e.g. data.npz)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Directory where the .csv files will be saved"
    )
    args = parser.parse_args()

    npz_to_csv(args.input, args.output_dir)


if __name__ == "__main__":
    main()
