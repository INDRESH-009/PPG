#!/usr/bin/env python3
"""
This script loads a signals.npz file (with 'ppg', 'abp', and 'fs'),
drops the first 30 minutes of data, downsamples to 50 Hz, reports NaN
statistics, and then saves the downsampled arrays into a new .npz.

Usage:
    python downsample_and_save.py /path/to/signals.npz
"""

import sys
import os
import numpy as np
from scipy.signal import decimate

def downsample_and_save(npz_path, drop_minutes=30, target_fs=50):
    # 1) Load .npz
    data = np.load(npz_path)
    raw_ppg = data.get("ppg", None)
    raw_abp = data.get("abp", None)
    fs_raw  = float(data.get("fs", 0.0))

    if raw_ppg is None or raw_abp is None:
        raise ValueError("The .npz must contain 'ppg' and 'abp' arrays.")
    if fs_raw != 500.0:
        raise ValueError(f"Expected fs=500, but got fs={fs_raw}.")

    total_samples = raw_ppg.shape[0]  # assume raw_ppg and raw_abp are same length

    # 2) Compute how many samples correspond to the first drop_minutes
    samples_to_drop = int(drop_minutes * 60 * fs_raw)  # 30*60*500 = 900_000

    if total_samples <= samples_to_drop:
        raise ValueError(
            f"Signal length ({total_samples} samples) is shorter than {drop_minutes} minutes."
        )

    # 3) Remove the first drop_minutes of data
    ppg_trimmed = raw_ppg[samples_to_drop:]
    abp_trimmed = raw_abp[samples_to_drop:]
    remain = ppg_trimmed.shape[0]  # number of samples left at 500 Hz

    # 4) Down-sample both signals to target_fs
    decim_factor = int(fs_raw // target_fs)  # 500//50 = 10
    if decim_factor < 1:
        raise ValueError(f"Original fs={fs_raw} < target_fs={target_fs}.")

    # Use zero-phase IIR decimation
    ppg_ds = decimate(ppg_trimmed, decim_factor, ftype='iir', zero_phase=True)
    abp_ds = decimate(abp_trimmed, decim_factor, ftype='iir', zero_phase=True)

    # 5) Count NaNs in the down-sampled arrays
    total_len = ppg_ds.size  # both ppg_ds and abp_ds have same length after decimation

    nan_count_ppg = int(np.isnan(ppg_ds).sum())
    nan_count_abp = int(np.isnan(abp_ds).sum())
    total_nans = nan_count_ppg + nan_count_abp
    total_values = total_len * 2  # PPG + ABP combined

    perc_ppg = 100.0 * nan_count_ppg / total_len
    perc_abp = 100.0 * nan_count_abp / total_len
    perc_total = 100.0 * total_nans / total_values

    # 6) Print results
    print(f"Analyzing '{os.path.basename(npz_path)}':")
    print(f"  Original sampling rate: {fs_raw:.1f} Hz")
    print(f"  Dropped first {drop_minutes} minutes → {remain} samples remain at 500 Hz")
    print(f"  Decimation factor: {decim_factor} (→ {target_fs} Hz)")
    print(f"  Down-sampled length per signal: {total_len} samples")
    print()
    print(f"  NaNs in down-sampled PPG: {nan_count_ppg} / {total_len}  ({perc_ppg:.4f} %)")
    print(f"  NaNs in down-sampled ABP: {nan_count_abp} / {total_len}  ({perc_abp:.4f} %)")
    print(f"  Combined NaNs: {total_nans} / {total_values}  ({perc_total:.4f} %)")

    # 7) Save down-sampled arrays into a new .npz
    base_dir, filename = os.path.split(npz_path)
    name, _ = os.path.splitext(filename)
    out_name = f"{name}.down{target_fs}.npz"
    out_path = os.path.join(base_dir, out_name)

    np.savez_compressed(
        out_path,
        ppg_ds=ppg_ds.astype(np.float32),
        abp_ds=abp_ds.astype(np.float32),
        fs= np.float32(target_fs)
    )
    print(f"\nSaved down-sampled data to: {out_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python downsample_and_save.py /path/to/signals.npz")
        sys.exit(1)

    npz_file = sys.argv[1]
    if not os.path.isfile(npz_file):
        print(f"Error: file '{npz_file}' not found.")
        sys.exit(1)

    try:
        downsample_and_save(npz_file)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
