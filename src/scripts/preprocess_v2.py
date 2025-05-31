#!/usr/bin/env python3
"""
preprocess_sliding.py

Same as the original preprocess.py but uses a 50%‐overlap sliding window
on the downsampled 50 Hz signals.  Each window is 10 s (500 samples) long,
and we step by 5 s (250 samples) each time.

Usage:
    python -m src.scripts.preprocess_sliding \
        --in  data/raw_api \
        --out data/processed
"""

import argparse
import pathlib
import numpy as np
import scipy.signal as sg
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAW_FS      = 500        # original waveform rate (Hz)
TARGET_FS   = 50         # downsample target (Hz)
SEG_SECONDS = 10
SEG_SAMPLES = TARGET_FS * SEG_SECONDS   # 500 samples per 10 s

# 50% overlap → stride = 5 s = 250 samples at 50 Hz
STRIDE_SEC  = SEG_SECONDS // 2           # 5 seconds
STRIDE_SAMPS= TARGET_FS * STRIDE_SEC     # 250 samples

# bandpass‐filter parameters for PPG
BP_ORDER    = 3
BP_LOW      = 0.5
BP_HIGH     = 8.0

# ── HELPERS ───────────────────────────────────────────────────────────────────
def bandpass_ppg(x: np.ndarray) -> np.ndarray:
    """
    Bandpass PPG between BP_LOW and BP_HIGH (Hz) using a Butterworth filter.
    """
    nyq = 0.5 * RAW_FS
    b, a = sg.butter(
        BP_ORDER,
        [BP_LOW / nyq, BP_HIGH / nyq],
        btype="band"
    )
    padlen = 3 * (max(b.size, a.size) - 1)
    return sg.filtfilt(b, a, x, padlen=padlen)

def downsample(x: np.ndarray) -> np.ndarray:
    """
    Downsample from 500 Hz → 50 Hz using a zero‐phase FIR decimator (factor=10).
    """
    return sg.decimate(x, int(RAW_FS / TARGET_FS), ftype="fir", zero_phase=True)

def hr_ok(ppg_seg: np.ndarray) -> bool:
    """
    Simple HR check on a 10 s PPG segment at 50 Hz:
      • Count peaks at least 250 ms apart → compute HR (peaks/10 s × 60 bpm).
      • Require 30 ≤ HR ≤ 180.
    """
    peaks, _ = sg.find_peaks(ppg_seg, distance=int(0.25 * TARGET_FS))
    hr = (len(peaks) / SEG_SECONDS) * 60
    return 30 <= hr <= 180

def bp_ok(sbp: float, dbp: float) -> bool:
    """
    Basic physiological sanity check for SBP/DBP.
      • 70 ≤ SBP ≤ 180
      • 40 ≤ DBP ≤ 120
      • SBP − DBP ≥ 10 is inherently true if above bounds hold.
    """
    return (70 <= sbp <= 180) and (40 <= dbp <= 120)

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(raw_root: pathlib.Path, out_root: pathlib.Path):
    out_root.mkdir(parents=True, exist_ok=True)

    # Iterate over every downloaded “signals.npz” (one per case ID)
    for npz_file in tqdm.tqdm(sorted(raw_root.rglob("signals.npz")), desc="subjects"):
        subj_id = npz_file.parent.name

        data    = np.load(npz_file)
        ppg_raw = data["ppg"]   # shape (n_samples,)
        abp_raw = data["abp"]

        # 1) Mask‐out NaN + extreme ABP artifact (ABP ≤ −300 is invalid)
        mask = (~np.isnan(ppg_raw)) & (~np.isnan(abp_raw)) & (abp_raw > -300)
        ppg = ppg_raw[mask]
        abp = abp_raw[mask]

        # 2) Require at least 30 s clean PPG (i.e. ≥ 500 × 30 = 15 000 samples at 500 Hz)
        if ppg.size < RAW_FS * 30:
            continue

        # 3) Band‐pass filter PPG, then downsample both PPG & ABP to 50 Hz
        ppg_f  = bandpass_ppg(ppg)           # still 500 Hz
        ppg_ds = downsample(ppg_f)           # now 50 Hz
        abp_ds = downsample(abp)             # now 50 Hz

        # 4) If the resulting arrays are shorter than 10 s (500 samples), skip
        if ppg_ds.size < SEG_SAMPLES or abp_ds.size < SEG_SAMPLES:
            continue

        # 5) Build a sliding‐window index list:
        #    start indices: 0, STRIDE_SAMPS, 2*STRIDE_SAMPS, ... until (len−SEG_SAMPLES).
        n_total = ppg_ds.size
        window_starts = np.arange(0, n_total - SEG_SAMPLES + 1, STRIDE_SAMPS, dtype=int)
        # window_starts is an array like [0, 250, 500, 750, ..., n_total–SEG_SAMPLES].

        records = []
        for i, start in enumerate(window_starts):
            end = start + SEG_SAMPLES
            seg_ppg = ppg_ds[start:end]    # length 500
            seg_abp = abp_ds[start:end]

            sbp = float(seg_abp.max())
            dbp = float(seg_abp.min())

            # 6) Quality gate: HR + BP must be physiologically plausible
            if not (hr_ok(seg_ppg) and bp_ok(sbp, dbp)):
                continue

            records.append({
                "subject_id": subj_id,
                "segment_id": f"{subj_id}_{i:05d}",
                "ppg":        seg_ppg.tolist(),
                "sbp":        sbp,
                "dbp":        dbp,
                # Mark the very first window (start==0) as calibration; all others False
                "is_calib":   (i == 0)
            })

        # 7) Write one Parquet file per subject (if any windows survived QC)
        if records:
            out_path = out_root / f"{subj_id}.parquet"
            pq.write_table(
                pa.Table.from_pylist(records),
                out_path,
                compression="zstd"
            )
            print(f"{subj_id}: saved {len(records)} segments")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in",   dest="raw_root", required=True,
                   help="Root folder containing downloaded signals.npz (e.g., data/raw_api)")
    p.add_argument("--out",  dest="out_root", required=True,
                   help="Destination for processed Parquet segments (e.g., data/processed)")
    args = p.parse_args()

    main(
        pathlib.Path(args.raw_root).expanduser(),
        pathlib.Path(args.out_root).expanduser()
    )
