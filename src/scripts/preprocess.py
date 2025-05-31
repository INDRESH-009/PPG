#python -m src.scripts/preprocess \
#  --in  data/raw_api \
#  --out data/processed



#!/usr/bin/env python3
import argparse, pathlib, numpy as np
import scipy.signal as sg, pyarrow as pa, pyarrow.parquet as pq, tqdm

# ── CONFIG ───────────────────────────────────────────────────────────────────
RAW_FS      = 500        # incoming sample rate (Hz)
TARGET_FS   = 50         # down-sample target (Hz)
SEG_SECONDS = 10
SEG_SAMPLES = TARGET_FS * SEG_SECONDS   # 500 samples per segment

# bandpass PPG between 0.5 and 8 Hz
BP_ORDER    = 3
BP_LOW      = 0.5
BP_HIGH     = 8.0

# ── HELPERS ──────────────────────────────────────────────────────────────────
def bandpass_ppg(x: np.ndarray) -> np.ndarray:
    nyq = 0.5 * RAW_FS
    b, a = sg.butter(
        BP_ORDER,
        [BP_LOW / nyq, BP_HIGH / nyq],
        btype="band"
    )
    # padlen = 3 * (max(len(b), len(a)) − 1)
    padlen = 3 * (max(b.size, a.size) - 1)
    return sg.filtfilt(b, a, x, padlen=padlen)

def downsample(x: np.ndarray) -> np.ndarray:
    # 500 → 50 Hz downsample by factor 10
    return sg.decimate(x, int(RAW_FS / TARGET_FS), ftype="fir", zero_phase=True)

def hr_ok(ppg_seg: np.ndarray) -> bool:
    # crude peak‐count HR check
    peaks, _ = sg.find_peaks(ppg_seg, distance=int(0.25 * TARGET_FS))
    hr = len(peaks) / SEG_SECONDS * 60
    return 30 <= hr <= 180

def bp_ok(sbp: float, dbp: float) -> bool:
    return (70 <= sbp <= 180) and (40 <= dbp <= 120)

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main(raw_root: pathlib.Path, out_root: pathlib.Path):
    out_root.mkdir(parents=True, exist_ok=True)

    for npz_file in tqdm.tqdm(sorted(raw_root.rglob("signals.npz")), desc="subjects"):
        subj_id = npz_file.parent.name
        data = np.load(npz_file)
        ppg_raw = data["ppg"]
        abp_raw = data["abp"]

        # mask out NaNs and extreme ABP artifacts
        mask = ~np.isnan(ppg_raw) & ~np.isnan(abp_raw) & (abp_raw > -300)
        ppg = ppg_raw[mask]
        abp = abp_raw[mask]

        # need at least 30 s of clean data
        if ppg.size < RAW_FS * 30:
            continue

        # 1) band-pass & down-sample both channels
        ppg_f  = bandpass_ppg(ppg)
        ppg_ds = downsample(ppg_f)
        abp_ds = downsample(abp)

        # 2) chop into non-overlapping 10-s segments
        n_seg = ppg_ds.size // SEG_SAMPLES
        ppg_ds = ppg_ds[:n_seg * SEG_SAMPLES].reshape(n_seg, SEG_SAMPLES)
        abp_ds = abp_ds[:n_seg * SEG_SAMPLES].reshape(n_seg, SEG_SAMPLES)

        # 3) compute SBP/DBP per segment
        sbp_arr = abp_ds.max(axis=1)
        dbp_arr = abp_ds.min(axis=1)

        # 4) quality‐gate & collect
        records = []
        for k in range(n_seg):
            seg_ppg = ppg_ds[k]
            sbp, dbp = float(sbp_arr[k]), float(dbp_arr[k])
            if not (hr_ok(seg_ppg) and bp_ok(sbp, dbp)):
                continue
            records.append({
                "subject_id": subj_id,
                "segment_id": f"{subj_id}_{k:05d}",
                "ppg":        seg_ppg.tolist(),
                "sbp":        sbp,
                "dbp":        dbp,
                "is_calib":   k == 0
            })

        # 5) write out one Parquet per subject
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
    p.add_argument("--in",  dest="raw_root", required=True)
    p.add_argument("--out", dest="out_root", required=True)
    args = p.parse_args()
    main(
        pathlib.Path(args.raw_root).expanduser(),
        pathlib.Path(args.out_root).expanduser()
    )
