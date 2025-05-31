#!/usr/bin/env python3
"""
Download VitalDB PPG+ABP via the HTTP API, using track‐specific string TIDs.

Usage:
    python src/scripts/download_vitaldb_api.py --num 10 --out data/raw_api
"""
import argparse, pathlib, io, sys
import requests
import numpy as np
import tqdm
import pandas as pd

API = "https://api.vitaldb.net"

def fetch_trks():
    """Fetch the master track list (CSV)."""
    r = requests.get(f"{API}/trks")
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))

def download_waveform(tid: str):
    """
    Download one waveform track by its (string) tid.
    Returns the values column as a 1D float32 array.
    """
    r = requests.get(f"{API}/{tid}")
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), header=None)
    # values are in column 1, row 1 onward
    return df.iloc[1:, 1].to_numpy(dtype=np.float32)

def main(n_cases: int, out_root: pathlib.Path):
    out_root.mkdir(parents=True, exist_ok=True)

    trks = fetch_trks()
    case_ids = sorted(trks["caseid"].unique())[:n_cases]

    pbar = tqdm.tqdm(case_ids, desc="cases")
    MIN_SAMPLES = 500 * 600  # ≥ 10 minutes @ 500 Hz
    collected = 0

    for cid in pbar:
        # isolate this case’s tracks
        sub = trks[trks.caseid == cid]
        # find exactly one PPG and one ART waveform
        pleth_rows = sub[sub.tname == "SNUADC/PLETH"]
        art_rows   = sub[sub.tname == "SNUADC/ART"]
        if pleth_rows.empty or art_rows.empty:
            pbar.write(f"[!] case {cid}: missing PPG or ABP track")
            continue

        pleth_tid = pleth_rows["tid"].iloc[0]
        art_tid   = art_rows["tid"].iloc[0]

        # download both
        try:
            ppg = download_waveform(pleth_tid)
            abp = download_waveform(art_tid)
        except Exception as e:
            pbar.write(f"[!] case {cid} download error: {e}")
            continue

        # check length
        if len(ppg) < MIN_SAMPLES or len(abp) < MIN_SAMPLES:
            pbar.write(f"[ ] case {cid}: only {len(ppg)} samples, skipping")
            continue

        # save to .npz
        d = out_root / str(cid)
        d.mkdir(exist_ok=True)
        np.savez_compressed(
            d / "signals.npz",
            ppg=ppg,
            abp=abp,
            fs=np.int16(500),
        )
        collected += 1
        pbar.write(f"✅ case {cid}: {len(ppg)} samples")
        if collected >= n_cases:
            break

    pbar.close()
    print(f"\n✅  collected {collected} usable case(s) into {out_root}")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=10,
                   help="number of cases to collect (default 10)")
    p.add_argument("--out", required=True,
                   help="output folder, e.g. data/raw_api")
    args = p.parse_args()

    main(args.num, pathlib.Path(args.out).expanduser())

