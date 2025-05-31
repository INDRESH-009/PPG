#!/usr/bin/env python3
"""
VitalDB downloader, targeting the true 500 Hz PPG + ABP waveforms.

Usage:
    python -m src.scripts.download_vitaldb --num 10 --out data/raw

This will fetch up to N cases that contain at least MIN_SECONDS of
both "SNUADC/PLETH" and "SNUADC/ART" tracks, saving each as:

    data/raw/<case-id>/signals.npz
      ├─ ppg : float32[n_samples]
      ├─ abp : float32[n_samples]
      └─ fs  : int16   (500 Hz)
"""
from __future__ import annotations
import argparse, pathlib, sys, concurrent.futures as cf

import numpy as np
import tqdm, vitaldb

# --------------------------------------------------------------------------- #
# CONFIGURATION                                                              #
# --------------------------------------------------------------------------- #
SIGS         = ["SNUADC/PLETH", "SNUADC/ART"]  # exact waveform track names
MIN_SECONDS  = 600                             # require ≥ 10 min of waveform
MIN_SAMPLES  = MIN_SECONDS * 500               # for a 500 Hz signal
THREADS      = 4
PROBE_PRINT  = 50                              # print a probe status every N IDs

# --------------------------------------------------------------------------- #
# HELPERS                                                                    #
# --------------------------------------------------------------------------- #
def _matrix(ret):
    """Unpack (array, info) if necessary."""
    return ret[0] if isinstance(ret, tuple) else ret

def long_enough(case_id: int) -> bool:
    """
    Probe the case at 1 Hz to quickly check if the two waveform
    tracks are present and >= MIN_SECONDS long.
    """
    try:
        probe = _matrix(vitaldb.load_case(case_id, SIGS, 1))
        return (
            probe.ndim == 2
            and probe.shape[1] == 2
            and probe.shape[0] >= MIN_SECONDS
        )
    except Exception:
        return False

def fetch_full(case_id: int, out_root: pathlib.Path) -> bool:
    """
    Download the full 500 Hz waveforms, validate length,
    and save to NPZ. Returns True if saved.
    """
    try:
        mat = _matrix(vitaldb.load_case(case_id, SIGS))
        if mat.ndim != 2 or mat.shape[1] != 2:
            return False

        ppg, abp = mat[:, 0], mat[:, 1]
        if ppg.shape[0] < MIN_SAMPLES:
            return False

        # write out
        out_dir = out_root / str(case_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "signals.npz",
            ppg=ppg.astype("float32"),
            abp=abp.astype("float32"),
            fs=np.int16(500),
        )
        return True

    except Exception as e:
        print(f"[!] case {case_id} skipped: {e}", file=sys.stderr)
        return False

# --------------------------------------------------------------------------- #
# MAIN                                                                      #
# --------------------------------------------------------------------------- #
def main(num_cases: int, out_root: pathlib.Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)

    ids = vitaldb.find_cases(SIGS)
    total = len(ids)
    print(f"Scanning {total} cases for {SIGS}")

    collected = 0
    probed    = 0
    bar       = tqdm.tqdm(total=num_cases, desc="downloaded")

    with cf.ThreadPoolExecutor(max_workers=THREADS) as pool:
        for cid in ids:
            if collected >= num_cases:
                break

            probed += 1
            if probed % PROBE_PRINT == 0:
                tqdm.tqdm.write(f"  probed {probed}/{total} cases, collected {collected}")

            # 1 Hz probe
            if not long_enough(cid):
                continue

            # download full waveforms
            # (we do this synchronously to avoid huge memory spikes)
            if fetch_full(cid, out_root):
                collected += 1
                bar.update(1)

    bar.close()
    print(f"\n✅  collected {collected} usable case(s) into {out_root}")

# --------------------------------------------------------------------------- #
# CLI                                                                       #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download VitalDB PPG + ABP waveform cases"
    )
    p.add_argument(
        "--num", type=int, default=10,
        help="number of cases to collect (default 10)"
    )
    p.add_argument(
        "--out", required=True,
        help="output folder for downloaded cases, e.g. data/raw"
    )
    args = p.parse_args()

    main(args.num, pathlib.Path(args.out).expanduser())
