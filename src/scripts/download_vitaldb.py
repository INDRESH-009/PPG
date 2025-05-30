#!/usr/bin/env python
import argparse, pathlib, numpy as np, tqdm, sys, vitaldb

SIGS = ["PLETH", "ART"]          # request both tracks
INTERVAL = 0.002                 # 1 / 500 Hz

def main(args):
    out_root = pathlib.Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)

    case_ids = vitaldb.find_cases(SIGS)[: args.num]

    for cid in tqdm.tqdm(case_ids, desc="Cases"):
        try:
            # NEW: ask for the names too
            waves, names = vitaldb.load_case(
                cid, SIGS, sample_interval=INTERVAL, return_signals=True
            )

            # Map column indices robustly
            idx_ppg = names.index("PLETH")
            idx_abp = names.index("ART")

            ppg = waves[:, idx_ppg].astype(np.float32)
            abp = waves[:, idx_abp].astype(np.float32)

            out_dir = out_root / str(cid)
            out_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                out_dir / "signals.npz",
                ppg=ppg,
                abp=abp,
                fs=np.int16(500),
            )
        except Exception as e:
            print(f"[!] Case {cid} skipped: {e}", file=sys.stderr)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--out", required=True)
    main(p.parse_args())
