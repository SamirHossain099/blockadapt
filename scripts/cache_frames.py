"""Cache DeepSense blockage scenarios as low-resolution RGB frames + per-frame labels (one npz each).

    python scripts/cache_frames.py --scenarios scenario21 scenario17
    python scripts/cache_frames.py --scenarios scenario17 scenario21 --ts-only

Writes data/cache/<scenario>.npz: img (N, 3, 72, 128) uint8, block (N,) int8, seq (N,) int32,
t (N,) int32 frame index within the sequence, pwr_max (N,) float32 (max received power of the 60 GHz
vector, a continuous signal the blockage label is derived from), and data/cache/<scenario>_ts.npy:
(N,) float64 seconds since the first frame of the sequence, from the index file's UTC time stamps
(`--ts-only` writes just this file, for scenarios cached before it existed).
"""
import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

RAW = Path(os.environ.get("DEEPSENSE_ROOT", r"N:\Datasets\DeepSense 6G\extracted"))
OUT = Path(__file__).resolve().parents[1] / "data" / "cache"
HW = (72, 128)


def load_image(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB").resize(HW[::-1]), np.uint8).transpose(2, 0, 1)


def seconds(df):
    """Seconds since the start of each sequence; time stamps read "['HH-MM-SS-ms']" and wrap at midnight."""
    col = [c for c in df.columns if c.startswith("time_stamp")][0]
    p = df[col].str.strip("[]'\" ").str.split("-", expand=True).astype(float)
    s = (p[0] * 3600 + p[1] * 60 + p[2] + p[3] / 1000.0).to_numpy()
    out = np.zeros(len(df))
    for q in df["seq_index"].unique():
        m = (df["seq_index"] == q).to_numpy()
        x = s[m] - s[m][0]
        out[m] = np.where(x < -43200, x + 86400, x)
    return out


def main():
    import pandas as pd
    from multiprocessing import Pool

    from tqdm import tqdm
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--ts-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.scenarios:
        out = OUT / f"{name}.npz"
        csv = [c for c in glob.glob(str(RAW / name / "**" / "*.csv"), recursive=True) if "resources" not in c][0]
        base = Path(csv).parent
        df = pd.read_csv(csv)
        np.save(OUT / f"{name}_ts.npy", seconds(df))
        if out.exists() or a.ts_only:
            print("cached" if out.exists() else "ts only", name, flush=True)
            continue
        p = lambda x: base / str(x).lstrip("./")  # noqa: E731

        def label(x):
            return int(float(open(p(x)).read().split()[0]))

        def power(x):
            return np.nanmax(np.loadtxt(p(x)))

        labels = list(df["unit1_blockage"])
        if not all(p(x).exists() for x in labels):
            # scenario 20: the label files are numbered by row (1..N) while the index file names them by the
            # `index` column, which jumps by 6,584 at sequence 4 (checked against received power: results/label_agreement.json)
            d0 = str(Path(str(labels[0])).parent).replace("\\", "/")
            labels = [f"{d0}/label_{k}.txt" for k in range(1, len(df) + 1)]
            print(f"{name}: label files taken by row number, not by the index file's paths", flush=True)
        with ThreadPoolExecutor(a.threads) as ex:
            block = np.array(list(ex.map(label, labels)), np.int8)
            pwr = np.array(list(ex.map(power, df["unit1_pwr_60ghz"])), np.float32)
        # images in worker processes: a thread pool degraded from 500 to 11 frames/s over a long scenario
        with Pool(a.procs) as pool:
            imgs = list(tqdm(pool.imap(load_image, [str(p(x)) for x in df["unit1_rgb"]], chunksize=64),
                             total=len(df), desc=name, mininterval=30))
        seq = df["seq_index"].to_numpy(np.int32)
        t = df.groupby("seq_index").cumcount().to_numpy(np.int32)
        np.savez_compressed(out, img=np.stack(imgs), block=block, seq=seq, t=t, pwr_max=pwr)
        print(f"wrote {out.name}: {len(df)} frames, blockage rate {block.mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
