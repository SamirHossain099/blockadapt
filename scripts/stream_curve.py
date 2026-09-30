"""Average precision along the stream, in consecutive blocks of frames, from the per-frame scores of the E2 runs.

For each target and method: AP inside each block of BLOCK valid frames (time order), per seed, then the mean and
standard deviation over seeds. Also the number of frames fine-tuning needs before its block AP first exceeds the
source's. Writes results/stream_curve.json, which src/figures.py plots.

    python scripts/stream_curve.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from blk.data import load_domain  # noqa: E402
from blk.metrics import avg_precision  # noqa: E402

TARGETS = ["scenario17:night", "scenario18", "scenario19", "scenario20", "scenario21"]
METHODS = ["source", "norm", "t3a", "selflabel"]
BLOCK = 5000
FIRST = 1000   # finer blocks at the start of the stream, for the adaptation-speed numbers
SEEDS = (0, 1, 2, 3, 4)


def main():
    out = {"block": BLOCK, "first": FIRST, "targets": {}}
    for tgt in TARGETS:
        files = [ROOT / "results" / "scores" / f"E2_F5_{tgt.replace(':', '-')}_s{s}.npz" for s in SEEDS]
        if not all(f.exists() for f in files):
            continue
        d = load_domain(tgt)
        z = [np.load(f) for f in files]
        idx = z[0]["idx"]
        y, seq = d["y"][idx], d["seq"][idx]
        edges = list(range(0, len(idx) - BLOCK + 1, BLOCK))
        t = {"frames": int(len(idx)), "x": [e + BLOCK for e in edges], "seq_starts": [int(np.flatnonzero(seq == q)[0]) for q in np.unique(seq)],
             "methods": {}, "early": {}}
        for m in METHODS:
            ap = np.array([[avg_precision(zz[f"sequential/{m}"][e:e + BLOCK], y[e:e + BLOCK]) for e in edges] for zz in z])
            t["methods"][m] = {"mean": np.round(ap.mean(0), 4).tolist(), "std": np.round(ap.std(0), 4).tolist()}
            e2 = list(range(0, min(len(idx), 10 * FIRST) - FIRST + 1, FIRST))
            ap2 = np.array([[avg_precision(zz[f"sequential/{m}"][e:e + FIRST], y[e:e + FIRST]) for e in e2] for zz in z])
            t["early"][m] = np.round(np.nanmean(ap2, 0), 4).tolist()
        out["targets"][tgt] = t
        s, f = np.array(t["methods"]["source"]["mean"]), np.array(t["methods"]["selflabel"]["mean"])
        print(f"{tgt:17s} blocks {len(edges):3d}  source {s.mean():.3f}  selflabel {f.mean():.3f}  first block: {s[0]:.3f} vs {f[0]:.3f}  "
              f"selflabel above source in {int((f > s).sum())}/{len(s)} blocks")
    (ROOT / "results" / "stream_curve.json").write_text(json.dumps(out, indent=1))
    print("wrote results/stream_curve.json")


if __name__ == "__main__":
    main()
