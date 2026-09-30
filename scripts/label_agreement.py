"""Domain table and the power-derived self-label, from the caches only (no model).

For each domain and each of its sequences: frames, measured frame rate, horizon in frames, image brightness,
blockage rate and episodes, agreement of the power rule (blk.data.power_blocked) with the dataset's blockage
labels frame by frame and for the horizon label the predictor is trained on, and a persistence baseline (the
score is the link's current power-derived state) scored like the predictors.
Also the sweep of the power fraction on the source recordings, which is where PWR_FRAC was chosen.

    python scripts/label_agreement.py      -> results/label_agreement.json, results/power_fraction_sweep.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from blk.data import PWR_FRAC, _future, load_domain, power_blocked  # noqa: E402
from blk.metrics import auroc, avg_precision, f1_at  # noqa: E402

DOMAINS = [d for d in ["scenario17:day", "scenario17:night", "scenario18", "scenario19", "scenario20", "scenario21"]
           if (ROOT / "data" / "cache" / (d.split(":")[0] + ".npz")).exists()]


def prf(h, b):
    tp = int((h & b).sum())
    return {"agreement": float((h == b).mean()), "precision": tp / max(int(h.sum()), 1), "recall": tp / max(int(b.sum()), 1)}


def episodes(b):
    e = np.flatnonzero(np.diff(np.r_[0, b.astype(int), 0]))
    return e[1::2] - e[0::2]


def describe(d, m):
    b = d["block"][m].astype(bool)
    v = d["valid"][m]
    dt = float(np.median(np.diff(d["ts"][m])))
    ep = episodes(b)
    out = {"frames": int(m.sum()), "valid": int(v.sum()), "fps": round(1 / dt, 2), "H": int(d["Hf"][m][0]),
           "duration_min": round(float(d["ts"][m][-1] - d["ts"][m][0]) / 60, 1),
           "brightness": round(float(d["img"][m][::50].mean()), 1),
           "block_rate": float(b.mean()), "episodes": int(len(ep)), "episode_median_s": round(float(np.median(ep)) * dt, 2) if len(ep) else None,
           "pos_rate": float(d["y"][m][v].mean()),
           "frame": prf(d["block_pwr"][m], b), "horizon": prf(d["y_pwr"][m][v].astype(bool), d["y"][m][v].astype(bool))}
    s, y = d["block_pwr"][m][v].astype(float), d["y"][m][v]
    out["persistence"] = {"auroc": auroc(s, y), "ap": avg_precision(s, y), "f1": f1_at(s, y, 0.5)}
    return out


def main():
    rows = []
    for name in DOMAINS:
        d = load_domain(name)
        row = {"target": name, **describe(d, np.ones(len(d["seq"]), bool)), "per_seq": {}}
        row["fps"] = [min(1 / np.median(np.diff(d["ts"][d["seq"] == q])) for q in np.unique(d["seq"])).round(2),
                      max(1 / np.median(np.diff(d["ts"][d["seq"] == q])) for q in np.unique(d["seq"])).round(2)]
        row["H"] = sorted({int(h) for h in d["Hf"]})
        for q in np.unique(d["seq"]):
            row["per_seq"][int(q)] = describe(d, d["seq"] == q)
        rows.append(row)
        print(f"{name:17s} frames {row['frames']:6d}  fps {row['fps']}  H {row['H']}  bright {row['brightness']:5.1f}  "
              f"blocked {100 * row['block_rate']:.2f}%  episodes {row['episodes']} ({row['episode_median_s']} s)  pos {100 * row['pos_rate']:.1f}%  "
              f"power rule: frame agree {row['frame']['agreement']:.4f} P {row['frame']['precision']:.3f} R {row['frame']['recall']:.3f}; "
              f"horizon agree {row['horizon']['agreement']:.4f} P {row['horizon']['precision']:.3f} R {row['horizon']['recall']:.3f}; "
              f"persistence AP {row['persistence']['ap']:.3f} F1 {row['persistence']['f1']:.3f}", flush=True)
        for q, s in row["per_seq"].items():
            print(f"     seq {q}: {s['frames']} frames, {s['fps']} fps, H {s['H']}, bright {s['brightness']}, blocked {100 * s['block_rate']:.2f}%, "
                  f"frame agree {s['frame']['agreement']:.4f} P {s['frame']['precision']:.3f} R {s['frame']['recall']:.3f}")
        if name == "scenario17:day":  # the fraction is chosen here and nowhere else
            sweep = []
            for frac in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
                h = np.zeros(len(d["seq"]), bool)
                for q in np.unique(d["seq"]):
                    m = d["seq"] == q
                    h[m] = power_blocked(d["pwr_max"][m], frac)
                r = prf(h, d["block"].astype(bool))
                r["f1"] = 2 * r["precision"] * r["recall"] / max(r["precision"] + r["recall"], 1e-9)
                sweep.append({"frac": frac, **r})
                print(f"     source sweep frac {frac}: P {r['precision']:.3f} R {r['recall']:.3f} F1 {r['f1']:.3f}")
            assert max(sweep, key=lambda r: r["f1"])["frac"] == PWR_FRAC
            (ROOT / "results" / "power_fraction_sweep.json").write_text(json.dumps(sweep, indent=1))
    (ROOT / "results" / "label_agreement.json").write_text(json.dumps(rows, indent=1, default=float))
    print("wrote results/label_agreement.json")


if __name__ == "__main__":
    main()
