"""Aggregate the two side experiments: per-frame latency and the fine-tuning learning-rate sensitivity.

    python scripts/run.py --seeds 0 --targets scenario21 --tag _lat          (alone on the GPU)
    python scripts/run.py --seeds 0 1 2 --methods source selflabel --ft_lr 3e-5 --tag _lr3e-5
    python scripts/run.py --seeds 0 1 2 --methods source selflabel --ft_lr 3e-4 --tag _lr3e-4
    python scripts/extras.py      -> results/latency.json, results/lr_sensitivity.json
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def runs(pattern):
    out = {}
    for f in sorted(R.glob(pattern)):
        for run in json.loads(f.read_text())["runs"]:
            out.setdefault(run["seed"], {}).update(run["targets"])
    return out


def main():
    lat = runs("E2_lat_*.json")
    if lat:
        r = lat[0]["scenario21"]["sequential"]
        out = {"target": "scenario21", "seed": 0, "batch": 16, "ms_per_frame": {m: round(v["latency_ms_per_frame"], 3) for m, v in r.items()}}
        (R / "latency.json").write_text(json.dumps(out, indent=1))
        print("latency ms/frame:", out["ms_per_frame"])
    sens = {"3e-5": runs("E2_lr3e-5_*.json"), "1e-4": runs("E2_F5_*.json"), "3e-4": runs("E2_lr3e-4_*.json")}
    if sens["3e-5"] and sens["3e-4"]:
        seeds = sorted(set(sens["3e-5"]) & set(sens["3e-4"]) & set(sens["1e-4"]))
        targets = sorted(set.intersection(*(set(sens[k][seeds[0]]) for k in sens)))
        ap = {k: float(np.mean([[sens[k][s][t]["sequential"]["selflabel"]["ap"] for t in targets] for s in seeds])) for k in sens}
        per = {k: {t: float(np.mean([sens[k][s][t]["sequential"]["selflabel"]["ap"] for s in seeds])) for t in targets} for k in sens}
        src = float(np.mean([[sens["1e-4"][s][t]["sequential"]["source"]["ap"] for t in targets] for s in seeds]))
        out = {"seeds": seeds, "targets": targets, "mean_ap": {k: round(v, 4) for k, v in ap.items()}, "per_target": per,
               "source_mean_ap": round(src, 4)}
        (R / "lr_sensitivity.json").write_text(json.dumps(out, indent=1))
        print("lr sensitivity, mean AP over", len(targets), "targets and", len(seeds), "seeds:", out["mean_ap"], "source", out["source_mean_ap"])


if __name__ == "__main__":
    main()
