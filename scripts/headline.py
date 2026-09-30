"""Headline numbers from the final E2 result files; writes results/headline.json.

Rows: one per (order, target, method) with the mean and standard deviation over seeds of AUROC, AP and F1, the
paired difference to the frozen source model (same seed, same target) and the number of seeds in which the
method's AP is above the source's; one per (target, sequence, method) for the sequential stream; the
within-domain reference; and the agreement of the power-derived labels with the dataset labels per target.

    python scripts/headline.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))
PATTERN = "E2_F5_*.json"
LABELS = {"source": "Source (frozen)", "norm": "Normalisation statistics", "tent": "Tent", "eata": "EATA", "sar": "SAR",
          "cotta": "CoTTA", "t3a": "T3A", "thr": "Threshold re-calibration, self-labels",
          "selflabel": "Online fine-tuning, self-labels", "selflabel-gt": "Online fine-tuning, dataset labels"}
METRICS = ("auroc", "ap", "f1")


def merged():
    """Runs of all result files matching PATTERN, merged by seed (a file may hold a subset of seeds)."""
    runs = {}
    for f in sorted(R.glob(PATTERN)):
        for run in json.loads(f.read_text())["runs"]:
            if run["targets"]:
                runs.setdefault(run["seed"], {"seed": run["seed"], "within": run["within"], "targets": {}})["targets"].update(run["targets"])
    if not runs:
        raise FileNotFoundError(PATTERN)
    return [runs[s] for s in sorted(runs)]


def stat(v):
    return round(float(np.mean(v)), 3), round(float(np.std(v)), 3)


def main():
    runs = merged()
    rows = []
    # a target counts once every seed has finished it (a run file is rewritten after each seed)
    targets = [t for t in runs[0]["targets"] if all(t in x["targets"] and x["targets"][t].keys() == runs[0]["targets"][t].keys() for x in runs)]
    for tgt in targets:
        for order in runs[0]["targets"][tgt]:
            for m in runs[0]["targets"][tgt][order]:
                row = {"kind": "target", "order": order, "target": tgt, "method": m, "label": LABELS[m], "seeds": len(runs)}
                for k in METRICS:
                    v = [x["targets"][tgt][order][m][k] for x in runs]
                    s = [x["targets"][tgt][order]["source"][k] for x in runs]
                    row[k], row[k + "_std"] = stat(v)
                    row["d_" + k], row["d_" + k + "_std"] = stat(np.array(v) - np.array(s))
                    if k == "ap":
                        row["seeds_above_source"] = int((np.array(v) > np.array(s)).sum())
                row["latency_ms"] = round(float(np.median([x["targets"][tgt][order][m]["latency_ms_per_frame"] for x in runs])), 2)
                row["pos_rate"] = round(runs[0]["targets"][tgt][order][m]["pos_rate"], 4)
                rows.append(row)
        for q in runs[0]["targets"][tgt]["sequential"]["source"]["per_seq"]:
            for m in runs[0]["targets"][tgt]["sequential"]:
                row = {"kind": "sequence", "order": "sequential", "target": tgt, "seq": int(q), "method": m, "seeds": len(runs)}
                for k in METRICS:
                    row[k], row[k + "_std"] = stat([x["targets"][tgt]["sequential"][m]["per_seq"][q][k] for x in runs])
                rows.append(row)
    w = {"kind": "within", "target": "scenario17:day (within)", "method": "within", "seeds": len(runs)}
    for k in METRICS:
        w[k], w[k + "_std"] = stat([x["within"][k] for x in runs])
    rows.append(w)
    # mean over the targets of the per-seed values (each target weighs the same), then statistics over seeds
    for group, names in (("all", targets), ("sessions", [t for t in targets if t != "scenario17:night"])):
        for order in runs[0]["targets"][targets[0]]:
            for m in runs[0]["targets"][targets[0]][order]:
                row = {"kind": "mean", "group": group, "order": order, "method": m, "label": LABELS[m], "seeds": len(runs), "targets": len(names)}
                for k in METRICS:
                    v = np.array([[x["targets"][t][order][m][k] for t in names] for x in runs]).mean(1)
                    s = np.array([[x["targets"][t][order]["source"][k] for t in names] for x in runs]).mean(1)
                    row[k], row[k + "_std"] = stat(v)
                    row["d_" + k], row["d_" + k + "_std"] = stat(v - s)
                rows.append(row)
    lab = R / "label_agreement.json"
    if lab.exists():
        rows += [{"kind": "labels", **x} for x in json.loads(lab.read_text())]
    for x in rows:
        if x["kind"] == "target":
            print(f"{x['order']:10s} {x['target']:17s} {x['label']:38s} AUROC {x['auroc']:.3f}  AP {x['ap']:.3f} ± {x['ap_std']:.3f}"
                  f"  F1 {x['f1']:.3f}  dAP {x['d_ap']:+.3f} ({x['seeds_above_source']}/{x['seeds']} seeds above source)")
    for x in rows:
        if x["kind"] == "mean":
            print(f"mean/{x['group']:8s} {x['order']:10s} {x['label']:38s} AUROC {x['auroc']:.3f}  AP {x['ap']:.3f}  F1 {x['f1']:.3f}  dAP {x['d_ap']:+.3f} ± {x['d_ap_std']:.3f}")
    (R / "headline.json").write_text(json.dumps(rows, indent=1))
    print("wrote results/headline.json")


if __name__ == "__main__":
    main()
