"""Every number quoted in the paper, computed from results/.

Writes results/manuscript_numbers.json: {name: {"value": float, "text": formatted string, "source": file}} plus the
two tables as Markdown under "table1" and "table2". The manuscript is rendered from this file, so its numbers cannot drift from results/.

    python scripts/manuscript_numbers.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = {}
TARGETS = ["scenario17:night", "scenario18", "scenario19", "scenario20", "scenario21"]
SESSIONS = TARGETS[1:]
NAME = {"scenario17:day": "17, afternoon (source)", "scenario17:night": "17, night", "scenario18": "18", "scenario19": "19",
        "scenario20": "20", "scenario21": "21"}
FREE = ["norm", "tent", "eata", "sar", "cotta", "t3a"]
# threshold re-calibration keeps the source's ranking, so its AP equals the source row and is not tabulated
ROWS = [("source", "Source, frozen"), ("norm", "Normalization statistics"), ("tent", "Tent"), ("eata", "EATA"), ("sar", "SAR"),
        ("cotta", "CoTTA"), ("t3a", "T3A"), ("selflabel", "Fine-tuning, self-labels"), ("selflabel-gt", "Fine-tuning, dataset labels")]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def put(name, value, fmt, source):
    OUT[name] = {"value": float(value), "text": fmt.format(value), "source": source}


def main():
    head = json.loads((R / "headline.json").read_text())
    lab = {r["target"]: r for r in json.loads((R / "label_agreement.json").read_text())}
    curve = json.loads((R / "stream_curve.json").read_text())
    tg = {(r["order"], r["target"], r["method"]): r for r in head if r["kind"] == "target"}
    mean = {(r["group"], r["order"], r["method"]): r for r in head if r["kind"] == "mean"}
    within = next(r for r in head if r["kind"] == "within")
    seeds = within["seeds"]
    put("seeds", seeds, "{:d}", "headline")
    put("within_ap", within["ap"], "{:.2f}", "headline")

    # domains and the power rule
    put("frames_targets", sum(lab[t]["frames"] for t in TARGETS), "{:,d}", "label_agreement")
    put("agree_frame_min", 100 * min(lab[t]["frame"]["agreement"] for t in lab), "{:.1f}", "label_agreement")
    put("agree_horizon_min", 100 * min(lab[t]["horizon"]["agreement"] for t in lab), "{:.1f}", "label_agreement")
    put("recall_min", 100 * min(lab[t]["frame"]["recall"] for t in lab), "{:.0f}", "label_agreement")
    put("precision_min", 100 * min(lab[t]["frame"]["precision"] for t in lab), "{:.0f}", "label_agreement")
    put("persist_ap_min", min(lab[t]["persistence"]["ap"] for t in TARGETS), "{:.2f}", "label_agreement")
    put("persist_ap_max", max(lab[t]["persistence"]["ap"] for t in TARGETS), "{:.2f}", "label_agreement")
    sweep = {r["frac"]: r["f1"] for r in json.loads((R / "power_fraction_sweep.json").read_text())}
    put("pwr_f1", sweep[0.4], "{:.3f}", "power_fraction_sweep")
    put("pos_min", 100 * min(lab[t]["pos_rate"] for t in lab), "{:.0f}", "label_agreement")
    put("pos_max", 100 * max(lab[t]["pos_rate"] for t in lab), "{:.0f}", "label_agreement")

    # label-free methods, sequential stream
    src_mean = mean[("all", "sequential", "source")]["ap"]
    put("src_ap_mean", src_mean, "{:.2f}", "headline")
    d = {m: mean[("all", "sequential", m)]["d_ap"] for m in FREE}
    put("free_dap_worst", -min(d.values()), "{:.2f}", "headline")
    put("free_dap_best", -max(d.values()), "{:.2f}", "headline")
    five = [m for m in FREE if m != "t3a"]
    put("five_dap_min", -max(d[m] for m in five), "{:.2f}", "headline")
    put("five_dap_max", -min(d[m] for m in five), "{:.2f}", "headline")
    put("t3a_dap", -d["t3a"], "{:.2f}", "headline")
    below = sum(tg[("sequential", t, m)]["ap"] < tg[("sequential", t, "source")]["ap"] for t in TARGETS for m in FREE)
    put("free_pairs_below", below, "{:d}", "headline")
    put("free_pairs", len(TARGETS) * len(FREE), "{:d}", "headline")
    put("five_seedpairs_above", sum(tg[("sequential", t, m)]["seeds_above_source"] for t in TARGETS for m in five), "{:d}", "headline")
    put("five_seedpairs", len(five) * len(TARGETS) * seeds, "{:d}", "headline")
    put("t3a_seedpairs_above", sum(tg[("sequential", t, "t3a")]["seeds_above_source"] for t in TARGETS), "{:d}", "headline")
    put("seedpairs", len(TARGETS) * seeds, "{:d}", "headline")
    put("tent_auroc_min", min(tg[("sequential", t, "tent")]["auroc"] for t in SESSIONS), "{:.2f}", "headline")
    put("tent_auroc_max", max(tg[("sequential", t, "tent")]["auroc"] for t in SESSIONS), "{:.2f}", "headline")

    # shuffled control
    four = ["norm", "eata", "sar", "cotta"]
    di = {m: mean[("all", "iid", m)]["d_ap"] for m in FREE}
    put("iid_four_dap_min", -max(di[m] for m in four), "{:.2f}", "headline")
    put("iid_four_dap_max", -min(di[m] for m in four), "{:.2f}", "headline")
    put("seq_four_dap_min", -max(d[m] for m in four), "{:.2f}", "headline")
    put("seq_four_dap_max", -min(d[m] for m in four), "{:.2f}", "headline")
    put("iid_tent_dap", -di["tent"], "{:.2f}", "headline")
    put("iid_t3a_dap", -di["t3a"], "{:.2f}", "headline")
    put("iid_pairs_above", sum(tg[("iid", t, m)]["ap"] > tg[("iid", t, "source")]["ap"] + 0.005 for t in TARGETS for m in FREE), "{:d}", "headline")

    # self-labels
    ft = mean[("all", "sequential", "selflabel")]
    put("ft_ap_mean", ft["ap"], "{:.2f}", "headline")
    put("ft_dap_mean", ft["d_ap"], "{:.2f}", "headline")
    put("ft_auroc_mean", ft["auroc"], "{:.2f}", "headline")
    put("src_auroc_mean", mean[("all", "sequential", "source")]["auroc"], "{:.2f}", "headline")
    dft = {t: tg[("sequential", t, "selflabel")]["d_ap"] for t in TARGETS}
    put("ft_dap_night", dft["scenario17:night"], "{:.2f}", "headline")
    put("ft_dap_sess_min", min(dft[t] for t in SESSIONS), "{:.2f}", "headline")
    put("ft_dap_sess_max", max(dft[t] for t in SESSIONS), "{:.2f}", "headline")
    put("ft_seedpairs_above", sum(tg[("sequential", t, "selflabel")]["seeds_above_source"] for t in TARGETS), "{:d}", "headline")
    put("ft_gt_gap_max", max(abs(tg[("sequential", t, "selflabel")]["ap"] - tg[("sequential", t, "selflabel-gt")]["ap"]) for t in TARGETS), "{:.3f}", "headline")
    put("ft_std_max", max(tg[("sequential", t, "selflabel")]["ap_std"] for t in SESSIONS), "{:.3f}", "headline")
    put("src_std_max", max(tg[("sequential", t, "source")]["ap_std"] for t in SESSIONS), "{:.3f}", "headline")
    put("thr_f1_src", mean[("sessions", "sequential", "source")]["f1"], "{:.2f}", "headline")
    put("thr_f1", mean[("sessions", "sequential", "thr")]["f1"], "{:.2f}", "headline")
    put("ft_f1", mean[("sessions", "sequential", "selflabel")]["f1"], "{:.2f}", "headline")
    put("thr_f1_night_src", tg[("sequential", "scenario17:night", "source")]["f1"], "{:.2f}", "headline")
    put("thr_f1_night", tg[("sequential", "scenario17:night", "thr")]["f1"], "{:.2f}", "headline")

    # along the stream
    e = curve["targets"]["scenario17:night"]["early"]
    put("early_night_src", e["source"][0], "{:.2f}", "stream_curve")
    put("early_night_ft", e["selflabel"][0], "{:.2f}", "stream_curve")
    put("early_frames", curve["first"], "{:,d}", "stream_curve")
    cross = next(i for i, (f_, s_) in enumerate(zip(e["selflabel"], e["source"])) if f_ >= s_)
    put("night_cross_frames", cross * curve["first"], "{:,d}", "stream_curve")
    put("night_cross_min", cross * curve["first"] / 12.05 / 60, "{:.0f}", "stream_curve")
    if "scenario19" in curve["targets"]:
        c = curve["targets"]["scenario19"]
        s, f = np.array(c["methods"]["source"]["mean"]), np.array(c["methods"]["selflabel"]["mean"])
        put("s19_blocks", len(s), "{:d}", "stream_curve")
        put("s19_blocks_above", int((f > s).sum()), "{:d}", "stream_curve")
        put("s19_src_min", s.min(), "{:.2f}", "stream_curve")
        put("s19_src_max", s.max(), "{:.2f}", "stream_curve")
        put("s19_ft_min", f[1:].min(), "{:.2f}", "stream_curve")
        put("s19_ft_max", f[1:].max(), "{:.2f}", "stream_curve")
    ct = curve["targets"]
    put("selfest_night", ct["scenario17:night"]["source_first_selflabel_ap"], "{:.2f}", "stream_curve")
    put("selfest_sess_min", min(ct[t]["source_first_selflabel_ap"] for t in SESSIONS), "{:.2f}", "stream_curve")
    put("selfest_sess_max", max(ct[t]["source_first_selflabel_ap"] for t in SESSIONS), "{:.2f}", "stream_curve")
    put("selfest_err_max", max(abs(ct[t]["source_first_selflabel_ap"] - ct[t]["source_first_true_ap"]) for t in TARGETS), "{:.2f}", "stream_curve")
    blocks = [(np.array(c["methods"]["selflabel"]["mean"]) > np.array(c["methods"]["source"]["mean"])) for t, c in curve["targets"].items() if t in SESSIONS]
    put("sess_blocks", sum(len(b) for b in blocks), "{:d}", "stream_curve")
    put("sess_blocks_above", sum(int(b.sum()) for b in blocks), "{:d}", "stream_curve")

    s20 = json.loads((R / "s20_index.json").read_text())
    put("s20_jump", s20["jump"], "{:,d}", "s20_index")
    put("s20_missing", s20["seq4_named_missing"], "{:,d}", "s20_index")
    put("s20_existing", s20["seq4_named_existing"], "{:,d}", "s20_index")
    put("s20_agree", 100 * s20["seq4_by_row_agreement"], "{:.1f}", "s20_index")

    # cost and sensitivity, if measured
    lat = R / "latency.json"
    if lat.exists():
        L = json.loads(lat.read_text())
        for m in ("source", "norm", "tent", "sar", "cotta", "selflabel", "thr"):
            put(f"lat_{m.replace('-', '_')}", L["ms_per_frame"][m], "{:.2f}", "latency")
        put("lat_free_max", max(L["ms_per_frame"][m] for m in FREE), "{:.2f}", "latency")
    sens = R / "lr_sensitivity.json"
    if sens.exists():
        S = json.loads(sens.read_text())
        put("lr_ap_min", min(S["mean_ap"].values()), "{:.2f}", "lr_sensitivity")
        put("lr_ap_max", max(S["mean_ap"].values()), "{:.2f}", "lr_sensitivity")
        put("lr_dev", max(abs(v - S["mean_ap"]["1e-4"]) for v in S["mean_ap"].values()), "{:.2f}", "lr_sensitivity")
        put("lr_src_ap", S["source_mean_ap"], "{:.2f}", "lr_sensitivity")

    # tables
    # four columns, so the IEEE build keeps the table inside one text column
    t1 = ["| Domain | Frames | Positive (%) | Agreement (%) |", "|---|---|---|---|"]
    for t in ["scenario17:day"] + TARGETS:
        r = lab[t]
        t1.append(f"| {NAME[t]} | {r['frames']:,d} | {100 * r['pos_rate']:.1f} | {100 * r['frame']['agreement']:.2f} |")
    put("fps_min", min(lab[t]["fps"][0] for t in lab), "{:.0f}", "label_agreement")
    put("fps_max", max(lab[t]["fps"][1] for t in lab), "{:.0f}", "label_agreement")
    put("bright_night", lab["scenario17:night"]["brightness"], "{:.0f}", "label_agreement")
    put("bright_min", min(lab[t]["brightness"] for t in lab if t != "scenario17:night"), "{:.0f}", "label_agreement")
    put("bright_max", max(lab[t]["brightness"] for t in lab if t != "scenario17:night"), "{:.0f}", "label_agreement")
    t2 = ["| Method | " + " | ".join(NAME[t] for t in TARGETS) + " | Mean |", "|---|" + "---|" * (len(TARGETS) + 1)]
    for m, label in ROWS:
        cells = [f"{tg[('sequential', t, m)]['ap']:.2f}" for t in TARGETS]
        t2.append(f"| {label} | " + " | ".join(cells) + f" | {mean[('all', 'sequential', m)]['ap']:.2f} |")
    OUT["table1"] = {"text": "\n".join(t1), "source": "label_agreement"}
    OUT["table2"] = {"text": "\n".join(t2), "source": "headline"}
    OUT["word_seeds"] = {"text": WORDS[seeds], "value": seeds, "source": "headline"}
    (R / "manuscript_numbers.json").write_text(json.dumps(OUT, indent=1))
    for k, v in OUT.items():
        if not k.startswith("table"):
            print(f"{k:24s} {v['text']}")
    print(OUT["table1"]["text"]); print(OUT["table2"]["text"])


if __name__ == "__main__":
    main()
