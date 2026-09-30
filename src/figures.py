"""Paper figures, one per file, 600 dpi PNG plus vector PDF, from results/*.json only.

Every series carries a redundant non-colour encoding (marker or line style) so the figures survive a
black-and-white print.

    python src/figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

PALETTE = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100", "magenta": "#e87ba4", "violet": "#4a3aa7",
           "grey": "#52514e"}
TARGETS = ["scenario17:night", "scenario18", "scenario19", "scenario20", "scenario21"]
TGT_STYLE = {"scenario17:night": (PALETTE["violet"], "o", "17 night"), "scenario18": (PALETTE["blue"], "s", "18"),
             "scenario19": (PALETTE["aqua"], "^", "19"), "scenario20": (PALETTE["orange"], "D", "20"),
             "scenario21": (PALETTE["magenta"], "v", "21")}
METHOD_LABEL = {"norm": "Norm. statistics", "tent": "Tent", "eata": "EATA", "sar": "SAR", "cotta": "CoTTA", "t3a": "T3A",
                "selflabel": "Fine-tuning,\nself-labels"}
LINE = {"source": (PALETTE["grey"], ":", "Source (frozen)"), "norm": (PALETTE["orange"], "--", "Normalization statistics"),
        "t3a": (PALETTE["yellow"], "-.", "T3A"), "selflabel": (PALETTE["blue"], "-", "Fine-tuning, self-labels")}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
                     "lines.linewidth": 1.3, "lines.markersize": 4})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_delta_ap():
    """Change in AP against the frozen source model, per method and target: (a) sequential stream, (b) shuffled."""
    rows = [r for r in json.loads((RES / "headline.json").read_text()) if r["kind"] == "target"]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3), sharey=True, sharex=True)
    for ax, order, title, methods in ((axes[0], "sequential", "(a) sequential stream", list(METHOD_LABEL)),
                                      (axes[1], "iid", "(b) shuffled batches", list(METHOD_LABEL)[:-1])):
        for k, m in enumerate(methods):
            for j, t in enumerate(TARGETS):
                r = [x for x in rows if x["order"] == order and x["target"] == t and x["method"] == m]
                if r:
                    c, mk, lab = TGT_STYLE[t]
                    ax.plot(r[0]["d_ap"], k + (j - 2) * 0.13, mk, color=c, markersize=3.6, label=lab if k == 0 else None,
                            markerfacecolor=c if order == "sequential" else "white", markeredgewidth=0.8)
        ax.axvline(0, color=PALETTE["grey"], linewidth=0.8)
        ax.set_title(title, fontsize=8)
        ax.set_yticks(range(len(METHOD_LABEL)))
        ax.set_yticklabels(list(METHOD_LABEL.values()))
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    fig.supxlabel("change in AP against the frozen source model", fontsize=8, x=0.56, y=-0.04)
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=7, handletextpad=0.1, columnspacing=0.8, bbox_to_anchor=(0.56, -0.2),
               title="target", title_fontsize=7)
    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig_delta_ap")


def fig_stream(target="scenario19"):
    """AP in consecutive blocks of the stream on the longest target."""
    d = json.loads((RES / "stream_curve.json").read_text())
    t = d["targets"][target]
    x = np.array(t["x"]) / 1000
    fig, ax = plt.subplots(figsize=(3.5, 1.9))
    for m, (c, ls, lab) in LINE.items():
        mu, sd = np.array(t["methods"][m]["mean"]), np.array(t["methods"][m]["std"])
        ax.plot(x, mu, color=c, linestyle=ls, label=lab)
        ax.fill_between(x, mu - sd, mu + sd, color=c, alpha=0.15, linewidth=0)
    for s in t["seq_starts"][1:]:
        ax.axvline(s / 1000, color="#b5b4b0", linewidth=0.5)
    ax.set_xlabel("frames into the stream (thousands)")
    ax.set_ylabel(f"AP per {d['block'] // 1000}k frames")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.3), columnspacing=1.0)
    save(fig, "fig_stream")


if __name__ == "__main__":
    fig_delta_ap()
    fig_stream()
    print("figures written to", FIG)
