"""The aggregated numbers are recomputable from the released result files and carry the paper's claims."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["scenario17:night", "scenario18", "scenario19", "scenario20", "scenario21"]
FREE = ["norm", "tent", "eata", "sar", "cotta", "t3a"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows():
    return json.loads((ROOT / "results" / "headline.json").read_text())


def test_headline_recomputable_with_five_seeds_and_five_targets():
    before = _rows()
    _load("headline").main()
    assert _rows() == before
    tg = [r for r in before if r["kind"] == "target"]
    assert {r["target"] for r in tg} == set(TARGETS)
    assert all(r["seeds"] == 5 for r in tg)


def test_every_method_scored_on_every_target():
    tg = {(r["order"], r["target"], r["method"]) for r in _rows() if r["kind"] == "target"}
    for t in TARGETS:
        for m in ["source"] + FREE + ["thr", "selflabel", "selflabel-gt"]:
            assert ("sequential", t, m) in tg
        for m in ["source"] + FREE:
            assert ("iid", t, m) in tg


def test_label_free_adaptation_never_beats_the_source_in_the_mean():
    mean = {(r["group"], r["order"], r["method"]): r for r in _rows() if r["kind"] == "mean"}
    for m in FREE:
        assert mean[("all", "sequential", m)]["d_ap"] < 0


def test_self_labels_beat_the_source_on_every_target():
    tg = {(r["order"], r["target"], r["method"]): r for r in _rows() if r["kind"] == "target"}
    for t in TARGETS:
        assert tg[("sequential", t, "selflabel")]["ap"] > tg[("sequential", t, "source")]["ap"]


def test_power_rule_agrees_with_dataset_labels():
    rows = json.loads((ROOT / "results" / "label_agreement.json").read_text())
    assert len(rows) == 6
    assert min(r["frame"]["agreement"] for r in rows) > 0.99
    sweep = json.loads((ROOT / "results" / "power_fraction_sweep.json").read_text())
    assert max(sweep, key=lambda r: r["f1"])["frac"] == 0.4
