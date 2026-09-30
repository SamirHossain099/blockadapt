"""Unit checks of the pieces the results rest on; none needs the dataset or a GPU."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from blk.data import PWR_FRAC, _future, power_blocked  # noqa: E402
from blk.metrics import auroc, avg_precision, best_threshold, f1_at  # noqa: E402


def test_future_label_looks_strictly_ahead():
    b = np.array([0, 0, 0, 1, 0, 0, 0, 0])
    y = _future(b, 2)
    # frame 1 and 2 see the blockage at frame 3 within two frames; frame 3 itself does not count its own state
    assert y.tolist() == [0, 1, 1, 0, 0, 0, 0, 0]


def test_future_label_is_zero_where_the_horizon_runs_out():
    b = np.ones(6, int)
    assert _future(b, 3).tolist() == [1, 1, 1, 0, 0, 0]


def test_power_rule_flags_a_drop_and_is_causal():
    rng = np.random.default_rng(0)
    p = 0.2 + 0.01 * rng.standard_normal(2000)
    p[1000:1005] = 0.02
    h = power_blocked(p)
    assert h[1000:1005].all() and h.sum() == 5
    # changing the future does not change the past
    q = p.copy()
    q[1500:] = 0.001
    assert (power_blocked(q)[:1500] == h[:1500]).all()


def test_power_rule_follows_a_slow_change_of_level():
    p = np.r_[np.full(1500, 0.2), np.full(1500, 0.3)]
    assert not power_blocked(p).any()
    assert PWR_FRAC == 0.4


def test_metrics_on_known_cases():
    y = np.array([0, 0, 1, 1])
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), y) == 0.0
    assert avg_precision(np.array([0.1, 0.2, 0.8, 0.9]), y) == 1.0
    assert avg_precision(np.array([0.9, 0.1, 0.8, 0.2]), y) == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert f1_at(np.array([0.1, 0.9, 0.8, 0.2]), y, 0.5) == pytest.approx(0.5)
    s = np.r_[np.linspace(0, 0.4, 50), np.linspace(0.6, 1, 50)]
    yy = np.r_[np.zeros(50, int), np.ones(50, int)]
    assert f1_at(s, yy, best_threshold(s, yy)) == 1.0


def test_label_free_methods_never_read_labels():
    """The six label-free methods take no feedback; only the self-label methods define one."""
    torch = pytest.importorskip("torch")
    from blk.tta import LABEL_FREE, METHODS, Method
    for m in LABEL_FREE:
        assert METHODS[m].feedback is Method.feedback, m
    assert METHODS["selflabel"].feedback is not Method.feedback
    assert torch is not None


def test_methods_run_on_cpu_and_keep_the_source_prediction_for_the_frozen_model():
    torch = pytest.importorskip("torch")
    from blk.model import BlockNet
    from blk.tta import METHODS
    torch.manual_seed(0)
    model = BlockNet().eval()
    b = {"x": torch.rand(16, 24, 72, 128)}
    ref = model(b).detach()
    for name, cls in METHODS.items():
        out = cls(model).step(b)
        assert out.shape == (16, 2) and torch.isfinite(out).all(), name
    assert torch.allclose(METHODS["source"](model).step(b), ref, atol=1e-5)
    # a method's adaptation never touches the model it was built from
    assert torch.allclose(model(b), ref, atol=1e-6)
