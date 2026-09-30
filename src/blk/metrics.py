"""Binary metrics for rare-event prediction: AUROC, average precision, F1 at the source-chosen threshold."""
import numpy as np


def auroc(score, y):
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    return float((ranks[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def avg_precision(score, y):
    y = np.asarray(y).astype(bool)
    if not y.any():
        return float("nan")
    order = np.argsort(-score)
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def f1_at(score, y, thr):
    y = np.asarray(y).astype(bool)
    p = score >= thr
    tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def best_threshold(score, y):
    qs = np.quantile(score, np.linspace(0.5, 0.999, 200))
    return float(max(qs, key=lambda t: f1_at(score, y, t)))
