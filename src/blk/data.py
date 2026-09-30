"""Blockage prediction samples from the cached scenarios (scripts/cache_frames.py).

Task: at frame t, from the last F frames (t-F+1..t), predict whether the LOS link is blocked at any frame in
(t, t+H]. The future label becomes observable from the received power H frames later, which is what a
deployed base station can use as a free, delayed self-label.

Domains (DeepSense 6G scenarios 17-21: one fixed 60 GHz link across a street, one camera at the receiver):
  scenario17:day   = scenario 17 sequences 1-3 (afternoon)           - source
  scenario17:night = scenario 17 sequence 4 (dark, brightness ~35)   - illumination shift, same session
  scenario18..21   = other recording sessions                        - session shift (and frame rate)
  scenarioNN:seqK  = one sequence of a scenario

The horizon is H_SEC seconds; H in frames is set per sequence from the measured median frame interval
(the recorded rates differ from the nominal ones of the dataset paper and change within scenario 19).

Two labels per frame:
  y      from the blockage label files of the dataset  - used for scoring, always
  y_pwr  from the received power alone (power_blocked)  - what the base station can compute itself; used as
         the delayed feedback by the self-label methods
"""
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"
F = 8             # input frames
H_SEC = 1.0       # prediction horizon in seconds
PWR_FRAC = 0.4    # blocked if power < PWR_FRAC x running median; chosen on the source domain only
PWR_WINDOW = 600  # frames of the causal running median


def power_blocked(pwr: np.ndarray, frac: float = PWR_FRAC, window: int = PWR_WINDOW) -> np.ndarray:
    """Blockage inferred from received power within one sequence: below `frac` of the median of the previous
    `window` frames. Causal (frame t uses frames before t only); the first frame is compared with itself."""
    import pandas as pd
    med = pd.Series(pwr).rolling(window, min_periods=1).median().shift(1).bfill().to_numpy()
    return pwr < frac * med


def _future(b: np.ndarray, H: int) -> np.ndarray:
    """any(b[j+1 .. j+H]) for each j (zero where the window runs past the end)."""
    c = np.r_[0, np.cumsum(b)]
    out = np.zeros(len(b), np.int64)
    j = np.arange(len(b) - H)
    out[j] = (c[j + H + 1] - c[j + 1]) > 0
    return out


def load_domain(name: str) -> dict:
    scen, _, part = name.partition(":")
    z = np.load(CACHE / f"{scen}.npz")
    d = {k: z[k] for k in z.files}
    d["ts"] = np.load(CACHE / f"{scen}_ts.npy")
    if part == "day":
        m = d["seq"] <= 3
    elif part == "night":
        m = d["seq"] == 4
    elif part.startswith("seq"):
        m = d["seq"] == int(part[3:])
    else:
        m = np.ones(len(d["block"]), bool)
    d = {k: v[m] for k, v in d.items()}
    n, seq = len(d["block"]), d["seq"]
    y, y_pwr, valid, Hf = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n, bool), np.zeros(n, np.int64)
    blk_pwr = np.zeros(n, bool)
    for q in np.unique(seq):
        idx = np.flatnonzero(seq == q)
        H = int(round(H_SEC / np.median(np.diff(d["ts"][idx]))))
        blk_pwr[idx] = power_blocked(d["pwr_max"][idx])
        y[idx] = _future(d["block"][idx].astype(np.int64), H)
        y_pwr[idx] = _future(blk_pwr[idx].astype(np.int64), H)
        valid[idx[F - 1:len(idx) - H]] = True
        Hf[idx] = H
    d.update(y=y, y_pwr=y_pwr, valid=valid, Hf=Hf, block_pwr=blk_pwr)
    return d


def to_gpu(d: dict, dev="cuda") -> dict:
    return {"img": torch.as_tensor(d["img"], device=dev), "y": torch.as_tensor(d["y"], device=dev),
            "y_pwr": torch.as_tensor(d["y_pwr"], device=dev),
            "seq": torch.as_tensor(d["seq"].astype(np.int64), device=dev)}


def gather(T: dict, idx: torch.Tensor, label: str = "y") -> dict:
    """Batch of F-frame clips ending at idx (all idx are valid, so the clip stays inside its sequence)."""
    offs = torch.arange(-F + 1, 1, device=idx.device)
    clip = T["img"][(idx[:, None] + offs[None, :]).clamp_min(0)]           # B x F x 3 x H x W uint8
    x = clip.float().div_(255.0)
    x = x.flatten(1, 2)                                                    # B x 3F x H x W
    return {"x": x, "y": T[label][idx], "seq": T["seq"][idx], "idx": idx}
