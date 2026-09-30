"""E2: blockage prediction under shift, five targets, six label-free TTA methods and delayed self-labels.

Source: scenario 17 sequences 1-2 + first 80% of sequence 3 (day); within-domain reference: last 20% of seq 3.
Targets, streamed in time order in batches of 16 frames (predict before any feedback):
  scenario17:night (illumination), scenario18, scenario19, scenario20, scenario21 (other sessions).
Methods (src/blk/tta.py):
  source        frozen
  norm, tent, eata, sar, cotta, t3a      label-free TTA, hyper-parameters of `beamrecal`
  thr           threshold re-calibration only, from the delayed self-labels (gradient-free): F1-optimal
                threshold on the last `window` labelled frames
  selflabel     online fine-tuning on delayed self-labels: whether frame t is followed by a blockage within
                H frames is known at t + H from the received power alone (blk.data.power_blocked)
  selflabel-gt  the same with the dataset labels as feedback (upper reference for the power-derived labels)
Scoring always uses the dataset labels. Metrics: AUROC, average precision (AP), F1 at the source threshold
(or the adapted one for `thr`), overall and per sequence; per-frame scores go to results/scores/.
`--orders iid` adds the diagnostic control for the label-free methods: target frames in random order.

    python scripts/run.py --seeds 0 1 2 3 4 --tag _F5 --orders sequential iid
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from blk.data import gather, load_domain, to_gpu  # noqa: E402
from blk.metrics import auroc, avg_precision, best_threshold, f1_at  # noqa: E402
from blk.model import BlockNet  # noqa: E402
from blk.tta import LABEL_FREE, METHODS  # noqa: E402

TARGETS = ["scenario17:night", "scenario18", "scenario19", "scenario20", "scenario21"]
ALL = ["source", "norm", "tent", "eata", "sar", "cotta", "t3a", "thr", "selflabel", "selflabel-gt"]


def train(T, idx, epochs, seed, lr=1e-3, bs=128, pos_weight=4.0):
    torch.manual_seed(seed)
    model = BlockNet().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    w = torch.tensor([1.0, pos_weight], device="cuda")
    idx = torch.as_tensor(idx, device="cuda")
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = idx[torch.randperm(len(idx), generator=g).cuda()]
        for i in range(0, len(perm) - bs + 1, bs):
            b = gather(T, perm[i:i + bs])
            x = b["x"]
            # light photometric augmentation: brightness/contrast jitter, per clip
            gain = torch.empty(len(x), 1, 1, 1, device="cuda").uniform_(0.6, 1.4)
            x = (x * gain + torch.empty_like(gain).uniform_(-0.1, 0.1)).clamp(0, 1)
            loss = F.cross_entropy(model({"x": x}), b["y"], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


@torch.no_grad()
def scores(model, T, idx):
    out = []
    idx = torch.as_tensor(idx, device="cuda")
    for i in range(0, len(idx), 512):
        out.append(model(gather(T, idx[i:i + 512])).softmax(1)[:, 1].cpu())
    return torch.cat(out).numpy()


def metrics(s, thr, y):
    return {"auroc": auroc(s, y), "ap": avg_precision(s, y), "f1": f1_at_each(s, thr, y), "pos_rate": float(y.mean())}


def f1_at_each(s, thr, y):
    p, y = s >= thr, y.astype(bool)
    tp, fp, fn = (p & y).sum(), (p & ~y).sum(), (~p & y).sum()
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def stream(model, T, d, idx, method, thr, lr=1e-4, replay=256, window=2000, seed=0, iid=False, bs=16):
    """Returns (metrics dict, per-frame scores, per-frame thresholds), all in the order of `idx` (time order)."""
    torch.manual_seed(seed)
    idx = np.asarray(idx)
    order = np.random.default_rng(seed).permutation(len(idx)) if iid else np.arange(len(idx))
    sidx = idx[order]
    M = METHODS[method](model, lr=lr) if method.startswith("selflabel") else METHODS[method](model)
    label = "y" if method == "selflabel-gt" else "y_pwr"
    Hs = d["Hf"][sidx]
    fb_y = d[label]
    out_s, out_thr, lat = np.zeros(len(sidx)), np.zeros(len(sidx)), []
    buf, hist_s, hist_y = np.zeros(0, np.int64), [], []
    cur_thr, ptr = thr, 0
    for i in range(0, len(sidx), bs):
        j = torch.as_tensor(sidx[i:i + bs], device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s = M.step(gather(T, j)).softmax(1)[:, 1].detach().cpu().numpy()
        e = i + len(s)
        out_s[i:e] = s
        out_thr[i:e] = cur_thr
        # feedback: a frame's label is known once its horizon has passed
        p0 = ptr
        while ptr < e and ptr + Hs[ptr] <= e:
            ptr += 1
        if ptr > p0 and not iid:
            ready = sidx[p0:ptr]
            if method.startswith("selflabel"):
                buf = np.r_[buf, ready][-replay:]
                M.feedback(gather(T, torch.as_tensor(buf, device="cuda"), label))
            if method == "thr":
                hist_s = (hist_s + list(out_s[p0:ptr]))[-window:]
                hist_y = (hist_y + list(fb_y[ready]))[-window:]
                if sum(hist_y) >= 5:
                    cur_thr = best_threshold(np.array(hist_s), np.array(hist_y))
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000 / len(s))
    inv = np.argsort(order)
    out_s, out_thr = out_s[inv], out_thr[inv]
    y, seq = d["y"][idx], d["seq"][idx]
    r = metrics(out_s, out_thr, y)
    r["latency_ms_per_frame"] = float(np.median(lat))
    r["per_seq"] = {int(q): metrics(out_s[seq == q], out_thr[seq == q], y[seq == q]) for q in np.unique(seq)}
    return r, out_s.astype(np.float32), out_thr.astype(np.float32)


def source_models(seeds, epochs):
    """Train (or load) one source model per seed; returns {seed: (model, threshold, within-domain metrics)}."""
    src = load_domain("scenario17:day")
    valid = np.flatnonzero(src["valid"])
    seq = src["seq"][valid]
    s3 = valid[seq == 3]
    cut = s3[int(0.8 * len(s3))]
    tr = valid[(seq < 3) | ((seq == 3) & (valid < cut))]
    va = valid[(seq == 3) & (valid >= cut + 32)]
    Ts = to_gpu(src)
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    out = {}
    for seed in seeds:
        ck = ROOT / "checkpoints" / f"src_e{epochs}_s{seed}.pt"
        if ck.exists():
            model = BlockNet().cuda()
            model.load_state_dict(torch.load(ck))
            model.eval()
        else:
            t = time.time()
            model = train(Ts, tr, epochs, seed)
            torch.save(model.state_dict(), ck)
            print(f"seed {seed}: trained in {time.time() - t:.0f}s", flush=True)
        sv = scores(model, Ts, va)
        thr = best_threshold(sv, src["y"][va])
        w = {"auroc": auroc(sv, src["y"][va]), "ap": avg_precision(sv, src["y"][va]),
             "f1": f1_at(sv, src["y"][va], thr), "pos_rate": float(src["y"][va].mean())}
        print(f"seed {seed}: within-domain AUROC {w['auroc']:.3f} AP {w['ap']:.3f} F1 {w['f1']:.3f} "
              f"(pos {w['pos_rate']:.3f})", flush=True)
        out[seed] = (model, thr, w)
    del Ts
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--methods", nargs="+", default=ALL)
    ap.add_argument("--orders", nargs="+", default=["sequential"], choices=["sequential", "iid"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--ft_lr", type=float, default=1e-4, help="lr of self-label online fine-tuning")
    a = ap.parse_args()
    models = source_models(a.seeds, a.epochs)
    res = {"epochs": a.epochs, "ft_lr": a.ft_lr,
           "runs": [{"seed": s, "threshold": models[s][1], "within": models[s][2], "targets": {}} for s in a.seeds]}
    out = ROOT / "results" / f"E2{a.tag}_{int(time.time())}.json"
    (ROOT / "results" / "scores").mkdir(parents=True, exist_ok=True)
    for name in a.targets:
        d = load_domain(name)
        Tt = to_gpu(d)
        idx = np.flatnonzero(d["valid"])
        for run in res["runs"]:
            seed = run["seed"]
            model, thr, _ = models[seed]
            run["targets"][name] = {}
            keep = {}
            for order in a.orders:
                run["targets"][name][order] = {}
                for meth in a.methods:
                    if order == "iid" and meth not in ("source",) + LABEL_FREE:
                        continue
                    r, s, th = stream(model, Tt, d, idx, meth, thr, lr=a.ft_lr, seed=seed, iid=order == "iid")
                    run["targets"][name][order][meth] = r
                    keep[f"{order}/{meth}"] = s
                    if meth == "thr":
                        keep[f"{order}/thr_threshold"] = th
                    print(f"s{seed} {name:17s} {order:10s} {meth:13s} AUROC {r['auroc']:.3f}  AP {r['ap']:.3f}  "
                          f"F1 {r['f1']:.3f}  (pos {r['pos_rate']:.3f})  {r['latency_ms_per_frame']:.2f} ms", flush=True)
            np.savez_compressed(ROOT / "results" / "scores" / f"E2{a.tag}_{name.replace(':', '-')}_s{seed}.npz",
                                idx=idx, **keep)
            out.write_text(json.dumps(res, indent=1))
        del Tt, d
        gc.collect()
        torch.cuda.empty_cache()
    print("wrote", out.name)


if __name__ == "__main__":
    main()
