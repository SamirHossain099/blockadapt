"""Online adaptation methods for blockage prediction.

`step(b) -> logits` gives the predictions that are scored, made before the method adapts on that batch.
`feedback(fb)` is called afterwards with a batch of frames whose delayed label has just become known
(`fb["y"]`); only the self-label methods use it.

Label-free methods are the six used for beam prediction in `beamrecal` (normalisation statistics, Tent, EATA, SAR, CoTTA, T3A) with
the same hyper-parameters, for a two-class output.
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

C = 2


class EMANorm(nn.Module):
    """Normalisation with test-time EMA statistics; `momentum` is per batch of `ref_batch` frames."""
    ref_batch = 16

    def __init__(self, bn, momentum):
        super().__init__()
        self.weight, self.bias, self.eps, self.m = bn.weight, bn.bias, bn.eps, momentum
        self.register_buffer("mean", bn.running_mean.clone())
        self.register_buffer("var", bn.running_var.clone())

    def forward(self, x):
        dims = [0] + list(range(2, x.dim()))
        shape = [1, -1] + [1] * (x.dim() - 2)
        if self.m > 0:
            m = 1 - (1 - self.m) ** (x.shape[0] / self.ref_batch)
            with torch.no_grad():
                self.mean.mul_(1 - m).add_(m * x.mean(dims))
                self.var.mul_(1 - m).add_(m * x.var(dims, unbiased=False))
        return (x - self.mean.view(shape)) / torch.sqrt(self.var.view(shape) + self.eps) * self.weight.view(shape) \
            + self.bias.view(shape)


def swap_norms(model, momentum):
    for n, m in model.named_children():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            setattr(model, n, EMANorm(m, momentum))
        else:
            swap_norms(m, momentum)
    return model


def norm_params(model):
    ps = [p for m in model.modules() if isinstance(m, EMANorm) for p in (m.weight, m.bias)]
    for p in model.parameters():
        p.requires_grad_(False)
    for p in ps:
        p.requires_grad_(True)
    return ps


def entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def augment(b):
    """Photometric jitter per clip plus pixel noise (the augmentation family the source model was trained with)."""
    x = b["x"]
    gain = torch.empty(len(x), 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
    return {**b, "x": (x * gain + 0.02 * torch.randn_like(x)).clamp(0, 1)}


class Method:
    momentum = 0.0

    def __init__(self, model, momentum=None, **kw):
        self.model = swap_norms(copy.deepcopy(model), self.momentum if momentum is None else momentum).eval()

    @torch.no_grad()
    def step(self, b):
        return self.model(b)

    def feedback(self, fb):
        pass


class Source(Method):
    pass


class Norm(Method):
    momentum = 0.05


class Tent(Method):
    momentum = 0.05

    def __init__(self, model, momentum=None, lr=1e-3, **kw):
        super().__init__(model, momentum)
        self.opt = torch.optim.Adam(norm_params(self.model), lr=lr)

    def step(self, b):
        logits = self.model(b)
        self.opt.zero_grad()
        entropy(logits).mean().backward()
        self.opt.step()
        return logits.detach()


class EATA(Method):
    """Entropy-filtered, re-weighted Tent with an anchor to the source weights."""
    momentum = 0.05

    def __init__(self, model, momentum=None, lr=1e-3, e_margin=0.4, d_margin=0.05, lam=1.0, **kw):
        super().__init__(model, momentum)
        self.ps = norm_params(self.model)
        self.anchor = [p.detach().clone() for p in self.ps]
        self.opt = torch.optim.Adam(self.ps, lr=lr)
        self.e0, self.d_margin, self.lam, self.avg_p = e_margin * math.log(C), d_margin, lam, None

    def step(self, b):
        logits = self.model(b)
        ent = entropy(logits)
        keep = ent < self.e0
        p = logits.softmax(1).detach()
        if self.avg_p is not None:
            keep &= F.cosine_similarity(p, self.avg_p[None], dim=1).abs() < 1 - self.d_margin
        if keep.any():
            self.avg_p = p[keep].mean(0) if self.avg_p is None else 0.9 * self.avg_p + 0.1 * p[keep].mean(0)
            w = torch.exp(self.e0 - ent[keep].detach())
            loss = (ent[keep] * w).mean() + self.lam * sum(((q - a) ** 2).sum() for q, a in zip(self.ps, self.anchor))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        return logits.detach()


class SAR(Method):
    """Sharpness-aware reliable entropy minimisation with model reset on collapse."""
    momentum = 0.05

    def __init__(self, model, momentum=None, lr=1e-3, rho=0.05, e_margin=0.4, reset_th=0.2, **kw):
        super().__init__(model, momentum)
        self.ps = norm_params(self.model)
        self.init = copy.deepcopy(self.model.state_dict())
        self.opt = torch.optim.SGD(self.ps, lr=lr * 10, momentum=0.9)
        # reset threshold of the original (0.2 nats; used with 64 classes in `beamrecal`) rescaled to 2 classes
        self.rho, self.e0, self.reset_th, self.ema = rho, e_margin * math.log(C), reset_th * math.log(C) / math.log(64), None

    def step(self, b):
        logits = self.model(b)
        ent = entropy(logits)
        keep = ent < self.e0
        if keep.any():
            self.opt.zero_grad()
            ent[keep].mean().backward()
            grads = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in self.ps]
            norm = torch.norm(torch.stack([g.norm() for g in grads])) + 1e-12
            eps = [self.rho * g / norm for g in grads]
            with torch.no_grad():
                for p, e in zip(self.ps, eps):
                    p.add_(e)
            self.opt.zero_grad()
            ent2 = entropy(self.model(b))
            keep2 = ent2 < self.e0
            if keep2.any():
                loss2 = ent2[keep2].mean()
                loss2.backward()
                self.ema = loss2.item() if self.ema is None else 0.9 * self.ema + 0.1 * loss2.item()
            with torch.no_grad():
                for p, e in zip(self.ps, eps):
                    p.sub_(e)
            self.opt.step()
            self.opt.zero_grad()
            if self.ema is not None and self.ema < self.reset_th:
                self.model.load_state_dict(self.init)
                self.ema = None
        return logits.detach()


class CoTTA(Method):
    """Mean teacher with augmentation-averaged pseudo-labels and stochastic restore."""
    momentum = 0.05

    def __init__(self, model, momentum=None, lr=1e-4, alpha=0.999, restore_p=0.01, n_aug=4, ap=0.9, **kw):
        super().__init__(model, momentum)
        self.teacher = copy.deepcopy(self.model)
        self.source = {n: p.detach().clone() for n, p in self.model.named_parameters()}
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.alpha, self.restore_p, self.n_aug, self.ap = alpha, restore_p, n_aug, ap

    def step(self, b):
        with torch.no_grad():
            t = self.teacher(b)
            if t.softmax(1).max(1).values.mean() < self.ap:
                t = torch.stack([self.teacher(augment(b)) for _ in range(self.n_aug)]).mean(0)
        s = self.model(b)
        loss = -(t.softmax(1) * s.log_softmax(1)).sum(1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            for pt, ps in zip(self.teacher.parameters(), self.model.parameters()):
                pt.mul_(self.alpha).add_((1 - self.alpha) * ps)
            for n, p in self.model.named_parameters():
                mask = (torch.rand_like(p) < self.restore_p).float()
                p.copy_(mask * self.source[n] + (1 - mask) * p)
        return t


class T3A(Method):
    """Forward-only: replace the last linear layer with prototypes built from confident test features."""

    def __init__(self, model, momentum=None, filter_k=20, **kw):
        super().__init__(model, momentum)
        self.lin = self.model.head[-1]
        W = self.lin.weight.detach()
        self.support = F.normalize(W, dim=1)
        self.labels = torch.eye(C, device=W.device)
        self.ent = entropy(self.lin(W))
        self.k = filter_k

    @torch.no_grad()
    def step(self, b):
        z = self.model.features(b)
        logits = self.lin(z)
        self.support = torch.cat([self.support, F.normalize(z, dim=1)])
        self.labels = torch.cat([self.labels, F.one_hot(logits.argmax(1), C).float()])
        self.ent = torch.cat([self.ent, entropy(logits)])
        keep = torch.zeros(len(self.ent), dtype=torch.bool, device=z.device)
        cls = self.labels.argmax(1)
        for c in cls.unique():
            i = (cls == c).nonzero().squeeze(1)
            keep[i[self.ent[i].argsort()[: self.k]]] = True
        self.support, self.labels, self.ent = self.support[keep], self.labels[keep], self.ent[keep]
        protos = F.normalize(self.labels.T @ self.support, dim=1)
        return F.normalize(z, dim=1) @ protos.T * 10.0


class SelfLabel(Method):
    """Online fine-tuning of all weights on delayed labels: one Adam step per batch on a replay buffer of the
    most recent frames whose label has become available. Normalisation statistics stay at the source values."""

    def __init__(self, model, momentum=None, lr=1e-4, pos_weight=4.0, **kw):
        super().__init__(model, momentum)
        for p in self.model.parameters():
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.w = torch.tensor([1.0, pos_weight], device=next(self.model.parameters()).device)

    def feedback(self, fb):
        loss = F.cross_entropy(self.model(fb), fb["y"], weight=self.w)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()


METHODS = {"source": Source, "norm": Norm, "tent": Tent, "eata": EATA, "sar": SAR, "cotta": CoTTA, "t3a": T3A,
           "thr": Source, "selflabel": SelfLabel, "selflabel-gt": SelfLabel}
LABEL_FREE = ("norm", "tent", "eata", "sar", "cotta", "t3a")
