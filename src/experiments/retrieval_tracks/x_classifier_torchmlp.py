"""A PyTorch MLP on sparse TF-IDF rows. Dense batches are scattered on the device, so the GPU sees
only [batch, vocab] at a time. Supports input (feature) dropout, hidden dropout, label smoothing,
weight decay, cosine schedule, optional auxiliary heads.
"""
from __future__ import annotations

import math
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device(prefer="mps"):
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Net(nn.Module):
    def __init__(self, n_in, hidden=(1024,), n_out=10, p_h=0.5, act="relu", aux_dims=()):
        super().__init__()
        dims = [n_in, *hidden]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(hidden))])
        self.out = nn.Linear(dims[-1], n_out)
        self.aux = nn.ModuleList([nn.Linear(dims[-1], d) for d in aux_dims])
        self.p_h = p_h
        self.act = {"relu": F.relu, "gelu": F.gelu, "tanh": torch.tanh,
                    "square": lambda z: z * z, "relu2": lambda z: F.relu(z) ** 2,
                    "relu4": lambda z: F.relu(z) ** 4, "exp": lambda z: torch.exp(z.clamp(max=8.0)) - 1.0}[act]

    def features(self, x):
        for lin in self.layers:
            x = self.act(lin(x))
            x = F.dropout(x, self.p_h, self.training)
        return x

    def forward(self, x):
        return self.out(self.features(x))


def _batch_dense(X, idx, device, p_in=0.0, rng=None):
    sub = X[idx]
    coo = sub.tocoo()
    vals = coo.data.astype(np.float32)
    if p_in > 0:
        keep = rng.random(len(vals)) >= p_in
        vals = vals * keep / (1.0 - p_in)
    rows = torch.from_numpy(coo.row.astype(np.int64)).to(device)
    cols = torch.from_numpy(coo.col.astype(np.int64)).to(device)
    v = torch.from_numpy(vals).to(device)
    dense = torch.zeros((len(idx), X.shape[1]), dtype=torch.float32, device=device)
    dense[rows, cols] = v
    return dense


def train_mlp(X, y, hidden=(1024,), epochs=30, lr=1e-3, wd=0.0, p_in=0.0, p_h=0.5, ls=0.0, bs=256,
              act="relu", seed=0, device=None, aux=None, aux_w=0.3, class_weight=None, verbose=False,
              X_eval=None, y_eval=None, renorm=False):
    """aux: optional list of (targets ndarray, kind) where kind is 'ce' (int labels) or 'bce' (multi-hot float)."""
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X = sp.csr_matrix(X, dtype=np.float32)
    n = X.shape[0]
    aux = aux or []
    aux_dims = [int(t.max()) + 1 if k == "ce" else t.shape[1] for t, k in aux]
    net = Net(X.shape[1], hidden, int(max(y)) + 1, p_h, act, aux_dims).to(device)
    decay = [p for n_, p in net.named_parameters() if p.ndim > 1]
    no_decay = [p for n_, p in net.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    steps = epochs * math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1, anneal_strategy="cos")
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(device)
    aux_t = [torch.from_numpy(np.asarray(t, dtype=np.int64 if k == "ce" else np.float32)).to(device) for t, k in aux]
    cw = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32, device=device)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, bs):
            idx = np.sort(perm[s:s + bs])
            xb = _batch_dense(X, idx, device, p_in, rng)
            if renorm and p_in > 0:
                xb = F.normalize(xb, dim=1)
            it = torch.from_numpy(idx).to(device)
            h = net.features(xb)
            loss = F.cross_entropy(net.out(h), yt[it], label_smoothing=ls, weight=cw)
            for head, tt, (_, kind) in zip(net.aux, aux_t, aux):
                if kind == "ce":
                    loss = loss + aux_w * F.cross_entropy(head(h), tt[it])
                else:
                    loss = loss + aux_w * F.binary_cross_entropy_with_logits(head(h), tt[it]) * tt.shape[1] / 10
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        if verbose and (ep % 5 == 4 or ep == epochs - 1):
            msg = f"  ep {ep + 1}/{epochs} loss {tot / n:.4f} {time.time() - t0:.0f}s"
            if X_eval is not None:
                acc = (predict_proba(net, X_eval, device).argmax(1) == np.asarray(y_eval)).mean()
                msg += f" eval_acc {acc:.4f}"
            print(msg, flush=True)
    return net


@torch.no_grad()
def predict_proba(net, X, device=None, bs=512):
    device = device or next(net.parameters()).device
    net.eval()
    X = sp.csr_matrix(X, dtype=np.float32)
    out = []
    for s in range(0, X.shape[0], bs):
        xb = _batch_dense(X, np.arange(s, min(s + bs, X.shape[0])), device)
        out.append(F.softmax(net(xb), dim=1).cpu().numpy())
    return np.concatenate(out)
