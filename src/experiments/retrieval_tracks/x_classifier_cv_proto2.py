"""Parametric "family detector" network, 6-fold-style CV (chosen folds).

hidden_k(x) = relu(<p_k, phi(x)>)^power   with phi(x) = the three L2-normalised sparse TF-IDF views concatenated
                                            (scaled by 1/sqrt(3) so <phi(x), phi(x')> is the mean cosine)
prototypes p_k = K medoid tickets chosen inside each class by greedy farthest-point coverage (sparse rows)
readout      = ridge least squares on the hidden activations, then optional end-to-end fine-tuning of the
               readout (and prototype rows) with cross-entropy in torch.
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold

from experiments.retrieval_tracks.x_classifier_final import VIEWS
from experiments.retrieval_tracks.x_classifier_lib import BASE_VEC, bucket_table, fmt_buckets, load_core_val, log_cv

warnings.filterwarnings("ignore")


def multiview(train_texts, eval_texts, views):
    A, B = [], []
    for name in views:
        v = TfidfVectorizer(**VIEWS[name])
        A.append(v.fit_transform(train_texts).astype(np.float32)); B.append(v.transform(eval_texts).astype(np.float32))
    s = 1.0 / np.sqrt(len(views))
    return sp.hstack(A).tocsr() * s, sp.hstack(B).tocsr() * s


def farthest_point_medoids(X, y, K, n_classes=10, seed=0):
    """Greedy coverage inside each class: repeatedly add the ticket least similar to the prototypes chosen so far."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    chosen = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        kc = int(min(len(idx), max(1, round(K * len(idx) / n))))
        Xc = X[idx]
        best = np.full(len(idx), -1.0, dtype=np.float32)
        first = rng.integers(len(idx))
        picks = [first]
        best = np.maximum(best, np.asarray((Xc @ Xc[first].T).todense()).ravel())
        for _ in range(kc - 1):
            j = int(best.argmin())
            picks.append(j)
            best = np.maximum(best, np.asarray((Xc @ Xc[j].T).todense()).ravel())
        chosen.append(idx[np.array(picks)])
    return np.concatenate(chosen)


def hidden(X, P, power):
    H = (X @ P.T).toarray().astype(np.float32)
    np.clip(H, 0, None, out=H)
    return H ** power


def ridge_readout(H, y, lam, n_classes=10):
    Y = -np.ones((H.shape[0], n_classes)); Y[np.arange(H.shape[0]), y] = 1
    H64 = H.astype(np.float64)
    A = H64.T @ H64; A[np.diag_indices_from(A)] += lam
    return sla.solve(A, H64.T @ Y, assume_a="pos")


def finetune(X, y, P, W, power, epochs=10, lr=1e-3, bs=256, temp=20.0, tune_protos=True, seed=0, verbose=False, X_eval=None, y_eval=None):
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Pt = torch.tensor(P.toarray(), device=dev, requires_grad=tune_protos)
    Wt = torch.tensor(W.astype(np.float32), device=dev, requires_grad=True)
    bt = torch.zeros(W.shape[1], device=dev, requires_grad=True)
    params = [Wt, bt] + ([Pt] if tune_protos else [])
    opt = torch.optim.Adam(params, lr=lr)
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64)).to(dev)
    n = X.shape[0]
    def fwd(idx):
        xb = torch.tensor(X[idx].toarray(), device=dev)
        h = torch.clamp(xb @ Pt.T, min=0) ** power
        return h @ Wt + bt
    for ep in range(epochs):
        perm = rng.permutation(n); tot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            loss = F.cross_entropy(temp * fwd(idx), yt[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        if verbose and X_eval is not None:
            with torch.no_grad():
                pred = np.concatenate([fwd_eval(X_eval[s:s + 1024], Pt, Wt, bt, power).argmax(1) for s in range(0, X_eval.shape[0], 1024)])
            print(f"    ft ep {ep + 1} loss {tot / n:.4f} eval {(pred == y_eval).mean():.4f}", flush=True)
    return Pt.detach().cpu().numpy(), Wt.detach().cpu().numpy(), bt.detach().cpu().numpy()


@torch.no_grad()
def fwd_eval(Xb, Pt, Wt, bt, power):
    xb = torch.tensor(Xb.toarray(), device=Pt.device)
    return ((torch.clamp(xb @ Pt.T, min=0) ** power) @ Wt + bt).cpu().numpy()


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]); folds = [int(f) for f in sys.argv[2].split(",")]
    xc, yc, _, _ = load_core_val(); xc = np.array(xc, dtype=object)
    splits = list(StratifiedKFold(n_splits=6, shuffle=True, random_state=1).split(xc, yc))
    out = []
    for fi in folds:
        tr, te = splits[fi]; t0 = time.time()
        bv = TfidfVectorizer(**BASE_VEC); a0 = bv.fit_transform(xc[tr]); b0 = bv.transform(xc[te])
        sims = (b0 @ a0.T).max(axis=1).toarray().ravel()
        A, B = multiview(list(xc[tr]), list(xc[te]), cfg.get("views", ["U", "B", "C"]))
        sel = farthest_point_medoids(A, yc[tr], cfg["K"]) if cfg["K"] < len(tr) else np.arange(len(tr))
        P = A[sel]
        Htr = hidden(A, P, cfg["power"]); Hte = hidden(B, P, cfg["power"])
        W = ridge_readout(Htr, yc[tr], cfg.get("lam", 0.01))
        pred0 = (Hte @ W).argmax(1)
        print(f"fold {fi}: K={P.shape[0]} ridge-init acc={(pred0 == yc[te]).mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
        pred = pred0
        if cfg.get("ft_epochs", 0) > 0:
            Pn, Wn, bn = finetune(A, yc[tr], P, W, cfg["power"], epochs=cfg["ft_epochs"], lr=cfg.get("lr", 1e-3), temp=cfg.get("temp", 20.0),
                                  tune_protos=cfg.get("tune_protos", True), verbose=True, X_eval=B, y_eval=yc[te])
            pred = np.concatenate([fwd_eval(B[s:s + 1024], torch.tensor(Pn), torch.tensor(Wn), torch.tensor(bn), cfg["power"]).argmax(1) for s in range(0, B.shape[0], 1024)])
            print(f"fold {fi}: fine-tuned acc={(pred == yc[te]).mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
        out.append((pred, yc[te], sims))
    P_ = np.concatenate([o[0] for o in out]); Y_ = np.concatenate([o[1] for o in out]); S_ = np.concatenate([o[2] for o in out])
    log_cv("proto2_" + cfg["name"], [(o[0] == o[1]).mean() for o in out], {"buckets": fmt_buckets(bucket_table(S_, P_, Y_)), "cfg": cfg, "cv": "6fold_seed1", "folds": folds})
