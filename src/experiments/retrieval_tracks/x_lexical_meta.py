"""Track 'lexical': meta-features and meta-models for stacking kNN evidence with text-classifier probabilities."""
from __future__ import annotations
import numpy as np

N_CLASSES = 10
VIEWS = ("w12", "c25", "wc")
PROB_KEYS = ("p_lr1", "p_lr10", "p_cnb", "d_svc", "p_mlp")
CAND_KEYS = ("cand_cnt", "cand_meansim", "cand_second", "cand_seq", "cand_lenratio", "cand_subjsim", "cand_bodysim",
             "cand_ptype", "cand_ppriority")
AUX_CTX = ("aux_ptype", "aux_ppriority")


def _votes(idx, val, y_pool, k, power):
    lab = y_pool[idx[:, :k]]
    w = np.maximum(val[:, :k], 0) ** power
    out = np.zeros((idx.shape[0], N_CLASSES))
    np.add.at(out, (np.repeat(np.arange(idx.shape[0]), k), lab.ravel()), w.ravel())
    return out / np.maximum(out.sum(1, keepdims=True), 1e-12)


def _maxsim(idx, val, y_pool):
    lab = y_pool[idx]
    out = np.zeros((idx.shape[0], N_CLASSES))
    for c in range(N_CLASSES):
        out[:, c] = np.where(lab == c, val, 0).max(1)
    return out


def knn_blocks(o, y_pool, views=VIEWS):
    """Per-view per-class blocks + per-view scalar context."""
    y_pool = np.asarray(y_pool)
    per_class, ctx, names_pc, names_ctx = [], [], [], []
    for v in views:
        idx, val = o[f"knn_{v}_idx"], o[f"knn_{v}_val"]
        ms = _maxsim(idx, val, y_pool)
        per_class += [ms, _votes(idx, val, y_pool, 20, 4), _votes(idx, val, y_pool, 50, 1)]
        names_pc += [f"{v}_maxsim", f"{v}_vote20p4", f"{v}_vote50p1"]
        srt = np.sort(ms, 1)
        nnlab = y_pool[idx[:, 0]]
        pur5 = (y_pool[idx[:, :5]] == nnlab[:, None]).mean(1)
        ctx += [val[:, 0], val[:, 1], val[:, 4], val[:, :10].mean(1), srt[:, -2], srt[:, -1] - srt[:, -2], pur5]
        names_ctx += [f"{v}_{n}" for n in ("s1", "s2", "s5", "mean10", "rival", "margin", "pur5")]
    agree = (y_pool[o["knn_w12_idx"][:, 0]] == y_pool[o["knn_c25_idx"][:, 0]]).astype(float)
    ctx += [agree, o["c25_sim_of_w12_nn"], o["w12_sim_of_c25_nn"], o["len_words"]]
    names_ctx += ["nn_agree", "c25_sim_of_w12_nn", "w12_sim_of_c25_nn", "len_words"]
    return per_class, names_pc, np.column_stack(ctx), names_ctx


def flat_features(o, y_pool, prob_keys=PROB_KEYS):
    pc, names_pc, ctx, names_ctx = knn_blocks(o, y_pool)
    blocks = pc + [o[k] for k in prob_keys if k in o]
    names = [f"{n}_{c}" for n in names_pc + [k for k in prob_keys if k in o] for c in range(N_CLASSES)]
    return np.column_stack(blocks + [ctx]).astype(np.float32), names + names_ctx


def pair_features(o, y_pool, prior, prob_keys=PROB_KEYS, cand_keys=(), aux_ctx=(), views=VIEWS, gaps=True):
    """One row per (ticket, class). Shared-parameter view: is class c the right queue for ticket i?"""
    pc, names_pc, ctx, names_ctx = knn_blocks(o, y_pool, views)
    blocks = pc + [o[k] for k in prob_keys if k in o] + [o[k] for k in cand_keys]
    names = names_pc + [k for k in prob_keys if k in o] + list(cand_keys)
    for k in aux_ctx:
        ctx = np.column_stack([ctx, o[k]]); names_ctx = names_ctx + [f"{k}_{j}" for j in range(o[k].shape[1])]
    n = ctx.shape[0]
    cols, colnames = [], []
    for b, nme in zip(blocks, names):       # value for class c, and its gap to the best other class
        b = np.asarray(b, dtype=np.float64)
        cols.append(b.ravel()); colnames.append(nme)
        if gaps:
            top = np.sort(b, 1)
            best_other = np.where(b >= top[:, [-1]], top[:, [-2]], top[:, [-1]])
            cols.append((b - best_other).ravel()); colnames.append(nme + "_gap")
    cols += [np.repeat(ctx[:, j], N_CLASSES) for j in range(ctx.shape[1])]
    colnames += names_ctx
    cols += [np.tile(np.arange(N_CLASSES), n).astype(float), np.tile(np.asarray(prior), n)]
    colnames += ["class_id", "class_prior"]
    return np.column_stack(cols).astype(np.float32), colnames
