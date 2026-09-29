"""The retrieval channels and the leak guard behind final/router.py.

Everything here is a pure function of (pool texts, pool labels, evaluation texts) plus the
German files under data/x_german/, which derive only from parquet rows labelled 'de'.
Evaluation texts are used for exactly one thing: the leak guard that REMOVES German
tickets too close to them.

A "channel" is one way of comparing a ticket with a pool: which similarity (word overlap
or sentence embedding) against which pool (English, translated German, original German).
Every channel produces the same two numbers per queue, the best similarity to a ticket of
that queue and how many of the 20 nearest tickets belong to it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from common import ROOT
from final.lib import N_CLASSES, baseline_vec, topk_sims

GDIR = ROOT / "data" / "x_german"
DIAG = GDIR / "diagnostics_not_used_by_final"   # experiment caches and validation-side diagnostics
# Leak guard thresholds. 0.60 is the project-wide cut used for synthetic data against
# evaluation tickets. 0.988 is the e5 cosine that English ticket pairs with TF-IDF cosine
# 0.60 have (median, measured inside core), so the two guards bite at the same closeness.
TFIDF_GUARD = 0.60
E5_GUARD = 0.988
# Neighbours retrieved per channel.
TOPK = 20


class German:
    """The translated German pool: texts, queue ids, and cached e5 embeddings.

    Built once by final/translate.py (texts) and final/embed.py (embeddings). The
    embeddings exist for both the English translation and the original German text, so
    a ticket can be matched against either.
    """

    def __init__(self):
        rows = [json.loads(l) for l in open(GDIR / "german_translated.jsonl")]
        self.texts = [r["text"] for r in rows]
        self.labels = np.asarray([r["label"] for r in rows])
        self.e5_trans = np.load(GDIR / "e5_german_translated.npy")
        self.e5_orig = np.load(GDIR / "e5_german_original.npy")
        assert len(self.texts) == len(self.e5_trans) == len(self.e5_orig)


_ENC = None   # the e5 encoder, loaded on first use and kept for the process
_ENC_LOCK = __import__("threading").Lock()   # two threads must never load it at once


def e5_embed(texts):
    """multilingual-e5-base embeddings of a list of texts (a fixed pretrained encoder, nothing is fitted).

    With X_GERMAN_E5_CACHE=1 the result is cached under a hash of the texts so repeated experiments
    do not re-embed the same list. The cache is a pure function of the texts, holds no labels, and is
    off by default, so a normal call of the final module computes everything from its arguments."""
    import hashlib
    global _ENC
    use_cache = os.environ.get("X_GERMAN_E5_CACHE") == "1"
    # SHA-1 here only names a cache file, it protects nothing, so it is marked as not for security.
    key = hashlib.sha1(("\n".join(texts)).encode(), usedforsecurity=False).hexdigest()[:16]  # nosemgrep
    path = DIAG / "e5_cache" / (key + ".npy")
    if use_cache and path.exists():
        return np.load(path)
    from final.embed import embed, encoder
    if _ENC is None:
        with _ENC_LOCK:
            if _ENC is None:
                _ENC = encoder()
    E = embed(_ENC, list(texts))
    if use_cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, E)
    return E


def dense_max(Q, P, block=2000):
    """Highest cosine of each row of Q against any row of P (both L2-normalised), in blocks to bound memory."""
    return np.concatenate([(Q[s:s + block] @ P.T).max(1) for s in range(0, len(Q), block)])


def dense_topk(Q, P, k, block=1000):
    """Top-k cosines of each row of Q against rows of P, with the matching row indices, sorted best first."""
    sims = np.zeros((len(Q), k), np.float32)
    idx = np.zeros((len(Q), k), np.int64)
    for s in range(0, len(Q), block):
        d = Q[s:s + block] @ P.T
        j = np.argpartition(-d, kth=k - 1, axis=1)[:, :k]          # the k best, unordered
        o = np.argsort(-np.take_along_axis(d, j, 1), axis=1)       # then sorted
        j = np.take_along_axis(j, o, 1)
        idx[s:s + block] = j
        sims[s:s + block] = np.take_along_axis(d, j, 1)
    return sims, idx


def guard_mask(g: German, G_tfidf, B_tfidf, E_eval, tfidf_thr=TFIDF_GUARD, e5_thr=E5_GUARD):
    """True for German tickets that may be used. Looks at evaluation TEXTS only, never labels.

    A German ticket is dropped if its nearest evaluation ticket is at or above either
    threshold, by word overlap or by embedding. Returns the mask and the drop counts.
    """
    keep = np.ones(len(g.texts), bool)
    info = {}
    if tfidf_thr is not None:
        near = topk_sims(G_tfidf, B_tfidf, 1)[0][:, 0]
        info["dropped_tfidf"] = int((near >= tfidf_thr).sum())
        keep &= near < tfidf_thr
    if e5_thr is not None and E_eval is not None:
        near_e = np.maximum(dense_max(g.e5_trans, E_eval), 0)
        drop_e = near_e >= e5_thr
        info["dropped_e5_only"] = int((drop_e & keep).sum())
        keep &= ~drop_e
    info["dropped_total"] = int((~keep).sum())
    info["kept"] = int(keep.sum())
    return keep, info


def per_class(sims, labs, floor=0.0):
    """[n, k] neighbour sims and labels -> [n, 10, 2]: best similarity and neighbour count per class.

    A queue with no ticket among the k neighbours gets `floor` as its best similarity.
    """
    n = sims.shape[0]
    out = np.zeros((n, N_CLASSES, 2), np.float32)
    out[:, :, 0] = floor
    for c in range(N_CLASSES):
        m = labs == c
        out[:, c, 0] = np.where(m, sims, floor).max(1)
        out[:, c, 1] = m.sum(1)
    return out


def channels(A, y_tr, E_tr, B, E_ev, G, g: German, keep):
    """Five retrieval channels -> features [n_eval, 10 queues, 10].

    The last axis is five channels times (best similarity, neighbour count), in this order:
        0  word overlap -> English pool
        1  word overlap -> translated German pool
        2  e5 embedding -> English pool
        3  e5 embedding -> translated German pool
        4  e5 embedding -> original German text
    final/router.py's ENGLISH_ONLY = [0, 2] refers to these indices.

    A, B, G are TF-IDF matrices of the training, evaluation and (all) German texts from one
    vectorizer that was fitted on training + German texts only. `keep` is the leak-guard mask.

    The e5 channels use a floor of 0.8 rather than 0 because e5 cosines sit around 0.8
    even for unrelated tickets. A queue absent from the top 20 therefore gets a value just
    below any real neighbour instead of a zero on a different scale.
    """
    y_tr = np.asarray(y_tr)
    gy = g.labels[keep]
    D = G[np.where(keep)[0]]
    k = TOPK
    feats = []
    s, i = topk_sims(B, A, k); feats.append(per_class(s, y_tr[i]))                 # tfidf -> English
    s, i = topk_sims(B, D, k); feats.append(per_class(s, gy[i]))                   # tfidf -> German (translated)
    s, i = dense_topk(E_ev, E_tr, k); feats.append(per_class(s, y_tr[i], 0.8))     # e5 -> English
    s, i = dense_topk(E_ev, g.e5_trans[keep], k); feats.append(per_class(s, gy[i], 0.8))   # e5 -> German translated
    s, i = dense_topk(E_ev, g.e5_orig[keep], k); feats.append(per_class(s, gy[i], 0.8))    # e5 -> German original
    return np.concatenate(feats, axis=2)
