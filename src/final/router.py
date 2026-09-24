"""The router that ships. Retrieval over English and translated German tickets, plus a small stacker.

How a ticket is routed
    1. The ticket is compared against every labelled ticket in the pool five ways
       (final/features.py): word overlap (TF-IDF) against the English tickets and against
       the translated German tickets, and sentence-embedding similarity (multilingual-e5)
       against the English tickets, the translated German tickets and the original German
       text. Each comparison returns the 20 closest tickets and their queues.
    2. For each of the ten queues, each comparison records two numbers: how similar the
       closest ticket of that queue was, and how many of the 20 belonged to it.
    3. A gradient-boosted model (final/stacker.py) reads those numbers and answers one
       question per queue, "is this the right one". The ten answers are scaled to add to
       one, which is the routing probability. The highest is the prediction and its size
       is the confidence.

How the stacker is trained without cheating
    Its training rows are the training tickets themselves, produced by a 5-fold split
    inside the training set. Each fold's tickets are scored against a pool made of the
    other four folds, which is exactly the situation a new ticket is in. The stacker
    therefore learns how far to trust an English neighbour at 0.45 similarity against a
    translated German neighbour at 0.35 (translation lowers word overlap by about 0.2)
    from rows it never memorised. Evaluation tickets are never seen while fitting.

Leak guard
    Many German tickets are rewrites of English ones. Before any German ticket enters the
    pool, every one whose TF-IDF cosine to ANY evaluation text is >= 0.60, or whose e5
    cosine is >= 0.988 (the e5 value that English pairs at TF-IDF 0.60 have), is dropped.
    The guard reads evaluation texts only, never labels, and it also runs against the
    held-out fold while the stacker's training rows are built.

Inputs at prediction time are the ticket text only. The German pool lives in
data/x_german/ (built by final/translate.py and final/embed.py) and is derived from
parquet rows labelled 'de' and nothing else.

Results: 89.9% on the validation slice (core -> validation) and 91.8% on the test set,
scored once by final/test_once.py. This file was named x_german_final.py when those
numbers were produced. The code is unchanged, only the names.

    PYTHONPATH=src python src/final/router.py            # core -> validation
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet downloads stall on this machine

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from final.features import E5_GUARD, TFIDF_GUARD, German, channels, e5_embed, guard_mask
from final.lib import baseline_vec, make_mlp
from final.stacker import fit_stacker, predict_stacker

# The baseline network's probabilities can be fed to the stacker as one more input.
# An ablation by cross-validation inside core showed no gain, so it is off.
USE_MLP = False
# Folds for building the stacker's out-of-fold training rows.
FOLDS = 5
# Channel indices (see final/features.py) that never touch the German pool, used by
# use_german=False to score the English-only variant of this same stacker.
ENGLISH_ONLY = [0, 2]


def _features(x_tr, y_tr, E_tr, x_ev, E_ev, g, tfidf_guard, e5_guard, use_mlp, seed=42):
    """Retrieval features of the evaluation tickets against a pool.

    The pool is the training tickets plus every German ticket that passes the leak guard
    against the evaluation texts.

    Args:
        x_tr, y_tr, E_tr: pool tickets, their queues and their e5 embeddings
        x_ev, E_ev:       evaluation tickets and their e5 embeddings
        g:                the German pool (texts, labels, cached embeddings)
        tfidf_guard, e5_guard: leak-guard thresholds, None disables one
        use_mlp:          also fit the baseline network on the pool and return its probabilities

    Returns:
        F     [n_eval, 10 queues, 10] retrieval features (final/features.channels)
        P     [n_eval, 10] network probabilities, or None
        info  how many German tickets the guard dropped
    """
    # One vectorizer over training + German texts so both pools share a vocabulary.
    # Fitting a vectorizer uses no labels, and the evaluation texts are never included.
    vec = baseline_vec().fit(list(x_tr) + g.texts)
    A, B, G = vec.transform(x_tr), vec.transform(x_ev), vec.transform(g.texts)
    keep, info = guard_mask(g, G, B, E_ev, tfidf_guard, e5_guard)
    F = channels(A, y_tr, E_tr, B, E_ev, G, g, keep)
    P = None
    if use_mlp:
        from scipy.sparse import vstack
        X = vstack([A, G[np.where(keep)[0]]]).tocsr()
        Y = np.concatenate([y_tr, g.labels[keep]])
        P = make_mlp(seed).fit(X, Y).predict_proba(B)
    return F, P, info


def fit_predict_proba(train_texts, train_labels, eval_texts, *, use_german=True,
                      tfidf_guard=TFIDF_GUARD, e5_guard=E5_GUARD, use_mlp=USE_MLP,
                      verbose=True, return_info=False):
    """Fit the router on the training tickets and return probabilities for the evaluation tickets.

    This is the one public entry point. It uses only its arguments, the fixed pretrained
    e5 encoder, and the German pool under data/x_german/. It never reads a split by name,
    so the same call serves core -> validation and train -> test.

    Args:
        train_texts, train_labels: the labelled pool, cleaned ticket texts and queue ids 0..9
        eval_texts:                the tickets to route
        use_german:                False scores the English-only variant of the same stacker
        tfidf_guard, e5_guard:     leak-guard thresholds (see the module docstring)
        use_mlp:                   add the baseline network's probabilities as stacker inputs
        return_info:               also return the guard counts and evaluation features

    Returns:
        [n_eval, 10] probabilities, rows sum to one
    """
    t0 = time.time()
    x_tr, x_ev = list(train_texts), list(eval_texts)
    y_tr = np.asarray(train_labels)
    g = German()
    # Sentence embeddings for every training and evaluation ticket, computed here from
    # the texts (nothing cached by split), about a minute on the GPU.
    E_tr, E_ev = e5_embed(x_tr), e5_embed(x_ev)
    use = None if use_german else ENGLISH_ONLY

    # Stage 1: build the stacker's training rows out-of-fold inside the training set.
    # Each fold is scored against the other folds exactly the way the evaluation
    # tickets are scored against the whole training set below, guard included.
    F_oof = None
    P_oof = np.zeros((len(x_tr), 10), np.float32) if use_mlp else None
    guard_log = []
    for k, (tr, ho) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(x_tr, y_tr)):
        F, P, info = _features([x_tr[i] for i in tr], y_tr[tr], E_tr[tr], [x_tr[i] for i in ho], E_tr[ho],
                               g, tfidf_guard, e5_guard, use_mlp)
        if F_oof is None:
            F_oof = np.zeros((len(x_tr),) + F.shape[1:], np.float32)
        F_oof[ho] = F
        if use_mlp:
            P_oof[ho] = P
        guard_log.append(info)
        if verbose:
            print(f"  fold {k}: guard {info}  [{time.time()-t0:.0f}s]", flush=True)
    stacker = fit_stacker(F_oof, y_tr, P_oof, use)

    # Stage 2: score the evaluation tickets against the full training pool and apply
    # the stacker. This is the only place evaluation texts are used, and only as queries.
    F, P, info = _features(x_tr, y_tr, E_tr, x_ev, E_ev, g, tfidf_guard, e5_guard, use_mlp)
    probs = predict_stacker(stacker, F, P, use)
    if verbose:
        print(f"  evaluation guard: {info}  [{time.time()-t0:.0f}s]", flush=True)
    if return_info:
        return probs, {"guard_eval": info, "guard_folds": guard_log, "features_eval": F,
                       "seconds": round(time.time() - t0, 1)}
    return probs


if __name__ == "__main__":
    # The development protocol: fit on the core of the training set, score the
    # validation slice. The test set is not touched here (see final/test_once.py).
    from final.lib import load_core_val

    xc, yc, xv, yv = load_core_val()
    t0 = time.time()
    probs, info = fit_predict_proba(xc, yc, xv, return_info=True)
    pred = probs.argmax(1)
    print(f"core -> validation: accuracy {accuracy_score(yv, pred)*100:.2f}%  "
          f"macro-F1 {f1_score(yv, pred, average='macro'):.4f}  ({(time.time()-t0)/60:.1f} min)")
    print("German tickets dropped by the leak guard:", json.dumps(info["guard_eval"]))
