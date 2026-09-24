"""The stacker: turns retrieval features (and optional network probabilities) into queue probabilities.

Rather than one ten-way classifier, every ticket becomes ten rows, one per candidate
queue, and a single binary gradient-boosted model scores each row: "is this the right
queue for this ticket?" The ten scores are then normalised per ticket. Framing it this
way lets one small model share what it learns about similarity across all queues, with
a categorical queue id column for anything queue-specific (such as how common it is).

It is trained on out-of-fold rows built inside the training set, never on evaluation tickets.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

N_CLASSES = 10


def pair_features(F, P=None, use=None):
    """Flatten [n, 10, 2*C] channel features into the (ticket, queue) pair rows the stacker reads.

    F: per ticket, per queue, per channel: best similarity and neighbour count (final/features.channels).
    P: optional [n, 10] network probabilities, added as two more columns.
    use: which channels to include, by index. None means all.

    Per channel each pair row gets four columns: the queue's best similarity, its neighbour
    count, its margin over the best OTHER queue, and the ticket's overall best similarity
    (the same for all ten rows of a ticket, it tells the model how familiar the ticket is).
    The last column is the queue id, marked categorical when the model is fitted.

    Returns X of shape [n * 10, columns], rows ordered ticket-major, queue 0..9 within a ticket.
    """
    n, _, w = F.shape
    C = w // 2
    use = list(range(C)) if use is None else use
    cols = []
    for c in use:
        best, cnt = F[:, :, 2 * c], F[:, :, 2 * c + 1]
        srt = np.sort(best, axis=1)
        top, second = srt[:, -1:], srt[:, -2:-1]
        other = np.where(best >= top, second, top)          # best similarity among the OTHER classes
        cols += [best, cnt, best - other, np.broadcast_to(top, best.shape)]
    if P is not None:
        srt = np.sort(P, axis=1)
        top, second = srt[:, -1:], srt[:, -2:-1]
        other = np.where(P >= top, second, top)
        cols += [P, P - other]
    cols.append(np.broadcast_to(np.arange(N_CLASSES, dtype=np.float32)[None, :], (n, N_CLASSES)))
    X = np.stack(cols, axis=2).reshape(n * N_CLASSES, len(cols)).astype(np.float32)
    return X


def fit_stacker(F, y, P=None, use=None, seed=0, **kw):
    """Fit the binary model. The target is 1 on the one row per ticket whose queue is the true label."""
    X = pair_features(F, P, use)
    t = (np.arange(N_CLASSES)[None, :] == np.asarray(y)[:, None]).reshape(-1).astype(int)
    args = dict(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
                l2_regularization=1.0, categorical_features=[X.shape[1] - 1], random_state=seed)
    args.update(kw)
    return HistGradientBoostingClassifier(**args).fit(X, t)


def predict_stacker(model, F, P=None, use=None):
    """Score every (ticket, queue) pair and normalise per ticket. Returns [n, 10] probabilities."""
    s = model.predict_proba(pair_features(F, P, use))[:, 1].reshape(-1, N_CLASSES)
    return s / s.sum(1, keepdims=True)
