"""Track 'lexical': candidate-level evidence. For each (ticket, class) take the most similar pool ticket of that
class (within the top-50 of the combined word+char view) and describe how much it looks like a PARAPHRASE of the
ticket: token-order alignment, length ratio, subject/body cosines, and whether the candidate's type/priority
(training-side metadata of the pool ticket) match what text-only classifiers predict for the ticket."""
from __future__ import annotations
import numpy as np
from rapidfuzz.distance import Indel
from sklearn.linear_model import LogisticRegression
from experiments.retrieval_tracks.x_lexical_views import TfidfView

N_CLASSES = 10
TYPES = ["Change", "Incident", "Problem", "Request"]
PRIOS = ["high", "low", "medium"]


def _subj(t): return t.split(" . ", 1)[0] if " . " in t else ""
def _body(t): return t.split(" . ", 1)[1] if " . " in t else t


def _rowdot(A, B):
    return np.asarray(A.multiply(B).sum(1)).ravel()


def cand_features(train_texts, y_train, eval_texts, knn_idx, knn_val, meta_train=None):
    """Returns dict name -> [n_eval, 10] arrays (0 where the class has no candidate in the top-50)."""
    y_train = np.asarray(y_train)
    n = len(eval_texts)
    lab = y_train[knn_idx]                                   # [n, 50]
    cand = np.full((n, N_CLASSES), -1, dtype=np.int64)
    cnt = np.zeros((n, N_CLASSES)); mean = np.zeros((n, N_CLASSES)); second = np.zeros((n, N_CLASSES))
    for c in range(N_CLASSES):
        m = lab == c
        has = m.any(1)
        first = m.argmax(1)
        cand[has, c] = knn_idx[np.arange(n), first][has]
        cnt[:, c] = m.sum(1)
        mean[:, c] = np.where(m, knn_val, 0).sum(1) / np.maximum(cnt[:, c], 1)
        v = np.sort(np.where(m, knn_val, 0), 1)
        second[:, c] = v[:, -2]
    out = {"cand_cnt": cnt / knn_idx.shape[1], "cand_meansim": mean, "cand_second": second}

    tok_tr = [t.split() for t in train_texts]; tok_ev = [t.split() for t in eval_texts]
    seq = np.zeros((n, N_CLASSES)); lenr = np.zeros((n, N_CLASSES))
    for i in range(n):
        a = tok_ev[i]
        for c in range(N_CLASSES):
            j = cand[i, c]
            if j < 0: continue
            b = tok_tr[j]
            seq[i, c] = Indel.normalized_similarity(a, b)
            lenr[i, c] = min(len(a), len(b)) / max(len(a), len(b), 1)
    out["cand_seq"], out["cand_lenratio"] = seq, lenr

    vs = TfidfView(ngram_range=(1, 2), min_df=1).fit([_subj(t) for t in train_texts])
    vb = TfidfView(ngram_range=(1, 2), min_df=2).fit([_body(t) for t in train_texts])
    Es, Eb = vs.transform([_subj(t) for t in eval_texts]).tocsr(), vb.transform([_body(t) for t in eval_texts]).tocsr()
    ss = np.zeros((n, N_CLASSES)); sb = np.zeros((n, N_CLASSES))
    for c in range(N_CLASSES):
        has = cand[:, c] >= 0
        r = np.where(has)[0]
        ss[r, c] = _rowdot(Es[r], vs.D[cand[r, c]]); sb[r, c] = _rowdot(Eb[r], vb.D[cand[r, c]])
    out["cand_subjsim"], out["cand_bodysim"] = ss, sb

    if meta_train is not None:
        w = TfidfView(ngram_range=(1, 2), min_df=2, max_features=200_000).fit(train_texts)
        Xe = w.transform(eval_texts)
        for name, vocab in (("type", TYPES), ("priority", PRIOS)):
            t = np.array([vocab.index(m[name]) for m in meta_train])
            clf = LogisticRegression(C=3.0, max_iter=3000).fit(w.D, t)
            P = np.zeros((n, len(vocab))); P[:, clf.classes_] = clf.predict_proba(Xe)
            f = np.zeros((n, N_CLASSES))
            for c in range(N_CLASSES):
                r = np.where(cand[:, c] >= 0)[0]
                f[r, c] = P[r, t[cand[r, c]]]
            out[f"cand_p{name}"] = f
            out[f"aux_p{name}"] = P
    return {k: v.astype(np.float32) for k, v in out.items()}
