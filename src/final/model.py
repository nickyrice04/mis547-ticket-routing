"""The final router as a fit, save, load, predict object, so it can run behind an API.

final/router.py holds the tested code path. Its fit_predict_proba fits the router and
scores a batch of tickets in one call, which is what an experiment needs. A service needs
the two halves apart: fit once on the training job (the GPU droplet), save the result,
then load it on the inference droplet and score one ticket at a time for months.

This class is that split and nothing more. It calls the same functions router.py calls
(the out-of-fold stage is router._features itself, and prediction uses the same
channels() and predict_stacker()), so a Router fitted on core and asked for validation
predictions with the leak guard on returns exactly the probabilities that were scored.
tests/test_model_parity.py checks that on the real data.

The leak guard, and why production turns it off
    The guard exists to keep an EVALUATION honest. It drops German tickets that are
    near-copies of the tickets being scored, so a translation of a test ticket cannot
    hand the model the answer. A deployed router has no test set to protect. A new
    ticket whose German twin is already in the labelled history is exactly the case
    retrieval is for. So the evaluation recipe fits and scores with the guard on
    (89.9% on validation, 91.8% on test), and the production recipe fits with it off.

    Router().fit(x, y)                               # production recipe
    Router().fit(x, y, guard=True)                   # the evaluated recipe
    router.predict_proba(texts)                      # production scoring
    router.predict_proba(texts, guard=True)          # evaluation scoring
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold

from final.features import E5_GUARD, TFIDF_GUARD, German, channels, e5_embed, guard_mask
from final.lib import baseline_vec
from final.router import FOLDS, _features
from final.stacker import fit_stacker, predict_stacker

ARTIFACT_FORMAT = 1


@dataclass
class GermanPool:
    """The translated German tickets the router retrieves against, with the same attributes as final.features.German.

    Kept as plain arrays so a fitted Router can be saved and loaded without data/x_german/
    on disk. The inference droplet never needs the raw German files, only this.
    """
    texts: list
    labels: np.ndarray
    e5_trans: np.ndarray
    e5_orig: np.ndarray

    @classmethod
    def from_disk(cls) -> "GermanPool":
        g = German()
        return cls(g.texts, g.labels, g.e5_trans, g.e5_orig)

    @classmethod
    def empty(cls, dim: int = 768) -> "GermanPool":
        return cls([], np.zeros(0, int), np.zeros((0, dim), np.float32), np.zeros((0, dim), np.float32))


@dataclass
class Router:
    """A fitted router: the vectorizer, both labelled pools, their embeddings, and the stacker."""
    labels: list = field(default_factory=list)          # queue names, index = class id
    meta: dict = field(default_factory=dict)            # version, sizes, timings, metrics
    vec: object = None                                  # TF-IDF fitted on training + German texts
    A: object = None                                    # TF-IDF of the English training tickets
    G: object = None                                    # TF-IDF of the German pool
    y: np.ndarray = None                                # queue ids of the English training tickets
    E: np.ndarray = None                                # e5 embeddings of the English training tickets
    german: GermanPool = None
    stacker: object = None

    # ------------------------------------------------------------------------------- fit
    def fit(self, texts, labels, *, german: GermanPool | None = None, guard: bool = False,
            embed_fn=e5_embed, verbose: bool = True) -> "Router":
        """Fit on labelled English tickets plus the German pool.

        Stage 1 builds the stacker's training rows out-of-fold, exactly as router.py does,
        with the leak guard against each held-out fold when guard=True. Stage 2 fits the
        vectorizer on everything and keeps the pools for prediction.
        """
        t0 = time.time()
        log = (lambda m: print(f"  [{time.time()-t0:6.0f}s] {m}", flush=True)) if verbose else (lambda m: None)
        x = list(texts)
        y = np.asarray(labels)
        g = german if german is not None else GermanPool.from_disk()
        tg, eg = (TFIDF_GUARD, E5_GUARD) if guard else (None, None)

        E = embed_fn(x)
        log(f"embedded {len(x)} training tickets")
        F_oof = None
        guard_log = []
        for k, (tr, ho) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(x, y)):
            F, _, info = _features([x[i] for i in tr], y[tr], E[tr], [x[i] for i in ho], E[ho],
                                   g, tg, eg, False)
            if F_oof is None:
                F_oof = np.zeros((len(x),) + F.shape[1:], np.float32)
            F_oof[ho] = F
            guard_log.append(info)
            log(f"fold {k} features done, guard {info}")
        self.stacker = fit_stacker(F_oof, y)
        log("stacker fitted")

        self.vec = baseline_vec().fit(x + list(g.texts))
        self.A, self.G = self.vec.transform(x), self.vec.transform(list(g.texts))
        self.y, self.E, self.german = y, E, g
        self.meta.update({"train_rows": len(x), "german_rows": len(g.texts), "fit_guard": guard,
                          "fit_seconds": round(time.time() - t0, 1), "fold_guard": guard_log})
        return self

    # ------------------------------------------------------------------------------- predict
    def features(self, texts, *, guard: bool = False, embed_fn=e5_embed, E_ev=None):
        """Retrieval features of new tickets against the stored pools. Returns (F, guard info)."""
        texts = list(texts)
        if E_ev is None:
            E_ev = embed_fn(texts)
        B = self.vec.transform(texts)
        if guard:
            keep, info = guard_mask(self.german, self.G, B, E_ev, TFIDF_GUARD, E5_GUARD)
        else:
            keep, info = np.ones(len(self.german.texts), bool), {"dropped_total": 0}
        AT, GT = self._transposed()
        return channels(self.A, self.y, self.E, B, E_ev, self.G, self.german, keep, AT=AT, GT=GT), info

    def _transposed(self):
        """A and G transposed, built once per process. Rebuilding them on every request was the
        slowest part of serving one ticket. Kept out of the saved artifact."""
        cache = self.__dict__.get("_cache")
        if cache is None:
            cache = self.__dict__["_cache"] = (self.A.T.tocsr(), self.G.T.tocsr())
        return cache

    def predict_proba(self, texts, *, guard: bool = False, embed_fn=e5_embed) -> np.ndarray:
        """[n, 10] queue probabilities for new tickets, rows sum to one."""
        F, _ = self.features(texts, guard=guard, embed_fn=embed_fn)
        return predict_stacker(self.stacker, F)

    def route(self, texts, *, embed_fn=e5_embed) -> list[dict]:
        """Probabilities plus the numbers the API reports and the drift monitor watches.

        nearest_similarity is the highest word-overlap cosine between the ticket and any
        labelled ticket in either pool. It is the familiarity signal the whole project is
        built on: above about 0.5 the router is nearly always right, below 0.3 it is
        guessing, so a rising share of low values means new kinds of tickets are arriving.
        """
        F, _ = self.features(texts, embed_fn=embed_fn)
        P = predict_stacker(self.stacker, F)
        # channel 0 (English) and channel 1 (German) word-overlap best similarity per queue
        nearest = np.maximum(F[:, :, 0].max(1), F[:, :, 2].max(1))
        out = []
        for p, s in zip(P, nearest):
            order = np.argsort(-p)
            out.append({"probs": p, "top": [(self.labels[i], float(p[i])) for i in order[:3]],
                        "queue": self.labels[order[0]], "confidence": float(p[order[0]]),
                        "nearest_similarity": float(s)})
        return out

    # ------------------------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        joblib.dump({"format": ARTIFACT_FORMAT, **state}, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Router":
        d = joblib.load(path)
        if d.pop("format", None) != ARTIFACT_FORMAT:
            raise ValueError(f"{path} is not a router artifact of format {ARTIFACT_FORMAT}")
        r = cls()
        r.__dict__.update(d)
        return r
