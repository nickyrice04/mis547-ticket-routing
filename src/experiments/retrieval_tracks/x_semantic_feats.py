"""Build candidate-class features. mode 'loo' = leave-one-out inside core (for model selection + meta training);
mode 'val' = validation queries against the full core pool (for the final confirmation only)."""
import sys, time, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from experiments.retrieval_tracks.x_semantic_common import *
from experiments.retrieval_tracks.x_semantic_stack import class_feats, assemble, NC

def build(dense_names, tag, lam=5.0, with_val=True):
    ctx, cy, vtx, vy = load_core_val()
    n = len(ctx)
    vec = baseline_tfidf(); Xc = vec.fit_transform(ctx); Xv = vec.transform(vtx)
    Ec = {d: np.load(CACHE / f"emb_{d}_core.npy") for d in dense_names}
    Ev = {d: np.load(CACHE / f"emb_{d}_val.npy") for d in dense_names}
    prior = np.bincount(cy, minlength=NC) / n

    # ---- classifier probabilities: 10-fold OOF inside core, full-core fit for validation
    t0 = time.time()
    Dc = np.hstack([Ec[d] for d in dense_names]); Dv = np.hstack([Ev[d] for d in dense_names])
    oof = {"lr_tfidf": np.zeros((n, NC), np.float32), "lr_dense": np.zeros((n, NC), np.float32)}
    for tr, te in StratifiedKFold(10, shuffle=True, random_state=1).split(ctx, cy):
        oof["lr_tfidf"][te] = LogisticRegression(C=10, max_iter=200).fit(Xc[tr], cy[tr]).predict_proba(Xc[te])
        oof["lr_dense"][te] = LogisticRegression(C=10, max_iter=300).fit(Dc[tr], cy[tr]).predict_proba(Dc[te])
    print("oof clf acc", {k: round(float((v.argmax(1) == cy).mean()), 4) for k, v in oof.items()}, f"{time.time()-t0:.0f}s", flush=True)
    valp = {}
    if with_val:
        valp["lr_tfidf"] = LogisticRegression(C=10, max_iter=200).fit(Xc, cy).predict_proba(Xv).astype(np.float32)
        valp["lr_dense"] = LogisticRegression(C=10, max_iter=300).fit(Dc, cy).predict_proba(Dv).astype(np.float32)

    def feats(Xq, Eq, self_offset, nq):
        per_view = None
        for a in range(0, nq, 2000):
            b = min(a + 2000, nq)
            self_idx = np.arange(a, b) if self_offset else None
            sims = {"lex": (Xq[a:b] @ Xc.T).toarray().astype(np.float32)}
            for d in dense_names:
                sims[d] = Eq[d][a:b] @ Ec[d].T
            sims["fused"] = sims["lex"] + lam * np.mean([sims[d] for d in dense_names], 0)
            chunk = {v: class_feats(S, cy, self_idx) for v, S in sims.items()}
            if per_view is None:
                per_view = {v: ({k: [x] for k, x in cd.items()}, {k: [x] for k, x in qd.items()}) for v, (cd, qd) in chunk.items()}
            else:
                for v, (cd, qd) in chunk.items():
                    for k, x in cd.items(): per_view[v][0][k].append(x)
                    for k, x in qd.items(): per_view[v][1][k].append(x)
        return {v: ({k: np.concatenate(x) for k, x in cd.items()}, {k: np.concatenate(x) for k, x in qd.items()}) for v, (cd, qd) in per_view.items()}

    t0 = time.time()
    pv = feats(Xc, Ec, True, n)
    X, names = assemble(pv, oof, prior, n)
    np.savez_compressed(CACHE / f"feats_{tag}_loo.npz", X=X, y=cy, names=np.array(names),
                        top1={v: pv[v][1]["top1"] for v in pv}["lex"], s1_lex=pv["lex"][1]["s1"],
                        top1_fused=pv["fused"][1]["top1"])
    print("loo feats", X.shape, f"{time.time()-t0:.0f}s", flush=True)
    if with_val:
        pvv = feats(Xv, Ev, False, len(vtx))
        Xval, _ = assemble(pvv, valp, prior, len(vtx))
        np.savez_compressed(CACHE / f"feats_{tag}_val.npz", X=Xval, y=vy, names=np.array(names),
                            top1=pvv["lex"][1]["top1"], s1_lex=pvv["lex"][1]["s1"], top1_fused=pvv["fused"][1]["top1"])

if __name__ == "__main__":
    tag = sys.argv[1]; dense = sys.argv[2].split(",")
    build(dense, tag)
