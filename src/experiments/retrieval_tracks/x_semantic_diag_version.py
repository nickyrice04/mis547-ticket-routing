"""Diagnostic inside core: stack and 1-NN accuracy by dataset version, and the share of low-similarity tickets per version."""
import json, numpy as np
from experiments.retrieval_tracks.x_semantic_cv3 import load_clf, cy, CACHE, NC
meta = json.load(open(CACHE / "core_meta.json"))
ver = np.array([str(m["version"]) for m in meta])
s1 = np.load(CACHE / "view_lex_loo.npz")["q_s1"]; top1 = np.load(CACHE / "view_lex_loo.npz")["q_top1"]
P = np.load(CACHE / "cv2_pre4_P_all.npy"); ok = P.argmax(1) == cy
pt = load_clf("lr_tfidf", "loo")
for v in np.unique(ver):
    m = ver == v; lo = m & (s1 < 0.3)
    print(f"version {v:>5}: n={m.sum():5d} share_low={lo.sum()/m.sum():.3f} median_s1={np.median(s1[m]):.3f} | stack acc {ok[m].mean():.4f} lex1nn {(top1==cy)[m].mean():.4f} | low band: stack {ok[lo].mean():.4f} lr_tfidf {(pt.argmax(1)==cy)[lo].mean():.4f} majority {(np.bincount(cy[lo], minlength=NC).max()/max(lo.sum(),1)):.3f}")
