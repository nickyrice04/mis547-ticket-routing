"""Diagnostic inside core: for tickets with no close relative (lexical sim < 0.3), how do the plain classifiers and the stack do, per queue?"""
import numpy as np
from experiments.retrieval_tracks.x_semantic_cv3 import load_clf, cy, CACHE, NC
from common import load_meta
L = load_meta()["labels"]
s1 = np.load(CACHE / "view_lex_loo.npz")["q_s1"]
low = s1 < 0.3
pt, pd = load_clf("lr_tfidf", "loo"), load_clf("lr_dense", "loo")
print("low band n", low.sum(), "majority", (cy[low] == np.bincount(cy[low]).argmax()).mean().round(3))
for nm, p in [("lr_tfidf", pt), ("lr_dense", pd), ("avg", (pt + pd) / 2), ("geo", np.exp(np.log(pt + 1e-9) + np.log(pd + 1e-9)))]:
    print(f"  {nm:10s} low-band acc {(p.argmax(1) == cy)[low].mean():.4f}   overall {(p.argmax(1) == cy).mean():.4f}")
P = np.load(CACHE / "cv2_pre4_P_all.npy")
print("stack pre4 low-band", (P.argmax(1) == cy)[low].mean().round(4))
print("class distribution in low band vs overall, and stack recall per class in low band")
for c in range(NC):
    m = low & (cy == c)
    print(f"  {L[c]:34s} share_low {m.sum()/low.sum():.3f} share_all {(cy==c).mean():.3f} recall_low {(P.argmax(1)[m] == c).mean():.3f}  pred_share_low {(P.argmax(1)[low] == c).mean():.3f}")
