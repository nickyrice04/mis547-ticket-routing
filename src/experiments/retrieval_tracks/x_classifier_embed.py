"""Embed core tickets (only core; validation embedded separately and only used for scoring) with a fixed pretrained sentence encoder."""
import sys, time, numpy as np, os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from sentence_transformers import SentenceTransformer
from experiments.retrieval_tracks.x_classifier_lib import load_core_val
name = sys.argv[1]; tag = sys.argv[2]
xc, yc, xv, yv = load_core_val()
m = SentenceTransformer(name, device="mps"); m.max_seq_length = 256
t0 = time.time()
for part, texts in (("core", xc), ("val", xv)):
    e = m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    np.save(f"data/x_classifier/emb_{tag}_{part}.npy", e.astype(np.float32))
    print(part, e.shape, f"{time.time()-t0:.0f}s", flush=True)
