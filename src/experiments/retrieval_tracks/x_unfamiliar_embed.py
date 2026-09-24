"""Embed core and validation texts with fixed pretrained sentence encoders (no fitting on any ticket)."""
import sys, time, os
import numpy as np
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import torch
from sentence_transformers import SentenceTransformer
from experiments.retrieval_tracks.x_unfamiliar_lib import load_core_val

torch.set_num_threads(3)
name = sys.argv[1]; tag = sys.argv[2]; prefix = sys.argv[3] if len(sys.argv) > 3 else ""
xc, yc, xv, yv = load_core_val()
m = SentenceTransformer(name, device="mps"); m.max_seq_length = 384
for part, texts in (("core", xc), ("val", xv)):
    t0 = time.time()
    order = np.argsort([len(t) for t in texts])[::-1]
    E = m.encode([prefix + texts[i] for i in order], batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    out = np.zeros_like(E); out[order] = E
    np.save(f"data/x_unfamiliar/emb_{tag}_{part}.npy", out.astype(np.float32))
    print(part, out.shape, f"{time.time()-t0:.0f}s", flush=True)
