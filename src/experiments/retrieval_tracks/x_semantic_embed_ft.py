"""Embed core + validation with a fine-tuned model directory. Usage: x_semantic_embed_ft.py <model_tag> <prefix?>"""
import sys
import numpy as np
from experiments.retrieval_tracks.x_semantic_common import *
tag = sys.argv[1]; prefix = sys.argv[2] if len(sys.argv) > 2 else ""
ctx, cy, vtx, vy = load_core_val()
path = str(ROOT / "models" / f"x_semantic_{tag}")
ec = embed(path, ctx, f"{tag}_core", prefix=prefix, max_seq_length=192)
ev = embed(path, vtx, f"{tag}_val", prefix=prefix, max_seq_length=192)
print("embedded", ec.shape, ev.shape)
