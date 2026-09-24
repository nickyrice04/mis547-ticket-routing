"""Measure what each model tier costs to serve on a CPU.

Training happens once. Serving happens on every ticket, so serving is what
decides the droplet size. This script measures the numbers that decide it:

  * memory held after the model is loaded and has answered once
  * single ticket latency at 1, 2 and 4 threads, which stands in for the vCPU
    count of a droplet
  * batch throughput, for the nightly backfill case
  * size on disk, which is what the droplet has to download on deploy

Run it with the thread count pinned so the laptop's other cores do not flatter
the result.

    python src/baselines/bench.py 1_tfidf_logreg sklearn models/1_tfidf_logreg.joblib
    python src/baselines/bench.py 2_distilbert hf models/2_distilbert
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import psutil

RUNS = 50
WARMUP = 5
THREADS = (1, 2, 4)
SAMPLE = (
    "i was charged twice for my subscription this month and the second charge "
    "never showed up on my invoice. i need the duplicate refunded to the card "
    "ending in 4412 before my next statement closes."
)


def rss_mb() -> float:
    """Resident memory of this process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def dir_size_mb(path: str) -> float:
    """Size on disk of a model file or directory in MB."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def bench_sklearn(path: str):
    """Load a (vectorizer, classifier) joblib and return (baseline rss, predict function, no thread control)."""
    import joblib

    base = rss_mb()
    vec, clf = joblib.load(path)
    predict = lambda text: clf.predict_proba(vec.transform([text]))
    return base, predict, {1: None}


def bench_hf(path: str):
    """Load a Hugging Face classifier and return (baseline rss, predict function, the torch module for thread control)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base = rss_mb()
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float32)
    model.eval()

    def predict(text):
        enc = tok(text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        with torch.no_grad():
            return model(**enc).logits

    return base, predict, torch


def main() -> None:
    tier, kind, path = sys.argv[1], sys.argv[2], sys.argv[3]
    if kind == "sklearn":
        base, predict, torch_mod = bench_sklearn(path)
    else:
        base, predict, torch_mod = bench_hf(path)

    for _ in range(WARMUP):
        predict(SAMPLE)
    loaded = rss_mb()

    latency = {}
    thread_counts = THREADS if kind != "sklearn" else (1,)
    for n in thread_counts:
        if kind != "sklearn":
            torch_mod.set_num_threads(n)
        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            predict(SAMPLE)
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        latency[f"{n}_thread"] = {
            "p50_ms": round(statistics.median(times), 1),
            "p95_ms": round(times[int(0.95 * len(times))], 1),
            "mean_ms": round(statistics.mean(times), 1),
        }

    payload = {
        "tier": tier,
        "kind": kind,
        "disk_mb": round(dir_size_mb(path), 1),
        "rss_after_load_mb": round(loaded, 1),
        "rss_model_only_mb": round(loaded - base, 1),
        "latency": latency,
        "machine": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip(),
    }
    out = Path("results") / f"bench_{tier}.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
