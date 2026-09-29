"""The training job. Runs on the GPU droplet, publishes a new model version to Spaces.

This is the burst of heavy compute the architecture rents a GPU for. Every stage is
timed and the timings go into the training_runs table, which is where the report's
compute estimates come from.

Stages
    data        the dataset from Spaces (or local), the deduplicated split, its fingerprint
    translate   German tickets to English with the neural translator (the GPU-heavy stage),
                or reuse the pool already in Spaces with --reuse-german
    embed       multilingual-e5 embeddings of the German pool
    validate    the EVALUATED recipe: fit on the core of the training set with the leak
                guard on, score the validation slice. This is the number the quality gate
                reads and the one that matches the report (89.9% when nothing changed).
                The same run records the reference statistics the drift monitor compares
                live traffic with.
    feedback    tickets people have corrected since the last run, from the audit database
    fit         the PRODUCTION recipe: all training tickets plus the corrections, guard off
    gate        promote only if validation accuracy clears MIN_ACCURACY and is no more than
                MAX_REGRESSION below the version in production
    publish     the artifact and its metrics to Spaces, latest.json moved if promoted,
                a training_runs row either way

    PYTHONPATH=src python -m mlops.train_pipeline                  # full run
    PYTHONPATH=src python -m mlops.train_pipeline --reuse-german   # skip translation
    PYTHONPATH=src python -m mlops.train_pipeline --local-only     # no Spaces, no database
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from common import ROOT, SPLITS, build_splits, load_meta, load_split, threshold_table
from mlops import db, drift, storage

MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.85"))
MAX_REGRESSION = float(os.environ.get("MAX_REGRESSION", "0.01"))
THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7"))
GERMAN_FILES = ["german_raw.jsonl", "german_translated.jsonl", "e5_german_translated.npy", "e5_german_original.npy"]
GDIR = ROOT / "data" / "x_german"
OUT = ROOT / "artifacts"


def device_name() -> str:
    import torch
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "apple-mps" if torch.backends.mps.is_available() else "cpu"


def git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def ece(conf, correct, bins=15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)


class Timer:
    def __init__(self):
        self.stages = {}

    def __call__(self, name):
        timer = self

        class _T:
            def __enter__(self):
                self.t0 = time.time()
                print(f"== {name}", flush=True)

            def __exit__(self, *exc):
                timer.stages[name] = round(time.time() - self.t0, 1)
                print(f"   {name} took {timer.stages[name]} s", flush=True)
        return _T()


def stage_data(remote: bool) -> dict:
    parquet = ROOT / "data_tickets.parquet"
    if not parquet.exists() and remote:
        storage.download("data/data_tickets.parquet", parquet)
    if not (SPLITS / "train.json").exists():
        build_splits()
    fp = hashlib.sha256((SPLITS / "train.json").read_bytes() + (SPLITS / "test.json").read_bytes()).hexdigest()
    return {"split_fingerprint": fp[:16], "dataset_sha256": storage.sha256(parquet)[:16]}


def stage_translate(remote: bool, reuse: bool) -> str:
    """Produce data/x_german/. Returns where the pool came from."""
    have = all((GDIR / f).exists() for f in GERMAN_FILES)
    if reuse and have:
        return "local cache"
    if reuse and remote and all(storage.exists(f"data/x_german/{f}") for f in GERMAN_FILES):
        for f in GERMAN_FILES:
            storage.download(f"data/x_german/{f}", GDIR / f)
        return "spaces"
    from final import translate
    translate.extract()
    translate.translate()
    translate.assemble()
    return "translated on this run"


def stage_embed(source: str) -> None:
    from final import embed
    if source == "translated on this run" or not (GDIR / "e5_german_translated.npy").exists():
        embed.main()


def stage_validate(labels) -> dict:
    """The evaluated recipe on core -> validation, plus the drift reference."""
    from final.lib import load_core_val
    from final.model import Router

    xc, yc, xv, yv = load_core_val()
    r = Router(labels=labels).fit(xc, yc, guard=True, verbose=False)
    F, _ = r.features(xv, guard=True)
    from final.stacker import predict_stacker
    P = predict_stacker(r.stacker, F)
    pred, conf = P.argmax(1), P.max(1)
    nearest = np.maximum(F[:, :, 0].max(1), F[:, :, 2].max(1))
    correct = (pred == yv).astype(float)
    per_queue = {q: round(float(correct[yv == i].mean()), 4) for i, q in enumerate(labels)}
    return {
        "protocol": "fit on core (16,147) with the leak guard, score the validation slice (2,850)",
        "accuracy": round(float(accuracy_score(yv, pred)), 4),
        "macro_f1": round(float(f1_score(yv, pred, average="macro")), 4),
        "ece": round(ece(conf, correct), 4),
        "per_queue_recall": per_queue,
        "thresholds": threshold_table(P, yv),
        "reference": drift.reference_stats(labels, pred, conf, nearest, THRESHOLD),
    }


def stage_feedback(remote: bool) -> list[tuple[str, str]]:
    if not remote or not db.url():
        return []
    try:
        return db.corrected_tickets()
    except Exception as e:
        print(f"   could not read corrections, training without them: {e}", flush=True)
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse-german", action="store_true", help="skip translation if the pool exists")
    ap.add_argument("--local-only", action="store_true", help="no Spaces, no database")
    ap.add_argument("--force-promote", action="store_true", help="promote even if the gate says no")
    args = ap.parse_args()
    remote = not args.local_only and storage.configured()

    started = datetime.now(timezone.utc)
    version = started.strftime("v%Y%m%d-%H%M%S")
    T = Timer()
    labels = None
    meta: dict = {"version": version, "trained_at": started.isoformat(), "host": socket.gethostname(),
                  "device": device_name(), "git_sha": git_sha(), "python": platform.python_version(),
                  "recipe": "production: all training tickets + corrections, German pool without the leak guard"}
    run_row = {"started_at": started, "model_version": version, "host": meta["host"], "device": meta["device"],
               "git_sha": meta["git_sha"], "status": "failed"}
    try:
        with T("data"):
            meta.update(stage_data(remote))
            labels = load_meta()["labels"]
        with T("translate"):
            meta["german_source"] = stage_translate(remote, args.reuse_german)
        with T("embed"):
            stage_embed(meta["german_source"])
        with T("validate"):
            val = stage_validate(labels)
            print(f"   validation accuracy {val['accuracy']:.4f}  macro-F1 {val['macro_f1']:.4f}  ECE {val['ece']:.4f}")
        with T("feedback"):
            corrections = stage_feedback(remote)
            print(f"   {len(corrections)} corrected tickets from the audit log")
        with T("fit"):
            from final.model import Router
            x, y = load_split("train")
            x = list(x) + [t for t, _ in corrections]
            y = np.concatenate([np.asarray(y), np.asarray([labels.index(q) for _, q in corrections], int)])
            router = Router(labels=labels).fit(x, y, guard=False, verbose=True)

        with T("gate"):
            current = storage.latest() if remote else None
            floor = MIN_ACCURACY
            if current and current.get("metrics", {}).get("accuracy") is not None:
                floor = max(floor, current["metrics"]["accuracy"] - MAX_REGRESSION)
            promote = val["accuracy"] >= floor or args.force_promote
            print(f"   gate: {val['accuracy']:.4f} vs floor {floor:.4f} -> {'promote' if promote else 'reject'}")

        with T("publish"):
            meta.update({"validation": {k: v for k, v in val.items() if k != "reference"},
                         "reference": val["reference"], "feedback_rows": len(corrections), "stages_seconds": T.stages})
            router.meta.update(meta)
            path = router.save(OUT / version / "router.joblib")
            digest = storage.sha256(path)
            size_mb = round(path.stat().st_size / 1e6, 1)
            metrics = {"version": version, "sha256": digest, "size_mb": size_mb, **meta}
            (OUT / version / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
            key = f"models/{version}/router.joblib"
            if remote:
                storage.upload(path, key)
                storage.upload(OUT / version / "metrics.json", f"models/{version}/metrics.json")
                if meta["german_source"] == "translated on this run":
                    for f in GERMAN_FILES:
                        storage.upload(GDIR / f, f"data/x_german/{f}")
                if promote:
                    storage.write_json(storage.LATEST, {
                        "version": version, "key": key, "sha256": digest, "size_mb": size_mb,
                        "promoted_at": datetime.now(timezone.utc).isoformat(),
                        "metrics": {k: val[k] for k in ("accuracy", "macro_f1", "ece")}})
            print(f"   artifact {path} ({size_mb} MB, sha256 {digest[:12]})")

        run_row.update({"status": "promoted" if promote else "rejected", "train_rows": len(x),
                        "german_rows": router.meta.get("german_rows"), "feedback_rows": len(corrections),
                        "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"], "val_ece": val["ece"],
                        "artifact_key": key if remote else str(path)})
    except Exception as e:
        run_row["notes"] = repr(e)[:2000]
        raise
    finally:
        run_row["finished_at"] = datetime.now(timezone.utc)
        run_row["stage_seconds"] = T.stages
        if remote and db.url():
            try:
                db.record_training_run(run_row)
            except Exception as e:
                print(f"   could not record the training run: {e}", flush=True)
        print(json.dumps({k: v for k, v in run_row.items() if k not in ("started_at", "finished_at")},
                         indent=2, default=str))


if __name__ == "__main__":
    main()
