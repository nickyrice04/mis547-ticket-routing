"""Publish a model version to Spaces and point production at it, or roll back to an older one.

The training pipeline publishes its own versions. This does the same thing for a version
that already exists on disk (for example the first one, trained on a laptop) and is the
rollback tool: publishing an older version moves models/latest.json back to it.

    SPACES_REGION=... SPACES_BUCKET=... SPACES_KEY=... SPACES_SECRET=... \
      PYTHONPATH=src python scripts/publish_model.py artifacts/<version>

The directory must hold router.joblib and metrics.json as the pipeline writes them.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")
from mlops import storage  # noqa: E402


def main() -> None:
    """Upload the version if it is not in Spaces yet, then point latest.json at it with its hash and metrics."""
    d = Path(sys.argv[1])
    version = d.name
    artifact, metrics_file = d / "router.joblib", d / "metrics.json"
    metrics = json.loads(metrics_file.read_text())
    digest = storage.sha256(artifact)
    key = f"models/{version}/router.joblib"
    if not storage.exists(key):
        storage.upload(artifact, key)
        storage.upload(metrics_file, f"models/{version}/metrics.json")
        print(f"uploaded {key} ({artifact.stat().st_size / 1e6:.1f} MB)")
    val = metrics.get("validation", {})
    storage.write_json(storage.LATEST, {
        "version": version, "key": key, "sha256": digest, "size_mb": round(artifact.stat().st_size / 1e6, 1),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {k: val.get(k) for k in ("accuracy", "macro_f1", "ece")}})
    print(f"models/latest.json -> {version} (sha256 {digest[:12]}, validation accuracy {val.get('accuracy')})")


if __name__ == "__main__":
    main()
