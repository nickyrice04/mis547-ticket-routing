"""Record a finished training run in the training_runs table after the fact.

The training job writes its own row. If the audit tables did not exist yet when a run
finished (the first cloud run, before the schema migration), this writes the row from the
run's saved artifact and metrics instead. Run it where the trainer's credentials are:

    sudo bash -c 'set -a; . /etc/ticket-routing/trainer.env; set +a; cd /srv/ticket-routing; \
      PYTHONPATH=src .venv-train/bin/python scripts/record_run.py artifacts/<version>'
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, "src")
from mlops import db, storage  # noqa: E402


def main() -> None:
    """Build the training_runs row from the saved metrics and artifact, and insert it."""
    d = Path(sys.argv[1])
    m = json.loads((d / "metrics.json").read_text())
    from final.model import Router
    meta = Router.load(d / "router.joblib").meta
    started = datetime.fromisoformat(m["trained_at"])
    stages = m.get("stages_seconds", {})
    pointer = storage.latest() if storage.configured() else None
    promoted = bool(pointer and pointer.get("version") == m["version"])
    val = m.get("validation", {})
    row = {
        "started_at": started, "finished_at": started + timedelta(seconds=sum(stages.values())),
        "model_version": m["version"], "status": "promoted" if promoted else "rejected",
        "host": m.get("host"), "device": m.get("device"), "git_sha": m.get("git_sha"),
        "train_rows": meta.get("train_rows"), "german_rows": meta.get("german_rows"),
        "feedback_rows": m.get("feedback_rows"), "val_accuracy": val.get("accuracy"),
        "val_macro_f1": val.get("macro_f1"), "val_ece": val.get("ece"), "stage_seconds": stages,
        "artifact_key": f"models/{m['version']}/router.joblib",
        "notes": "recorded after the run by scripts/record_run.py, the table did not exist when it finished",
    }
    db.record_training_run(row)
    print(json.dumps({k: str(v) for k, v in row.items()}, indent=2))


if __name__ == "__main__":
    main()
