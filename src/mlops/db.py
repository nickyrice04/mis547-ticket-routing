"""The database: an audit log of every routing decision, human corrections, and training runs.

Tickets are fixed-schema records with one text field, so a relational database fits, and
production uses DigitalOcean Managed PostgreSQL on the private network. Locally and in
tests the same code runs against SQLite, because SQLAlchemy Core hides the difference.

Three tables
    predictions     one row per routed ticket. This is the audit log: who asked (the API
                    key's name, never the key), which model version answered, what it
                    decided, how sure it was, how familiar the ticket was, and how long
                    each stage took. The ticket text is stored with long digit runs masked,
                    because corrected tickets become training data at the next refresh.
    feedback        a person's correction of a routing decision. The live accuracy signal
                    and the source of new labelled tickets.
    training_runs   every run of the training job, promoted or not, with its metrics.

Least privilege
    The tables are created once by an admin connection (python -m mlops.db migrate). The
    API connects as router_api, which can only read and insert predictions and feedback
    and read training runs. The training job connects as router_trainer, which can read
    predictions and feedback and write training runs. Neither can create or drop tables.
"""
from __future__ import annotations

import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import (JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData, String,
                        Table, Text, create_engine, func, insert, select, text)

metadata = MetaData()

predictions = Table(
    "predictions", metadata,
    Column("id", String(36), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, index=True),
    Column("request_id", String(36), nullable=False),
    Column("api_key_name", String(64), nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("language", String(8), nullable=False),
    Column("translated", Boolean, nullable=False),
    Column("ticket_text", Text, nullable=False),
    Column("queue", String(64), nullable=False),
    Column("confidence", Float, nullable=False),
    Column("auto_routed", Boolean, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("top3", JSON, nullable=False),
    Column("nearest_similarity", Float, nullable=False),
    Column("latency_ms", Float, nullable=False),
    Column("stage_ms", JSON, nullable=False),
)

feedback = Table(
    "feedback", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("ticket_id", String(36), ForeignKey("predictions.id"), nullable=False, index=True),
    Column("predicted_queue", String(64), nullable=False),
    Column("correct_queue", String(64), nullable=False),
    Column("agreed", Boolean, nullable=False),
    Column("reviewer", String(64), nullable=True),
    Column("api_key_name", String(64), nullable=False),
)

training_runs = Table(
    "training_runs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True), nullable=True),
    Column("model_version", String(64), nullable=False),
    Column("status", String(16), nullable=False),          # promoted, rejected, failed
    Column("host", String(128), nullable=True),
    Column("device", String(64), nullable=True),
    Column("git_sha", String(40), nullable=True),
    Column("train_rows", Integer, nullable=True),
    Column("german_rows", Integer, nullable=True),
    Column("feedback_rows", Integer, nullable=True),
    Column("val_accuracy", Float, nullable=True),
    Column("val_macro_f1", Float, nullable=True),
    Column("val_ece", Float, nullable=True),
    Column("stage_seconds", JSON, nullable=True),
    Column("artifact_key", String(256), nullable=True),
    Column("notes", Text, nullable=True),
)

_DIGITS = re.compile(r"\d[\d\s-]{6,}\d")     # card, account and phone numbers
_engine = None


def redact(text_: str) -> str:
    """Mask long digit runs before a ticket is stored. The model input is not changed."""
    return _DIGITS.sub("[number]", text_)


def url() -> str | None:
    return os.environ.get("DATABASE_URL")


def engine():
    global _engine
    if _engine is None:
        u = url()
        if not u:
            raise RuntimeError("DATABASE_URL is not set")
        kw = {"pool_pre_ping": True}
        if not u.startswith("sqlite"):
            kw.update(pool_size=5, max_overflow=5, pool_recycle=1800)
        _engine = create_engine(u, **kw)
        if u.startswith("sqlite"):
            metadata.create_all(_engine)         # local development only
    return _engine


def ping() -> bool:
    try:
        with engine().connect() as c:
            c.execute(text("select 1"))
        return True
    except Exception:
        return False


def now():
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


def record_prediction(row: dict) -> None:
    with engine().begin() as c:
        c.execute(insert(predictions).values(**row))


def get_prediction(ticket_id: str) -> dict | None:
    with engine().connect() as c:
        r = c.execute(select(predictions).where(predictions.c.id == ticket_id)).mappings().first()
    return dict(r) if r else None


def record_feedback(row: dict) -> int:
    with engine().begin() as c:
        res = c.execute(insert(feedback).values(**row))
        return int(res.inserted_primary_key[0])


def record_training_run(row: dict) -> None:
    with engine().begin() as c:
        c.execute(insert(training_runs).values(**row))


def recent_predictions(days: int = 7, limit: int = 50_000) -> list[dict]:
    since = now() - timedelta(days=days)
    q = (select(predictions.c.queue, predictions.c.confidence, predictions.c.auto_routed,
                predictions.c.nearest_similarity, predictions.c.language, predictions.c.model_version)
         .where(predictions.c.created_at >= since).order_by(predictions.c.created_at.desc()).limit(limit))
    with engine().connect() as c:
        return [dict(r) for r in c.execute(q).mappings()]


def recent_feedback(days: int = 30) -> list[dict]:
    since = now() - timedelta(days=days)
    q = select(feedback.c.agreed, feedback.c.correct_queue, feedback.c.predicted_queue).where(
        feedback.c.created_at >= since)
    with engine().connect() as c:
        return [dict(r) for r in c.execute(q).mappings()]


def corrected_tickets(exclude_keys=("replay",)) -> list[tuple[str, str]]:
    """(ticket text, correct queue) for every human-reviewed ticket, newest correction winning.

    Traffic replayed from the held-out test set (API key "replay") is excluded, so the
    test tickets never leak into a future training set.
    """
    q = (select(predictions.c.ticket_text, feedback.c.correct_queue, feedback.c.created_at)
         .join(feedback, feedback.c.ticket_id == predictions.c.id)
         .where(predictions.c.api_key_name.notin_(exclude_keys))
         .order_by(feedback.c.created_at))
    latest = {}
    with engine().connect() as c:
        for t, qn, _ in c.execute(q):
            latest[t] = qn
    return list(latest.items())


def counts() -> dict:
    with engine().connect() as c:
        return {t.name: c.execute(select(func.count()).select_from(t)).scalar_one()
                for t in (predictions, feedback, training_runs)}


# ------------------------------------------------------------------------------------------ admin
GRANTS = """
GRANT CONNECT ON DATABASE {db} TO router_api, router_trainer;
GRANT USAGE ON SCHEMA public TO router_api, router_trainer;
GRANT SELECT, INSERT ON predictions, feedback TO router_api;
GRANT SELECT ON training_runs TO router_api;
GRANT USAGE, SELECT ON SEQUENCE feedback_id_seq TO router_api;
GRANT SELECT ON predictions, feedback TO router_trainer;
GRANT SELECT, INSERT, UPDATE ON training_runs TO router_trainer;
GRANT USAGE, SELECT ON SEQUENCE training_runs_id_seq TO router_trainer;
"""


def migrate(admin_url: str) -> None:
    """Create the tables and grant the two service users their narrow permissions. Idempotent."""
    eng = create_engine(admin_url)
    metadata.create_all(eng)
    if eng.dialect.name == "postgresql":
        with eng.begin() as c:
            # The database name comes from the server itself and is quoted as an identifier,
            # the statements are fixed text, so nothing user-supplied reaches this SQL.
            db = c.execute(text("select current_database()")).scalar_one()
            quoted = eng.dialect.identifier_preparer.quote(db)
            for stmt in GRANTS.format(db=quoted).strip().splitlines():
                c.execute(text(stmt))  # nosemgrep: avoid-sqlalchemy-text
    print(f"schema ready on {eng.url.host or 'sqlite'}: {', '.join(metadata.tables)}")


if __name__ == "__main__":
    # python -m mlops.db migrate      (reads ADMIN_DATABASE_URL, used once per database)
    if sys.argv[1:] == ["migrate"]:
        migrate(os.environ["ADMIN_DATABASE_URL"])
    elif sys.argv[1:] == ["counts"]:
        print(counts())
    else:
        sys.exit("usage: python -m mlops.db migrate|counts")
