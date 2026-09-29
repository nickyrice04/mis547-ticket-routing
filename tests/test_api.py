"""The API contract: authentication, validation errors, routing, feedback, the audit log, rate limits.

A fake router stands in for the real one so these tests run in seconds on any machine,
including the GitHub Actions runner. The audit log is a throwaway SQLite database.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

QUEUES = ["Billing and Payments", "Technical Support", "Human Resources"]


class FakeRouter:
    labels = QUEUES
    meta = {"version": "test", "reference": None}

    def route(self, texts, embed_fn=None):
        out = []
        for t in texts:
            p = np.array([0.9, 0.08, 0.02]) if "charge" in t else np.array([0.4, 0.35, 0.25])
            order = np.argsort(-p)
            out.append({"probs": p, "queue": QUEUES[order[0]], "confidence": float(p[order[0]]),
                        "top": [(QUEUES[i], float(p[i])) for i in order], "nearest_similarity": 0.62})
        return out


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEYS", "team:team-key,grader:grader-key")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'audit.db'}")
    monkeypatch.setenv("ARTIFACT_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("TRANSLATE", "false")
    import importlib
    from mlops import db
    db._engine = None
    from serve import app as app_module
    app_module = importlib.reload(app_module)
    monkeypatch.setattr(app_module, "embed_texts", lambda texts: np.zeros((len(texts), 8), np.float32))
    with TestClient(app_module.app) as c:
        app_module.STATE.router, app_module.STATE.version, app_module.STATE.error = FakeRouter(), "test", None
        app_module.STATE.ready = True
        c.app_module = app_module
        yield c
    db._engine = None


def route(c, key="team-key", **body):
    return c.post("/v1/route", json=body or {"subject": "double charge", "body": "I see a charge twice"},
                  headers={"X-API-Key": key})


def test_health_endpoints(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["database"] == "ok"


def test_missing_and_bad_keys_get_clear_errors(client):
    r = client.post("/v1/route", json={"subject": "x"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "missing_api_key"
    r = route(client, key="wrong")
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"


def test_invalid_tickets_are_rejected_with_a_message(client):
    r = client.post("/v1/route", json={"subject": "", "body": " "}, headers={"X-API-Key": "team-key"})
    assert r.status_code == 422 and "subject or a body" in r.json()["error"]["message"]
    r = client.post("/v1/route", json={"subject": "a", "extra": 1}, headers={"X-API-Key": "team-key"})
    assert r.status_code == 422 and "extra" in r.json()["error"]["message"]
    r = client.post("/v1/route", json={"subject": "a" * 301}, headers={"X-API-Key": "team-key"})
    assert r.status_code == 422


def test_confident_ticket_is_auto_routed_and_audited(client):
    r = route(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["queue"] == "Billing and Payments" and body["auto_routed"] is True
    assert body["audit_logged"] is True and len(body["top_queues"]) == 3
    assert r.headers["X-Request-ID"] == body["request_id"]
    from mlops import db
    row = db.get_prediction(body["ticket_id"])
    assert row["api_key_name"] == "team" and row["queue"] == "Billing and Payments"


def test_unsure_ticket_goes_to_a_person(client):
    r = route(client, subject="hello", body="something vague")
    assert r.json()["auto_routed"] is False and r.json()["confidence"] < 0.7


def test_feedback_round_trip(client):
    tid = route(client).json()["ticket_id"]
    h = {"X-API-Key": "grader-key"}
    r = client.post("/v1/feedback", json={"ticket_id": tid, "correct_queue": "Technical Support"}, headers=h)
    assert r.status_code == 201 and r.json()["agreed"] is False
    r = client.post("/v1/feedback", json={"ticket_id": tid, "correct_queue": "Nope"}, headers=h)
    assert r.status_code == 422 and r.json()["error"]["code"] == "unknown_queue"
    r = client.post("/v1/feedback", json={"ticket_id": "0" * 36, "correct_queue": "Technical Support"}, headers=h)
    assert r.status_code == 404


def test_drift_needs_enough_traffic(client):
    route(client)
    r = client.get("/v1/drift", headers={"X-API-Key": "team-key"})
    assert r.status_code == 200 and r.json()["status"] in ("unknown", "insufficient_data")


def test_rate_limit_answers_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(client.app_module, "RATE_PER_MIN", 3)
    codes = [route(client).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
    r = route(client)
    assert r.status_code == 429 and "Retry-After" in r.headers


def test_metrics_are_exposed(client):
    route(client)
    text = client.get("/metrics").text
    assert "router_predictions_total" in text and "router_confidence_bucket" in text
