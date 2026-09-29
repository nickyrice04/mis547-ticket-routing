"""The inference service: routes a support ticket to one of ten queues.

This is the API the instructor calls. It runs in a container on the inference droplet
behind Caddy (which terminates HTTPS) and serves the final router (src/final/model.py).

Request path for one ticket
    1. authenticate the X-API-Key header and apply the per-key rate limit
    2. validate the input (subject and body lengths, unknown fields rejected)
    3. translate German sentences to English (final/translate.py, same rules as training)
    4. score the ticket against the labelled pools (final/model.py)
    5. apply the confidence threshold: confident tickets are auto-routed, the rest go to a person
    6. write the decision to the audit log in PostgreSQL and update the metrics
    7. return the queue, the confidence, the top three queues and a ticket id for feedback

Fail safe
    If the audit log cannot be written, the ticket is still answered but auto_routed is
    forced to false, so no automated routing decision is ever made without a record.
    /readyz reports the database as down, which a load balancer uses to stop sending
    traffic to this node.

Endpoints
    GET  /healthz          liveness: the process is up
    GET  /readyz           readiness: the model is loaded and the audit database answers
    GET  /                 service description and the endpoints below
    POST /v1/route         route one ticket                          (API key)
    POST /v1/feedback      record a person's correction of a ticket  (API key)
    GET  /v1/queues        the ten queue names                       (API key)
    GET  /v1/model         the served model's version and metrics    (API key)
    GET  /v1/drift         drift report over recent traffic          (API key)
    GET  /metrics          Prometheus metrics, blocked from the internet by Caddy

Configuration is all environment variables, see deploy/api.env.example.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # src/ on the path inside the container

from common import clean  # noqa: E402
from mlops import db, drift, storage  # noqa: E402

SERVICE = "ticket-router"
THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7"))
RATE_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))
MODEL_CACHE = Path(os.environ.get("MODEL_CACHE", "/models"))
ARTIFACT_PATH = os.environ.get("ARTIFACT_PATH")          # a local artifact, used instead of Spaces
MODEL_POLL_SECONDS = int(os.environ.get("MODEL_POLL_SECONDS", "600"))
TRANSLATE = os.environ.get("TRANSLATE", "true").lower() == "true"
MAX_SUBJECT, MAX_BODY = 300, 8000

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(SERVICE)


def jlog(event: str, **fields) -> None:
    """One JSON object per line, so the logs can be searched and shipped anywhere."""
    log.info(json.dumps({"ts": round(time.time(), 3), "service": SERVICE, "event": event, **fields}, default=str))


# ------------------------------------------------------------------------------------------ metrics
REGISTRY = CollectorRegistry()
REQUESTS = Counter("router_requests_total", "HTTP requests", ["endpoint", "status"], registry=REGISTRY)
LATENCY = Histogram("router_request_seconds", "Request latency", ["endpoint"],
                    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10), registry=REGISTRY)
STAGE = Histogram("router_stage_seconds", "Time per stage of one prediction", ["stage"],
                  buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5), registry=REGISTRY)
PREDICTIONS = Counter("router_predictions_total", "Routed tickets", ["queue", "auto_routed"], registry=REGISTRY)
CONFIDENCE = Histogram("router_confidence", "Top-queue probability", buckets=[i / 10 for i in range(1, 11)], registry=REGISTRY)
NEAREST = Histogram("router_nearest_similarity", "Similarity to the nearest labelled ticket (familiarity)",
                    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), registry=REGISTRY)
TRANSLATED = Counter("router_translated_total", "Tickets that needed translation", registry=REGISTRY)
FEEDBACK = Counter("router_feedback_total", "Human corrections", ["agreed"], registry=REGISTRY)
AUDIT_FAILURES = Counter("router_audit_failures_total", "Predictions that could not be written to the audit log", registry=REGISTRY)
RATE_LIMITED = Counter("router_rate_limited_total", "Requests refused by the rate limit", ["key"], registry=REGISTRY)
MODEL_INFO = Gauge("router_model_info", "The served model version", ["version"], registry=REGISTRY)


# ------------------------------------------------------------------------------------------ model state
class State:
    """The loaded router, translator and pointer. Swapped atomically when a new version appears."""
    router = None
    translator = None
    ready = False            # true only after warm-up has routed a ticket end to end
    pointer: dict | None = None
    version = "none"
    error: str | None = None
    lock = threading.Lock()


STATE = State()


def load_model() -> None:
    """Load the artifact named by ARTIFACT_PATH, else the version Spaces' latest.json points to."""
    from final.model import Router

    if ARTIFACT_PATH:
        path, pointer = Path(ARTIFACT_PATH), {"version": Path(ARTIFACT_PATH).stem, "source": "local"}
    elif storage.configured():
        pointer = storage.latest()
        if not pointer:
            raise RuntimeError("Spaces has no models/latest.json yet, run the training job first")
        path = storage.fetch_verified(pointer, MODEL_CACHE)
    else:
        raise RuntimeError("set ARTIFACT_PATH or the SPACES_* variables so the service can find a model")
    router = Router.load(path)
    version = router.meta.get("version", pointer["version"])
    with STATE.lock:
        STATE.router, STATE.pointer, STATE.version, STATE.error = router, pointer, version, None
    MODEL_INFO.clear()
    MODEL_INFO.labels(version=version).set(1)
    jlog("model_loaded", version=version, source=pointer.get("source", "spaces"),
         train_rows=router.meta.get("train_rows"), german_rows=router.meta.get("german_rows"))


def poll_for_new_models() -> None:
    """Background thread: when the training job promotes a new version, load it without a restart."""
    while True:
        time.sleep(MODEL_POLL_SECONDS)
        try:
            pointer = storage.latest()
            if pointer and pointer.get("version") != (STATE.pointer or {}).get("version"):
                jlog("model_update_found", version=pointer["version"])
                load_model()
                if not STATE.ready:          # the first model arrived after startup
                    prepare()
        except Exception as e:                     # keep serving the current model
            jlog("model_poll_failed", error=str(e))


def prepare() -> None:
    """Load the translator and route one ticket end to end, then declare the service ready."""
    if TRANSLATE and STATE.translator is None:
        from final.translate import Translator
        STATE.translator = Translator(device="cpu")
    _predict("Warm up . Please ignore this ticket, it checks the service at startup.")
    if STATE.translator is not None:
        STATE.translator.to_english("Test", "Bitte ignorieren Sie dieses Ticket.")
    STATE.ready = True
    jlog("ready", version=STATE.version, threshold=THRESHOLD)


def warm_up() -> None:
    """Load everything and route one ticket, so the first real request is not slow."""
    try:
        load_model()
        prepare()
    except Exception as e:
        STATE.error = str(e)
        jlog("startup_failed", error=str(e))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load the model in the background so /healthz answers at once, and start the model poller."""
    threading.Thread(target=warm_up, daemon=True).start()
    if not ARTIFACT_PATH and storage.configured():
        threading.Thread(target=poll_for_new_models, daemon=True).start()
    yield


app = FastAPI(
    title="Support ticket router",
    version="1.0",
    lifespan=lifespan,
    description="Routes a customer support ticket to one of ten queues. Team 3, MIS 547. "
                "Send the API key in the X-API-Key header.",
)


# ------------------------------------------------------------------------------------------ errors
def error(status: int, code: str, message: str, request: Request | None = None, headers=None) -> JSONResponse:
    """The one error shape every failure uses: an HTTP status, a machine-readable code, a human message, the request id."""
    rid = getattr(getattr(request, "state", None), "request_id", None) if request else None
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"code": code, "message": message}, "request_id": rid})


class ApiError(Exception):
    """An expected failure (bad key, unknown ticket, rate limit) that becomes a clean JSON error response."""
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers


@app.exception_handler(ApiError)
async def _api_error(request: Request, exc: ApiError):
    """Turn an ApiError into the standard JSON error response."""
    return error(exc.status, exc.code, exc.message, request, exc.headers)


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    """Invalid request bodies (422) get a readable message naming the field, instead of FastAPI's raw list."""
    def one(e):
        where = ".".join(str(p) for p in e["loc"] if p != "body")
        msg = e["msg"].removeprefix("Value error, ")
        return f"{where}: {msg}" if where else msg
    problems = "; ".join(one(e) for e in exc.errors())
    return error(422, "invalid_request", problems or "the request body is not valid", request)


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException):
    """Framework errors such as 404 or 405 use the same JSON error shape."""
    return error(exc.status_code, "http_error", str(exc.detail), request)


@app.exception_handler(Exception)
async def _unexpected(request: Request, exc: Exception):
    """Anything unforeseen is logged with its request id and answered with a generic 500, never a stack trace."""
    jlog("unhandled_error", request_id=getattr(request.state, "request_id", None), error=repr(exc))
    return error(500, "internal_error", "the service hit an unexpected error, it has been logged", request)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """A request id on every request and response, and latency and status metrics per endpoint."""
    request.state.request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    t0 = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    endpoint = getattr(route, "path", "unmatched")
    LATENCY.labels(endpoint).observe(time.perf_counter() - t0)
    REQUESTS.labels(endpoint, str(response.status_code)).inc()
    response.headers["X-Request-ID"] = request.state.request_id
    return response


# ------------------------------------------------------------------------------------------ auth
def _keys() -> dict[str, str]:
    """API_KEYS="team:secret1,grader:secret2" -> {"team": "secret1", ...}. Names are logged, keys never."""
    out = {}
    for pair in filter(None, os.environ.get("API_KEYS", "").split(",")):
        name, _, key = pair.partition(":")
        if name and key:
            out[name.strip()] = key.strip()
    return out


API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
_WINDOWS: dict[str, deque] = {}


def authenticate(request: Request, key: str | None = Depends(API_KEY_HEADER)) -> str:
    """Check the X-API-Key header against API_KEYS and apply the per-key rate limit. Returns the key's name for the audit log."""
    keys = _keys()
    if not keys:
        raise ApiError(503, "not_configured", "no API keys are configured on this server")
    if not key:
        raise ApiError(401, "missing_api_key", "send your API key in the X-API-Key header")
    name = next((n for n, k in keys.items() if hmac.compare_digest(k.encode(), key.encode())), None)
    if name is None:
        jlog("auth_failed", request_id=request.state.request_id, client=request.client.host if request.client else None)
        raise ApiError(401, "invalid_api_key", "the API key is not valid")
    # sliding one-minute window per key
    limit = RATE_PER_MIN * (20 if name == "replay" else 1)
    window, now = _WINDOWS.setdefault(name, deque()), time.monotonic()
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= limit:
        RATE_LIMITED.labels(name).inc()
        retry = int(60 - (now - window[0])) + 1
        raise ApiError(429, "rate_limited", f"more than {limit} requests in a minute, retry in {retry} s",
                       headers={"Retry-After": str(retry)})
    window.append(now)
    request.state.api_key_name = name
    return name


# ------------------------------------------------------------------------------------------ schemas
class Ticket(BaseModel):
    """The request body for /v1/route. Subject and body are cleaned exactly like the training tickets."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{
        "subject": "Charged twice for my subscription",
        "body": "I was billed twice this month for the same plan and need the duplicate charge refunded."}]})

    subject: str = Field("", max_length=MAX_SUBJECT, description="Ticket subject line")
    body: str = Field("", max_length=MAX_BODY, description="Ticket body")
    language: Literal["auto", "en", "de"] = Field("auto", description="auto detects German sentences")

    @model_validator(mode="after")
    def not_empty(self):
        """Reject a ticket with neither a subject nor a body."""
        if not (self.subject.strip() or self.body.strip()):
            raise ValueError("a ticket needs a subject or a body")
        return self


class QueueScore(BaseModel):
    """One queue and the probability the router gives it."""
    queue: str
    probability: float


class RouteResult(BaseModel):
    """The response of /v1/route: the decision, how sure the router is, and what was recorded."""
    ticket_id: str
    queue: str
    confidence: float
    auto_routed: bool
    threshold: float
    top_queues: list[QueueScore]
    familiarity: float = Field(description="similarity to the nearest labelled ticket, below 0.3 means unfamiliar")
    language: str
    translated: bool
    model_version: str
    audit_logged: bool
    latency_ms: float
    request_id: str


class Feedback(BaseModel):
    """The request body for /v1/feedback: a person's verdict on a routed ticket."""
    model_config = ConfigDict(extra="forbid")
    ticket_id: str = Field(min_length=36, max_length=36)
    correct_queue: str = Field(max_length=64)
    reviewer: str | None = Field(None, max_length=64)


# ------------------------------------------------------------------------------------------ routes
def embed_texts(texts: list[str]):
    """The sentence embedder. A function of its own so tests can swap in a fake one."""
    from final.features import e5_embed
    return e5_embed(texts)


def _predict(text: str) -> dict:
    """Score one cleaned English ticket, timing each stage."""
    router = STATE.router
    t = time.perf_counter()
    embedded = embed_texts([text])
    t_embed = time.perf_counter() - t
    t = time.perf_counter()
    result = router.route([text], embed_fn=lambda _: embedded)[0]
    t_route = time.perf_counter() - t
    STAGE.labels("embed").observe(t_embed)
    STAGE.labels("retrieve_and_stack").observe(t_route)
    return {**result, "stage_ms": {"embed": round(t_embed * 1000, 1), "retrieve_and_stack": round(t_route * 1000, 1)}}


def require_model():
    """Answer 503 with Retry-After until the model is loaded and warmed up."""
    if STATE.router is None or not STATE.ready:
        raise ApiError(503, "model_loading", STATE.error or "the model is still loading, retry shortly",
                       headers={"Retry-After": "30"})


@app.get("/healthz", tags=["health"])
def healthz():
    """Liveness. The process is up. Docker restarts the container when this stops answering."""
    return {"status": "ok"}


@app.get("/readyz", tags=["health"])
def readyz():
    """Readiness. The model is loaded and the audit database answers. A load balancer routes only to ready nodes."""
    model_ok = STATE.router is not None and STATE.ready
    db_ok = db.ping() if db.url() else None
    ready = model_ok and db_ok is not False
    body = {"status": "ready" if ready else "not_ready", "model_version": STATE.version,
            "model_loaded": model_ok, "database": {None: "not_configured", True: "ok", False: "down"}[db_ok],
            "error": STATE.error}
    return JSONResponse(status_code=200 if ready else 503, content=body)


@app.get("/", tags=["info"])
def root():
    """A short description of the service and its endpoints."""
    return {"service": SERVICE, "model_version": STATE.version, "docs": "/docs",
            "auth": "X-API-Key header", "confidence_threshold": THRESHOLD,
            "endpoints": {"POST /v1/route": "route a ticket", "POST /v1/feedback": "correct a routing decision",
                          "GET /v1/queues": "queue names", "GET /v1/model": "model metadata",
                          "GET /v1/drift": "drift report", "GET /healthz": "liveness", "GET /readyz": "readiness"}}


@app.post("/v1/route", response_model=RouteResult, tags=["routing"])
def route(ticket: Ticket, request: Request, key_name: str = Depends(authenticate)):
    """Route one support ticket to one of ten queues. Tickets at or above the confidence threshold are routed automatically, the rest are flagged for a person. German sentences are translated first. Every decision is written to the audit log and the response carries a ticket_id for feedback."""
    require_model()
    t0 = time.perf_counter()
    stage_ms = {}

    t = time.perf_counter()
    if TRANSLATE and ticket.language != "en" and STATE.translator is not None:
        tr = STATE.translator.to_english(ticket.subject, ticket.body)
        text, language, translated = tr["text"], tr["language"], tr["translated_sentences"] > 0
    else:
        text, language, translated = clean(ticket.subject, ticket.body), "en", False
    stage_ms["translate_and_clean"] = round((time.perf_counter() - t) * 1000, 1)
    STAGE.labels("translate_and_clean").observe(stage_ms["translate_and_clean"] / 1000)
    if not text:
        raise ApiError(422, "empty_after_cleaning", "the ticket has no text left after removing links and emails")

    with STATE.lock:
        version = STATE.version
    res = _predict(text)
    stage_ms.update(res["stage_ms"])
    auto = res["confidence"] >= THRESHOLD
    ticket_id = db.new_id()
    latency = round((time.perf_counter() - t0) * 1000, 1)

    audit_ok = False
    if db.url():
        try:
            db.record_prediction({
                "id": ticket_id, "created_at": db.now(), "request_id": request.state.request_id,
                "api_key_name": key_name, "model_version": version, "language": language,
                "translated": translated, "ticket_text": db.redact(text), "queue": res["queue"],
                "confidence": res["confidence"], "auto_routed": auto, "threshold": THRESHOLD,
                "top3": [[q, round(p, 4)] for q, p in res["top"]], "nearest_similarity": res["nearest_similarity"],
                "latency_ms": latency, "stage_ms": stage_ms})
            audit_ok = True
        except Exception as e:
            AUDIT_FAILURES.inc()
            jlog("audit_write_failed", request_id=request.state.request_id, error=str(e))
    if not audit_ok and db.url():
        auto = False                     # fail safe: no automated decision without an audit record

    PREDICTIONS.labels(res["queue"], str(auto).lower()).inc()
    CONFIDENCE.observe(res["confidence"])
    NEAREST.observe(res["nearest_similarity"])
    if translated:
        TRANSLATED.inc()
    jlog("routed", request_id=request.state.request_id, ticket_id=ticket_id, key=key_name, model_version=version,
         queue=res["queue"], confidence=round(res["confidence"], 4), auto_routed=auto,
         familiarity=round(res["nearest_similarity"], 3), language=language, latency_ms=latency, stage_ms=stage_ms)
    return RouteResult(
        ticket_id=ticket_id, queue=res["queue"], confidence=round(res["confidence"], 4), auto_routed=auto,
        threshold=THRESHOLD, top_queues=[QueueScore(queue=q, probability=round(p, 4)) for q, p in res["top"]],
        familiarity=round(res["nearest_similarity"], 4), language=language, translated=translated,
        model_version=version, audit_logged=audit_ok, latency_ms=latency, request_id=request.state.request_id)


@app.post("/v1/feedback", status_code=201, tags=["routing"])
def post_feedback(fb: Feedback, request: Request, key_name: str = Depends(authenticate)):
    """A person corrected (or confirmed) a routing decision. Feeds live accuracy and the next retraining."""
    require_model()
    if fb.correct_queue not in STATE.router.labels:
        raise ApiError(422, "unknown_queue", f"correct_queue must be one of: {', '.join(STATE.router.labels)}")
    if not db.url():
        raise ApiError(503, "no_database", "feedback needs the audit database, which is not configured")
    pred = db.get_prediction(fb.ticket_id)
    if pred is None:
        raise ApiError(404, "unknown_ticket", f"no routed ticket has id {fb.ticket_id}")
    agreed = pred["queue"] == fb.correct_queue
    fid = db.record_feedback({"created_at": db.now(), "ticket_id": fb.ticket_id, "predicted_queue": pred["queue"],
                              "correct_queue": fb.correct_queue, "agreed": agreed, "reviewer": fb.reviewer,
                              "api_key_name": key_name})
    FEEDBACK.labels(str(agreed).lower()).inc()
    jlog("feedback", request_id=request.state.request_id, ticket_id=fb.ticket_id, agreed=agreed, key=key_name)
    return {"feedback_id": fid, "ticket_id": fb.ticket_id, "predicted_queue": pred["queue"],
            "correct_queue": fb.correct_queue, "agreed": agreed}


@app.get("/v1/queues", tags=["info"])
def queues(_: str = Depends(authenticate)):
    """The ten queue names, the valid values for correct_queue in /v1/feedback."""
    require_model()
    return {"queues": STATE.router.labels}


@app.get("/v1/model", tags=["info"])
def model_info(_: str = Depends(authenticate)):
    """The model version being served, where and when it was trained, and its validation metrics."""
    require_model()
    m = STATE.router.meta
    return {"version": STATE.version, "pointer": STATE.pointer,
            **{k: m.get(k) for k in ("trained_at", "train_rows", "german_rows", "feedback_rows", "recipe",
                                      "validation", "device", "host", "git_sha")}}


@app.get("/v1/drift", tags=["monitoring"])
def drift_report(days: int = 7, _: str = Depends(authenticate)):
    """Compare the last N days of routed tickets with the reference recorded at training time: unfamiliar share, queue mix (PSI), confidence, and live accuracy from corrections. Status is ok, warn or drift."""
    require_model()
    if not db.url():
        raise ApiError(503, "no_database", "drift is computed from the audit log, which is not configured")
    days = max(1, min(days, 90))
    rows = [r for r in db.recent_predictions(days) if r["model_version"] == STATE.version]
    return drift.report(rows, db.recent_feedback(days), STATE.router.meta.get("reference"), days)


@app.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus scrapes this on the private Docker network. Caddy returns 404 for it from the internet."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)
