"""The inference API from the midterm architecture diagram.

One container, one model, one route. The model and its vectorizer or tokenizer
load once at startup, exactly as the report describes, and every request is
logged with the model version so a prediction can be audited later.

    MODEL_KIND=sklearn MODEL_PATH=models/1_tfidf_logreg.joblib uvicorn serve:app
    MODEL_KIND=hf MODEL_PATH=models/2_distilbert uvicorn serve:app

This service serves the baseline tiers (a scikit-learn joblib or a Hugging Face
classifier). Serving the final router (src/final/router.py) needs the export step
described in the README: fit once, save the vectorizer, the pools, the embeddings
and the stacker as one artifact, and load that here in place of the joblib.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

import sys

sys.path.insert(0, str(Path(__file__).parent))
from common import clean, load_meta  # noqa: E402

MODEL_KIND = os.environ.get("MODEL_KIND", "sklearn")
MODEL_PATH = os.environ.get("MODEL_PATH", "models/1_tfidf_logreg.joblib")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
# Below this confidence a human does the routing instead. See the threshold
# tables in results/ for how this number was chosen.
THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

app = FastAPI(title="Ticket routing", version=MODEL_VERSION)
LABELS = load_meta()["labels"]
_model = {}


class Ticket(BaseModel):
    subject: str = ""
    body: str = ""


class Prediction(BaseModel):
    queue: str
    confidence: float
    auto_routed: bool
    model_version: str
    latency_ms: float


@app.on_event("startup")
def load_model() -> None:
    if MODEL_KIND == "sklearn":
        import joblib

        vec, clf = joblib.load(MODEL_PATH)
        _model["predict"] = lambda text: clf.predict_proba(vec.transform([text]))[0]
    else:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
        tok = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()

        def predict(text):
            enc = tok(text, truncation=True, max_length=128, return_tensors="pt")
            with torch.no_grad():
                logits = model(**enc).logits
            return torch.softmax(logits[0].float(), dim=-1).numpy()

        _model["predict"] = predict
    print(f"loaded {MODEL_KIND} model from {MODEL_PATH}", flush=True)


@app.get("/health")
def health() -> dict:
    """The droplet's own health check, and what a load balancer would poll."""
    return {"status": "ok" if "predict" in _model else "loading", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
def predict(ticket: Ticket) -> Prediction:
    t0 = time.perf_counter()
    probs = _model["predict"](clean(ticket.subject, ticket.body))
    idx = int(probs.argmax())
    confidence = float(probs[idx])
    latency = (time.perf_counter() - t0) * 1000

    result = Prediction(
        queue=LABELS[idx],
        confidence=round(confidence, 4),
        auto_routed=confidence >= THRESHOLD,
        model_version=MODEL_VERSION,
        latency_ms=round(latency, 1),
    )
    # Stand-in for the audit table in PostgreSQL. Same fields, so swapping this
    # line for an INSERT is the only change needed.
    print(json.dumps({"ts": time.time(), **result.model_dump()}), flush=True)
    return result
