"""Is the model still seeing the kind of tickets it was trained on, and is it still right?

The training job records a reference from the validation slice: the mix of queues the model
predicted, how familiar the tickets were, how confident it was, and how many it would
auto-route. This module compares recent production traffic from the audit log with that
reference. Four signals, chosen because of what this project found:

    unfamiliar share   the share of tickets whose nearest labelled ticket is below 0.3
                       similarity. The router is nearly perfect on familiar tickets and near
                       a guess on unfamiliar ones, so this is the leading indicator. A rise
                       means new kinds of tickets are arriving before accuracy visibly falls.
    queue mix (PSI)    population stability index between the predicted queue mix and the
                       reference. Above 0.2 is the usual "significant shift" line.
    confidence         a falling mean confidence or auto-route rate at the same threshold.
    live accuracy      agreement with human corrections. The only direct accuracy signal,
                       available for reviewed tickets only.

Each signal is compared with a threshold and the report says ok, warn, or drift, and why.
"""
from __future__ import annotations

import math
from collections import Counter

UNFAMILIAR = 0.3        # below this nearest similarity a ticket has no relative in the pool
MIN_ROWS = 200          # fewer predictions than this and the window says nothing yet
PSI_WARN, PSI_DRIFT = 0.1, 0.2
UNFAMILIAR_WARN, UNFAMILIAR_DRIFT = 0.05, 0.10        # absolute rise over the reference share
CONFIDENCE_WARN = 0.05                                # absolute fall in mean confidence
LIVE_ACCURACY_FLOOR, MIN_FEEDBACK = 0.85, 50


def reference_stats(queue_names, pred, confidence, nearest, threshold) -> dict:
    """What "normal" looks like, computed by the training job on the validation slice."""
    n = len(pred)
    mix = Counter(queue_names[i] for i in pred)
    return {
        "rows": n,
        "queue_mix": {q: mix.get(q, 0) / n for q in queue_names},
        "unfamiliar_share": float(sum(s < UNFAMILIAR for s in nearest) / n),
        "mean_confidence": float(sum(confidence) / n),
        "auto_route_rate": float(sum(c >= threshold for c in confidence) / n),
        "threshold": threshold,
    }


def psi(expected: dict, actual: dict, eps: float = 1e-4) -> float:
    """Population stability index between two distributions over the same categories."""
    total = 0.0
    for k in expected:
        e, a = max(expected[k], eps), max(actual.get(k, 0.0), eps)
        total += (a - e) * math.log(a / e)
    return total


def report(rows: list[dict], feedback_rows: list[dict], reference: dict | None, days: int) -> dict:
    """Compare a window of audit-log rows with the reference. rows come from db.recent_predictions."""
    out = {"window_days": days, "predictions": len(rows), "feedback": len(feedback_rows)}
    if not reference:
        return {**out, "status": "unknown", "reasons": ["the served model has no reference statistics"]}
    if len(rows) < MIN_ROWS:
        return {**out, "status": "insufficient_data",
                "reasons": [f"{len(rows)} predictions in the window, drift is judged from {MIN_ROWS}"],
                "reference": reference}

    n = len(rows)
    mix = Counter(r["queue"] for r in rows)
    live = {
        "queue_mix": {q: mix.get(q, 0) / n for q in reference["queue_mix"]},
        "unfamiliar_share": sum(r["nearest_similarity"] < UNFAMILIAR for r in rows) / n,
        "mean_confidence": sum(r["confidence"] for r in rows) / n,
        "auto_route_rate": sum(r["auto_routed"] for r in rows) / n,
        "languages": dict(Counter(r["language"] for r in rows)),
    }
    signals, status = {}, "ok"

    def flag(level):
        nonlocal status
        order = {"ok": 0, "warn": 1, "drift": 2}
        status = level if order[level] > order[status] else status

    reasons = []
    p = psi(reference["queue_mix"], live["queue_mix"])
    signals["queue_mix_psi"] = round(p, 4)
    if p >= PSI_DRIFT:
        flag("drift"); reasons.append(f"queue mix shifted, PSI {p:.2f} >= {PSI_DRIFT}")
    elif p >= PSI_WARN:
        flag("warn"); reasons.append(f"queue mix moving, PSI {p:.2f} >= {PSI_WARN}")

    rise = live["unfamiliar_share"] - reference["unfamiliar_share"]
    signals["unfamiliar_share_change"] = round(rise, 4)
    if rise >= UNFAMILIAR_DRIFT:
        flag("drift"); reasons.append(f"unfamiliar tickets up {rise*100:.1f} points, new kinds of tickets arriving")
    elif rise >= UNFAMILIAR_WARN:
        flag("warn"); reasons.append(f"unfamiliar tickets up {rise*100:.1f} points")

    fall = reference["mean_confidence"] - live["mean_confidence"]
    signals["mean_confidence_change"] = round(-fall, 4)
    if fall >= CONFIDENCE_WARN:
        flag("warn"); reasons.append(f"mean confidence down {fall:.3f}")

    if len(feedback_rows) >= MIN_FEEDBACK:
        acc = sum(f["agreed"] for f in feedback_rows) / len(feedback_rows)
        signals["live_accuracy"] = round(acc, 4)
        if acc < LIVE_ACCURACY_FLOOR:
            flag("drift"); reasons.append(f"live accuracy {acc:.3f} below {LIVE_ACCURACY_FLOOR} on reviewed tickets")

    return {**out, "status": status, "reasons": reasons or ["all signals within their thresholds"],
            "signals": signals, "live": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in live.items()},
            "reference": reference}
