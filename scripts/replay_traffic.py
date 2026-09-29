"""Replay held-out tickets through the live endpoint, to exercise observability end to end.

Sends N tickets from the test split to /v1/route with the "replay" API key and, for a
share of them, posts the true queue to /v1/feedback, the way a support agent would
confirm or correct a routing decision. Afterwards /v1/drift has enough traffic to judge,
and its live accuracy comes from real corrections.

With --drift-demo it then sends off-topic text (recipes, sports scores, weather) that has
no relative in the labelled history, to show the unfamiliar-share signal firing.

Replayed tickets are tagged with the "replay" key and the training job never learns from
them (src/mlops/db.py corrected_tickets), so the test set stays out of future training.

    PYTHONPATH=src python scripts/replay_traffic.py --url https://... --key ... -n 300 --feedback 0.5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, "src")
from common import load_meta, load_split  # noqa: E402

OFF_TOPIC = [
    "Recipe . Whisk two eggs with milk, add flour slowly and let the batter rest for ten minutes before frying.",
    "Match report . The home side won three to one after two late goals in the second half.",
    "Weather . Expect scattered showers in the afternoon with winds from the northwest.",
    "Gardening . Water tomato plants deeply twice a week and prune the lower leaves in July.",
    "Travel . The museum opens at nine and the guided tour of the old town starts from the main square.",
]


def call(url, key, path, payload=None, method="POST"):
    req = urllib.request.Request(url + path, method=method, headers={"X-API-Key": key, "Content-Type": "application/json"},
                                 data=json.dumps(payload).encode() if payload is not None else None)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read() or b"{}") | {"http_status": e.code}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--feedback", type=float, default=0.5, help="share of tickets that get a correction")
    ap.add_argument("--drift-demo", type=int, default=0, help="off-topic tickets to send afterwards")
    args = ap.parse_args()

    labels = load_meta()["labels"]
    x, y = load_split("test")
    rng = random.Random(7)
    idx = rng.sample(range(len(x)), args.n)
    right = sent = fb = 0
    t0 = time.time()
    for i in idx:
        # cleaned tickets read "subject . body", and a ticket without a subject starts with ". "
        text = x[i]
        subject, sep, body = ("", ". ", text[2:]) if text.startswith(". ") else text.partition(" . ")
        if not sep or len(subject) > 300:
            subject, body = "", text
        r = call(args.url, args.key, "/v1/route", {"subject": subject, "body": body, "language": "en"})
        if "queue" not in r:
            print("error:", r)
            continue
        sent += 1
        right += r["queue"] == labels[y[i]]
        if rng.random() < args.feedback:
            call(args.url, args.key, "/v1/feedback", {"ticket_id": r["ticket_id"], "correct_queue": labels[y[i]],
                                                     "reviewer": "replay"})
            fb += 1
    print(f"replayed {sent} held-out tickets in {time.time()-t0:.0f} s, accuracy {right/max(sent,1):.3f}, "
          f"{fb} corrections sent")
    for k in range(args.drift_demo):
        call(args.url, args.key, "/v1/route", {"subject": "", "body": OFF_TOPIC[k % len(OFF_TOPIC)] + f" ({k})"})
    if args.drift_demo:
        print(f"sent {args.drift_demo} off-topic tickets")
    print(json.dumps(call(args.url, args.key, "/v1/drift?days=1", method="GET"), indent=2))


if __name__ == "__main__":
    main()
