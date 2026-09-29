"""Drift signals: population stability index and the report's status rules."""
from mlops import drift

QUEUES = ["a", "b", "c"]
REF = {"rows": 1000, "queue_mix": {"a": 0.5, "b": 0.3, "c": 0.2}, "unfamiliar_share": 0.2,
       "mean_confidence": 0.85, "auto_route_rate": 0.8, "threshold": 0.7}


def rows(n, mix, unfamiliar, conf):
    out = []
    for i in range(n):
        q = "a" if i < n * mix[0] else ("b" if i < n * (mix[0] + mix[1]) else "c")
        out.append({"queue": q, "confidence": conf, "auto_routed": conf >= 0.7,
                    "nearest_similarity": 0.1 if i % 100 < unfamiliar * 100 else 0.8, "language": "en"})
    return out


def test_psi_is_zero_for_identical_mixes():
    assert drift.psi(REF["queue_mix"], REF["queue_mix"]) == 0.0


def test_same_traffic_is_ok():
    r = drift.report(rows(1000, (0.5, 0.3), 0.2, 0.85), [], REF, 7)
    assert r["status"] == "ok", r["reasons"]


def test_new_kinds_of_tickets_are_flagged():
    r = drift.report(rows(1000, (0.5, 0.3), 0.45, 0.85), [], REF, 7)
    assert r["status"] == "drift"
    assert any("unfamiliar" in x for x in r["reasons"])


def test_queue_mix_shift_is_flagged():
    r = drift.report(rows(1000, (0.1, 0.1), 0.2, 0.85), [], REF, 7)
    assert r["status"] == "drift" and r["signals"]["queue_mix_psi"] >= drift.PSI_DRIFT


def test_low_live_accuracy_is_flagged():
    fb = [{"agreed": i % 10 < 7} for i in range(100)]
    r = drift.report(rows(1000, (0.5, 0.3), 0.2, 0.85), fb, REF, 7)
    assert r["status"] == "drift" and r["signals"]["live_accuracy"] == 0.7


def test_small_windows_are_not_judged():
    assert drift.report(rows(10, (0.5, 0.3), 0.2, 0.85), [], REF, 7)["status"] == "insufficient_data"
