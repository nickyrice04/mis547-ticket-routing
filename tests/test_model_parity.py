"""The served Router must be the evaluated router, bit for bit.

Fits final.model.Router on the core of the training set with the leak guard on, scores
the validation slice with the guard on, and compares with the probabilities that
final/router.py produced when the 89.9% validation number was recorded. Any difference
means the API would be serving a different model from the one in the report.

Slow (about two minutes on the laptop GPU) and needs data/x_german/, so it is skipped in
CI and run by hand:  PYTHONPATH=src pytest -m slow tests/test_model_parity.py
"""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SAVED = ROOT / "results" / "x_german_val_probs.npy"


@pytest.mark.slow
@pytest.mark.skipif(not (ROOT / "data" / "x_german" / "german_translated.jsonl").exists(),
                    reason="needs the translated German pool under data/x_german/")
def test_router_matches_the_evaluated_model(tmp_path):
    from final.lib import load_core_val
    from final.model import Router

    xc, yc, xv, yv = load_core_val()
    r = Router(labels=list(range(10))).fit(xc, yc, guard=True, verbose=False)
    path = r.save(tmp_path / "router.joblib")
    P = Router.load(path).predict_proba(xv, guard=True)
    saved = np.load(SAVED)
    assert np.abs(P - saved).max() == 0.0
    assert (P.argmax(1) == yv).mean() == pytest.approx(0.8993, abs=5e-5)
