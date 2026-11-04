"""Tests for the API in lendsmart.api.

TestClient sends requests to the app directly in Python, so no server needs
to be running. A fake scorer replaces the real model, so the tests don't need
models/ (which is not in git) and their results are predictable.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lendsmart.api import Scorer, app, get_scorer
from lendsmart.models import FEATURES


class FakeModel:
    """Pretends to be the trained model: risk = 10% per time 90+ days late.

    It also checks it receives the same columns, in the same order and as
    numbers, that the real model was trained on.
    """

    def predict_proba(self, X):
        assert list(X.columns) == FEATURES
        # Every column must be a number: "i" = whole number, "f" = decimal.
        # (An "object" column is what crashed the real model before the fix.)
        assert all(dtype.kind in "if" for dtype in X.dtypes)
        risk = 0.1 * X["NumberOfTimes90DaysLate"].to_numpy()
        return np.column_stack([1 - risk, risk])


# A complete, valid request. Tests copy it and change one thing.
GOOD_BORROWER = {
    "credit_utilization": 0.3,
    "age": 40,
    "times_30_59_days_late": 0,
    "debt_ratio": 0.4,
    "monthly_income": 5000,
    "open_credit_lines": 6,
    "times_90_days_late": 0,
    "real_estate_loans": 1,
    "times_60_89_days_late": 0,
    "dependents": 2,
}


@pytest.fixture
def client():
    """A test client whose /score uses the fake model with a 0.14 cutoff."""
    # dependency_overrides tells FastAPI: wherever get_scorer is used, call this instead.
    app.dependency_overrides[get_scorer] = lambda: Scorer(model=FakeModel(), cutoff=0.14)
    yield TestClient(app)
    # Undo the override after the test, so other tests start clean.
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # The database may or may not be running during tests; either answer is fine,
    # as long as /health reports it instead of crashing.
    assert body["database"] in ("ok", "unreachable")


def test_low_risk_borrower_is_approved(client):
    response = client.post("/score", json=GOOD_BORROWER)
    assert response.status_code == 200
    body = response.json()
    assert body["probability_of_default"] == 0.0
    assert body["decision"] == "approve"
    assert body["cutoff"] == 0.14


def test_high_risk_borrower_is_rejected(client):
    # 2 times 90+ days late -> the fake model says 20%, above the 0.14 cutoff.
    response = client.post("/score", json={**GOOD_BORROWER, "times_90_days_late": 2})
    assert response.json()["decision"] == "reject"


def test_income_and_dependents_can_be_left_out(client):
    # This used to crash (None made the column non-numeric). It must work now.
    borrower = {k: v for k, v in GOOD_BORROWER.items() if k not in ("monthly_income", "dependents")}
    response = client.post("/score", json=borrower)
    assert response.status_code == 200


def test_under_18_is_refused(client):
    response = client.post("/score", json={**GOOD_BORROWER, "age": 15})
    # 422 = "Unprocessable Content": the request's data failed the checks.
    assert response.status_code == 422


def test_wrong_type_is_refused(client):
    response = client.post("/score", json={**GOOD_BORROWER, "debt_ratio": "banana"})
    assert response.status_code == 422


def test_missing_required_field_is_refused(client):
    borrower = {k: v for k, v in GOOD_BORROWER.items() if k != "age"}
    response = client.post("/score", json=borrower)
    assert response.status_code == 422
