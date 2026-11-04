"""A small web API that scores one borrower and returns approve or reject.

Start it from the main project folder (after scripts/train.py and
scripts/final_evaluation.py) with:
    .venv/bin/uvicorn lendsmart.api:app --reload

Then open http://127.0.0.1:8000/docs in a browser to try it out.
"""

import json
from dataclasses import dataclass
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field
from sqlalchemy import text

from lendsmart.config import MODELS_DIR, REPORTS_DIR
from lendsmart.data import clean_data
from lendsmart.db import DB_TO_CSV_COLUMNS, get_engine
from lendsmart.models import FEATURES

# The FastAPI app. uvicorn looks for this "app" object and serves it.
app = FastAPI(
    title="LendSmart",
    description="Scores a loan applicant's risk of default and suggests approve or reject.",
    version="0.1.0",
)


class Borrower(BaseModel):
    """What a request to /score must contain.

    The names match the database columns (snake_case). FastAPI checks every
    request against this description: a missing field, or a value of the wrong
    type or outside the allowed range, is rejected with a clear error message.
    ge = "greater than or equal to", le = "less than or equal to".
    """

    credit_utilization: float = Field(
        ge=0, description="Share of the credit limit in use (0.3 = 30%)"
    )
    age: int = Field(ge=18, le=120, description="Age in years")
    times_30_59_days_late: int = Field(ge=0, description="Times 30-59 days late, last 2 years")
    debt_ratio: float = Field(ge=0, description="Monthly debts and costs divided by income")
    # "float | None = None" means the field may be left out; it then counts as missing.
    monthly_income: float | None = Field(
        default=None, ge=0, description="Monthly income (leave out if unknown)"
    )
    open_credit_lines: int = Field(ge=0, description="Open loans and credit lines")
    times_90_days_late: int = Field(ge=0, description="Times 90+ days late")
    real_estate_loans: int = Field(ge=0, description="Mortgages and home equity loans")
    times_60_89_days_late: int = Field(ge=0, description="Times 60-89 days late, last 2 years")
    dependents: int | None = Field(
        default=None, ge=0, description="Number of dependents (leave out if unknown)"
    )


class Decision(BaseModel):
    """What /score sends back."""

    probability_of_default: float
    decision: str
    cutoff: float


@dataclass
class Scorer:
    """The trained model and the approval cutoff, kept together."""

    model: object
    cutoff: float


# lru_cache means this runs once: the model is loaded on the first request and
# then reused, instead of being read from disk for every borrower.
@lru_cache
def get_scorer():
    """Load the model and cutoff that were chosen in scripts/final_evaluation.py."""
    results = json.loads((REPORTS_DIR / "final_results.json").read_text())
    model = joblib.load(MODELS_DIR / f"{results['model']}.joblib")
    return Scorer(model=model, cutoff=results["cutoff"])


@lru_cache
def get_db_engine():
    """Create the database engine once, then reuse it (like get_scorer)."""
    return get_engine()


@app.get("/health")
def health():
    """A quick check that the API is running, and whether it can reach the database."""
    try:
        # "SELECT 1" is the smallest possible query: it only proves the
        # connection works.
        with get_db_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        # The API still answers, but reports that the database is unreachable
        # (e.g. it isn't running, or the settings are missing).
        database = "unreachable"
    return {"status": "ok", "database": database}


@app.post("/score", response_model=Decision)
def score(borrower: Borrower, scorer: Scorer = Depends(get_scorer)):
    """Score one borrower: chance of default, and approve or reject.

    Depends(get_scorer) asks FastAPI to hand this function the scorer. Tests can
    swap in a fake one, so they don't need the real trained model.
    """
    # 1. Turn the request into a one-row table with the original CSV column names,
    #    exactly like the data the model was trained on.
    #    astype(float) makes every column numeric. Without it, a field that was
    #    left out (None) would make its column type "object", which XGBoost
    #    refuses. As a float, None becomes NaN: the same "missing" as in training.
    row = (
        pd.DataFrame([borrower.model_dump()])
        .rename(columns=DB_TO_CSV_COLUMNS)
        .astype(float)
    )

    # 2. Apply the same cleaning as in training (this adds the income_missing flag).
    row = clean_data(row)

    # 3. Predict. [0, 1] = first (only) row, chance of class 1 (default).
    probability = float(scorer.model.predict_proba(row[FEATURES])[0, 1])

    # 4. The same rule as in final_evaluation.py: approve if below the cutoff.
    decision = "approve" if probability < scorer.cutoff else "reject"

    return Decision(
        probability_of_default=round(probability, 4),
        decision=decision,
        cutoff=scorer.cutoff,
    )
