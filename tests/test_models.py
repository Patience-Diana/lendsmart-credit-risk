"""Tests for lendsmart.models and lendsmart.evaluate.

Run them from the main project folder with:
    .venv/bin/python -m pytest
"""

import numpy as np
import pandas as pd
import pytest

from lendsmart.evaluate import score_model
from lendsmart.models import (
    FEATURES,
    Capper,
    build_logistic_regression,
    build_xgboost,
)


def make_fake_features(n_rows=500):
    """Build a small made-up table with all the model's feature columns.

    The target depends on late payments, so there is a real pattern to learn.
    """
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {column: rng.uniform(0, 10, size=n_rows) for column in FEATURES}
    )
    # A couple of blank incomes, like the real data.
    X.loc[[3, 4], "MonthlyIncome"] = np.nan
    # Default (1) when the borrower has many late payments.
    y = (X["NumberOfTimes90DaysLate"] > 7).astype(int)
    return X, y


# ---------- Capper ----------


def test_capper_learns_caps_from_fit_data_only():
    train = pd.DataFrame({"income": [1.0, 2.0, 3.0, 4.0, 100.0]})
    other = pd.DataFrame({"income": [1_000_000.0]})
    capper = Capper(quantile=0.99).fit(train)
    # The cap comes from the training data (just under 100), so a huge value
    # in other data is cut down to it, and doesn't change the cap itself.
    assert capper.transform(other)["income"].iloc[0] < 100
    assert capper.caps_["income"] < 100


def test_capper_leaves_normal_values_and_blanks_alone():
    train = pd.DataFrame({"income": [1.0, 2.0, 3.0, 4.0, 100.0]})
    new = pd.DataFrame({"income": [2.5, np.nan]})
    result = Capper().fit(train).transform(new)
    assert result["income"].iloc[0] == 2.5
    assert np.isnan(result["income"].iloc[1])


# ---------- both models ----------


# parametrize runs the same test once for each model builder in the list.
@pytest.mark.parametrize("build", [build_logistic_regression, build_xgboost])
def test_model_gives_probabilities_between_0_and_1(build):
    X, y = make_fake_features()
    model = build().fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    assert len(probs) == len(X)
    assert ((probs >= 0) & (probs <= 1)).all()


@pytest.mark.parametrize("build", [build_logistic_regression, build_xgboost])
def test_model_learns_a_simple_pattern(build):
    X, y = make_fake_features()
    model = build().fit(X, y)
    probs = model.predict_proba(X)[:, 1]
    # The pattern is easy (many late payments = default), so ranking should be
    # far better than a coin toss.
    assert score_model(y, probs)["roc_auc"] > 0.9


# ---------- score_model ----------


def test_score_model_perfect_and_backwards():
    y = [0, 0, 1, 1]
    # Defaulters get the highest scores: a perfect ranking.
    assert score_model(y, [0.1, 0.2, 0.8, 0.9])["roc_auc"] == 1.0
    # Defaulters get the lowest scores: exactly backwards.
    assert score_model(y, [0.9, 0.8, 0.2, 0.1])["roc_auc"] == 0.0
