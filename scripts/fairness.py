"""Check whether the final model treats age groups fairly, on the test set.

Run it from the main project folder (after scripts/final_evaluation.py) with:
    .venv/bin/python scripts/fairness.py

It uses the model and cutoff that were locked in final_evaluation.py, so
nothing here changes the model. It only looks at how its decisions fall.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lendsmart.config import MODELS_DIR, REPORTS_DIR
from lendsmart.data import TARGET, clean_data, load_data, split_data
from lendsmart.eda import age_band, bar_chart
from lendsmart.models import FEATURES


def band_summary(group):
    """Work out the fairness numbers for one age band.

    group is a DataFrame of test borrowers with columns: TARGET, prob, approved.
    """
    good = group[TARGET] == 0  # borrowers who really repaid
    return pd.Series(
        {
            "borrowers": len(group),
            # How often this band really defaults (the "ground truth").
            "actual_default_rate": group[TARGET].mean(),
            # How often the model says yes.
            "approval_rate": group["approved"].mean(),
            # Of those approved, how many still defaulted?
            "default_rate_if_approved": group.loc[group["approved"], TARGET].mean(),
            # Of the people who WOULD have repaid, how many did we wrongly turn away?
            # This is the key fairness number: it's the harm to good borrowers.
            "good_borrowers_rejected": (~group.loc[good, "approved"]).mean(),
            # How well the model ranks risk WITHIN this band.
            "roc_auc": roc_auc_score(group[TARGET], group["prob"]),
        }
    )


def main():
    # The locked choices from final_evaluation.py.
    results = json.loads((REPORTS_DIR / "final_results.json").read_text())
    model = joblib.load(MODELS_DIR / f"{results['model']}.joblib")
    cutoff = results["cutoff"]

    _train, _val, test = split_data(clean_data(load_data()))
    test = test.copy()
    test["prob"] = model.predict_proba(test[FEATURES])[:, 1]
    test["approved"] = test["prob"] < cutoff

    # One row of numbers per age band.
    table = test.groupby(age_band(test["age"]), observed=True)[
        [TARGET, "prob", "approved"]
    ].apply(band_summary)

    # Print it as percentages, which are easier to read.
    readable = table.copy()
    for column in readable.columns:
        if column == "borrowers":
            readable[column] = readable[column].astype(int)
        elif column == "roc_auc":
            readable[column] = readable[column].round(3)
        else:
            readable[column] = (readable[column] * 100).round(1).astype(str) + "%"
    print(f"Fairness by age band (test set, cutoff {cutoff:.2f}):\n")
    print(readable.T.to_string())

    # Chart: approval rate by age band.
    bar_chart(
        table["approval_rate"],
        title="Approval rate by age band",
        xlabel="Age band",
        ylabel="Approval rate (%)",
        file_name="approval_rate_by_age_band.png",
    )

    # Save the numbers for the README.
    path = REPORTS_DIR / "fairness_by_age.json"
    path.write_text(json.dumps(table.round(4).to_dict(orient="index"), indent=2))
    print(f"Saved {path.name}")


if __name__ == "__main__":
    main()
