"""Explain the XGBoost model with SHAP: which features push risk up or down.

Run it from the main project folder (after scripts/train.py) with:
    .venv/bin/python scripts/explain.py

SHAP splits each borrower's prediction into one contribution per feature.
A positive contribution pushes that borrower's risk UP, a negative one DOWN.
"""

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import xgboost

from lendsmart.config import FIGURES_DIR, MODELS_DIR
from lendsmart.data import clean_data, load_data, split_data
from lendsmart.models import FEATURES

# SHAP is slow on all 29,878 test rows, and 2,000 random rows give the same picture.
SAMPLE_SIZE = 2000


def main():
    _train, _val, test = split_data(clean_data(load_data()))
    model = joblib.load(MODELS_DIR / "xgboost.joblib")

    # A random sample of test borrowers. random_state makes it the same sample every run.
    sample = test[FEATURES].sample(SAMPLE_SIZE, random_state=42)

    # Ask XGBoost itself for the SHAP values. pred_contribs=True returns, for each
    # borrower, one contribution per feature plus a final "starting point" column.
    # (shap.TreeExplainer would do the same, but shap 0.49 can't read models saved
    # by xgboost 3.x, so we let XGBoost do the maths and use shap only for the chart.)
    contributions = model.get_booster().predict(
        xgboost.DMatrix(sample), pred_contribs=True
    )
    # Package the numbers the way shap's charts expect them.
    shap_values = shap.Explanation(
        values=contributions[:, :-1],  # one column per feature
        base_values=contributions[:, -1],  # the starting point, the same for everyone
        data=sample.values,  # the borrowers' actual values, used for the colors
        feature_names=FEATURES,
    )

    # Beeswarm chart: one row per feature, one dot per borrower.
    # - Left/right position: did this feature push the borrower's risk down or up?
    # - Color: was the feature's value low (blue) or high (red) for that borrower?
    # show=False lets us save the chart instead of opening a window.
    shap.plots.beeswarm(shap_values, max_display=len(FEATURES), show=False)
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    # fig.axes[0] is the main chart (fig.axes[1] is the color bar on the right).
    fig.axes[0].set_title("What drives the predicted risk of default", loc="left")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "shap_summary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.name}\n")

    # Rank features by their average impact (the average size of their contribution,
    # ignoring whether it pushed risk up or down).
    impact = abs(shap_values.values).mean(axis=0)
    ranking = sorted(zip(FEATURES, impact), key=lambda pair: pair[1], reverse=True)
    print("Average impact on the model's output, biggest first:")
    for feature, value in ranking:
        print(f"  {feature:40} {value:.3f}")


if __name__ == "__main__":
    main()
