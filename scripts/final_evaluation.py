"""Choose the approval cutoff on validation, then score the final model ONCE on test.

Run it from the main project folder (after scripts/train.py) with:
    .venv/bin/python scripts/final_evaluation.py

Order matters here:
1. Every choice (which model, which cutoff) is made on the VALIDATION set.
2. Those choices are then locked, and the TEST set is scored once.
Because the test set never influenced a choice, its score is an honest
estimate of how the model would do on brand-new borrowers.
"""

import json

import joblib
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lendsmart.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from lendsmart.cost import best_cutoff, cost_by_cutoff, total_cost
from lendsmart.data import TARGET, clean_data, load_data, split_data
from lendsmart.eda import BAR_COLOR, GRID_COLOR, TEXT_COLOR
from lendsmart.evaluate import score_model
from lendsmart.models import FEATURES

# XGBoost was the winner on validation (higher ROC-AUC, PR-AUC and lower cost).
CHOSEN_MODEL = "xgboost"


def save_cost_chart(costs, chosen_cutoff):
    """Save a line chart of total cost at each cutoff, marking the chosen one."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(costs["cutoff"], costs["cost"], color=BAR_COLOR, linewidth=2)

    # Mark the cheapest cutoff with a dot and a label.
    chosen_cost = costs.loc[costs["cutoff"] == chosen_cutoff, "cost"].iloc[0]
    ax.scatter([chosen_cutoff], [chosen_cost], color=BAR_COLOR, s=60, zorder=3)
    # The label sits above the dot, with a thin arrow pointing down to it.
    ax.annotate(
        f"Lowest cost: cutoff {chosen_cutoff:.2f}",
        (chosen_cutoff, chosen_cost),
        xytext=(10, 70),
        textcoords="offset points",
        color=TEXT_COLOR,
        arrowprops={"arrowstyle": "-", "color": TEXT_COLOR, "linewidth": 1},
    )
    # Show y-axis numbers with commas (8,000 instead of 8000).
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))

    ax.set_title("Total cost of mistakes at each approval cutoff", loc="left", fontsize=13)
    ax.set_xlabel("Cutoff (reject if chance of default is at least this)", color=TEXT_COLOR)
    ax.set_ylabel("Total cost (validation set)", color=TEXT_COLOR)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_COLOR, length=0)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "cost_by_cutoff.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.name}")


def decision_summary(y_true, probs, cutoff):
    """Describe what happens when the cutoff is applied to a group of borrowers."""
    y_true = np.asarray(y_true)
    approved = np.asarray(probs) < cutoff
    return {
        "approval_rate": float(approved.mean()),
        # Of the people we approve, how many still default?
        "default_rate_among_approved": float(y_true[approved].mean()),
        "total_cost": int(total_cost(y_true, probs, cutoff)),
        # The cost if we had no model and approved everyone (cutoff above 1).
        "cost_if_approve_everyone": int(total_cost(y_true, probs, 1.01)),
    }


def main():
    _train, val, test = split_data(clean_data(load_data()))
    model = joblib.load(MODELS_DIR / f"{CHOSEN_MODEL}.joblib")

    # ---- Step 1: choose the cutoff on VALIDATION ----
    val_probs = model.predict_proba(val[FEATURES])[:, 1]
    cutoff = best_cutoff(val[TARGET], val_probs)
    save_cost_chart(cost_by_cutoff(val[TARGET], val_probs), cutoff)
    print(f"Chosen on validation: model = {CHOSEN_MODEL}, cutoff = {cutoff:.2f}\n")

    # ---- Step 2: score ONCE on TEST, with everything locked ----
    test_probs = model.predict_proba(test[FEATURES])[:, 1]
    results = {
        "model": CHOSEN_MODEL,
        "cutoff": float(cutoff),
        "test_rows": len(test),
        **{k: float(v) for k, v in score_model(test[TARGET], test_probs).items()},
        **decision_summary(test[TARGET], test_probs, cutoff),
    }

    print("Final results on the TEST set:")
    for key, value in results.items():
        print(f"  {key:30} {value}")

    # Save the numbers so the README (and you) can refer back to them.
    path = REPORTS_DIR / "final_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {path.name}")


if __name__ == "__main__":
    main()
