"""Train both models, compare them on the validation set, and save them.

Run it from the main project folder with:
    .venv/bin/python scripts/train.py

The test set is NOT touched here. It is kept for one final, honest score
after the model and approval cutoff have been chosen.
"""

# joblib saves Python objects (like a trained model) to a file and loads them back.
import joblib

from lendsmart.config import MODELS_DIR
from lendsmart.data import TARGET, clean_data, load_data, split_data
from lendsmart.evaluate import score_model
from lendsmart.models import FEATURES, build_logistic_regression, build_xgboost


def main():
    # 1. Load, clean and split the data (the same split every time).
    train, val, _test = split_data(clean_data(load_data()))
    print(f"Train: {len(train):,} rows   Validation: {len(val):,} rows\n")

    # 2. The two models to compare, by name.
    models = {
        "logistic_regression": build_logistic_regression(),
        "xgboost": build_xgboost(),
    }

    # Make the models/ folder if it doesn't exist yet (it is ignored by git).
    MODELS_DIR.mkdir(exist_ok=True)

    print(f"{'Model':<22}{'ROC-AUC':>10}{'PR-AUC':>10}")
    for name, model in models.items():
        # 3. Learn from the training data only.
        model.fit(train[FEATURES], train[TARGET])

        # 4. Predict the chance of default for each validation borrower.
        # predict_proba gives two columns (chance of 0, chance of 1); [:, 1] keeps the second.
        probs = model.predict_proba(val[FEATURES])[:, 1]

        # 5. Score those predictions against what really happened.
        scores = score_model(val[TARGET], probs)
        print(f"{name:<22}{scores['roc_auc']:>10.4f}{scores['pr_auc']:>10.4f}")

        # 6. Save the trained model, so later steps (cutoff, SHAP, API) can reuse it
        #    without training again.
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    print(f"\nRandom guessing would score ROC-AUC 0.5000 and PR-AUC {val[TARGET].mean():.4f}")
    print(f"Models saved to {MODELS_DIR}")


# Only run main() when this file is run directly, not when it is imported.
if __name__ == "__main__":
    main()
