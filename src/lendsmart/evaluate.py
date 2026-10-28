"""Score how well a model's predicted probabilities separate defaulters from non-defaulters."""

from sklearn.metrics import average_precision_score, roc_auc_score


def score_model(y_true, probs):
    """Return a dictionary with two scores for a set of predictions.

    y_true: the real outcomes (1 = defaulted, 0 = did not).
    probs: the model's predicted chance of default for the same borrowers.

    - ROC-AUC: pick one random defaulter and one random non-defaulter. ROC-AUC is
      the chance the model gives the defaulter the higher risk score.
      0.5 = coin toss, 1.0 = perfect ranking.
    - PR-AUC (average precision): how precise the model's "high risk" list stays
      as you go down it. It focuses on the rare defaulters, so it's a tougher,
      more honest score when only 6.7% of borrowers default.
      Guessing at random scores about 0.067 (the default rate), 1.0 = perfect.
    """
    return {
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
    }
