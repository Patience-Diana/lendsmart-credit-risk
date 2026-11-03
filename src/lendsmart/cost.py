"""Costs of wrong lending decisions, used to choose the approval cutoff."""

import numpy as np
import pandas as pd

# ASSUMED COSTS: the dataset has no loan amounts or interest rates, so we can't
# know what each mistake really costs. Instead we assume how bad each mistake is
# compared with the other. These are relative units, not real money.
#
# The two mistakes a lender can make:
# - Approving a "bad loan": lending to someone who then defaults.
#   The lender can lose most of the money it lent.
# - Rejecting a "good" borrower: turning away someone who would have repaid.
#   The lender only loses the profit (interest) it would have earned.
#
# So a bad loan is assumed to cost 5 times as much as a rejected good borrower.
# Change these two numbers to see how the best cutoff moves.
COST_BAD_LOAN = 5
COST_REJECTED_GOOD = 1


def total_cost(y_true, probs, cutoff):
    """Return the total cost of the approve/reject decisions at one cutoff.

    The rule: APPROVE a borrower if their predicted chance of default is
    below the cutoff, otherwise REJECT them.

    y_true: the real outcomes (1 = defaulted, 0 = repaid).
    probs: the model's predicted chance of default for the same borrowers.
    """
    # np.asarray turns lists or pandas columns into plain arrays we can compare.
    y_true = np.asarray(y_true)
    approved = np.asarray(probs) < cutoff

    # Mistake 1: approved, but they defaulted.
    bad_loans = (approved & (y_true == 1)).sum()
    # Mistake 2: rejected, but they would have repaid.
    # "~approved" means "not approved", i.e. rejected.
    rejected_good = (~approved & (y_true == 0)).sum()

    return COST_BAD_LOAN * bad_loans + COST_REJECTED_GOOD * rejected_good


def cost_by_cutoff(y_true, probs):
    """Try every cutoff from 0.01 to 0.99 and return the cost of each.

    Returns a DataFrame with columns: cutoff, cost, approval_rate.
    """
    rows = []
    # np.arange(0.01, 1.0, 0.01) gives 0.01, 0.02, ..., 0.99.
    for cutoff in np.arange(0.01, 1.0, 0.01):
        rows.append(
            {
                "cutoff": round(cutoff, 2),
                "cost": total_cost(y_true, probs, cutoff),
                # The share of borrowers who would be approved at this cutoff.
                "approval_rate": (np.asarray(probs) < cutoff).mean(),
            }
        )
    return pd.DataFrame(rows)


def best_cutoff(y_true, probs):
    """Return the cutoff with the lowest total cost."""
    costs = cost_by_cutoff(y_true, probs)
    # idxmin() finds the row with the smallest cost.
    return costs.loc[costs["cost"].idxmin(), "cutoff"]
