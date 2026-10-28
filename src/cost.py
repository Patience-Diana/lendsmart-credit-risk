"""Costs of wrong lending decisions, used to choose the approval cutoff."""

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
