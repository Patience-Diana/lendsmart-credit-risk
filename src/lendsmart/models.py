"""Build the two models: logistic regression (simple baseline) and XGBoost."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Same fixed seed as the split, so XGBoost's random choices repeat exactly.
RANDOM_STATE = 42

# The columns the models learn from. These are all the columns except the
# target, plus the income_missing flag that clean_data() adds.
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "income_missing",
]


class Capper(BaseEstimator, TransformerMixin):
    """Cap every column at its 99th percentile, learned from the training data.

    This is a pipeline step, so it follows scikit-learn's two-stage pattern:
    - fit(X): LEARN the caps (only ever called on training data).
    - transform(X): APPLY the caps (to train, validation, test, and new borrowers).
    Keeping the learning inside fit() is what stops validation/test data leaking in.
    """

    def __init__(self, quantile=0.99):
        self.quantile = quantile

    def fit(self, X, y=None):
        # Work out each column's cap. Blank values are skipped automatically.
        # The trailing underscore in caps_ is a scikit-learn convention:
        # it marks something learned during fit().
        self.caps_ = X.quantile(self.quantile)
        return self

    def transform(self, X):
        # Replace anything above a column's cap with the cap.
        # axis=1 tells clip to line the caps up with the columns.
        return X.clip(upper=self.caps_, axis=1)


def build_logistic_regression():
    """Return an untrained logistic regression pipeline.

    A Pipeline chains steps together. When you call pipeline.fit(train), every
    step learns from the training data only. When you call
    pipeline.predict_proba(other_data), every step reuses what it learned.
    """
    return Pipeline(
        steps=[
            # 1. Cap extreme values. Logistic regression is thrown off by them
            #    (e.g. DebtRatio up to 329,664, or the 96/98 codes in the late columns).
            ("cap", Capper(quantile=0.99)),
            # 2. Fill blanks (MonthlyIncome, NumberOfDependents) with the median
            #    of the training data. The income_missing column still remembers
            #    which incomes were blank.
            ("impute", SimpleImputer(strategy="median")),
            # 3. Put every column on the same scale (average 0, spread 1).
            #    Otherwise a column in the thousands (income) would swamp a
            #    column in single digits (late payments).
            ("scale", StandardScaler()),
            # 4. The model itself. It learns one weight per column and turns the
            #    weighted sum into a probability of default between 0 and 1.
            #    max_iter gives it enough attempts to finish learning.
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )


def build_xgboost():
    """Return an untrained XGBoost model.

    XGBoost builds many small decision trees, one after another. Each new tree
    focuses on the borrowers the earlier trees got most wrong, and the final
    prediction adds up all the trees' answers.

    Unlike logistic regression, it needs no capping, filling or scaling:
    - Trees split on "is this value above or below X?", so a DebtRatio of
      329,664 is simply "above X". Extreme values do no harm.
    - It handles blank values itself, by learning which side of each split
      the blanks should go to.
    """
    return XGBClassifier(
        # How many trees to build.
        n_estimators=300,
        # How much each new tree is allowed to change the prediction.
        # Small steps (0.05) learn more slowly but more reliably.
        learning_rate=0.05,
        # How many questions each tree can ask in a row. Shallow trees (4)
        # are less likely to memorize the training data (overfitting).
        max_depth=4,
        # Each tree sees a random 80% of the rows and 80% of the columns.
        # This variety also helps against overfitting.
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )
