"""Functions for loading the LendSmart loan data."""

# Path helps us build file paths that work on any computer (Mac, Windows, Linux).
from pathlib import Path

# pandas is the library we use to work with tables of data (called DataFrames).
import pandas as pd

# Find the project's main folder.
# __file__ is this file (src/data.py), .parent is the src/ folder,
# and .parent again is the main LENDSMART/ folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The full path to the data file, built from the main folder.
# Building it this way means load_data() works no matter where you run it from,
# for example from inside notebooks/ or from the main folder.
DATA_PATH = PROJECT_ROOT / "data" / "cs-training.csv"


def load_data(path=DATA_PATH):
    """Read the loan CSV file and return it as a pandas DataFrame.

    Each row is one borrower. The target column is SeriousDlqin2yrs:
    1 means the person defaulted, 0 means they did not.
    """
    # Read the CSV file into a DataFrame.
    # index_col=0 tells pandas that the first column (a plain row number
    # with no name) is just a label for each row, not real information.
    # Without it, we'd get an extra useless column called "Unnamed: 0".
    df = pd.read_csv(path, index_col=0)

    # Give the data back to whoever called this function.
    return df


# The columns we cap at the 99th percentile (for logistic regression).
CAP_COLUMNS = ["MonthlyIncome", "DebtRatio"]


def clean_data(df):
    """Apply the fixed cleaning rules that are safe to use on all the data.

    "Fixed" means the rule doesn't learn anything from the data, so it can't
    cause leakage. It's the same rule for train, validation and test.

    - Drop rows with age under 18 (impossible for a borrower, a data error).
    - Add an 'income_missing' column: 1 if MonthlyIncome is blank, 0 if not.
      Missing income may itself help predict default, and it marks the rows
      where DebtRatio looks like a dollar amount instead of a ratio.

    Very old ages and high incomes are kept, because they are real.
    """
    # Work on a copy so the original DataFrame passed in is left unchanged.
    df = df.copy()

    # Keep only rows where age is 18 or more.
    df = df[df["age"] >= 18]

    # isna() gives True/False for each row; astype(int) turns it into 1/0.
    # This must be done BEFORE missing incomes are filled in later,
    # otherwise we'd lose track of which ones were blank.
    df["income_missing"] = df["MonthlyIncome"].isna().astype(int)

    return df


def fit_caps(train_df):
    """Work out the 99th-percentile cap for each column in CAP_COLUMNS.

    Only pass the TRAINING data here. If we used all the data, the caps
    would "see" the test rows, which is leakage.
    Returns a dictionary like {"MonthlyIncome": 25000.0, "DebtRatio": 4979.0}.
    """
    caps = {}
    for column in CAP_COLUMNS:
        # quantile(0.99) is the value that 99% of rows are at or below.
        # Blank values are skipped automatically.
        caps[column] = train_df[column].quantile(0.99)
    return caps


def apply_caps(df, caps):
    """Cap each column at the given limit (values above it become the limit).

    Use the caps from fit_caps(train_df) on train, validation and test alike.
    This is for logistic regression, which is thrown off by extreme values.
    XGBoost can use the uncapped data.
    """
    df = df.copy()
    for column, cap in caps.items():
        # clip(upper=cap) replaces anything bigger than cap with cap.
        # Blank values stay blank; they get filled in later in the model pipeline.
        df[column] = df[column].clip(upper=cap)
    return df
