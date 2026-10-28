"""Functions for loading the LendSmart loan data."""

# pandas is the library we use to work with tables of data (called DataFrames).
import pandas as pd

# train_test_split is scikit-learn's tool for randomly dividing data into parts.
from sklearn.model_selection import train_test_split

# The path to the data file comes from config.py, where all paths live.
# Because it's a full path, load_data() works no matter where you run it from.
from lendsmart.config import DATA_PATH

# The column we want to predict: 1 = defaulted, 0 = did not.
TARGET = "SeriousDlqin2yrs"

# A fixed "seed" for the random split. Using the same number every time
# means the split comes out exactly the same on every run and every computer.
RANDOM_STATE = 42


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


def clean_data(df):
    """Apply the fixed cleaning rules that are safe to use on all the data.

    "Fixed" means the rule doesn't learn anything from the data, so it can't
    cause leakage. It's the same rule for train, validation and test.

    - Drop rows with age under 18 (impossible for a borrower, a data error).
    - Add an 'income_missing' column: 1 if MonthlyIncome is blank, 0 if not.
      Missing income may itself help predict default, and it marks the rows
      where DebtRatio looks like a dollar amount instead of a ratio.

    Very old ages and high incomes are kept, because they are real.
    Capping extreme values is NOT done here: it learns from the data, so it
    happens inside the logistic regression pipeline (see Capper in models.py).
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


def split_data(df):
    """Split the data into three parts: 60% train, 20% validation, 20% test.

    - train: the models learn from this.
    - validation: used to compare the two models and choose the approval cutoff.
    - test: used ONCE at the very end, for an honest final score.

    Returns three DataFrames: train, val, test.
    """
    # Remove exact duplicate rows first. Otherwise the same borrower could
    # end up in both train and test, and the model would be "tested" on a row
    # it already learned from, which makes the score look better than it is.
    df = df.drop_duplicates()

    # First cut: take 20% of the rows away as the test set.
    # stratify=df[TARGET] keeps the default rate (about 6.7%) the same in
    # both parts, so neither part ends up with unusually many or few defaults.
    rest, test = train_test_split(
        df, test_size=0.20, stratify=df[TARGET], random_state=RANDOM_STATE
    )

    # Second cut: split the remaining 80% into train and validation.
    # 0.25 of the remaining 80% is 20% of the original data.
    train, val = train_test_split(
        rest, test_size=0.25, stratify=rest[TARGET], random_state=RANDOM_STATE
    )

    return train, val, test
