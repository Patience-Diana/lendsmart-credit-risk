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
