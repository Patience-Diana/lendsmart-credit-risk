"""Quick first look at the LendSmart loan data (EDA = exploratory data analysis)."""

# Reuse the loader we already wrote, so the data is read the same way everywhere.
from src.data import load_data

# The name of the column we want to predict: 1 = defaulted, 0 = did not.
TARGET = "SeriousDlqin2yrs"


def overview(df):
    """Print a short summary of a loan DataFrame."""
    # 1. Shape: how big the table is.
    # df.shape gives (number of rows, number of columns).
    rows, cols = df.shape
    print(f"Shape: {rows:,} rows and {cols} columns")
    print()

    # 2. Column types: what kind of values each column holds.
    # int64 = whole numbers, float64 = numbers with decimals.
    # Note: a column with missing values is shown as float64 even if it
    # holds whole numbers, because pandas needs decimals to store "missing".
    print("Column types:")
    print(df.dtypes.to_string())
    print()

    # 3. Missing values: how many blanks each column has.
    # isna() marks each blank cell as True; sum() counts the Trues per column.
    missing = df.isna().sum()
    # Also show it as a percentage of all rows, which is easier to judge.
    missing_pct = missing / rows * 100
    print("Missing values per column:")
    for column in df.columns:
        print(f"  {column}: {missing[column]:,} ({missing_pct[column]:.1f}%)")
    print()

    # 4. Default rate: the share of borrowers who defaulted.
    # The target is 0 or 1, so its average is the fraction of 1s.
    defaults = df[TARGET].sum()
    default_rate = df[TARGET].mean() * 100
    print(f"Default rate: {default_rate:.2f}% ({defaults:,} of {rows:,} borrowers defaulted)")


# This block only runs when you run the file directly, not when you import it.
# Run it from the main project folder with:
#   .venv/bin/python -m src.eda
if __name__ == "__main__":
    data = load_data()
    overview(data)
