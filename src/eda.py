"""Quick first look at the LendSmart loan data (EDA = exploratory data analysis)."""

# matplotlib draws charts. "Agg" makes it save charts to files without
# trying to open a window, which is what we want when running a script.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Reuse the loader we already wrote, so the data is read the same way everywhere.
from src.data import PROJECT_ROOT, clean_data, load_data

# The name of the column we want to predict: 1 = defaulted, 0 = did not.
TARGET = "SeriousDlqin2yrs"

# Where the charts are saved.
REPORTS_DIR = PROJECT_ROOT / "reports"

# Chart colors: one blue for the bars, gray for text and gridlines.
BAR_COLOR = "#2a78d6"
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"


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


def iqr_bounds(series):
    """Return the (low, high) limits of the IQR rule for one column.

    The IQR rule is a common way to spot outliers (unusually small or large values):
    - Q1 is the value 25% of the way through the sorted data, Q3 is 75% of the way.
    - IQR (interquartile range) = Q3 - Q1, the spread of the middle half of the data.
    - Anything below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR is called an outlier.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def count_iqr_outliers(df, column):
    """Print how many values in a column fall outside the IQR limits."""
    # dropna() skips the blanks, because a missing value isn't an outlier.
    values = df[column].dropna()
    low, high = iqr_bounds(values)
    n_low = (values < low).sum()
    n_high = (values > high).sum()
    print(f"  IQR limits: {low:,.2f} to {high:,.2f}")
    print(f"  Below the low limit: {n_low:,}   Above the high limit: {n_high:,}")


def check_outliers(df):
    """Count outliers in age, MonthlyIncome and DebtRatio and suggest a fix for each.

    This only looks at the data. It does not change anything.
    Any limits we later use to fix outliers (for example the 99th percentile)
    must be worked out on the training set only, to avoid leakage.
    """
    # --- age ---
    print("age")
    count_iqr_outliers(df, "age")
    # Besides the IQR rule, check values that make no sense for a borrower.
    under_18 = (df["age"] < 18).sum()
    over_100 = (df["age"] > 100).sum()
    print(f"  Under 18 (impossible for a loan): {under_18:,}   Over 100: {over_100:,}")
    print("  Suggestion: drop the under-18 row(s), they are data errors.")
    print("  Keep the very old ages: they are rare but possible, and there are few of them.")
    print()

    # --- MonthlyIncome ---
    print("MonthlyIncome")
    count_iqr_outliers(df, "MonthlyIncome")
    zero_income = (df["MonthlyIncome"] == 0).sum()
    missing_income = df["MonthlyIncome"].isna().sum()
    # The 99th percentile: 99% of incomes are at or below this value.
    p99_income = df["MonthlyIncome"].quantile(0.99)
    print(f"  Exactly 0: {zero_income:,}   Missing: {missing_income:,}")
    print(f"  99th percentile: {p99_income:,.0f}")
    print("  Suggestion: high incomes are mostly real (well-paid people), so don't delete them.")
    print("  Cap them at the 99th percentile for logistic regression, which is")
    print("  sensitive to extreme values. XGBoost can use them as they are.")
    print("  Fill missing incomes with the median and add a yes/no 'income_missing' column,")
    print("  because whether income is missing may itself help predict default.")
    print()

    # --- DebtRatio ---
    print("DebtRatio")
    count_iqr_outliers(df, "DebtRatio")
    # A ratio above 1 means debts and costs are bigger than income.
    above_1 = df["DebtRatio"] > 1
    # Check how many of those rows also have no income recorded.
    above_1_no_income = (above_1 & df["MonthlyIncome"].isna()).sum()
    p99_debt = df["DebtRatio"].quantile(0.99)
    print(f"  Above 1: {above_1.sum():,}   Of those, income is missing: {above_1_no_income:,}")
    print(f"  99th percentile: {p99_debt:,.2f}")
    print("  Suggestion: most huge ratios come from rows with no income, where the")
    print("  'ratio' looks like a raw dollar amount of debt instead of a ratio.")
    print("  Keep the 'income_missing' column so the model can tell these rows apart,")
    print("  and cap DebtRatio at the 99th percentile for logistic regression.")


def bar_chart(rates, title, xlabel, file_name):
    """Draw a bar chart of default rates and save it as a PNG in reports/.

    rates is a pandas Series: the index holds the group names (e.g. "18-29")
    and the values hold the default rate for each group (e.g. 0.117).
    """
    # Make the reports/ folder if it doesn't exist yet.
    REPORTS_DIR.mkdir(exist_ok=True)

    # Turn rates like 0.117 into percentages like 11.7, which are easier to read.
    percents = rates * 100

    # Create an empty chart (fig is the whole image, ax is the plotting area).
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Draw one bar per group. width=0.5 leaves space between the bars.
    labels = [str(group) for group in percents.index]
    bars = ax.bar(labels, percents.values, width=0.5, color=BAR_COLOR)

    # Write each bar's value just above it, so readers don't have to guess.
    ax.bar_label(bars, fmt="%.1f%%", padding=3, color=TEXT_COLOR)
    # Leave some empty space above the tallest bar so its label doesn't hit the title.
    ax.margins(y=0.12)

    # Titles and axis labels.
    ax.set_title(title, loc="left", fontsize=13, color="#0b0b0b")
    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel("Default rate (%)", color=TEXT_COLOR)

    # Make the frame and gridlines light, so the bars stand out.
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1)
    ax.set_axisbelow(True)  # draw gridlines behind the bars
    ax.tick_params(colors=TEXT_COLOR, length=0)

    # Save the image and close it to free up memory.
    path = REPORTS_DIR / file_name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


def make_charts(df):
    """Save two charts to reports/: default rate by age band and by 30-59 days late count.

    Pass in cleaned data (from clean_data), so the impossible age-0 row is gone.
    """
    # --- Chart 1: default rate by age band ---
    # Takeaway: default risk falls steadily with age, and 18-29 year olds default
    # almost 4 times as often as people aged 60+ (about 11.7% vs 3.1%).
    # pd.cut sorts each age into a band. The edges mean: 18-29, 30-44, 45-59, 60+.
    age_band = pd.cut(
        df["age"],
        bins=[17, 29, 44, 59, 200],
        labels=["18-29", "30-44", "45-59", "60+"],
    )
    # groupby puts rows in the same band together; mean() of the 0/1 target
    # gives the share who defaulted in each band.
    rates_by_age = df.groupby(age_band, observed=True)[TARGET].mean()
    bar_chart(
        rates_by_age,
        title="Younger borrowers default more often",
        xlabel="Age band",
        file_name="default_rate_by_age_band.png",
    )

    # --- Chart 2: default rate by number of times 30-59 days late ---
    # Takeaway: even one 30-59 day late payment nearly quadruples the default
    # rate (about 4% to 15%), and it keeps climbing to about 45% at 4 or more.
    late = df["NumberOfTime30-59DaysPastDueNotWorse"]
    # 96 and 98 are error codes, not real counts, so leave them out of this chart.
    # where(late < 90) turns them into blanks, which groupby skips.
    late = late.where(late < 90)
    # Very few people were late 5+ times, so group 4 and above into one "4+" bar.
    late_group = late.clip(upper=4).map(
        {0: "0", 1: "1", 2: "2", 3: "3", 4: "4+"}
    )
    rates_by_late = df.groupby(late_group)[TARGET].mean()
    # Put the bars in a sensible order (groupby sorts text, which happens to
    # match here, but being explicit is safer).
    rates_by_late = rates_by_late.reindex(["0", "1", "2", "3", "4+"])
    bar_chart(
        rates_by_late,
        title="Each past late payment raises the default rate",
        xlabel="Number of times 30-59 days late (last 2 years)",
        file_name="default_rate_by_30_59_days_late.png",
    )


# This block only runs when you run the file directly, not when you import it.
# Run it from the main project folder with:
#   .venv/bin/python -m src.eda
if __name__ == "__main__":
    data = load_data()
    overview(data)
    print()
    print("=" * 60)
    print("Outlier check")
    print("=" * 60)
    check_outliers(data)
    print()
    # The charts use cleaned data, so the impossible age-0 row is left out.
    make_charts(clean_data(data))
