"""Tests for lendsmart.data: cleaning and splitting.

Run them from the main project folder with:
    .venv/bin/python -m pytest

pytest finds every function whose name starts with "test_" and runs it.
Each test uses "assert": if the statement after it is False, the test fails.
"""

import numpy as np
import pandas as pd

from lendsmart.data import TARGET, clean_data, split_data


def make_fake_data(n_rows=1000):
    """Build a small made-up loan table, so the tests don't need the real CSV.

    It has exactly 10% defaults, plus two planted problems the code should catch:
    one borrower aged 0 and one duplicated row.
    """
    # A random number generator with a fixed seed, so the fake data is the same every run.
    rng = np.random.default_rng(0)

    df = pd.DataFrame(
        {
            # 1 in every 10 rows is a default (rows 0, 10, 20, ...).
            TARGET: [1 if i % 10 == 0 else 0 for i in range(n_rows)],
            "age": rng.integers(21, 90, size=n_rows),
            "MonthlyIncome": rng.uniform(1000, 20000, size=n_rows),
            "DebtRatio": rng.uniform(0, 2, size=n_rows),
        }
    )

    # Planted problem 1: an impossible age.
    df.loc[5, "age"] = 0
    # Planted problem 2: a blank income.
    df.loc[7, "MonthlyIncome"] = np.nan
    # Planted problem 3: an exact copy of row 1, added at the end with a new label.
    duplicate = df.loc[[1]].rename(index={1: n_rows})
    df = pd.concat([df, duplicate])

    return df


# ---------- clean_data ----------


def test_clean_data_drops_under_18():
    cleaned = clean_data(make_fake_data())
    # The age-0 row should be gone, so every remaining age is 18 or more.
    assert (cleaned["age"] >= 18).all()


def test_clean_data_flags_missing_income():
    cleaned = clean_data(make_fake_data())
    # Row 7 had a blank income, so its flag should be 1. Row 8 had an income, so 0.
    assert cleaned.loc[7, "income_missing"] == 1
    assert cleaned.loc[8, "income_missing"] == 0


def test_clean_data_does_not_change_the_original():
    raw = make_fake_data()
    clean_data(raw)
    # The original table should still have its age-0 row and no new column.
    assert (raw["age"] == 0).any()
    assert "income_missing" not in raw.columns


# ---------- split_data ----------


def test_split_removes_duplicates():
    train, val, test = split_data(make_fake_data())
    all_rows = pd.concat([train, val, test])
    # After the split, no row should appear twice anywhere.
    assert not all_rows.duplicated().any()


def test_split_parts_do_not_overlap():
    train, val, test = split_data(make_fake_data())
    # Each row has a unique label (index). No label should be in two parts.
    # "&" between two sets gives the labels they have in common.
    assert not set(train.index) & set(val.index)
    assert not set(train.index) & set(test.index)
    assert not set(val.index) & set(test.index)


def test_split_sizes_are_60_20_20():
    train, val, test = split_data(make_fake_data())
    total = len(train) + len(val) + len(test)
    # Allow a small difference (0.01 = 1 percentage point) for rounding.
    assert abs(len(train) / total - 0.60) < 0.01
    assert abs(len(val) / total - 0.20) < 0.01
    assert abs(len(test) / total - 0.20) < 0.01


def test_split_keeps_the_default_rate():
    df = make_fake_data()
    overall_rate = df[TARGET].mean()
    train, val, test = split_data(df)
    # Thanks to stratify, each part's default rate should match the overall one.
    for part in [train, val, test]:
        assert abs(part[TARGET].mean() - overall_rate) < 0.01


def test_split_is_the_same_every_time():
    df = make_fake_data()
    first_train, _, _ = split_data(df)
    second_train, _, _ = split_data(df)
    # Thanks to random_state, running it twice picks exactly the same rows.
    assert list(first_train.index) == list(second_train.index)
