"""Talk to the Postgres database: save borrowers into it and read them back."""

import pandas as pd
from sqlalchemy import create_engine, text

from lendsmart.config import database_url

# The name of the table that holds the borrower data.
BORROWERS_TABLE = "borrowers"

# The CSV's column names have capitals and hyphens, which Postgres only accepts
# inside "quotes" in every query. So in the database we use simple snake_case
# names instead. This dictionary translates CSV name -> database name.
CSV_TO_DB_COLUMNS = {
    "SeriousDlqin2yrs": "defaulted",
    "RevolvingUtilizationOfUnsecuredLines": "credit_utilization",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "times_30_59_days_late",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_credit_lines",
    "NumberOfTimes90DaysLate": "times_90_days_late",
    "NumberRealEstateLoansOrLines": "real_estate_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "times_60_89_days_late",
    "NumberOfDependents": "dependents",
}
# The same dictionary, flipped: database name -> CSV name.
DB_TO_CSV_COLUMNS = {db: csv for csv, db in CSV_TO_DB_COLUMNS.items()}


def get_engine():
    """Return a SQLAlchemy "engine": the object that manages connections to Postgres.

    pandas uses it for to_sql() (write a DataFrame to a table) and
    read_sql() (read a query's result into a DataFrame).
    """
    return create_engine(database_url())


def save_borrowers(df, engine):
    """Write the raw borrower data into the borrowers table, replacing what's there.

    df is the raw data from load_data(): its index holds the borrower ids.
    """
    # Rename to database-friendly names, and turn the index into an "id" column.
    table = df.rename(columns=CSV_TO_DB_COLUMNS)
    table.index.name = "id"

    # to_sql creates the table (choosing column types from the data) and inserts the rows.
    # - if_exists="replace": drop any old table first, so re-running gives a clean copy.
    # - index=True: save the index as the "id" column.
    # - chunksize: send 10,000 rows at a time instead of all 150,000 at once.
    table.to_sql(
        BORROWERS_TABLE,
        engine,
        if_exists="replace",
        index=True,
        chunksize=10_000,
    )

    # Make "id" the primary key: a unique label for each row, which Postgres
    # also uses to find rows quickly. to_sql already made a plain index on "id",
    # which the primary key replaces, so we drop it to avoid keeping two.
    # "with engine.begin()" opens a connection and saves (commits) the changes
    # when the block finishes.
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {BORROWERS_TABLE} ADD PRIMARY KEY (id)"))
        connection.execute(text(f"DROP INDEX IF EXISTS ix_{BORROWERS_TABLE}_id"))


def load_borrowers(engine):
    """Read the borrowers table back into a DataFrame with the original CSV column names.

    The result looks exactly like load_data()'s, so the rest of the code
    (clean_data, split_data, the models) works on it unchanged.
    """
    df = pd.read_sql(
        f"SELECT * FROM {BORROWERS_TABLE} ORDER BY id", engine, index_col="id"
    )
    # Translate the column names back, and drop the index name to match load_data().
    df = df.rename(columns=DB_TO_CSV_COLUMNS)
    df.index.name = None
    return df
