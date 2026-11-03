"""Copy the borrower data from the CSV file into the Postgres database.

Start the database first (docker compose up -d), then run from the main project folder:
    .venv/bin/python scripts/load_to_db.py

Running it again is safe: it replaces the table with a fresh copy.
"""

from lendsmart.data import load_data
from lendsmart.db import BORROWERS_TABLE, get_engine, load_borrowers, save_borrowers


def main():
    # 1. Read the raw CSV. We store the RAW data: the database is the "source of
    #    truth", and cleaning still happens in Python with clean_data().
    df = load_data()
    print(f"Read {len(df):,} rows from the CSV")

    # 2. Connect and write the table.
    engine = get_engine()
    save_borrowers(df, engine)
    print(f"Saved them to the '{BORROWERS_TABLE}' table in Postgres")

    # 3. Read it back and check nothing was lost or changed on the way.
    from_db = load_borrowers(engine)
    same = from_db.equals(df)
    print(f"Read {len(from_db):,} rows back. Identical to the CSV: {same}")


if __name__ == "__main__":
    main()
