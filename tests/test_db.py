"""Tests for lendsmart.db and the database settings in lendsmart.config.

These tests don't need the database to be running: they check the column-name
translation and the connection address, which are pure Python.
"""

from lendsmart.config import database_url
from lendsmart.db import CSV_TO_DB_COLUMNS, DB_TO_CSV_COLUMNS
from lendsmart.models import FEATURES


def test_every_column_has_a_database_name():
    # Every feature the model uses (except the income_missing flag, which
    # clean_data() adds later) must be stored in the database.
    for column in FEATURES:
        if column != "income_missing":
            assert column in CSV_TO_DB_COLUMNS


def test_database_names_are_simple_snake_case():
    # Lowercase letters, digits and underscores only, so no quotes are needed in SQL.
    for name in CSV_TO_DB_COLUMNS.values():
        assert name == name.lower()
        assert all(ch.isalnum() or ch == "_" for ch in name)


def test_column_names_translate_back_exactly():
    # CSV name -> database name -> CSV name should give back the original.
    for csv_name, db_name in CSV_TO_DB_COLUMNS.items():
        assert DB_TO_CSV_COLUMNS[db_name] == csv_name


def test_database_url_uses_environment_settings(monkeypatch):
    # monkeypatch temporarily sets environment variables just for this test.
    monkeypatch.setenv("POSTGRES_USER", "alice")
    monkeypatch.setenv("POSTGRES_PASSWORD", "secret")
    monkeypatch.setenv("POSTGRES_HOST", "db")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "loans")
    assert database_url() == "postgresql+psycopg://alice:secret@db:5432/loans"
