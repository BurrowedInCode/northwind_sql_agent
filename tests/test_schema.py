from contextlib import closing
from db.schema import get_schema
import sqlite3 as db
import pytest


def test_get_schema(tmp_path):
    database_path = tmp_path / "test.db"
    with closing(db.connect(database_path)) as con:
        con.executescript("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        );

        CREATE VIEW available_products AS
        SELECT id, name FROM products;
        """)
        con.commit()

    schema = get_schema(database_path)

    assert isinstance(schema, str)
    assert "products" in schema
    assert "available_products" in schema
    assert "sqlite_sequence" not in schema


def test_get_schema_rejects_missing_database(tmp_path):
    missing_database = tmp_path / "missing_database"

    assert not missing_database.exists()

    with pytest.raises(db.OperationalError):
        get_schema(missing_database)

    assert not missing_database.exists()
