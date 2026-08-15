import sqlite3 as db
import pytest
from contextlib import closing

from db.execute import execute_query


def test_execute(tmp_path):
    database_path = tmp_path / "test.db"
    with closing(db.connect(database_path)) as con:
        con.executescript("""   
        CREATE TABLE products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        );

        INSERT INTO products (name) VALUES ('milk');

        INSERT INTO products (name) VALUES ('bread');
        """)

        con.commit()

    sql = "SELECT name FROM products"
    columns, rows = execute_query(database_path, sql)
    assert columns == ["name"]
    assert rows == [("milk",), ("bread",)]

    with pytest.raises(db.OperationalError, match="readonly"):
        execute_query(database_path, "DELETE FROM products")

    _, limited_rows = execute_query(database_path, sql, row_limit=1)
    assert limited_rows == [("milk",)]
