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


def test_execute_query_rejects_internal_schema(tmp_path):
    database_path = tmp_path / "test.db"

    with closing(db.connect(database_path)) as con:
        pass

    with pytest.raises(db.DatabaseError, match="prohibited"):
        execute_query(database_path, "SELECT name FROM sqlite_master")


def test_execute_query_rejects_employee_photos(tmp_path):
    database_path = tmp_path / "test.db"

    with closing(db.connect(database_path)) as con:
        con.executescript("""
                        CREATE TABLE employees (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL,
                            photo BLOB);

                        INSERT INTO Employees (id, name, photo) VALUES (1, 'Nancy', X'89504E47');
                        """)
        con.commit()

    columns, rows = execute_query(database_path, "SELECT name FROM employees")

    assert columns == ["name"]
    assert rows == [("Nancy",)]
    with pytest.raises(db.DatabaseError, match="prohibited"):
        execute_query(database_path, "SELECT photo FROM employees")


def test_execute_query_rejects_attach(tmp_path):
    database_path = tmp_path / "test.db"
    attached_database_path = tmp_path / "attached.db"

    with closing(db.connect(database_path)) as con:
        con.executescript("""
                        CREATE TABLE products (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL
                        );

                        INSERT INTO products (name) VALUES ('milk');
                        """)

        con.commit()
    assert not attached_database_path.exists()

    with pytest.raises(db.DatabaseError, match="not authorized"):
        execute_query(
            database_path, f"ATTACH DATABASE '{attached_database_path}' AS external"
        )

    assert not attached_database_path.exists()


def test_execute_query_rejects_pragma(tmp_path):
    database_path = tmp_path / "test.db"

    with closing(db.connect(database_path)) as con:
        con.executescript("""
                        CREATE TABLE products (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT NOT NULL
                        );

                        INSERT INTO products (name) VALUES ('milk');
                        """)
        con.commit()
    with pytest.raises(db.DatabaseError, match="not authorized"):
        execute_query(database_path, "PRAGMA table_list")
