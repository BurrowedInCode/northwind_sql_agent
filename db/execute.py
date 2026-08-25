import sqlite3 as db
from contextlib import closing
from pathlib import Path


def execute_query(
    database_path: str | Path, sql: str, row_limit: int = 100
) -> tuple[list[str], list[tuple]]:
    path = Path(database_path)
    absolute_path = path.resolve()
    database_uri = absolute_path.as_uri() + "?mode=ro"
    with closing(db.connect(database_uri, uri=True)) as con:

        def authorize(
            action_code: int,
            table_name: str | None,
            column_name: str | None,
            database_name: str | None,
            source: str | None,
        ) -> int:
            if (
                action_code == db.SQLITE_READ
                and table_name is not None
                and table_name.startswith("sqlite_")
            ):
                return db.SQLITE_DENY
            if (
                action_code == db.SQLITE_READ
                and table_name is not None
                and column_name is not None
                and table_name.casefold() == "employees"
                and column_name.casefold() == "photo"
            ):
                return db.SQLITE_DENY
            if action_code == db.SQLITE_ATTACH:
                return db.SQLITE_DENY

            return db.SQLITE_OK

        con.set_authorizer(authorize)
        cur: db.Cursor = con.cursor()

        cur.execute(sql)

        columns = [column[0] for column in cur.description]
        rows = cur.fetchmany(row_limit)

        return columns, rows
