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
        cur: db.Cursor = con.cursor()

        cur.execute(sql)

        columns = [column[0] for column in cur.description]
        rows = cur.fetchmany(row_limit)

        return columns, rows
