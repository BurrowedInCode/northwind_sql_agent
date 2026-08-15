import sqlite3 as db
from pathlib import Path
from contextlib import closing

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_PATH = PROJECT_ROOT / "northwind.db"


def get_schema(database_path: str | Path) -> str:
    path = Path(database_path)
    absolute_path = path.resolve()
    database_uri = absolute_path.as_uri() + "?mode=ro"
    with closing(db.connect(database_uri, uri=True)) as con:
        cur = con.cursor()
        data = cur.execute("""
        SELECT sql 
        FROM sqlite_schema 
        WHERE type IN ('table', 'view') 
            AND name NOT LIKE 'sqlite_%' 
            AND sql IS NOT NULL 
        ORDER BY name
        """).fetchall()
        return "\n\n".join(row[0] for row in data)


if __name__ == "__main__":
    print(get_schema(DATABASE_PATH))
