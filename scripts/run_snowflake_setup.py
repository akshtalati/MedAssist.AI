#!/usr/bin/env python3
"""
Execute snowflake_setup.sql to create warehouse, database, schemas, and tables.

Requires SNOWFLAKE_PASSWORD in .env. Run from project root.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.snowflake_client import get_connection

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SQL_PATH = PROJECT_ROOT / "scripts" / "snowflake_setup.sql"


def _split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    for chunk in sql.split(";"):
        cleaned_lines: list[str] = []
        for line in chunk.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue
            cleaned_lines.append(line)
        stmt = "\n".join(cleaned_lines).strip()
        if stmt:
            statements.append(stmt)
    return statements


def main():
    if not SQL_PATH.exists():
        print(f"SQL file not found: {SQL_PATH}")
        sys.exit(1)

    sql = SQL_PATH.read_text(encoding="utf-8")
    statements = _split_sql_statements(sql)

    conn = get_connection()
    cursor = conn.cursor()

    for i, stmt in enumerate(statements):
        if not stmt or stmt.upper().startswith("--"):
            continue
        try:
            cursor.execute(stmt + ";")
            print(f"OK: {stmt[:60]}...")
        except Exception as e:
            print(f"Error ({i+1}): {e}")
            print(f"  Statement: {stmt[:200]}...")
            conn.rollback()
            cursor.close()
            conn.close()
            sys.exit(1)

    conn.commit()
    cursor.close()
    conn.close()
    print("Snowflake setup complete.")


if __name__ == "__main__":
    main()
