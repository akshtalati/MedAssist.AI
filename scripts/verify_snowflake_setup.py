#!/usr/bin/env python3
"""
Verify Snowflake setup and data load status.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.snowflake_client import get_connection


def main():
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("USE WAREHOUSE MEDASSIST_WH")
        cursor.execute("USE DATABASE MEDASSIST_DB")

        print("=" * 60)
        print("Snowflake Setup Verification")
        print("=" * 60)

        # Check role
        cursor.execute("SELECT CURRENT_ROLE()")
        role = cursor.fetchone()[0]
        print(f"\nCurrent Role: {role}")

        # Check schemas
        cursor.execute("SHOW SCHEMAS IN DATABASE MEDASSIST_DB")
        schemas = [s[1] for s in cursor.fetchall()]
        print(f"\nSchemas: {', '.join(schemas)}")

        # Check tables in each schema
        for schema in ['RAW', 'NORMALIZED', 'VECTORS']:
            if schema not in schemas:
                continue
            cursor.execute(f"SHOW TABLES IN SCHEMA {schema}")
            tables = cursor.fetchall()
            print(f"\n{schema} Schema Tables ({len(tables)}):")
            for table in tables:
                table_name = table[1]
                cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table_name}")
                count = cursor.fetchone()[0]
                print(f"  - {table_name}: {count:,} rows")

        print("\n" + "=" * 60)
        print("Verification complete!")
        print("=" * 60)

    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
