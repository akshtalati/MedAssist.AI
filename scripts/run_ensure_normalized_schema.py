#!/usr/bin/env python3
"""
Create NORMALIZED schema and SYMPTOM_DISEASE_MAP + DISEASES tables if they don't exist.
Uses the same Snowflake connection as the loader (SNOWFLAKE_* from .env).
If this fails with "insufficient privileges", run scripts/ensure_normalized_schema.sql
in Snowflake as ACCOUNTADMIN.

Usage: python scripts/run_ensure_normalized_schema.py
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
        cursor.execute("""
            CREATE SCHEMA IF NOT EXISTS NORMALIZED
            COMMENT = 'Processed indices: symptom→disease, disease catalog'
        """)
        print("NORMALIZED schema: OK (created or already exists)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS NORMALIZED.SYMPTOM_DISEASE_MAP (
              symptom VARCHAR(500),
              orpha_code VARCHAR(20),
              disease_name VARCHAR(500),
              frequency VARCHAR(100),
              hpo_id VARCHAR(50),
              _loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)
        print("NORMALIZED.SYMPTOM_DISEASE_MAP: OK (created or already exists)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS NORMALIZED.DISEASES (
              orpha_code VARCHAR(20) PRIMARY KEY,
              disease_name VARCHAR(500),
              _loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)
        print("NORMALIZED.DISEASES: OK (created or already exists)")

        conn.commit()
        print("Done. Run: python scripts/load_to_snowflake.py --all --skip-loaded")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        print("If you see 'insufficient privileges', run scripts/ensure_normalized_schema.sql in Snowflake as ACCOUNTADMIN.", file=sys.stderr)
        sys.exit(1)
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
