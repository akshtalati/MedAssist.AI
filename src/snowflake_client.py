"""Snowflake connection helper for MedAssist.AI."""

import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_snowflake_conn_params() -> dict[str, Any]:
    """Get connection params from env. See .env.example for required vars."""
    account = os.environ.get("SNOWFLAKE_ACCOUNT", "").strip()
    user = os.environ.get("SNOWFLAKE_USER", "").strip()
    password = os.environ.get("SNOWFLAKE_PASSWORD", "")
    authenticator = os.environ.get("SNOWFLAKE_AUTHENTICATOR", "")
    role = os.environ.get("SNOWFLAKE_ROLE", "TRAINING_ROLE")
    warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE", "MEDASSIST_WH")
    database = os.environ.get("SNOWFLAKE_DATABASE", "MEDASSIST_DB")
    schema = os.environ.get("SNOWFLAKE_SCHEMA", "RAW")

    if not account or not user:
        raise ValueError(
            "Set SNOWFLAKE_ACCOUNT and SNOWFLAKE_USER in the environment (see .env.example)."
        )

    params: dict[str, Any] = {
        "account": account,
        "user": user,
        "role": role,
        "warehouse": warehouse,
        "database": database,
        "schema": schema,
    }
    if authenticator.lower() in ("externalbrowser", "external_browser"):
        params["authenticator"] = "externalbrowser"
        # Don't pass password - browser handles auth (including MFA)
    elif password:
        params["password"] = password
    return params


def get_connection():
    """Return a Snowflake connection. Requires snowflake-connector-python.
    
    Uses SNOWFLAKE_ROLE from .env (e.g. MEDASSIST_TEAM_ROLE or TRAINING_ROLE).
    That role must have USAGE on schemas and INSERT/SELECT on the tables you load.
    """
    import snowflake.connector

    params = get_snowflake_conn_params()
    # Use role from env (e.g. MEDASSIST_TEAM_ROLE); fallback to TRAINING_ROLE
    if not params.get("role"):
        params["role"] = "TRAINING_ROLE"

    if "password" not in params and "authenticator" not in params:
        raise ValueError(
            "Set SNOWFLAKE_PASSWORD in .env, or SNOWFLAKE_AUTHENTICATOR=externalbrowser for MFA (opens browser)."
        )
    
    return snowflake.connector.connect(**params)
