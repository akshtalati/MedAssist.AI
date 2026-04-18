"""
Call Snowflake Cortex Agents via the public REST API (agent object :run).

Docs: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-agents-run
Auth: PAT — https://docs.snowflake.com/en/developer-guide/snowflake-rest-api/authentication
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import requests


class CortexAgentRequestError(RuntimeError):
    """Snowflake agent :run failed after HTTP; carries optional diagnostics (no secrets)."""

    def __init__(self, message: str, *, debug: dict[str, Any] | None = None):
        super().__init__(message)
        self.debug = debug


def _account_from_env() -> str:
    return (os.environ.get("SNOWFLAKE_ACCOUNT") or "").strip()


def snowflake_rest_base_url() -> str:
    explicit = (os.environ.get("SNOWFLAKE_REST_URL") or "").strip().rstrip("/")
    if explicit:
        return explicit
    account = _account_from_env()
    if not account:
        raise ValueError("Set SNOWFLAKE_ACCOUNT or SNOWFLAKE_REST_URL for Cortex Agent REST calls.")
    return f"https://{account}.snowflakecomputing.com"


def _rest_pat() -> str:
    return (
        (os.environ.get("SNOWFLAKE_REST_PAT") or os.environ.get("SNOWFLAKE_PROGRAMMATIC_ACCESS_TOKEN") or "")
        .strip()
    )


def rest_authorization_headers() -> dict[str, str]:
    token = _rest_pat()
    if not token:
        raise ValueError(
            "Set SNOWFLAKE_REST_PAT (or SNOWFLAKE_PROGRAMMATIC_ACCESS_TOKEN) for Cortex Agent REST auth."
        )
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Snowflake-Authorization-Token-Type": "PROGRAMMATIC_ACCESS_TOKEN",
    }


def agent_object_run_url(database: str, schema: str, name: str) -> str:
    """Official long-form :run URL (short /api/v2/agents/FQN:run often returns 404)."""
    base = snowflake_rest_base_url()
    db = quote(database.strip(), safe="")
    sch = quote(schema.strip(), safe="")
    ag = quote(name.strip(), safe="")
    return f"{base}/api/v2/databases/{db}/schemas/{sch}/agents/{ag}:run"


def extract_text_from_agent_response(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for item in payload.get("content") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text" and isinstance(item.get("text"), str):
            t = item["text"].strip()
            if t:
                chunks.append(t)
    return "\n\n".join(chunks).strip()


def _redact_headers(h: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in h.items():
        if k.lower() == "authorization":
            out[k] = "Bearer <redacted>"
        else:
            out[k] = v
    return out


def run_cortex_agent_object(
    *,
    database: str,
    schema: str,
    agent_name: str,
    user_text: str,
    messages: list[dict[str, Any]] | None = None,
    thread_id: int | None = None,
    parent_message_id: int | None = None,
    stream: bool = False,
    timeout_sec: int = 900,
    collect_debug: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    url = agent_object_run_url(database, schema, agent_name)
    headers = rest_authorization_headers()

    if messages is None:
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            }
        ]

    body: dict[str, Any] = {"messages": messages, "stream": stream}
    if thread_id is not None:
        body["thread_id"] = int(thread_id)
    if parent_message_id is not None:
        body["parent_message_id"] = int(parent_message_id)

    debug: dict[str, Any] | None = None
    if collect_debug:
        debug = {
            "request_url": url,
            "request_method": "POST",
            "request_headers": _redact_headers(headers),
            "request_body": body,
        }

    r = requests.post(url, headers=headers, json=body, timeout=timeout_sec)
    if collect_debug and debug is not None:
        debug["response_http_status"] = r.status_code
        debug["response_content_type"] = r.headers.get("Content-Type", "")

    if not r.ok:
        if collect_debug and debug is not None:
            debug["response_body_snippet"] = (r.text or "")[:8000]
        detail = (r.text or "")[:4000]
        raise CortexAgentRequestError(
            f"Cortex Agent HTTP {r.status_code}: {detail}",
            debug=debug if collect_debug else None,
        )

    try:
        js = r.json()
    except Exception as e:
        if collect_debug and debug is not None:
            debug["response_body_snippet"] = (r.text or "")[:8000]
            debug["response_json_error"] = str(e)
        raise CortexAgentRequestError(
            f"Cortex Agent response is not JSON: {e}",
            debug=debug if collect_debug else None,
        ) from e

    if not isinstance(js, dict):
        raise CortexAgentRequestError(
            "Cortex Agent JSON root is not an object",
            debug=debug if collect_debug else None,
        )

    if collect_debug and debug is not None:
        debug["response_json_top_keys"] = list(js.keys())
        content = js.get("content")
        if isinstance(content, list):
            debug["response_content_item_types"] = [
                c.get("type") for c in content if isinstance(c, dict)
            ]
        debug["extracted_text_length"] = len(extract_text_from_agent_response(js))

    return js, debug


def agent_config_from_env() -> dict[str, str]:
    fqn = (os.environ.get("SNOWFLAKE_CORTEX_AGENT_FQN") or "").strip()
    if fqn:
        parts = fqn.split(".")
        if len(parts) != 3:
            raise ValueError(
                "SNOWFLAKE_CORTEX_AGENT_FQN must be exactly DATABASE.SCHEMA.AGENT_NAME (three segments)."
            )
        return {"database": parts[0], "schema": parts[1], "agent_name": parts[2]}

    database = (
        os.environ.get("SNOWFLAKE_CORTEX_AGENT_DATABASE") or os.environ.get("SNOWFLAKE_DATABASE") or ""
    ).strip()
    schema = (os.environ.get("SNOWFLAKE_CORTEX_AGENT_SCHEMA") or "").strip()
    name = (os.environ.get("SNOWFLAKE_CORTEX_AGENT_NAME") or "").strip()
    if not database or not schema or not name:
        raise ValueError(
            "Set SNOWFLAKE_CORTEX_AGENT_FQN, or SNOWFLAKE_CORTEX_AGENT_DATABASE, "
            "SNOWFLAKE_CORTEX_AGENT_SCHEMA, and SNOWFLAKE_CORTEX_AGENT_NAME."
        )
    return {"database": database, "schema": schema, "agent_name": name}


def is_cortex_agent_ready() -> bool:
    """True when PAT + agent coordinates are present (REST :run can be attempted)."""
    try:
        agent_config_from_env()
    except ValueError:
        return False
    return bool(_rest_pat())
