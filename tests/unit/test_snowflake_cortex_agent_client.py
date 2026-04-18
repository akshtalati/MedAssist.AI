"""Unit tests for Snowflake Cortex Agent REST client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.snowflake_cortex_agent_client import (
    agent_config_from_env,
    agent_object_run_url,
    extract_text_from_agent_response,
    is_cortex_agent_ready,
    run_cortex_agent_object,
    snowflake_rest_base_url,
)


def test_snowflake_rest_base_url_from_account(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SNOWFLAKE_REST_URL", raising=False)
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "xy.us-east-1.aws")
    assert snowflake_rest_base_url() == "https://xy.us-east-1.aws.snowflakecomputing.com"


def test_agent_object_run_url_long_form(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    u = agent_object_run_url("MEDASSIST_DB", "PUBLIC", "MEDASSIST_AI")
    assert u.endswith("/api/v2/databases/MEDASSIST_DB/schemas/PUBLIC/agents/MEDASSIST_AI:run")


def test_extract_text_from_agent_response() -> None:
    payload = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Hello."},
            {"type": "text", "text": "World."},
        ],
    }
    assert extract_text_from_agent_response(payload) == "Hello.\n\nWorld."


def test_agent_config_from_env_fqn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_CORTEX_AGENT_FQN", "DB.SC.AG")
    assert agent_config_from_env() == {"database": "DB", "schema": "SC", "agent_name": "AG"}


def test_is_cortex_agent_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_CORTEX_AGENT_FQN", "A.B.C")
    monkeypatch.setenv("SNOWFLAKE_REST_PAT", "tok")
    assert is_cortex_agent_ready() is True
    monkeypatch.delenv("SNOWFLAKE_REST_PAT", raising=False)
    assert is_cortex_agent_ready() is False


@patch("src.snowflake_cortex_agent_client.requests.post")
@patch("src.snowflake_cortex_agent_client.rest_authorization_headers")
def test_run_cortex_agent_object(mock_headers: MagicMock, mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SNOWFLAKE_ACCOUNT", "acct")
    mock_headers.return_value = {"Authorization": "Bearer x"}
    mock_post.return_value.ok = True
    mock_post.return_value.status_code = 200
    mock_post.return_value.headers = {}
    mock_post.return_value.json.return_value = {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}
    out, dbg = run_cortex_agent_object(
        database="D", schema="S", agent_name="A", user_text="q", stream=False, collect_debug=False
    )
    assert out["role"] == "assistant"
    assert dbg is None
