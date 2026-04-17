"""Pytest defaults: disable JWT auth so API tests do not need Bearer tokens."""

import os

import pytest


def pytest_configure(config):
    os.environ["AUTH_DISABLED"] = "1"
    os.environ.setdefault("JWT_SECRET", "test-secret")
