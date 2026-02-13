"""Configuration loader for MedAssist.AI data ingestion."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

# Project root (parent of src)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    """Load config.yaml from project root."""
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_data_paths() -> dict:
    """Return resolved data directory paths."""
    config = load_config()
    paths = config.get("paths", {})
    return {
        "raw": PROJECT_ROOT / paths.get("raw", "data/raw"),
        "normalized": PROJECT_ROOT / paths.get("normalized", "data/normalized"),
        "metadata": PROJECT_ROOT / paths.get("metadata", "data/metadata"),
        "schema": PROJECT_ROOT / paths.get("schema", "data/schema"),
    }


def get_env(key: str, default: str = "") -> str:
    """Get environment variable with optional default."""
    return os.environ.get(key, default)
