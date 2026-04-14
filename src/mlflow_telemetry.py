"""Optional MLflow metrics — never raises; no-op if mlflow missing or disabled."""

from __future__ import annotations

import os
from typing import Any


def log_assessment_metrics(
    *,
    run_name: str,
    metrics: dict[str, Any],
    tags: dict[str, str] | None = None,
) -> None:
    if os.environ.get("MLFLOW_DISABLE", "").lower() in ("1", "true", "yes"):
        return
    try:
        import mlflow
    except Exception:
        return
    uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    try:
        mlflow.set_tracking_uri(uri)
        with mlflow.start_run(run_name=run_name[:250]):
            for k, v in (metrics or {}).items():
                if v is None:
                    continue
                try:
                    if isinstance(v, bool):
                        mlflow.log_metric(k, 1.0 if v else 0.0)
                    elif isinstance(v, (int, float)):
                        mlflow.log_metric(k, float(v))
                    else:
                        mlflow.log_param(str(k), str(v)[:500])
                except Exception:
                    continue
            if tags:
                mlflow.set_tags({k: v[:500] for k, v in tags.items()})
    except Exception:
        return
