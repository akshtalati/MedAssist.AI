"""
MedAssist.AI eval DAG — runs the HTTP eval suite against a running API.

Airflow Variables:
- medassist_project_root: path to MedAssist.AI project root (required)
- medassist_api_base: API base URL (optional, default http://127.0.0.1:8000)

Optional: set an Airflow Connection or Variable for Slack webhook in a follow-up task (not required here).
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "medassist",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="medassist_eval",
    default_args=DEFAULT_ARGS,
    description="Run evals/run_evals.py against MEDASSIST_API_BASE",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    tags=["medassist", "eval"],
) as dag:
    _common_env = {
        "PROJECT_ROOT": "{{ var.value.medassist_project_root }}",
        "VENV_PYTHON": "{{ var.value.medassist_project_root }}/venv/bin/python",
        "MEDASSIST_API_BASE": "{{ var.value.medassist_api_base | default('http://127.0.0.1:8000') }}",
    }
    _run_eval = (
        'cd "$PROJECT_ROOT" && '
        'if [ -x "$VENV_PYTHON" ]; then PY="$VENV_PYTHON"; else PY=python3; fi && '
        '"$PY" evals/run_evals.py --base-url "$MEDASSIST_API_BASE"'
    )

    run_evals = BashOperator(
        task_id="run_evals_http",
        env=_common_env,
        bash_command=_run_eval,
    )
