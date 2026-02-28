"""
MedAssist.AI ingestion DAG.

Orchestrates: fetch (checkpointed) -> cleanup duplicates -> symptom index.

Set Airflow Variable (Admin -> Variables):
- medassist_project_root: path to MedAssist.AI project root (required)
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

# Default args for all tasks
DEFAULT_ARGS = {
    "owner": "medassist",
    "depends_on_past": False,
    "retries": 0,
}


with DAG(
    dag_id="medassist_ingestion",
    default_args=DEFAULT_ARGS,
    description="MedAssist.AI ingestion pipeline (fetch -> cleanup -> symptom index)",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    tags=["medassist", "ingestion"],
) as dag:
    # Pass paths via env so paths with spaces (e.g. "Data Engineering") work
    _common_env = {
        "PROJECT_ROOT": "{{ var.value.medassist_project_root }}",
        "VENV_PYTHON": "{{ var.value.medassist_project_root }}/venv/bin/python",
    }
    _run_script = 'cd "$PROJECT_ROOT" && "$VENV_PYTHON"'

    fetch_checkpointed = BashOperator(
        task_id="fetch_checkpointed",
        env=_common_env,
        bash_command=_run_script + " scripts/fetch_all_checkpointed.py",
    )

    cleanup_duplicates = BashOperator(
        task_id="cleanup_duplicates",
        env=_common_env,
        bash_command=_run_script + " scripts/cleanup_duplicate_raw.py",
    )

    build_symptom_index = BashOperator(
        task_id="build_symptom_index",
        env=_common_env,
        bash_command=_run_script + " scripts/build_symptom_index.py",
    )

    fetch_checkpointed >> cleanup_duplicates >> build_symptom_index
