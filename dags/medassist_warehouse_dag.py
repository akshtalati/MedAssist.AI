"""
MedAssist.AI warehouse / dbt DAG (runs after data is loaded into Snowflake).

Prerequisite: normalized + clinical tables populated (e.g. load_to_snowflake job — not part of medassist_ingestion today).

Airflow Variables:
- medassist_project_root: path to MedAssist.AI project root (required)
- medassist_load_snowflake: set to "1" to un-pause / enable this DAG in your environment (optional convention)

Configure dbt `profiles.yml` on the Airflow worker or inject env vars for Snowflake (same as CI secrets).
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
    dag_id="medassist_warehouse",
    default_args=DEFAULT_ARGS,
    description="dbt run + test (staging+ marts) after Snowflake load",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    tags=["medassist", "dbt", "warehouse"],
) as dag:
    _common_env = {
        "PROJECT_ROOT": "{{ var.value.medassist_project_root }}",
    }

    dbt_run = BashOperator(
        task_id="dbt_run_staging_plus",
        env=_common_env,
        bash_command='cd "$PROJECT_ROOT/dbt" && dbt run --select staging+ --profiles-dir .',
    )

    dbt_test = BashOperator(
        task_id="dbt_test",
        env=_common_env,
        bash_command='cd "$PROJECT_ROOT/dbt" && dbt test --profiles-dir .',
    )

    dbt_run >> dbt_test
