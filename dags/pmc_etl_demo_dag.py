"""
PMC-only ETL demo DAG.

Flow:
1) extract PMC
2) transform/quality report
3) load to Snowflake (PMC + manifests)
4) print postcheck query commands

Required Airflow Variable:
- medassist_project_root

Optional Airflow Variables:
- pmc_term (default: "rare disease")
- pmc_max_records (default: "1000")
- pmc_enable_load (default: "true")
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import BranchPythonOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.sdk import Variable

DEFAULT_ARGS = {
    "owner": "medassist",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _branch_load() -> str:
    flag = str(Variable.get("pmc_enable_load", default="true")).strip().lower()
    return "pmc_load_snowflake" if flag in {"1", "true", "yes"} else "skip_load"


with DAG(
    dag_id="pmc_etl_demo",
    default_args=DEFAULT_ARGS,
    description="PMC-only ETL demo (extract -> transform -> load -> postcheck)",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    start_date=datetime(2025, 1, 1),
    tags=["medassist", "pmc", "etl"],
) as dag:
    common_env = {
        "PROJECT_ROOT": "{{ var.value.medassist_project_root }}",
        "VENV_PYTHON": "{{ var.value.medassist_project_root }}/venv/bin/python",
        "PMC_TERM": "{{ var.value.get('pmc_term', 'rare disease') }}",
        "PMC_MAX": "{{ var.value.get('pmc_max_records', '1000') }}",
    }
    run = 'cd "$PROJECT_ROOT" && "$VENV_PYTHON"'

    pmc_extract = BashOperator(
        task_id="pmc_extract",
        env=common_env,
        execution_timeout=timedelta(minutes=45),
        bash_command=run + ' scripts/fetch_source.py pmc --term "$PMC_TERM" --max_records "$PMC_MAX"',
    )

    pmc_transform = BashOperator(
        task_id="pmc_transform",
        env=common_env,
        execution_timeout=timedelta(minutes=10),
        bash_command=run + " scripts/transform_pmc_curated.py --strict --min-valid-ratio 0.80",
    )

    branch_load = BranchPythonOperator(
        task_id="branch_load",
        python_callable=_branch_load,
    )

    pmc_load_snowflake = BashOperator(
        task_id="pmc_load_snowflake",
        env=common_env,
        execution_timeout=timedelta(minutes=45),
        bash_command=run + " scripts/load_to_snowflake.py --pmc --manifests",
    )

    skip_load = EmptyOperator(task_id="skip_load")

    pmc_postcheck = BashOperator(
        task_id="pmc_postcheck",
        env=common_env,
        bash_command=(
            "echo \"Run in Snowflake Worksheet:\" && "
            "echo \"SELECT COUNT(*) AS manifests FROM MEDASSIST_DB.RAW.FETCH_MANIFESTS;\" && "
            "echo \"SELECT COUNT(*) AS pmc_rows FROM MEDASSIST_DB.RAW.PMC_ARTICLES;\""
        ),
    )

    pmc_extract >> pmc_transform >> branch_load
    branch_load >> pmc_load_snowflake >> pmc_postcheck
    branch_load >> skip_load >> pmc_postcheck
