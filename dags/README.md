# MedAssist.AI Airflow DAGs

This folder contains the **medassist_ingestion** DAG, which automates the full ingestion pipeline without changing any existing project code.

## Setup

1. **Create the project venv** (from project root): run `./setup_venv.sh` so that `venv/bin/python` exists and has project dependencies installed.
2. **Install and run Airflow** (from project root): run `./start_airflow.sh`. This uses a separate `venv_airflow/` and `airflow_home/` in the project; the script sets `AIRFLOW__CORE__DAGS_FOLDER` to this repo's `dags/` automatically. On first run, `airflow standalone` will create an admin user and print the password in the terminal—use it to log in at http://localhost:8080.

## Required Airflow Variables

Set in Airflow UI: **Admin -> Variables**.

| Variable | Description |
|----------|-------------|
| `medassist_project_root` | **Required.** Absolute path to the MedAssist.AI project root (e.g. `/path/to/MedAssist.AI`). |
| `medassist_venv_python` | Optional. Path to the project venv Python. Default: `{{ medassist_project_root }}/venv/bin/python`. |

## Optional Variables (enable extra steps)

| Variable | Description |
|----------|-------------|
| `medassist_run_rag` | Set to `true` (or `1`, `yes`) to run `build_rag_index` after the symptom index. |
| `medassist_run_snowflake` | Set to `true` to run Snowflake setup and `load_to_snowflake --all` after the index steps. |

## DAG flow

1. **fetch_checkpointed** -> **cleanup_duplicates** -> **build_symptom_index**
2. Then either **build_rag_index** or **skip_rag** (depending on `medassist_run_rag`)
3. Then either **snowflake_setup** -> **load_to_snowflake** or **skip_snowflake** (depending on `medassist_run_snowflake`)

For a full fetch reset, run once manually from project root:  
`source venv/bin/activate && python scripts/fetch_all_checkpointed.py --reset`
