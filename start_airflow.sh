#!/usr/bin/env bash
# Start Airflow (webserver + scheduler). Run from project root.
# First run: airflow standalone will create an admin user and print the password to the terminal.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Subprocesses spawned by standalone (scheduler, api-server, etc.) run "airflow" and need it on PATH
export PATH="$(pwd)/venv_airflow/bin:${PATH}"
export AIRFLOW_HOME="${AIRFLOW_HOME:-$(pwd)/airflow_home}"
export AIRFLOW__CORE__DAGS_FOLDER="$(pwd)/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"

if [[ ! -d "venv_airflow" ]]; then
  echo "Run: python3 -m venv venv_airflow && source venv_airflow/bin/activate && pip install apache-airflow==3.1.7 --constraint ... (see dags/README.md)"
  exit 1
fi

# Ensure variable is set so the MedAssist DAG can find the project
./venv_airflow/bin/airflow variables set medassist_project_root "$(pwd)" 2>/dev/null || true

echo "Starting Airflow standalone (UI at http://localhost:8080)..."
exec ./venv_airflow/bin/airflow standalone
