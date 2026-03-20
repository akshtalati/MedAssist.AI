# PMC-Only ETL to Near-Production + Airflow Orchestration Plan

## Summary
Build a dedicated **PMC-only ETL pipeline** that is demo-ready and structured for production hardening:
- **Run mode:** manual trigger now, with schedule configured but not enabled.
- **Load target:** `Snowflake RAW.PMC_ARTICLES`.
- **Goal:** reproducible, observable, idempotent ETL with data quality checks and clear Airflow steps.

## 1. Environment and Dependency Readiness

### Current-state findings
- `venv` setup exists via `setup_venv.sh` and installs `requirements.txt`.
- `venv_airflow` setup is separate and expected by `start_airflow.sh`.
- `requirements.txt` includes core + Snowflake + RAG deps; Airflow is installed separately.
- `README.md` references `.env.example`, but `.env.example` is missing.

### Plan
1. Keep **two-venv model**:
- `venv`: data pipeline runtime.
- `venv_airflow`: Airflow runtime.

2. Define install commands explicitly in docs:
- Project runtime: `./setup_venv.sh`
- Airflow runtime: create `venv_airflow`, install pinned Airflow + provider.

3. Add `.env.example` (implementation phase):
- `EMAIL`, optional `NCBI_API_KEY`, `SNOWFLAKE_*`, `SNOWFLAKE_AUTHENTICATOR`.

4. Dependency hygiene:
- Pin critical versions for reproducibility.

## 2. PMC ETL Scope (Near-Prod)

### Extract (PMC only)
- Source: NCBI E-Utilities (`esearch` + `efetch`) via `PMCFetcher`.
- Configurable inputs: `term`, `max_records`.
- Existing retry/rate-limit support comes from `BaseFetcher`.

### Transform (quality + curation)
Rules:
1. Validate input shape has `data.articles[]`.
2. Keep only rows with non-empty `pmcid`.
3. Keep rows where at least one of `title` or `abstract` is non-empty.
4. Normalize whitespace and dates where possible.
5. De-duplicate by `pmcid`:
- Keep record with richest abstract; fallback latest file timestamp.
6. Produce:
- `data/normalized/pmc_curated_<run_id>.json`
- `data/normalized/pmc_quality_report_<run_id>.json`

### Load (Snowflake RAW)
- Load curated rows to `RAW.PMC_ARTICLES` with idempotent upsert logic (`NOT EXISTS` by `pmcid`).
- Load manifest metadata to `RAW.FETCH_MANIFESTS`.
- Emit load summary counts.

## 3. Airflow DAG Design (PMC-only)

### DAG name
`pmc_etl_demo`

### Trigger/schedule strategy
- Keep `schedule=None` for demo.
- Keep future daily schedule in config/comments, not enabled yet.

### Task graph
1. `pmc_extract`
- Fetch only PMC (`term`, `max_records` from Airflow params/variables).

2. `pmc_transform`
- Run PMC transform script; generate curated output + quality report.

3. `pmc_load_snowflake`
- Load PMC + manifests to Snowflake.

4. `pmc_postcheck` (recommended)
- Validate row counts and non-null `pmcid`.

### Runtime controls
- `retries=2`
- `retry_delay=5m`
- `execution_timeout` for each task
- `max_active_runs=1`
- path/env via `medassist_project_root`

## 4. Production-Readiness Additions

1. Idempotency
- Enforce uniqueness behavior by `pmcid`.

2. Observability
- Persist quality report per run.
- Log counts for extracted/valid/dropped/loaded.

3. Failure handling
- Fail transform if valid-row ratio below threshold.
- Fail load cleanly on Snowflake auth issues.

4. Configurability
- Airflow variables/params:
- `pmc_term`
- `pmc_max_records`
- `medassist_project_root`
- `pmc_enable_load` (optional)

## 5. Interface/API Changes

1. New script (planned):
- `scripts/transform_pmc_curated.py`
- Inputs: latest/raw PMC JSON or explicit `--input-file`
- Outputs: curated JSON + quality report
- Flags: `--strict`, `--min-valid-ratio`, `--output`

2. New DAG (planned):
- `dags/pmc_etl_demo.py`

3. Docs updates:
- `README.md`, `dags/README.md` with PMC-only ETL flow and two-venv setup.

## 6. Test Cases and Validation

### Functional
1. Extract writes a raw file in `data/raw/pmc/`.
2. Transform creates curated + quality report files.
3. Load inserts into `RAW.PMC_ARTICLES`.
4. Re-run is idempotent (no duplicate growth by `pmcid`).

### Failure/edge
1. Missing `EMAIL` causes actionable extract failure.
2. Missing Snowflake creds causes clear load failure (or controlled skip).
3. Empty extract creates zero-row curated/report safely.
4. Transient API errors retry before failing.

### Demo acceptance criteria
- One manual DAG run completes.
- Show raw file, curated file, quality report, and Snowflake row count.
- Re-run demonstrates idempotency.

## 7. Assumptions and Defaults

1. Load target is Snowflake `RAW.PMC_ARTICLES`.
2. Manual run now; schedule not enabled.
3. Separate `venv` and `venv_airflow` retained.
4. Airflow 3-compatible imports should be used.
5. `.env.example` must be added for clean setup reproducibility.
