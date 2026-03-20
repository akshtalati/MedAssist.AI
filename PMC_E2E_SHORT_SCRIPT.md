# PMC End-to-End Short Script + Run Checklist

## A) 60-90 sec speaking script
- We implemented a PMC-only ETL pipeline to prove end-to-end reliability before scaling to all sources.
- Extract pulls PMC records from NCBI and stores raw JSON locally.
- Transform does four things: normalizes text fields, handles missing/null PMCID values, removes duplicate records using stable keys, and generates a quality report with input/valid/drop counts.
- Load writes manifests and PMC rows into Snowflake RAW tables with idempotent behavior (`NOT EXISTS`-based inserts).
- Airflow orchestrates all stages with clear dependencies, retries, and run history.
- We validated at three levels: Airflow task status, local artifacts, and Snowflake row counts.
- We fixed key issues: VARIANT type mismatch, missing PMCID handling, and strict transform ratio bug.
- Final result: repeatable, auditable PMC pipeline ready to extend to other sources.

---

## B) Run steps (demo)

### 1) Start Airflow
From project root:

```bash
./start_airflow.sh
```

Open UI: `http://localhost:8080`

### 2) Airflow variables
In `Admin -> Variables`:
- `medassist_project_root` = project absolute path
- `pmc_term` = `rare disease`
- `pmc_max_records` = `100` (fast demo) or `1000`
- `pmc_enable_load` = `true`

### 3) Trigger DAG
- DAG: `pmc_etl_demo`
- Click `Trigger`

Expected flow:
- `pmc_extract` -> `pmc_transform` -> `branch_load` -> `pmc_load_snowflake` -> `pmc_postcheck`

---

## C) Verify in Airflow
Run is correct when:
- `pmc_extract` = Success
- `pmc_transform` = Success
- `branch_load` = Success
- `pmc_load_snowflake` = Success
- `skip_load` = Skipped (expected when load enabled)
- `pmc_postcheck` = Success

---

## D) Verify locally (IDE/terminal)

```bash
ls -lt data/raw/pmc | head -n 3
ls -lt data/normalized/pmc_curated_*.json | head -n 3
ls -lt data/normalized/pmc_quality_report_*.json | head -n 3
```

Quick sanity check:

```bash
python - << 'PY'
import json,glob
raw=sorted(glob.glob('data/raw/pmc/*.json'))[-1]
cur=sorted(glob.glob('data/normalized/pmc_curated_*.json'))[-1]
rep=sorted(glob.glob('data/normalized/pmc_quality_report_*.json'))[-1]
print('raw file:', raw)
print('curated file:', cur)
print('report file:', rep)
print('raw count:', len(json.load(open(raw)).get('data',{}).get('articles',[])))
print('curated count:', len(json.load(open(cur)).get('data',{}).get('articles',[])))
print('report:', json.load(open(rep)))
PY
```

---

## E) Verify in Snowflake

```sql
USE ROLE TRAINING_ROLE;
USE WAREHOUSE MEDASSIST_WH;

SELECT COUNT(*) AS manifests FROM MEDASSIST_DB.RAW.FETCH_MANIFESTS;
SELECT COUNT(*) AS pmc_rows FROM MEDASSIST_DB.RAW.PMC_ARTICLES;
SELECT pmcid, title, pub_date
FROM MEDASSIST_DB.RAW.PMC_ARTICLES
LIMIT 10;
```

Expected:
- manifests count increases after runs
- pmc_rows reflects loaded records
- sample rows show populated title/pub_date and stable ids

---

## F) ELI5: What each file/task is doing (and why it matters)

`dags/pmc_etl_demo_dag.py` is the traffic controller. Think of it like a checklist manager that says: first fetch data, then clean it, then load it, then verify. Business-wise, this gives reliability and visibility. Instead of “someone forgot step 2,” the order is enforced and visible in one screen for the team and professor.

`scripts/fetch_source.py` is the command switchboard for source-level runs. For PMC, it calls the PMC fetcher with inputs like search term and max records. Business-wise, this helps us run only what we need (PMC-only demo) without running the whole platform, which saves time and reduces risk during presentations and testing.

`src/fetchers/pmc.py` is the extractor. It talks to NCBI E-Utilities, gets PMC article data, parses XML into usable JSON fields (title, abstract, authors, date), and writes raw output. Business reason: raw ingestion proves data lineage. We can always show where data came from before we transform it.

`scripts/transform_pmc_curated.py` is the cleaner and quality gate. It normalizes messy text, handles missing PMCID safely, removes duplicates with stable keys, and writes a quality report. Business reason: this is where raw data becomes trustworthy. Without this step, analytics and downstream decisions are noisy and less credible.

`scripts/load_to_snowflake.py` is the loader. It takes curated/local outputs and inserts them into Snowflake RAW tables with idempotent logic so reruns don’t duplicate rows. Business reason: the warehouse is our shared source for reporting and future products, so this step turns file-based data into queryable, team-usable assets.

`src/snowflake_client.py` is the connection helper. It reads credentials from `.env`, picks auth mode, and opens the Snowflake session. Business reason: centralizing connection logic avoids copy-paste mistakes and makes security/config changes easier in one place.

Task `pmc_extract` in Airflow is the “collect” step. It creates the latest raw snapshot. Business value: we can prove freshness and source provenance.

Task `pmc_transform` is the “clean and standardize” step. It creates curated records and quality metrics. Business value: improves trust, consistency, and explainability of data quality.

Task `branch_load` is the “decision” step. It checks if load is enabled and routes the run. Business value: gives controlled operation (for example, transform-only runs in test mode).

Task `pmc_load_snowflake` is the “publish” step. It writes final records to Snowflake so analysts/apps can consume them. Business value: makes data operational and reusable across stakeholders.

Task `pmc_postcheck` is the “verify” step. It prints checks so we confirm counts after load. Business value: closes the loop with evidence that the pipeline did what it was supposed to do.
