# MedAssist Ingestion Airflow Demo Runbook (Today)

## 1) Demo Goal
Run the **`medassist_ingestion`** DAG end-to-end and verify local outputs.

DAG stages:
- `fetch_checkpointed`
- `cleanup_duplicates`
- `build_symptom_index`

---

## 2) Why Airflow
- Clear task dependency flow and run history in UI.
- Retry/failure handling better than manual script chaining.
- Easy to explain in class: one DAG, visible stage status, reproducible run.

---

## 3) Pre-check
From project root:

```bash
source venv/bin/activate
python -V
```

Required env for ingestion fetchers:

```bash
EMAIL=your_email@domain.com
```

Optional for better NCBI throughput:

```bash
NCBI_API_KEY=...
```

---

## 4) Start Airflow

```bash
./start_airflow.sh
```

Open:
- `http://localhost:8080`

---

## 5) Airflow Variables
In `Admin -> Variables`:

Required:
- `medassist_project_root` = absolute project path

No PMC-specific variables needed for this DAG.

---

## 6) Trigger the Correct DAG
1. Open DAG: **`medassist_ingestion`**
2. Click `Trigger`
3. Expected flow:
   - `fetch_checkpointed` -> `cleanup_duplicates` -> `build_symptom_index`

---

## 7) Verify Each Stage

### A) Airflow
- All three tasks should be `Success`.

### B) Local files

```bash
ls -la data/.fetch_checkpoint.json
ls -la data/normalized/symptom_index.json
ls -la data/raw
```

Quick checks:

```bash
python - << 'PY'
import json
cp=json.load(open('data/.fetch_checkpoint.json'))
idx=json.load(open('data/normalized/symptom_index.json'))
print('completed_sources=', cp.get('completed', []))
print('last_error=', cp.get('last_error'))
print('symptom_keys=', len(idx.get('symptom_to_diseases', {})))
PY
```

---

## 8) Issues Faced and Fixes (for this DAG)

1. DAG import failed initially
- Error: `schedule_interval` not accepted in current Airflow version.
- Fix: switched DAG to `schedule=None` style.

2. Missing Airflow variable caused task template failure
- Error: `Variable medassist_venv_python not found`.
- Fix: set/derive Python path from `medassist_project_root` and simplified env usage.

3. Paths with spaces caused command issues
- Fix: pass paths via env vars and quote in bash command (`cd "$PROJECT_ROOT"`).

4. Large fetch interruptions (timeouts/rate limits)
- Fix: checkpointed fetch (`data/.fetch_checkpoint.json`) so rerun resumes instead of restarting.

---

## 9) If DAG Appears Stuck
- Keep `./start_airflow.sh` terminal running.
- Refresh Airflow UI after 20-40 seconds.
- If needed, clear and retrigger the run from run details.

---

## 10) 2-Minute Script (medassist_ingestion only)
Today we ran our main ingestion DAG, `medassist_ingestion`, which has three stages: checkpointed fetch, duplicate cleanup, and symptom index build. We used Airflow to make the process reliable and observable instead of running scripts manually.

In `fetch_checkpointed`, we pull data from configured medical sources with resume checkpoints, so network or rate-limit interruptions do not force a full restart. In `cleanup_duplicates`, we keep the best raw file per source and remove duplicates. In `build_symptom_index`, we generate the symptom-to-disease lookup used in downstream querying.

We faced a few orchestration issues and fixed them: a DAG import parameter mismatch, missing runtime variable for venv path, and path quoting problems due to spaces in directories. We also handled timeout risks through checkpointing. After fixes, all three tasks run in sequence and produce verifiable outputs, including checkpoint metadata and the final symptom index file.
