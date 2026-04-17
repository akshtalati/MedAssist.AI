"""Clinical workflow storage and retrieval helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4
from urllib import request as urllib_request

from snowflake.connector.errors import ProgrammingError

from .snowflake_client import get_connection

# region agent log
_PERF064_LOG = Path(__file__).resolve().parent.parent / ".cursor" / "debug-064a4f.log"
_PERF064_SESSION = "064a4f"


def _perf064_tx(encounter_id: str, turn_no: int, n_candidates: int, phase: str, t0: float, **extra: Any) -> None:
    payload = {
        "sessionId": _PERF064_SESSION,
        "timestamp": int(time.time() * 1000),
        "hypothesisId": "H3b",
        "location": "clinical_workflow.py:answer_and_update_tx",
        "message": "tx_subphase",
        "data": {
            "runId": "post-fix",
            "phase": phase,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 2),
            "encounter_id": encounter_id,
            "turn_no": turn_no,
            "n_candidates": n_candidates,
            **extra,
        },
    }
    try:
        _PERF064_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_PERF064_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass
    try:
        req = urllib_request.Request(
            _DEBUG_ENDPOINT,
            data=json.dumps(payload, default=str).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Debug-Session-Id": _PERF064_SESSION,
            },
            method="POST",
        )
        urllib_request.urlopen(req, timeout=0.7).read()
    except Exception:
        pass


# endregion

_DEBUG_LOG_PATH = "/Users/atharvakurlekar/Library/CloudStorage/OneDrive-NortheasternUniversity/Data Engineering/med/MedAssist.AI/.cursor/debug-742e45.log"
_DEBUG_ENDPOINT = "http://127.0.0.1:7299/ingest/6c1651b6-79fe-48a7-a0a7-0d0f9a35fdde"


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    try:
        payload = {
            "sessionId": "742e45",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        req = urllib_request.Request(
            _DEBUG_ENDPOINT,
            data=json.dumps(payload, default=str).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Debug-Session-Id": "742e45",
            },
            method="POST",
        )
        urllib_request.urlopen(req, timeout=0.7).read()
    except Exception:
        return


def _add_column_if_missing(cur, table: str, col: str, col_type: str) -> None:
    """Snowflake has no CREATE TABLE IF NOT EXISTS column; evolve old tables."""
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {col_type}")
    except ProgrammingError as e:
        msg = str(e).lower()
        if "already exists" in msg or "duplicate" in msg:
            return
        raise


def _evolve_clinical_table_columns(cur) -> None:
    encounter_tbl = "CLINICAL.PATIENT_ENCOUNTER"
    _add_column_if_missing(cur, encounter_tbl, "org_id", "STRING")
    _add_column_if_missing(cur, encounter_tbl, "created_by_user_id", "STRING")
    diff = "CLINICAL.ENCOUNTER_DIFFERENTIAL"
    _add_column_if_missing(cur, diff, "kg_version", "STRING")
    _add_column_if_missing(cur, diff, "kg_build_id", "STRING")
    _add_column_if_missing(cur, diff, "source_snapshot_ts", "TIMESTAMP_NTZ")
    _add_column_if_missing(cur, diff, "confidence_score", "FLOAT")
    audit = "CLINICAL.ENCOUNTER_AUDIT"
    _add_column_if_missing(cur, audit, "context_hash", "STRING")
    _add_column_if_missing(cur, audit, "output_hash", "STRING")
    _add_column_if_missing(cur, audit, "kg_version", "STRING")
    _add_column_if_missing(cur, audit, "kg_build_id", "STRING")
    _add_column_if_missing(cur, audit, "error_flags", "VARIANT")
    _add_column_if_missing(cur, audit, "pipeline_metrics", "VARIANT")
    _add_column_if_missing(cur, audit, "actor_user_id", "STRING")
    _add_column_if_missing(cur, audit, "prompt_hash", "STRING")


@dataclass
class EncounterInput:
    age: int | None
    sex: str | None
    known_conditions: list[str]
    medications: list[str]
    allergies: list[str]
    history_summary: str
    symptoms: list[dict[str, str]]
    org_id: str | None = None
    created_by_user_id: str | None = None


@dataclass
class AssessmentResult:
    encounter_id: str
    provider_used: str
    model_name: str
    degraded_mode: str
    errors: list[str]
    contract: dict[str, Any]
    assessment_markdown: str
    top_candidates: list[dict[str, Any]]
    kg_version: str
    kg_build_id: str


def ensure_clinical_tables() -> None:
    """Create required CLINICAL and KNOWLEDGE_GRAPH tables if missing."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        ddl_statements = [
            "CREATE SCHEMA IF NOT EXISTS CLINICAL",
            "CREATE SCHEMA IF NOT EXISTS KNOWLEDGE_GRAPH",
            """
            CREATE TABLE IF NOT EXISTS CLINICAL.PATIENT_ENCOUNTER (
              encounter_id STRING PRIMARY KEY,
              created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
              updated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
              age NUMBER(3,0),
              sex STRING,
              known_conditions ARRAY,
              medications ARRAY,
              allergies ARRAY,
              history_summary STRING
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS CLINICAL.ENCOUNTER_SYMPTOMS (
              encounter_id STRING,
              symptom STRING,
              onset STRING,
              severity STRING,
              duration STRING,
              created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS CLINICAL.ENCOUNTER_QA (
              encounter_id STRING,
              turn_no NUMBER(10,0),
              question_text STRING,
              answer_text STRING,
              asked_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
              answered_at TIMESTAMP_NTZ
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS CLINICAL.ENCOUNTER_DIFFERENTIAL (
              encounter_id STRING,
              iteration NUMBER(10,0),
              rank_no NUMBER(10,0),
              disease_name STRING,
              disease_code STRING,
              kg_version STRING,
              kg_build_id STRING,
              source_snapshot_ts TIMESTAMP_NTZ,
              score FLOAT,
              rationale STRING,
              source STRING,
              confidence_score FLOAT,
              created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS CLINICAL.IDEMPOTENCY_KEYS (
              endpoint STRING,
              idem_key STRING,
              request_hash STRING,
              response_json VARIANT,
              created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
              expires_at TIMESTAMP_NTZ,
              PRIMARY KEY (endpoint, idem_key)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS CLINICAL.ENCOUNTER_AUDIT (
              audit_id STRING DEFAULT UUID_STRING(),
              encounter_id STRING,
              turn_no NUMBER(10,0),
              action STRING,
              provider STRING,
              model_name STRING,
              degraded_mode STRING,
              context_hash STRING,
              output_hash STRING,
              kg_version STRING,
              kg_build_id STRING,
              error_flags VARIANT,
              created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS CLINICAL.ENCOUNTER_DIFF_DELTA (
              encounter_id STRING,
              turn_no NUMBER(10,0),
              added_diseases ARRAY,
              removed_diseases ARRAY,
              rank_changes VARIANT,
              created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS KNOWLEDGE_GRAPH.KG_NODES (
              node_id STRING PRIMARY KEY,
              node_type STRING,
              node_label STRING,
              attributes VARIANT,
              source STRING,
              created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS KNOWLEDGE_GRAPH.KG_EDGES (
              edge_id STRING PRIMARY KEY,
              edge_type STRING,
              from_node_id STRING,
              to_node_id STRING,
              weight FLOAT DEFAULT 1.0,
              attributes VARIANT,
              source STRING,
              created_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS KNOWLEDGE_GRAPH.KG_BUILD_META (
              kg_version STRING,
              build_id STRING,
              source_snapshot_ts TIMESTAMP_NTZ,
              built_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
              notes STRING,
              PRIMARY KEY (kg_version, build_id)
            )
            """,
            """
            MERGE INTO KNOWLEDGE_GRAPH.KG_BUILD_META t
            USING (
              SELECT
                'v1' AS kg_version,
                'seed_symptom_map' AS build_id,
                CURRENT_TIMESTAMP() AS source_snapshot_ts,
                'Initial symptom-disease KG seed from NORMALIZED.SYMPTOM_DISEASE_MAP' AS notes
            ) s
            ON t.kg_version = s.kg_version AND t.build_id = s.build_id
            WHEN NOT MATCHED THEN
              INSERT (kg_version, build_id, source_snapshot_ts, notes)
              VALUES (s.kg_version, s.build_id, s.source_snapshot_ts, s.notes)
            """,
        ]
        for ddl in ddl_statements:
            cur.execute(ddl)
        _evolve_clinical_table_columns(cur)
        try:
            cur.execute(
                """
                ALTER TABLE CLINICAL.ENCOUNTER_QA
                ADD CONSTRAINT uq_encounter_turn UNIQUE (encounter_id, turn_no)
                """
            )
        except ProgrammingError as e:
            msg = str(e).lower()
            if "already exists" not in msg and "duplicate" not in msg:
                raise
    finally:
        cur.close()
        conn.close()


def seed_graph_from_symptom_map() -> None:
    """Idempotently seed disease/symptom nodes and edges from normalized map."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO KNOWLEDGE_GRAPH.KG_NODES (node_id, node_type, node_label, attributes, source)
            SELECT DISTINCT
              'DIS:' || COALESCE(NULLIF(TRIM(orpha_code), ''), SHA2(disease_name, 256)),
              'DISEASE',
              disease_name,
              OBJECT_CONSTRUCT('orpha_code', orpha_code),
              'NORMALIZED.SYMPTOM_DISEASE_MAP'
            FROM NORMALIZED.SYMPTOM_DISEASE_MAP
            WHERE disease_name IS NOT NULL
              AND NOT EXISTS (
                SELECT 1
                FROM KNOWLEDGE_GRAPH.KG_NODES n
                WHERE n.node_id = 'DIS:' || COALESCE(NULLIF(TRIM(NORMALIZED.SYMPTOM_DISEASE_MAP.orpha_code), ''), SHA2(NORMALIZED.SYMPTOM_DISEASE_MAP.disease_name, 256))
              )
            """
        )
        cur.execute(
            """
            INSERT INTO KNOWLEDGE_GRAPH.KG_NODES (node_id, node_type, node_label, attributes, source)
            SELECT DISTINCT
              'SYM:' || SHA2(LOWER(TRIM(symptom)), 256),
              'SYMPTOM',
              symptom,
              OBJECT_CONSTRUCT(),
              'NORMALIZED.SYMPTOM_DISEASE_MAP'
            FROM NORMALIZED.SYMPTOM_DISEASE_MAP
            WHERE symptom IS NOT NULL
              AND NOT EXISTS (
                SELECT 1
                FROM KNOWLEDGE_GRAPH.KG_NODES n
                WHERE n.node_id = 'SYM:' || SHA2(LOWER(TRIM(NORMALIZED.SYMPTOM_DISEASE_MAP.symptom)), 256)
              )
            """
        )
        cur.execute(
            """
            INSERT INTO KNOWLEDGE_GRAPH.KG_EDGES (edge_id, edge_type, from_node_id, to_node_id, weight, attributes, source)
            SELECT DISTINCT
              'EDGE:' || SHA2(
                'HAS_SYMPTOM|'
                || COALESCE(NULLIF(TRIM(orpha_code), ''), SHA2(disease_name, 256))
                || '|'
                || LOWER(TRIM(symptom)),
                256
              ),
              'HAS_SYMPTOM',
              'DIS:' || COALESCE(NULLIF(TRIM(orpha_code), ''), SHA2(disease_name, 256)),
              'SYM:' || SHA2(LOWER(TRIM(symptom)), 256),
              1.0,
              OBJECT_CONSTRUCT('frequency', frequency),
              'NORMALIZED.SYMPTOM_DISEASE_MAP'
            FROM NORMALIZED.SYMPTOM_DISEASE_MAP
            WHERE symptom IS NOT NULL
              AND disease_name IS NOT NULL
              AND NOT EXISTS (
                SELECT 1
                FROM KNOWLEDGE_GRAPH.KG_EDGES e
                WHERE e.edge_id = 'EDGE:' || SHA2(
                  'HAS_SYMPTOM|'
                  || COALESCE(NULLIF(TRIM(NORMALIZED.SYMPTOM_DISEASE_MAP.orpha_code), ''), SHA2(NORMALIZED.SYMPTOM_DISEASE_MAP.disease_name, 256))
                  || '|'
                  || LOWER(TRIM(NORMALIZED.SYMPTOM_DISEASE_MAP.symptom)),
                  256
                )
              )
            """
        )
    finally:
        cur.close()
        conn.close()


def create_encounter(inp: EncounterInput) -> str:
    encounter_id = str(uuid4())
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO CLINICAL.PATIENT_ENCOUNTER
              (encounter_id, age, sex, known_conditions, medications, allergies, history_summary, org_id, created_by_user_id)
            SELECT
              %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), %s, %s, %s
            """,
            (
                encounter_id,
                inp.age,
                inp.sex,
                _json_array(inp.known_conditions),
                _json_array(inp.medications),
                _json_array(inp.allergies),
                inp.history_summary,
                inp.org_id,
                inp.created_by_user_id,
            ),
        )
        for s in inp.symptoms:
            cur.execute(
                """
                INSERT INTO CLINICAL.ENCOUNTER_SYMPTOMS (encounter_id, symptom, onset, severity, duration)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    encounter_id,
                    (s.get("symptom") or "").strip().lower(),
                    s.get("onset"),
                    s.get("severity"),
                    s.get("duration"),
                ),
            )
    finally:
        cur.close()
        conn.close()
    return encounter_id


def create_encounter_tx(inp: EncounterInput) -> str:
    """Transaction-safe encounter creation."""
    encounter_id = str(uuid4())
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("BEGIN")
        cur.execute(
            """
            INSERT INTO CLINICAL.PATIENT_ENCOUNTER
              (encounter_id, age, sex, known_conditions, medications, allergies, history_summary, org_id, created_by_user_id)
            SELECT
              %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), %s, %s, %s
            """,
            (
                encounter_id,
                inp.age,
                inp.sex,
                _json_array(inp.known_conditions),
                _json_array(inp.medications),
                _json_array(inp.allergies),
                inp.history_summary,
                inp.org_id,
                inp.created_by_user_id,
            ),
        )
        for s in inp.symptoms:
            cur.execute(
                """
                INSERT INTO CLINICAL.ENCOUNTER_SYMPTOMS (encounter_id, symptom, onset, severity, duration)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    encounter_id,
                    (s.get("symptom") or "").strip().lower(),
                    s.get("onset"),
                    s.get("severity"),
                    s.get("duration"),
                ),
            )
        cur.execute("COMMIT")
        return encounter_id
    except Exception:
        cur.execute("ROLLBACK")
        raise
    finally:
        cur.close()
        conn.close()


def save_differential(
    encounter_id: str,
    iteration: int,
    rows: list[dict[str, Any]],
    *,
    kg_version: str = "v1",
    kg_build_id: str = "seed_symptom_map",
    source_snapshot_ts: str | None = None,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "DELETE FROM CLINICAL.ENCOUNTER_DIFFERENTIAL WHERE encounter_id = %s AND iteration = %s",
            (encounter_id, iteration),
        )
        for rank_no, r in enumerate(rows, start=1):
            cur.execute(
                """
                INSERT INTO CLINICAL.ENCOUNTER_DIFFERENTIAL
                  (encounter_id, iteration, rank_no, disease_name, disease_code, kg_version, kg_build_id, source_snapshot_ts, score, rationale, source, confidence_score)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    encounter_id,
                    iteration,
                    rank_no,
                    r.get("disease_name"),
                    r.get("disease_code"),
                    kg_version,
                    kg_build_id,
                    source_snapshot_ts,
                    float(r.get("score", 0.0)),
                    r.get("rationale", ""),
                    r.get("source", "knowledge_graph"),
                    float(r["confidence_score"]) if r.get("confidence_score") is not None else None,
                ),
            )
    finally:
        cur.close()
        conn.close()


def next_turn_no(encounter_id: str) -> int:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COALESCE(MAX(turn_no), 0) + 1 FROM CLINICAL.ENCOUNTER_QA WHERE encounter_id = %s", (encounter_id,))
        return int(cur.fetchone()[0] or 1)
    finally:
        cur.close()
        conn.close()


def add_question(encounter_id: str, turn_no: int, question_text: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO CLINICAL.ENCOUNTER_QA (encounter_id, turn_no, question_text)
            VALUES (%s, %s, %s)
            """,
            (encounter_id, turn_no, question_text),
        )
    finally:
        cur.close()
        conn.close()


def add_answer(encounter_id: str, turn_no: int, answer_text: str) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            UPDATE CLINICAL.ENCOUNTER_QA
            SET answer_text = %s, answered_at = CURRENT_TIMESTAMP()
            WHERE encounter_id = %s AND turn_no = %s
            """,
            (answer_text, encounter_id, turn_no),
        )
    finally:
        cur.close()
        conn.close()


def answer_and_update_tx(
    encounter_id: str,
    turn_no: int,
    answer_text: str,
    candidates: list[dict[str, Any]],
    *,
    kg_version: str,
    kg_build_id: str,
    source_snapshot_ts: str | None,
    diff_previous: list[dict[str, Any]] | None = None,
) -> None:
    """Single transaction for differential update after ENCOUNTER_QA was persisted (e.g. add_answer).

    When ``diff_previous`` is set, also inserts ENCOUNTER_DIFF_DELTA in the same transaction
    (same semantics as :func:`save_diff_delta`).
    """
    # region agent log
    t_tx = time.perf_counter()
    _ = answer_text  # retained for API compatibility; answer row is updated by add_answer before this runs.
    _perf064_tx(encounter_id, turn_no, len(candidates), "tx_begin", t_tx)
    # endregion
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("BEGIN")
        cur.execute(
            "DELETE FROM CLINICAL.ENCOUNTER_DIFFERENTIAL WHERE encounter_id = %s AND iteration = %s",
            (encounter_id, turn_no + 1),
        )
        # region agent log
        _perf064_tx(encounter_id, turn_no, len(candidates), "after_delete", t_tx)
        # endregion
        if candidates:
            placeholders = ", ".join(
                ["(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"] * len(candidates)
            )
            insert_sql = f"""
                INSERT INTO CLINICAL.ENCOUNTER_DIFFERENTIAL
                  (encounter_id, iteration, rank_no, disease_name, disease_code, kg_version, kg_build_id, source_snapshot_ts, score, rationale, source, confidence_score)
                VALUES {placeholders}
            """
            params: list[Any] = []
            for rank_no, r in enumerate(candidates, start=1):
                params.extend(
                    [
                        encounter_id,
                        turn_no + 1,
                        rank_no,
                        r.get("disease_name"),
                        r.get("disease_code"),
                        kg_version,
                        kg_build_id,
                        source_snapshot_ts,
                        float(r.get("score", 0.0)),
                        r.get("rationale", ""),
                        r.get("source", "knowledge_graph"),
                        float(r["confidence_score"]) if r.get("confidence_score") is not None else None,
                    ]
                )
            cur.execute(insert_sql, params)
        # region agent log
        _perf064_tx(encounter_id, turn_no, len(candidates), "after_batch_insert", t_tx)
        # endregion
        if diff_previous is not None:
            save_diff_delta(
                encounter_id,
                turn_no,
                diff_previous,
                candidates,
                cursor=cur,
            )
            # region agent log
            _perf064_tx(encounter_id, turn_no, len(candidates), "after_diff_delta_insert", t_tx)
            # endregion
        cur.execute("COMMIT")
        # region agent log
        _perf064_tx(encounter_id, turn_no, len(candidates), "after_commit", t_tx)
        # endregion
    except Exception:
        cur.execute("ROLLBACK")
        raise
    finally:
        cur.close()
        conn.close()


def get_encounter(encounter_id: str) -> dict[str, Any] | None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT age, sex, known_conditions, medications, allergies, history_summary, org_id, created_by_user_id
            FROM CLINICAL.PATIENT_ENCOUNTER
            WHERE encounter_id = %s
            """,
            (encounter_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        cur.execute(
            """
            SELECT symptom, onset, severity, duration
            FROM CLINICAL.ENCOUNTER_SYMPTOMS
            WHERE encounter_id = %s
            ORDER BY created_at
            """,
            (encounter_id,),
        )
        symptoms = [
            {"symptom": s, "onset": o, "severity": sev, "duration": d}
            for s, o, sev, d in cur.fetchall()
        ]
        cur.execute(
            """
            SELECT turn_no, question_text, answer_text
            FROM CLINICAL.ENCOUNTER_QA
            WHERE encounter_id = %s
            ORDER BY turn_no
            """,
            (encounter_id,),
        )
        qa = [{"turn_no": int(t), "question": q, "answer": a} for t, q, a in cur.fetchall()]
        cur.execute(
            """
            SELECT iteration, rank_no, disease_name, disease_code, score, rationale, source, confidence_score
            FROM CLINICAL.ENCOUNTER_DIFFERENTIAL
            WHERE encounter_id = %s
            ORDER BY iteration DESC, rank_no ASC
            """,
            (encounter_id,),
        )
        differential = [
            {
                "iteration": int(i),
                "rank_no": int(r),
                "disease_name": dn,
                "disease_code": dc,
                "score": float(sc or 0.0),
                "rationale": rs,
                "source": src,
                **({"confidence_score": float(cf)} if cf is not None else {}),
            }
            for i, r, dn, dc, sc, rs, src, cf in cur.fetchall()
        ]
        return normalize_encounter_dict(
            {
                "encounter_id": encounter_id,
                "age": row[0],
                "sex": row[1],
                "known_conditions": _to_list(row[2]),
                "medications": _to_list(row[3]),
                "allergies": _to_list(row[4]),
                "history_summary": row[5] or "",
                "org_id": row[6],
                "created_by_user_id": row[7],
                "symptoms": symptoms,
                "qa_history": qa,
                "differential": differential,
            }
        )
    finally:
        cur.close()
        conn.close()


def rank_diseases_from_graph(encounter_id: str, limit: int = 12) -> list[dict[str, Any]]:
    """
    Rank candidate diseases using KG edges + encounter symptoms.

    Confidence uses weighted symptom coverage (rarer symptoms carry more weight)
    to reduce large tie groups where many diseases match the same symptom count.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            WITH symptom_nodes AS (
              SELECT DISTINCT 'SYM:' || SHA2(LOWER(TRIM(symptom)), 256) AS sym_node
              FROM CLINICAL.ENCOUNTER_SYMPTOMS
              WHERE encounter_id = %s
            ),
            symptom_df AS (
              SELECT
                s.sym_node,
                NULLIF(COUNT(DISTINCT e.from_node_id), 0)::FLOAT AS disease_cnt
              FROM symptom_nodes s
              JOIN KNOWLEDGE_GRAPH.KG_EDGES e
                ON e.to_node_id = s.sym_node
               AND e.edge_type = 'HAS_SYMPTOM'
              GROUP BY s.sym_node
            ),
            disease_scores AS (
              SELECT
                n.node_label AS disease_name,
                COALESCE(n.attributes:orpha_code::STRING, '') AS disease_code,
                COUNT(*)::FLOAT AS matched_symptoms,
                COUNT(*)::FLOAT / NULLIF((
                  SELECT COUNT(*) FROM symptom_nodes
                ), 0) AS coverage,
                SUM(COALESCE(1.0 / sd.disease_cnt, 0.0))::FLOAT AS weighted_match,
                (
                  SELECT SUM(COALESCE(1.0 / disease_cnt, 0.0))::FLOAT FROM symptom_df
                ) AS total_weight
              FROM symptom_nodes s
              JOIN KNOWLEDGE_GRAPH.KG_EDGES e
                ON e.to_node_id = s.sym_node
               AND e.edge_type = 'HAS_SYMPTOM'
              JOIN KNOWLEDGE_GRAPH.KG_NODES n
                ON n.node_id = e.from_node_id
               AND n.node_type = 'DISEASE'
              LEFT JOIN symptom_df sd
                ON sd.sym_node = s.sym_node
              GROUP BY n.node_label, n.attributes:orpha_code::STRING
            )
            SELECT
              disease_name,
              disease_code,
              matched_symptoms,
              coverage,
              weighted_match,
              total_weight,
              (SELECT COUNT(*)::FLOAT FROM symptom_nodes) AS n_encounter_symptoms
            FROM disease_scores
            ORDER BY weighted_match DESC, matched_symptoms DESC, coverage DESC, disease_name
            LIMIT %s
            """,
            (encounter_id, limit),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    out: list[dict[str, Any]] = []
    for disease_name, disease_code, matched, coverage, weighted_match, total_weight, n_enc in rows or []:
        cov = float(coverage or 0.0)
        m = float(matched or 0.0)
        wm = float(weighted_match or 0.0)
        tw = float(total_weight or 0.0)
        ns = float(n_enc or 0.0)
        damp = m / max(1.0, ns) if ns else 0.0
        weighted_coverage = (wm / tw) if tw else 0.0
        confidence_score = round(min(1.0, max(0.0, weighted_coverage * (0.6 + 0.4 * damp))), 6)
        out.append(
            {
                "disease_name": disease_name,
                "disease_code": disease_code,
                "score": round(float(m + cov + 0.35 * wm), 6),
                "rationale": (
                    f"Matched symptoms: {int(m)}; coverage={cov:.2f}; "
                    f"weighted_match={wm:.3f}"
                ),
                "source": "knowledge_graph",
                "confidence_score": confidence_score,
            }
        )
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H3",
        location="src/clinical_workflow.py:rank_diseases_from_graph",
        message="graph_ranking_summary",
        data={
            "encounter_id": encounter_id,
            "rows_count": len(rows or []),
            "top_confidence": [float(x.get("confidence_score") or 0.0) for x in out[:3]],
            "top_score": [float(x.get("score") or 0.0) for x in out[:3]],
        },
    )
    # endregion
    return out


def rank_diseases_from_symptom_map(encounter_id: str, limit: int = 12) -> list[dict[str, Any]]:
    """Fallback ranking when KG tables are unavailable."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            WITH symptoms AS (
              SELECT DISTINCT LOWER(TRIM(symptom)) AS symptom
              FROM CLINICAL.ENCOUNTER_SYMPTOMS
              WHERE encounter_id = %s
            )
            SELECT
              m.disease_name,
              COALESCE(m.orpha_code, '') AS disease_code,
              COUNT(*)::FLOAT AS matched,
              (SELECT COUNT(*)::FLOAT FROM symptoms) AS n_encounter_symptoms
            FROM NORMALIZED.SYMPTOM_DISEASE_MAP m
            JOIN symptoms s
              ON LOWER(m.symptom) LIKE '%%' || s.symptom || '%%'
            GROUP BY m.disease_name, COALESCE(m.orpha_code, '')
            ORDER BY matched DESC, m.disease_name
            LIMIT %s
            """,
            (encounter_id, limit),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    out: list[dict[str, Any]] = []
    for dn, dc, matched, n_enc in rows or []:
        ns = float(n_enc or 0.0)
        m = float(matched or 0.0)
        cov = (m / ns) if ns else 0.0
        confidence_score = round(min(1.0, max(0.0, cov)), 6)
        out.append(
            {
                "disease_name": dn,
                "disease_code": dc,
                "score": float(matched or 0.0),
                "rationale": f"Symptom-map matches: {int(matched or 0)}",
                "source": "symptom_map",
                "confidence_score": confidence_score,
            }
        )
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H5",
        location="src/clinical_workflow.py:rank_diseases_from_symptom_map",
        message="symptom_map_ranking_summary",
        data={
            "encounter_id": encounter_id,
            "rows_count": len(rows or []),
            "top_confidence": [float(x.get("confidence_score") or 0.0) for x in out[:3]],
            "top_score": [float(x.get("score") or 0.0) for x in out[:3]],
        },
    )
    # endregion
    return out


_STOP = frozenset(
    "that with from this have been weeks days years month months patient history noted "
    "reports states presents denies without acute chronic severe mild moderate".split()
)

_FOLLOWUP_NEGATIONS = frozenset(
    {
        "no",
        "not",
        "without",
        "denies",
        "deny",
        "negative",
        "none",
        "never",
    }
)

_FOLLOWUP_PHRASES = (
    "shortness of breath",
    "chest pain",
    "abdominal pain",
    "visual aura",
    "neck stiffness",
    "focal weakness",
    "thunderclap onset",
    "altered mental status",
    "weight loss",
    "joint pain",
)


def _extract_affirmed_followup_terms(answer_text: str, *, max_terms: int = 12) -> list[str]:
    """
    Pull symptom-like terms from free-text follow-up responses.

    We skip clauses that are explicitly negated ("no fever", "denies confusion") so
    negative findings do not get inserted as positive encounter symptoms.
    """
    text = (answer_text or "").strip().lower()
    if len(text) < 2:
        return []

    clauses = [
        c.strip()
        for c in re.split(r"[.;]|\bbut\b|\bhowever\b|\bthough\b", text)
        if c.strip()
    ]
    out: list[str] = []
    seen: set[str] = set()
    for clause in clauses:
        words = re.findall(r"[a-z][a-z0-9]{2,}", clause)
        if not words:
            continue
        has_negation = any(w in _FOLLOWUP_NEGATIONS for w in words) or ("negative for" in clause)
        if has_negation:
            continue

        # Prefer medically meaningful multi-word phrases when present.
        for phrase in _FOLLOWUP_PHRASES:
            if phrase in clause and phrase not in seen:
                seen.add(phrase)
                out.append(phrase)
                if len(out) >= max_terms:
                    return out

        filtered = [w for w in words if len(w) >= 4 and w not in _STOP and w not in _FOLLOWUP_NEGATIONS]

        # Preserve short phrase context before falling back to single words.
        for i in range(len(filtered) - 1):
            bg = f"{filtered[i]} {filtered[i + 1]}"
            if bg not in seen:
                seen.add(bg)
                out.append(bg)
                if len(out) >= max_terms:
                    return out

        for w in filtered:
            if w not in seen:
                seen.add(w)
                out.append(w)
                if len(out) >= max_terms:
                    return out
    return out


def extract_encounter_search_tokens(encounter: dict[str, Any], max_tokens: int = 12) -> list[str]:
    """Terms from symptoms, history, QA, conditions, meds, allergies for map + context ranking."""
    parts: list[str] = []
    for s in encounter.get("symptoms") or []:
        if isinstance(s, dict):
            parts.append(str(s.get("symptom") or ""))
    parts.append(str(encounter.get("history_summary") or ""))
    for key in ("known_conditions", "medications", "allergies"):
        for x in encounter.get(key) or []:
            parts.append(str(x))
    for qa in encounter.get("qa_history") or []:
        if isinstance(qa, dict):
            parts.append(str(qa.get("question") or ""))
            parts.append(str(qa.get("answer") or ""))
    blob = " ".join(parts).lower()
    words = re.findall(r"[a-z][a-z0-9]{3,}", blob)
    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w in _STOP or w in seen:
            continue
        seen.add(w)
        out.append(w)
        if len(out) >= max_tokens:
            break
    return out


def merge_candidate_rankings(*lists: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    """Merge ranked disease lists by (name, code); keep highest score, add partial scores from extras."""
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for lst in lists:
        for c in lst or []:
            dn = str(c.get("disease_name") or "").strip()
            dc = str(c.get("disease_code") or "").strip()
            if not dn:
                continue
            key = (dn, dc)
            sc = float(c.get("score") or 0.0)
            if key not in by_key:
                by_key[key] = dict(c)
            else:
                by_key[key]["score"] = max(float(by_key[key].get("score") or 0), sc)
                prev_cf = float(by_key[key].get("confidence_score") or 0.0)
                new_cf = float(c.get("confidence_score") or 0.0)
                if new_cf or prev_cf:
                    by_key[key]["confidence_score"] = max(prev_cf, new_cf)
                src = by_key[key].get("source", "")
                if c.get("source") and c["source"] not in src:
                    by_key[key]["source"] = f"{src}+{c['source']}" if src else c["source"]
    out = sorted(by_key.values(), key=lambda x: -float(x.get("score") or 0.0))
    return out[:limit]


def ingest_followup_answer_tokens(encounter_id: str, answer_text: str, *, max_new: int = 12) -> None:
    """
    Insert salient tokens from clinician free-text into ENCOUNTER_SYMPTOMS so KG ranking
    (HAS_SYMPTOM edges) and symptom-node queries incorporate follow-up answers.
    """
    terms = _extract_affirmed_followup_terms(answer_text, max_terms=max_new)
    # region agent log
    _debug_log(
        run_id="pre-fix",
        hypothesis_id="H2",
        location="src/clinical_workflow.py:ingest_followup_answer_tokens:terms",
        message="extracted_affirmed_terms",
        data={
            "encounter_id": encounter_id,
            "max_new": max_new,
            "terms_preview": terms[:8],
            "terms_count": len(terms),
        },
    )
    # endregion
    if not terms:
        return
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT LOWER(TRIM(symptom)) FROM CLINICAL.ENCOUNTER_SYMPTOMS WHERE encounter_id = %s",
            (encounter_id,),
        )
        seen: set[str] = {r[0] for r in cur.fetchall() if r[0]}
        added = 0
        added_terms: list[str] = []
        for term in terms:
            if term in seen:
                continue
            if added >= max_new:
                break
            cur.execute(
                """
                INSERT INTO CLINICAL.ENCOUNTER_SYMPTOMS (encounter_id, symptom, onset, severity, duration)
                VALUES (%s, %s, NULL, NULL, NULL)
                """,
                (encounter_id, term),
            )
            seen.add(term)
            added += 1
            added_terms.append(term)
        # region agent log
        _debug_log(
            run_id="pre-fix",
            hypothesis_id="H2",
            location="src/clinical_workflow.py:ingest_followup_answer_tokens:insert",
            message="ingested_followup_terms",
            data={
                "encounter_id": encounter_id,
                "added_count": added,
                "added_terms": added_terms[:8],
                "seen_size_after": len(seen),
            },
        )
        # endregion
    finally:
        cur.close()
        conn.close()


def rank_diseases_from_context_tokens(encounter: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    """Use medications, conditions, allergies, and history text to find extra rows in SYMPTOM_DISEASE_MAP."""
    tokens = extract_encounter_search_tokens(encounter, max_tokens=10)
    if not tokens:
        return []
    conn = get_connection()
    cur = conn.cursor()
    try:
        conditions_sql: list[str] = []
        params: list[Any] = []
        for _ in tokens:
            conditions_sql.append("(LOWER(m.symptom) LIKE %s OR LOWER(m.disease_name) LIKE %s)")
            params.extend(["%{}%".format(t), "%{}%".format(t)])
        where = " OR ".join(conditions_sql)
        cur.execute(
            f"""
            SELECT
              m.disease_name,
              COALESCE(m.orpha_code, '') AS disease_code,
              COUNT(*)::FLOAT AS matched
            FROM NORMALIZED.SYMPTOM_DISEASE_MAP m
            WHERE {where}
            GROUP BY m.disease_name, COALESCE(m.orpha_code, '')
            ORDER BY matched DESC, m.disease_name
            LIMIT %s
            """,
            (*params, limit),
        )
        rows = cur.fetchall()
    except Exception:
        return []
    finally:
        cur.close()
        conn.close()

    return [
        {
            "disease_name": dn,
            "disease_code": dc,
            "score": max(0.25, float(matched or 0.0) * 0.2),
            "rationale": f"Context/history token match in map: {int(matched or 0)}",
            "source": "symptom_map_context",
            "confidence_score": round(min(1.0, max(0.0, 0.15 + 0.05 * float(matched or 0.0))), 6),
        }
        for dn, dc, matched in (rows or [])
    ]


_KG_META_CACHE_TTL_S = 15.0
_KG_META_CACHE: tuple[float, tuple[str, str, str | None]] | None = None


def get_latest_kg_build_meta() -> tuple[str, str, str | None]:
    global _KG_META_CACHE
    now = time.monotonic()
    if _KG_META_CACHE is not None and (now - _KG_META_CACHE[0]) < _KG_META_CACHE_TTL_S:
        return _KG_META_CACHE[1]
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT kg_version, build_id, source_snapshot_ts::STRING
            FROM KNOWLEDGE_GRAPH.KG_BUILD_META
            ORDER BY built_at DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            out: tuple[str, str, str | None] = ("v1", "seed_symptom_map", None)
        else:
            out = (str(row[0]), str(row[1]), row[2])
        _KG_META_CACHE = (now, out)
        return out
    except Exception:
        out = ("v1", "seed_symptom_map", None)
        _KG_META_CACHE = (now, out)
        return out
    finally:
        cur.close()
        conn.close()


def get_encounter_kg_preview(
    encounter_id: str,
    *,
    max_edges: int = 2500,
) -> dict[str, Any]:
    """
    Return all HAS_SYMPTOM edges from this encounter's symptoms to diseases in the KG
    (no artificial cap on distinct diseases). Rows are limited only by max_edges for safety.
    """
    kg_version, kg_build_id, _snap = get_latest_kg_build_meta()
    max_edges = max(1, min(int(max_edges), 5000))

    rows: list[Any] = []
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            WITH symptom_nodes AS (
              SELECT DISTINCT
                'SYM:' || SHA2(LOWER(TRIM(symptom)), 256) AS sym_node,
                LOWER(TRIM(symptom)) AS symptom_text
              FROM CLINICAL.ENCOUNTER_SYMPTOMS
              WHERE encounter_id = %s
            )
            SELECT
              dn.node_label AS disease_name,
              COALESCE(sn.node_label, s.symptom_text) AS symptom_label
            FROM symptom_nodes s
            JOIN KNOWLEDGE_GRAPH.KG_EDGES e
              ON e.to_node_id = s.sym_node
             AND e.edge_type = 'HAS_SYMPTOM'
            JOIN KNOWLEDGE_GRAPH.KG_NODES dn
              ON dn.node_id = e.from_node_id
             AND dn.node_type = 'DISEASE'
            LEFT JOIN KNOWLEDGE_GRAPH.KG_NODES sn
              ON sn.node_id = e.to_node_id
             AND sn.node_type = 'SYMPTOM'
            ORDER BY dn.node_label, symptom_label
            LIMIT %s
            """,
            (encounter_id, max_edges),
        )
        rows = cur.fetchall() or []
    except Exception:
        return {
            "encounter_id": encounter_id,
            "edges": [],
            "distinct_diseases": 0,
            "kg_version": kg_version,
            "kg_build_id": kg_build_id,
            "truncated": False,
            "note": "Knowledge graph query unavailable or tables empty.",
        }
    finally:
        cur.close()
        conn.close()

    edges = [{"disease": str(d or ""), "symptom": str(s or "")} for d, s in rows]
    distinct_diseases = len({e["disease"] for e in edges if e.get("disease")})
    truncated = len(rows) >= max_edges
    return {
        "encounter_id": encounter_id,
        "edges": edges,
        "distinct_diseases": distinct_diseases,
        "kg_version": kg_version,
        "kg_build_id": kg_build_id,
        "truncated": truncated,
        "max_edges": max_edges,
    }


def store_idempotent_response(endpoint: str, idem_key: str, request_hash: str, response: dict[str, Any]) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            MERGE INTO CLINICAL.IDEMPOTENCY_KEYS t
            USING (
              SELECT %s AS endpoint, %s AS idem_key, %s AS request_hash, PARSE_JSON(%s) AS response_json
            ) s
            ON t.endpoint = s.endpoint AND t.idem_key = s.idem_key
            WHEN MATCHED THEN UPDATE SET request_hash = s.request_hash, response_json = s.response_json
            WHEN NOT MATCHED THEN
              INSERT (endpoint, idem_key, request_hash, response_json, expires_at)
              VALUES (s.endpoint, s.idem_key, s.request_hash, s.response_json, DATEADD('day', 1, CURRENT_TIMESTAMP()))
            """,
            (endpoint, idem_key, request_hash, json.dumps(response)),
        )
    finally:
        cur.close()
        conn.close()


def fetch_idempotent_response(endpoint: str, idem_key: str, request_hash: str) -> tuple[dict[str, Any] | None, bool]:
    """
    Returns (response_json, conflict). conflict=True means same key with different body hash.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT request_hash, response_json
            FROM CLINICAL.IDEMPOTENCY_KEYS
            WHERE endpoint = %s AND idem_key = %s
            """,
            (endpoint, idem_key),
        )
        row = cur.fetchone()
        if not row:
            return None, False
        saved_hash = row[0]
        saved_json = row[1]
        if saved_hash != request_hash:
            return None, True
        if isinstance(saved_json, dict):
            return saved_json, False
        return json.loads(saved_json), False
    finally:
        cur.close()
        conn.close()


def hash_payload(payload: dict[str, Any]) -> str:
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def append_audit_row(
    encounter_id: str,
    turn_no: int,
    action: str,
    provider: str,
    model_name: str,
    degraded_mode: str,
    context_text: str,
    output_text: str,
    kg_version: str,
    kg_build_id: str,
    error_flags: list[str] | None = None,
    pipeline_metrics: dict[str, Any] | None = None,
    actor_user_id: str | None = None,
    prompt_hash: str | None = None,
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Use INSERT...SELECT so PARSE_JSON is not in a VALUES clause (Snowflake rejects PARSE_JSON in VALUES).
        cur.execute(
            """
            INSERT INTO CLINICAL.ENCOUNTER_AUDIT
              (encounter_id, turn_no, action, provider, model_name, degraded_mode, context_hash, output_hash, kg_version, kg_build_id, error_flags, pipeline_metrics, actor_user_id, prompt_hash)
            SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), %s, %s
            """,
            (
                encounter_id,
                turn_no,
                action,
                provider,
                model_name,
                degraded_mode,
                hashlib.sha256((context_text or "").encode("utf-8")).hexdigest(),
                hashlib.sha256((output_text or "").encode("utf-8")).hexdigest(),
                kg_version,
                kg_build_id,
                json.dumps(error_flags or []),
                json.dumps(pipeline_metrics or {}),
                actor_user_id,
                prompt_hash,
            ),
        )
    finally:
        cur.close()
        conn.close()


def save_diff_delta(
    encounter_id: str,
    turn_no: int,
    previous: list[dict[str, Any]],
    current: list[dict[str, Any]],
    *,
    cursor: Any | None = None,
) -> None:
    prev_names = [str(x.get("disease_name")) for x in previous]
    cur_names = [str(x.get("disease_name")) for x in current]
    added = [x for x in cur_names if x not in prev_names]
    removed = [x for x in prev_names if x not in cur_names]
    rank_changes: dict[str, dict[str, int]] = {}
    for name in set(prev_names).intersection(cur_names):
        rank_changes[name] = {"from": prev_names.index(name) + 1, "to": cur_names.index(name) + 1}
    if cursor is not None:
        cursor.execute(
            """
            INSERT INTO CLINICAL.ENCOUNTER_DIFF_DELTA
              (encounter_id, turn_no, added_diseases, removed_diseases, rank_changes)
            SELECT %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s)
            """,
            (
                encounter_id,
                turn_no,
                json.dumps(added),
                json.dumps(removed),
                json.dumps(rank_changes),
            ),
        )
        return
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO CLINICAL.ENCOUNTER_DIFF_DELTA
              (encounter_id, turn_no, added_diseases, removed_diseases, rank_changes)
            SELECT %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s)
            """,
            (
                encounter_id,
                turn_no,
                json.dumps(added),
                json.dumps(removed),
                json.dumps(rank_changes),
            ),
        )
    finally:
        cur.close()
        conn.close()


def _json_array(values: list[str]) -> str:
    cleaned = [v.strip() for v in values if v and v.strip()]
    return json.dumps(cleaned)


def _to_list(value: Any) -> list[str]:
    """
    Normalize Snowflake ARRAY / Python list / JSON string into a clean list of strings.
    Must not iterate a str with list(s) — e.g. list('[]') -> ['[', ']'] which joins to '[, ]'.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v is not None and str(v).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s or s in ("[]", "null", "{}"):
            return []
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if v is not None and str(v).strip()]
            except (json.JSONDecodeError, TypeError):
                pass
        return [s] if s else []
    if isinstance(value, (dict,)):
        return []
    try:
        return [str(v).strip() for v in list(value) if v is not None and str(v).strip()]
    except Exception:
        out = str(value).strip()
        return [out] if out else []


def normalize_encounter_dict(enc: dict[str, Any]) -> dict[str, Any]:
    """Sanitize encounter payloads from DB or JSON (fixes stringified arrays and junk rows)."""
    out = dict(enc)
    for k in ("known_conditions", "medications", "allergies"):
        if k in out:
            out[k] = _to_list(out.get(k))
    sy = out.get("symptoms")
    if isinstance(sy, list):
        clean: list[dict[str, Any]] = []
        for item in sy:
            if not isinstance(item, dict):
                continue
            name = (item.get("symptom") or "").strip()
            if not name:
                continue
            clean.append(
                {
                    "symptom": name,
                    "onset": item.get("onset"),
                    "severity": item.get("severity"),
                    "duration": item.get("duration"),
                }
            )
        out["symptoms"] = clean
    return out


def purge_encounter_data(encounter_id: str) -> None:
    """Remove one encounter and dependent clinical rows (admin / governance)."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        for stmt, params in [
            ("DELETE FROM CLINICAL.ENCOUNTER_DIFF_DELTA WHERE encounter_id = %s", (encounter_id,)),
            ("DELETE FROM CLINICAL.ENCOUNTER_AUDIT WHERE encounter_id = %s", (encounter_id,)),
            ("DELETE FROM CLINICAL.ENCOUNTER_DIFFERENTIAL WHERE encounter_id = %s", (encounter_id,)),
            ("DELETE FROM CLINICAL.ENCOUNTER_QA WHERE encounter_id = %s", (encounter_id,)),
            ("DELETE FROM CLINICAL.ENCOUNTER_SYMPTOMS WHERE encounter_id = %s", (encounter_id,)),
            ("DELETE FROM CLINICAL.PATIENT_ENCOUNTER WHERE encounter_id = %s", (encounter_id,)),
        ]:
            cur.execute(stmt, params)
    finally:
        cur.close()
        conn.close()


def fetch_audit_rows_for_export(
    *,
    since_iso: str | None,
    until_iso: str | None,
    limit: int = 5000,
) -> list[dict[str, Any]]:
    """Return recent audit rows for CSV/JSON export (bounded)."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        where = []
        params: list[Any] = []
        if since_iso:
            where.append("created_at >= %s")
            params.append(since_iso)
        if until_iso:
            where.append("created_at <= %s")
            params.append(until_iso)
        wclause = (" WHERE " + " AND ".join(where)) if where else ""
        cur.execute(
            f"""
            SELECT audit_id, encounter_id, turn_no, action, provider, model_name, degraded_mode,
                   created_at, actor_user_id, kg_version, kg_build_id, pipeline_metrics
            FROM CLINICAL.ENCOUNTER_AUDIT
            {wclause}
            ORDER BY created_at DESC
            LIMIT %s
            """,
            tuple(params + [limit]),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "audit_id": r[0],
                "encounter_id": r[1],
                "turn_no": r[2],
                "action": r[3],
                "provider": r[4],
                "model_name": r[5],
                "degraded_mode": r[6],
                "created_at": str(r[7]) if r[7] is not None else None,
                "actor_user_id": r[8],
                "kg_version": r[9],
                "kg_build_id": r[10],
                "pipeline_metrics": r[11],
            }
        )
    return out


def fetch_audit_action_summary(
    *,
    since_iso: str | None,
    until_iso: str | None,
) -> list[dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        where = []
        params: list[Any] = []
        if since_iso:
            where.append("created_at >= %s")
            params.append(since_iso)
        if until_iso:
            where.append("created_at <= %s")
            params.append(until_iso)
        wclause = (" WHERE " + " AND ".join(where)) if where else ""
        cur.execute(
            f"""
            SELECT action, COUNT(*) AS n
            FROM CLINICAL.ENCOUNTER_AUDIT
            {wclause}
            GROUP BY action
            ORDER BY n DESC
            """,
            tuple(params),
        )
        return [{"action": a, "count": int(n)} for a, n in cur.fetchall()]
    finally:
        cur.close()
        conn.close()
