"""Clinical workflow storage and retrieval helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any
from uuid import uuid4

from .snowflake_client import get_connection


@dataclass
class EncounterInput:
    age: int | None
    sex: str | None
    known_conditions: list[str]
    medications: list[str]
    allergies: list[str]
    history_summary: str
    symptoms: list[dict[str, str]]


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
            "ALTER TABLE CLINICAL.ENCOUNTER_QA ADD CONSTRAINT IF NOT EXISTS uq_encounter_turn UNIQUE (encounter_id, turn_no)",
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
              (encounter_id, age, sex, known_conditions, medications, allergies, history_summary)
            SELECT
              %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), %s
            """,
            (
                encounter_id,
                inp.age,
                inp.sex,
                _json_array(inp.known_conditions),
                _json_array(inp.medications),
                _json_array(inp.allergies),
                inp.history_summary,
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
              (encounter_id, age, sex, known_conditions, medications, allergies, history_summary)
            SELECT
              %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), %s
            """,
            (
                encounter_id,
                inp.age,
                inp.sex,
                _json_array(inp.known_conditions),
                _json_array(inp.medications),
                _json_array(inp.allergies),
                inp.history_summary,
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
                  (encounter_id, iteration, rank_no, disease_name, disease_code, kg_version, kg_build_id, source_snapshot_ts, score, rationale, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
) -> None:
    """Single transaction for answer + differential update."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("BEGIN")
        cur.execute(
            """
            UPDATE CLINICAL.ENCOUNTER_QA
            SET answer_text = %s, answered_at = CURRENT_TIMESTAMP()
            WHERE encounter_id = %s AND turn_no = %s
            """,
            (answer_text, encounter_id, turn_no),
        )
        cur.execute(
            "DELETE FROM CLINICAL.ENCOUNTER_DIFFERENTIAL WHERE encounter_id = %s AND iteration = %s",
            (encounter_id, turn_no + 1),
        )
        for rank_no, r in enumerate(candidates, start=1):
            cur.execute(
                """
                INSERT INTO CLINICAL.ENCOUNTER_DIFFERENTIAL
                  (encounter_id, iteration, rank_no, disease_name, disease_code, kg_version, kg_build_id, source_snapshot_ts, score, rationale, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
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
                ),
            )
        cur.execute("COMMIT")
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
            SELECT age, sex, known_conditions, medications, allergies, history_summary
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
            SELECT iteration, rank_no, disease_name, disease_code, score, rationale, source
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
            }
            for i, r, dn, dc, sc, rs, src in cur.fetchall()
        ]
        return {
            "encounter_id": encounter_id,
            "age": row[0],
            "sex": row[1],
            "known_conditions": _to_list(row[2]),
            "medications": _to_list(row[3]),
            "allergies": _to_list(row[4]),
            "history_summary": row[5] or "",
            "symptoms": symptoms,
            "qa_history": qa,
            "differential": differential,
        }
    finally:
        cur.close()
        conn.close()


def rank_diseases_from_graph(encounter_id: str, limit: int = 12) -> list[dict[str, Any]]:
    """
    Rank candidate diseases using KG edges + encounter symptoms.
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
            disease_scores AS (
              SELECT
                n.node_label AS disease_name,
                COALESCE(n.attributes:orpha_code::STRING, '') AS disease_code,
                COUNT(*)::FLOAT AS matched_symptoms,
                COUNT(*)::FLOAT / NULLIF((
                  SELECT COUNT(*) FROM symptom_nodes
                ), 0) AS coverage
              FROM symptom_nodes s
              JOIN KNOWLEDGE_GRAPH.KG_EDGES e
                ON e.to_node_id = s.sym_node
               AND e.edge_type = 'HAS_SYMPTOM'
              JOIN KNOWLEDGE_GRAPH.KG_NODES n
                ON n.node_id = e.from_node_id
               AND n.node_type = 'DISEASE'
              GROUP BY n.node_label, n.attributes:orpha_code::STRING
            )
            SELECT disease_name, disease_code, matched_symptoms, coverage
            FROM disease_scores
            ORDER BY matched_symptoms DESC, coverage DESC, disease_name
            LIMIT %s
            """,
            (encounter_id, limit),
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    out: list[dict[str, Any]] = []
    for disease_name, disease_code, matched, coverage in rows or []:
        out.append(
            {
                "disease_name": disease_name,
                "disease_code": disease_code,
                "score": float((matched or 0) + (coverage or 0)),
                "rationale": f"Matched symptoms: {int(matched or 0)}; coverage={float(coverage or 0):.2f}",
                "source": "knowledge_graph",
            }
        )
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
              COUNT(*)::FLOAT AS matched
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

    return [
        {
            "disease_name": dn,
            "disease_code": dc,
            "score": float(matched or 0.0),
            "rationale": f"Symptom-map matches: {int(matched or 0)}",
            "source": "symptom_map",
        }
        for dn, dc, matched in (rows or [])
    ]


def get_latest_kg_build_meta() -> tuple[str, str, str | None]:
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
            return "v1", "seed_symptom_map", None
        return str(row[0]), str(row[1]), row[2]
    except Exception:
        return "v1", "seed_symptom_map", None
    finally:
        cur.close()
        conn.close()


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
) -> None:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            INSERT INTO CLINICAL.ENCOUNTER_AUDIT
              (encounter_id, turn_no, action, provider, model_name, degraded_mode, context_hash, output_hash, kg_version, kg_build_id, error_flags)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s))
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
            ),
        )
    finally:
        cur.close()
        conn.close()


def save_diff_delta(encounter_id: str, turn_no: int, previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> None:
    prev_names = [str(x.get("disease_name")) for x in previous]
    cur_names = [str(x.get("disease_name")) for x in current]
    added = [x for x in cur_names if x not in prev_names]
    removed = [x for x in prev_names if x not in cur_names]
    rank_changes: dict[str, dict[str, int]] = {}
    for name in set(prev_names).intersection(cur_names):
        rank_changes[name] = {"from": prev_names.index(name) + 1, "to": cur_names.index(name) + 1}
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
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    try:
        return [str(v) for v in list(value)]
    except Exception:
        return [str(value)]
