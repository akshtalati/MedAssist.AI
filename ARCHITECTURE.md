# MedAssist.AI Architecture

This document has two views:

1. **Data platform** — ingestion, local staging, Snowflake, indexing (original diagram, refined).
2. **Application runtime** — FastAPI + Streamlit doctor workspace, evidence tiers, and KG differential (current product path).

---

## 1) Data platform (Snowflake-first)

```mermaid
flowchart TB
    subgraph EXT[External Medical Data Sources]
      S1[PubMed E-Utilities]
      S2[PubMed Central]
      S3[OpenFDA API]
      S4[Orphanet XML]
      S5[RxNorm RxNav]
      S6[WHO API]
      S7[NCBI Bookshelf]
      S8[OpenStax PDFs]
    end

    subgraph ING[Ingestion Layer]
      F1[Source Fetchers src/fetchers]
      W[DataWriter + Manifests]
    end

    subgraph STG[Local Staging]
      R[(data/raw)]
      M[(data/metadata)]
      N[(data/normalized)]
      C[(data/vectors ChromaDB)]
    end

    subgraph IDX[Indexing]
      I1[build_symptom_index.py]
      I2[build_rag_index.py]
    end

    subgraph ORCH[Orchestration]
      A1[Airflow medassist_ingestion]
      A2[CLI fetch scripts]
    end

    subgraph SF[Snowflake]
      WH[Warehouse]
      DB[(MEDASSIST_DB)]
      RAW_S[RAW PUBMED PMC BOOKSHELF OPENSTAX]
      NOR_S[NORMALIZED SYMPTOM_DISEASE_MAP DISEASES]
      VEC_S[VECTORS RAG_CHUNKS]
      CLIN[CLINICAL encounters audit]
      KG[KNOWLEDGE_GRAPH nodes edges meta]
    end

    subgraph CON[Consumption]
      Q1[query_symptoms.py]
      Q2[query_rag.py]
      Q3[SQL BI analytics]
    end

    S1 --> F1
    S2 --> F1
    S3 --> F1
    S4 --> F1
    S5 --> F1
    S6 --> F1
    S7 --> F1
    S8 --> F1
    F1 --> W
    W --> R
    W --> M
    A1 --> A2
    A2 --> F1
    R --> I1 --> N
    R --> I2 --> C
    R --> L1[load_to_snowflake.py]
    N --> L1
    C --> L1
    L1 --> WH --> DB
    DB --> RAW_S
    DB --> NOR_S
    DB --> VEC_S
    DB --> CLIN
    DB --> KG
    N --> Q1
    C --> Q2
    RAW_S --> Q3
    NOR_S --> Q3
    VEC_S --> Q3
```

### Notes

- Snowflake setup: `scripts/snowflake_setup.sql`, loads via `scripts/load_to_snowflake.py`.
- **Clinical + KG:** `scripts/snowflake_setup_clinical_kg.sql` and `src/clinical_workflow.py` (`ensure_clinical_tables`, `seed_graph_from_symptom_map`).
- Airflow DAG currently focuses on fetch → cleanup → symptom index; see `dags/README.md` for extensions.

---

## 2) Application runtime (unified doctor workspace)

This is what runs in demo/production for clinicians: **Streamlit UI** + **FastAPI** + **Snowflake** + optional **Vertex/Cortex**.

```mermaid
flowchart TB
    subgraph user [Clinician]
      BR[Browser]
    end

    subgraph ui [Streamlit streamlit_app.py]
      T1[Differential + Evidence tab]
      T2[Doctor Q and A tab]
    end

    subgraph api [FastAPI api/main.py]
      E1[POST /encounters/start]
      E2[POST assess-fast]
      E3[POST next-question / answer]
      A1[POST /ask]
    end

    subgraph diff [Differential engine]
      RK[rank_diseases_from_graph]
      FB[SYMPTOM_DISEASE_MAP + context tokens]
    end

    subgraph ev [Evidence tiers]
      V1[ChromaDB RAG]
      V2[Snowflake ILIKE on RAW literature]
      V3[Live NCBI E-utilities PubMed]
      V4[Live Europe PMC REST]
      V5[Constrained web last resort]
      VQ[evidence quality scoring]
    end

    subgraph llm [Generation]
      CRX[Snowflake Cortex]
      GEM[Vertex Gemini]
    end

    subgraph sn [Snowflake]
      ENC[CLINICAL encounters]
      KGG[KNOWLEDGE_GRAPH]
      RAWL[RAW literature tables]
    end

    BR --> T1
    BR --> T2
    T1 --> E1
    T1 --> E2
    T1 --> E3
    T2 --> A1
    E1 --> ENC
    E2 --> RK
    E2 --> FB
    RK --> KGG
    FB --> ENC
    E2 --> V1
    E2 --> V2
    E2 --> V3
    E2 --> V4
    E2 --> V5
    V1 --> VQ
    V2 --> VQ
    V3 --> VQ
    V4 --> VQ
    V5 --> VQ
    V2 --> RAWL
    A1 --> V1
    A1 --> V2
    A1 --> V3
    A1 --> V4
    A1 --> V5
    A1 --> VQ
    VQ -->|sufficient| CRX
    VQ -->|sufficient| GEM
    VQ -->|insufficient| RESP[Insufficient response and suggested follow-ups]
```

### Flow summary

1. **Intake** creates rows in **`CLINICAL`** and links symptoms.
2. **`assess-fast`** ranks diseases via **KG first**, merges fallbacks, attaches **`evidence_summary`** / **`evidence_sources`** / **`fallback_mode`** after journal-first + live medical APIs.
3. **Follow-up** is **chat-style** in Streamlit; each **`answer`** recomputes the differential and **`next-question`** advances the dialogue (policy in `followup_policy.py`).
4. **`/ask`** uses the same evidence stack; if **`evidence_quality`** is not sufficient, the API returns **no LLM hallucination path**—only structured insufficiency + suggested questions.

---

## Repository cross-reference

| Doc | Topic |
|-----|--------|
| [README.md](README.md) | Features, env vars, quick start, API table |
| [dbt/README.md](dbt/README.md) | dbt on Snowflake |
| [dags/README.md](dags/README.md) | Airflow |
| [data/schema/README.md](data/schema/README.md) | JSON schemas |
