-- MedAssist.AI Snowflake Setup
-- Run as TRAINING_ROLE (or another role with CREATE privileges)
-- Connection: account=SFEDU02-PGB87192, user=CRICKET

-- ============ WAREHOUSE ============
CREATE OR REPLACE WAREHOUSE MEDASSIST_WH
  WITH
  WAREHOUSE_SIZE = 'X-SMALL'
  AUTO_SUSPEND = 300
  AUTO_RESUME = TRUE
  INITIALLY_SUSPENDED = TRUE
  COMMENT = 'MedAssist.AI compute - symptom index, RAG, analytics';

-- ============ DATABASE ============
CREATE OR REPLACE DATABASE MEDASSIST_DB
  COMMENT = 'MedAssist.AI medical data';

-- ============ SCHEMAS ============
CREATE OR REPLACE SCHEMA MEDASSIST_DB.RAW
  COMMENT = 'Raw fetch manifests and API response metadata';

CREATE OR REPLACE SCHEMA MEDASSIST_DB.NORMALIZED
  COMMENT = 'Processed indices: symptom→disease, disease catalog';

CREATE OR REPLACE SCHEMA MEDASSIST_DB.VECTORS
  COMMENT = 'RAG chunks and embeddings for semantic search';

-- Use the warehouse
USE WAREHOUSE MEDASSIST_WH;
USE DATABASE MEDASSIST_DB;

-- ============ RAW TABLES ============

CREATE OR REPLACE TABLE RAW.FETCH_MANIFESTS (
  source VARCHAR(50),
  fetch_id VARCHAR(100),
  fetched_at TIMESTAMP_NTZ,
  api_endpoint VARCHAR(500),
  query_params VARIANT,
  record_count INTEGER,
  total_available INTEGER,
  file_path VARCHAR(500),
  checksum_sha256 VARCHAR(64),
  status VARCHAR(20),
  error VARCHAR(1000),
  _loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
  PRIMARY KEY (source, fetch_id)
);

CREATE OR REPLACE TABLE RAW.PUBMED_ARTICLES (
  pmid INTEGER,
  title VARCHAR(1000),
  abstract TEXT,
  journal VARCHAR(200),
  pub_date VARCHAR(50),
  source_file VARCHAR(200),
  _loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE OR REPLACE TABLE RAW.PMC_ARTICLES (
  pmcid VARCHAR(50),
  title VARCHAR(1000),
  abstract TEXT,
  journal VARCHAR(200),
  pub_date VARCHAR(50),
  source_file VARCHAR(200),
  _loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ============ NORMALIZED TABLES ============

CREATE OR REPLACE TABLE NORMALIZED.SYMPTOM_DISEASE_MAP (
  symptom VARCHAR(500),
  orpha_code VARCHAR(20),
  disease_name VARCHAR(500),
  frequency VARCHAR(100),
  hpo_id VARCHAR(50),
  _loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE OR REPLACE TABLE NORMALIZED.DISEASES (
  orpha_code VARCHAR(20) PRIMARY KEY,
  disease_name VARCHAR(500),
  _loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

-- ============ VECTORS (RAG chunks with embeddings) ============
-- VECTOR(FLOAT, 384) for all-MiniLM-L6-v2. GA since May 2024.
-- Fallback: use VARIANT if VECTOR not available in your region.

CREATE OR REPLACE TABLE VECTORS.RAG_CHUNKS (
  chunk_id VARCHAR(100) PRIMARY KEY,
  source VARCHAR(50),
  document_text TEXT,
  metadata VARIANT,
  embedding VECTOR(FLOAT, 384),
  _loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);
