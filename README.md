# MedAssist.AI

A Smart Doctor's Assistant—helping doctors find rare diseases and avoid medical errors.

## Data Ingestion (Phase 1)

This repository includes a data ingestion pipeline that fetches from 6 free medical data sources and stores them locally with proper metadata.

### Data Sources

| Source | Content |
|--------|---------|
| PubMed | 39M medical article metadata/abstracts |
| PubMed Central | 6M full-text papers (open-access subset) |
| OpenFDA | Drug labels, adverse events, NDC directory |
| Orphanet | 10k rare diseases (symptoms, genes, classifications) |
| RxNorm | Drug names and classes |
| WHO | Treatment guidelines and documents |
| NCBI Bookshelf | StatPearls, clinical guidelines, pharmacology (titles + abstracts) |
| OpenStax | Free textbooks: Pharmacology, Anatomy & Physiology, Microbiology, Nursing (PDF + extracted text) |

### Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set EMAIL (required for NCBI). Optionally add NCBI_API_KEY, OPENFDA_API_KEY.
```

### Usage

Fetch from all sources (small sample):

```bash
python scripts/fetch_all.py
```

**Larger data with checkpoint (resume on timeout/rate limit):**

```bash
python scripts/fetch_all_checkpointed.py --reset   # full run with higher limits
# If it stops (timeout/rate limit), run again to resume:
python scripts/fetch_all_checkpointed.py
python scripts/cleanup_duplicate_raw.py            # keep best file per source
```

Fetch from a single source:

```bash
python scripts/fetch_source.py pubmed --term "acute porphyria" --max_records 200
python scripts/fetch_source.py openfda --endpoint label --max_records 1000
python scripts/fetch_source.py orphanet --dataset phenotypes
python scripts/fetch_source.py rxnorm --query "ibuprofen"
python scripts/fetch_source.py pmc --term "rare disease" --max_records 500
python scripts/fetch_source.py who --endpoint documents --limit 50
python scripts/fetch_source.py ncbi_bookshelf --term "pharmacology" --max_records 100
python scripts/fetch_source.py openstax --book pharmacology
# OpenStax books: pharmacology, anatomy-physiology-2e, biology-2e, microbiology,
# medical-surgical-nursing, fundamentals-nursing, clinical-nursing-skills
```

### Data Layout

```
data/
├── raw/           # Raw API responses by source
│   ├── pubmed/
│   ├── pmc/
│   ├── openfda/
│   ├── orphanet/
│   ├── rxnorm/
│   ├── who/
│   ├── ncbi_bookshelf/
│   └── openstax/
│       ├── raw/        # Original PDFs
│       └── extracted/  # Text by page
├── normalized/    # Symptom index, etc.
├── metadata/      # Fetch manifests
├── vectors/       # ChromaDB RAG index (Phase 2)
└── schema/        # JSON schemas
```

Each fetch produces a JSON file and a manifest with: `source`, `fetch_id`, `fetched_at`, `api_endpoint`, `query_params`, `record_count`, `checksum_sha256`, `status`.

## Phase 2: Symptom Queries & RAG

### Symptom → Disease Index (Orphanet)

Build an inverted index mapping symptoms to rare diseases:

```bash
python scripts/build_symptom_index.py
python scripts/query_symptoms.py "fever" "vomiting"
python scripts/query_symptoms.py "nausea" "headache" --any   # union instead of intersection
```

### RAG (Semantic Search)

Chunk and embed text from PubMed, PMC, OpenStax, and NCBI Bookshelf for semantic retrieval:

```bash
pip install sentence-transformers chromadb   # if not already installed
python scripts/build_rag_index.py            # first-time: downloads model, indexes chunks
python scripts/build_rag_index.py --max-chunks 20000   # index more
python scripts/query_rag.py "fever and vomiting differential diagnosis"
python scripts/query_rag.py "acute porphyria symptoms" -n 10
```

Index is stored in `data/vectors/` (ChromaDB). Uses `all-MiniLM-L6-v2` for embeddings.

## Snowflake (Phase 2 - optional)

Create warehouse, database, and tables, then load data:

```bash
# 1. Add SNOWFLAKE_PASSWORD to .env (copy from .env.example)
# 2. Create objects (warehouse, database, schemas, tables)
python scripts/run_snowflake_setup.py

# Or run the SQL manually in Snowflake Worksheets:
#   scripts/snowflake_setup.sql

# 3. Load data from local storage
python scripts/load_to_snowflake.py --all
python scripts/load_to_snowflake.py --symptoms --pubmed   # selective load
```

**Objects created:**
- Warehouse: `MEDASSIST_WH` (X-Small, auto-suspend 5 min)
- Database: `MEDASSIST_DB`
- Schemas: `RAW` (manifests, articles), `NORMALIZED` (symptom index), `VECTORS` (RAG chunks)

**If MFA blocks Python connector:** Run `scripts/snowflake_setup.sql` manually in Snowflake Worksheets (you're already logged in). Then use `load_to_snowflake.py` when programmatic access is available.
