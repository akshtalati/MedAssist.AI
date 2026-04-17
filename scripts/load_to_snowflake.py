#!/usr/bin/env python3
"""
Load MedAssist.AI data into Snowflake.

Prerequisites:
  - Run snowflake_setup.sql to create warehouse, database, and tables.
  - If you see "NORMALIZED.SYMPTOM_DISEASE_MAP does not exist", run scripts/ensure_normalized_schema.sql.
  - Set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD in .env.

Usage:
  python scripts/load_to_snowflake.py --all              # Load all sources
  python scripts/load_to_snowflake.py --all --skip-loaded # Load only steps whose tables are empty (skip rest)
  python scripts/load_to_snowflake.py --pubmed --pmc      # Load only PubMed and PMC

Uses batch inserts (executemany) where safe; single-row inserts for large text/JSON to avoid Snowflake 252001.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_data_paths, PROJECT_ROOT
from src.snowflake_client import get_connection

# Batch size for executemany(); larger = fewer round-trips but more memory per batch
BATCH_SIZE = 1000
# Smaller batch for tables with large text (abstract/title) to avoid Snowflake 252001 rewrite limit
BATCH_SIZE_LARGE_TEXT = 25

# For --skip-loaded: (step_key, list of tables to check; if any has rows, step is considered "loaded")
SKIP_LOADED_CHECK = [
    ("manifests", ["RAW.FETCH_MANIFESTS"]),
    ("symptoms", ["NORMALIZED.SYMPTOM_DISEASE_MAP"]),
    ("pubmed", ["RAW.PUBMED_ARTICLES"]),
    ("pmc", ["RAW.PMC_ARTICLES"]),
    ("openfda", ["RAW.OPENFDA_LABELS", "RAW.OPENFDA_EVENTS", "RAW.OPENFDA_NDC"]),
    ("rxnorm", ["RAW.RXNORM_DRUGS"]),
    ("who", ["RAW.WHO_DOCUMENTS"]),
    ("ncbi_bookshelf", ["RAW.NCBI_BOOKSHELF"]),
    ("openstax", ["RAW.OPENSTAX_BOOKS"]),
    ("orphanet", ["RAW.ORPHANET_DISEASES", "RAW.ORPHANET_PHENOTYPES", "RAW.ORPHANET_GENES"]),
    ("orphanet_web", ["RAW.ORPHANET_WEB_PAGES"]),
    ("rag", ["VECTORS.RAG_CHUNKS"]),
]

TABLE_MIN_ROWS = {
    "RAW.FETCH_MANIFESTS": 1,
    "NORMALIZED.SYMPTOM_DISEASE_MAP": 1000,
    "RAW.PUBMED_ARTICLES": 1000,
    "RAW.PMC_ARTICLES": 500,
    "RAW.OPENFDA_LABELS": 100,
    "RAW.OPENFDA_EVENTS": 100,
    "RAW.OPENFDA_NDC": 100,
    "RAW.RXNORM_DRUGS": 100,
    "RAW.WHO_DOCUMENTS": 50,
    "RAW.NCBI_BOOKSHELF": 100,
    "RAW.OPENSTAX_BOOKS": 100,
    "RAW.ORPHANET_DISEASES": 100,
    "RAW.ORPHANET_PHENOTYPES": 100,
    "RAW.ORPHANET_GENES": 100,
    "RAW.ORPHANET_WEB_PAGES": 10,
    "VECTORS.RAG_CHUNKS": 100,
}


def _table_row_count(cursor, table: str) -> int:
    """Return row count for a single table (e.g. RAW.PUBMED_ARTICLES). Returns 0 on error."""
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0] or 0
    except Exception:
        return 0


def _step_already_loaded(cursor, tables: list) -> tuple[bool, int, list[str]]:
    """Return True only when all tables satisfy minimum row thresholds."""
    total = 0
    incomplete = []
    for t in tables:
        cnt = _table_row_count(cursor, t)
        total += cnt
        if cnt < TABLE_MIN_ROWS.get(t, 1):
            incomplete.append(f"{t}={cnt}<{TABLE_MIN_ROWS.get(t, 1)}")
    return (not incomplete, total, incomplete)


def load_manifests(cursor) -> int:
    """Load fetch manifests from data/metadata/."""
    print(">>> Step 1/12: Loading manifests...", flush=True)
    meta_dir = PROJECT_ROOT / "data" / "metadata"
    if not meta_dir.exists():
        return 0
    count = 0
    for f in meta_dir.glob("*_manifest.json"):
        try:
            with open(f) as fp:
                m = json.load(fp)
            cursor.execute(
                """
                INSERT INTO RAW.FETCH_MANIFESTS
                (source, fetch_id, fetched_at, api_endpoint, query_params, record_count,
                 total_available, file_path, checksum_sha256, status, error)
                SELECT %s, %s, %s, %s, PARSE_JSON(%s), %s, %s, %s, %s, %s, %s
                WHERE NOT EXISTS (SELECT 1 FROM RAW.FETCH_MANIFESTS WHERE source=%s AND fetch_id=%s)
                """,
                (
                    m.get("source", ""),
                    m.get("fetch_id", ""),
                    m.get("fetched_at"),
                    m.get("api_endpoint", ""),
                    json.dumps(m.get("query_params", {})),
                    m.get("record_count", 0),
                    m.get("total_available"),
                    m.get("file_path", ""),
                    m.get("checksum_sha256"),
                    m.get("status", "success"),
                    m.get("error"),
                    m.get("source", ""),
                    m.get("fetch_id", ""),
                ),
            )
            count += 1
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
    return count


def load_symptom_index(cursor) -> int:
    """Load symptom→disease map from data/normalized/symptom_index.json."""
    print(">>> Step 2/12: Loading symptom index...", flush=True)
    path = PROJECT_ROOT / "data" / "normalized" / "symptom_index.json"
    if not path.exists():
        return 0
    with open(path) as f:
        data = json.load(f)
    stod = data.get("symptom_to_diseases", {})
    sql = """
        INSERT INTO NORMALIZED.SYMPTOM_DISEASE_MAP
        (symptom, orpha_code, disease_name, frequency, hpo_id)
        VALUES (%s, %s, %s, %s, %s)
    """
    batch = []
    count = 0
    for symptom, entries in stod.items():
        for entry in entries:
            if isinstance(entry, (list, tuple)):
                orpha = str(entry[0]) if len(entry) > 0 else ""
                name = str(entry[1]) if len(entry) > 1 else ""
                freq = str(entry[2]) if len(entry) > 2 else ""
                hpo = str(entry[3]) if len(entry) > 3 else ""
            else:
                orpha, name, freq, hpo = "", "", "", ""
            batch.append((symptom[:500], orpha[:20], name[:500], freq[:100], hpo[:50]))
            if len(batch) >= BATCH_SIZE:
                cursor.executemany(sql, batch)
                count += len(batch)
                print(f"  Symptom-disease rows: {count}...", flush=True)
                batch = []
    if batch:
        cursor.executemany(sql, batch)
        count += len(batch)
    return count


def load_pubmed(cursor) -> int:
    """Load PubMed articles from data/raw/pubmed/. Uses single-row inserts to avoid Snowflake 252001."""
    print(">>> Step 3/12: Loading PubMed articles...", flush=True)
    raw_dir = PROJECT_ROOT / "data" / "raw" / "pubmed"
    if not raw_dir.exists():
        return 0
    sql = """
        INSERT INTO RAW.PUBMED_ARTICLES (pmid, title, abstract, journal, pub_date, source_file)
        SELECT %s, %s, %s, %s, %s, %s
        WHERE NOT EXISTS (SELECT 1 FROM RAW.PUBMED_ARTICLES WHERE pmid=%s)
    """
    count = 0
    files = list(raw_dir.glob("*.json"))
    for fi, f in enumerate(files):
        if files:
            print(f"  PubMed file {fi + 1}/{len(files)}: {f.name}", flush=True)
        try:
            with open(f) as fp:
                data = json.load(fp)
            articles = data.get("data", {}).get("articles", [])
            for a in articles:
                pmid = a.get("pmid")
                if pmid is None:
                    continue
                row = (
                    pmid,
                    str(a.get("title", ""))[:1000],
                    str(a.get("abstract", ""))[:100000],
                    str(a.get("journal", ""))[:200],
                    str(a.get("pub_date", ""))[:50],
                    f.name,
                    pmid,
                )
                cursor.execute(sql, row)
                count += 1
                if count % 5000 == 0:
                    print(f"  PubMed rows: {count}...", flush=True)
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
    return count


def load_pmc(cursor) -> int:
    """Load PMC articles from data/raw/pmc/. Uses single-row inserts to avoid Snowflake 252001."""
    print(">>> Step 4/12: Loading PMC articles...", flush=True)
    raw_dir = PROJECT_ROOT / "data" / "raw" / "pmc"
    if not raw_dir.exists():
        return 0
    sql = """
        INSERT INTO RAW.PMC_ARTICLES (pmcid, title, abstract, journal, pub_date, source_file)
        SELECT %s, %s, %s, %s, %s, %s
        WHERE NOT EXISTS (SELECT 1 FROM RAW.PMC_ARTICLES WHERE pmcid=%s)
    """
    count = 0
    files = list(raw_dir.glob("*.json"))
    for fi, f in enumerate(files):
        if files:
            print(f"  PMC file {fi + 1}/{len(files)}: {f.name}", flush=True)
        try:
            with open(f) as fp:
                data = json.load(fp)
            articles = data.get("data", {}).get("articles", [])
            for a in articles:
                pmcid = a.get("pmcid")
                if not pmcid:
                    continue
                row = (
                    str(pmcid)[:50],
                    str(a.get("title", ""))[:1000],
                    str(a.get("abstract", ""))[:100000],
                    str(a.get("journal", ""))[:200],
                    str(a.get("pub_date", ""))[:50],
                    f.name,
                    str(pmcid)[:50],
                )
                cursor.execute(sql, row)
                count += 1
                if count % 5000 == 0:
                    print(f"  PMC rows: {count}...", flush=True)
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
    return count


def load_openfda(cursor) -> int:
    """Load OpenFDA data from data/raw/openfda/{endpoint}/. Uses single-row inserts to avoid Snowflake 252001."""
    print(">>> Step 5/12: Loading OpenFDA data...", flush=True)
    raw_dir = PROJECT_ROOT / "data" / "raw" / "openfda"
    if not raw_dir.exists():
        return 0

    sql_label = """
        INSERT INTO RAW.OPENFDA_LABELS
        (application_number, product_ndc, brand_name, generic_name, manufacturer_name,
         product_type, active_ingredient, inactive_ingredient, purpose,
         indications_and_usage, warnings, dosage_and_administration, full_data, source_file)
        SELECT %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s),
               PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), %s
        WHERE NOT EXISTS (
            SELECT 1 FROM RAW.OPENFDA_LABELS 
            WHERE application_number=%s AND product_ndc=%s
        )
    """
    sql_event = """
        INSERT INTO RAW.OPENFDA_EVENTS
        (safetyreportid, receivedate, serious, seriousnessdeath, seriousnesslifethreatening,
         seriousnesshospitalization, seriousnessdisabling, seriousnessother,
         patient, drug, reaction, full_data, source_file)
        SELECT %s, %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), PARSE_JSON(%s), %s
        WHERE NOT EXISTS (SELECT 1 FROM RAW.OPENFDA_EVENTS WHERE safetyreportid=%s)
    """
    sql_ndc = """
        INSERT INTO RAW.OPENFDA_NDC
        (product_ndc, product_type, proprietary_name, non_proprietary_name, labeler_name, full_data, source_file)
        SELECT %s, %s, %s, %s, %s, PARSE_JSON(%s), %s
        WHERE NOT EXISTS (SELECT 1 FROM RAW.OPENFDA_NDC WHERE product_ndc=%s)
    """
    count = 0
    for endpoint_dir in ["label", "event", "ndc"]:
        endpoint_path = raw_dir / endpoint_dir
        if not endpoint_path.exists():
            continue
        sql = sql_label if endpoint_dir == "label" else (sql_event if endpoint_dir == "event" else sql_ndc)
        files = list(endpoint_path.glob("*.json"))
        for file_idx, f in enumerate(files):
            if files:
                print(f"  OpenFDA {endpoint_dir}: file {file_idx + 1}/{len(files)} ({f.name})", flush=True)
            try:
                with open(f) as fp:
                    data = json.load(fp)
                payload = data.get("data", {})
                results = payload.get("results", [])
                for record in results:
                    try:
                        if endpoint_dir == "label":
                            app_num = record.get("application_number", "")
                            product_ndc = record.get("product_ndc", "")
                            if not app_num and not product_ndc:
                                continue
                            row = (
                                app_num[:50],
                                product_ndc[:50],
                                str(record.get("brand_name", [""])[0] if isinstance(record.get("brand_name"), list) else record.get("brand_name", ""))[:500],
                                str(record.get("generic_name", [""])[0] if isinstance(record.get("generic_name"), list) else record.get("generic_name", ""))[:500],
                                str(record.get("manufacturer_name", [""])[0] if isinstance(record.get("manufacturer_name"), list) else record.get("manufacturer_name", ""))[:500],
                                str(record.get("product_type", ""))[:100],
                                json.dumps(record.get("active_ingredient", [])),
                                json.dumps(record.get("inactive_ingredient", [])),
                                json.dumps(record.get("purpose", [])),
                                json.dumps(record.get("indications_and_usage", [])),
                                json.dumps(record.get("warnings", [])),
                                json.dumps(record.get("dosage_and_administration", [])),
                                json.dumps(record),
                                f.name,
                                app_num[:50],
                                product_ndc[:50],
                            )
                        elif endpoint_dir == "event":
                            safety_id = record.get("safetyreportid", "")
                            if not safety_id:
                                continue
                            row = (
                                str(safety_id)[:100],
                                str(record.get("receivedate", ""))[:50],
                                record.get("serious", 0),
                                record.get("seriousnessdeath", 0),
                                record.get("seriousnesslifethreatening", 0),
                                record.get("seriousnesshospitalization", 0),
                                record.get("seriousnessdisabling", 0),
                                record.get("seriousnessother", 0),
                                json.dumps(record.get("patient", {})),
                                json.dumps(record.get("drug", [])),
                                json.dumps(record.get("reaction", [])),
                                json.dumps(record),
                                f.name,
                                str(safety_id)[:100],
                            )
                        elif endpoint_dir == "ndc":
                            product_ndc = record.get("product_ndc", "")
                            if not product_ndc:
                                continue
                            row = (
                                product_ndc[:50],
                                str(record.get("product_type", ""))[:100],
                                str(record.get("proprietary_name", ""))[:500],
                                str(record.get("non_proprietary_name", ""))[:500],
                                str(record.get("labeler_name", ""))[:500],
                                json.dumps(record),
                                f.name,
                                product_ndc[:50],
                            )
                        else:
                            continue
                        cursor.execute(sql, row)
                        count += 1
                        if count % 2000 == 0:
                            print(f"  OpenFDA {endpoint_dir}: {count} records...", flush=True)
                    except Exception as e:
                        print(f"  Skip record in {f.name}: {e}")
            except Exception as e:
                print(f"  Skip {f.name}: {e}")
    return count


def load_rxnorm(cursor) -> int:
    """Load RxNorm data from data/raw/rxnorm/."""
    print(">>> Step 6/12: Loading RxNorm data...", flush=True)
    raw_dir = PROJECT_ROOT / "data" / "raw" / "rxnorm"
    if not raw_dir.exists():
        return 0
    sql = """
        INSERT INTO RAW.RXNORM_DRUGS (rxcui, name, tty, synonym, query_term, full_data, source_file)
        SELECT %s, %s, %s, %s, %s, PARSE_JSON(%s), %s
        WHERE NOT EXISTS (
            SELECT 1 FROM RAW.RXNORM_DRUGS 
            WHERE rxcui=%s AND name=%s AND query_term=%s
        )
    """
    batch = []
    count = 0
    for f in raw_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            payload = data.get("data", {})
            query = payload.get("query", "")
            drugs = payload.get("drugs", [])
            for drug in drugs:
                rxcui = str(drug.get("rxcui", ""))
                name = str(drug.get("name", drug.get("synonym", "")))
                if not rxcui and not name:
                    continue
                try:
                    batch.append((
                        rxcui[:50],
                        name[:500],
                        str(drug.get("tty", ""))[:50],
                        str(drug.get("synonym", ""))[:500],
                        query[:200],
                        json.dumps(drug),
                        f.name,
                        rxcui[:50],
                        name[:500],
                        query[:200],
                    ))
                    if len(batch) >= BATCH_SIZE:
                        cursor.executemany(sql, batch)
                        count += len(batch)
                        batch = []
                except Exception as e:
                    print(f"  Skip drug {rxcui}: {e}")
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
    if batch:
        cursor.executemany(sql, batch)
        count += len(batch)
    return count


def load_who(cursor) -> int:
    """Load WHO documents from data/raw/who/."""
    print(">>> Step 7/12: Loading WHO documents...", flush=True)
    raw_dir = PROJECT_ROOT / "data" / "raw" / "who"
    if not raw_dir.exists():
        return 0
    sql = """
        INSERT INTO RAW.WHO_DOCUMENTS
        (document_id, title, url, language, publication_date, document_type, full_data, source_file)
        SELECT %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s
        WHERE NOT EXISTS (SELECT 1 FROM RAW.WHO_DOCUMENTS WHERE document_id=%s)
    """
    batch = []
    count = 0
    for f in raw_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            payload = data.get("data", {})
            records = payload.get("records", [])
            if not records:
                records = payload.get("value", [])
            for record in records:
                if not isinstance(record, dict):
                    continue
                doc_id = str(record.get("id", record.get("document_id", "")))
                if not doc_id:
                    doc_id = str(record.get("title", ""))[:200]
                if not doc_id:
                    continue
                try:
                    batch.append((
                        doc_id[:200],
                        str(record.get("title", record.get("name", "")))[:1000],
                        str(record.get("url", record.get("link", "")))[:500],
                        str(record.get("language", "en"))[:10],
                        str(record.get("publication_date", record.get("date", "")))[:50],
                        str(record.get("document_type", record.get("type", "")))[:100],
                        json.dumps(record),
                        f.name,
                        doc_id[:200],
                    ))
                    if len(batch) >= BATCH_SIZE:
                        cursor.executemany(sql, batch)
                        count += len(batch)
                        batch = []
                except Exception as e:
                    print(f"  Skip document {doc_id}: {e}")
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
    if batch:
        cursor.executemany(sql, batch)
        count += len(batch)
    return count


def load_ncbi_bookshelf(cursor) -> int:
    """Load NCBI Bookshelf data from data/raw/ncbi_bookshelf/."""
    print(">>> Step 8/12: Loading NCBI Bookshelf...", flush=True)
    raw_dir = PROJECT_ROOT / "data" / "raw" / "ncbi_bookshelf"
    if not raw_dir.exists():
        return 0
    sql = """
        INSERT INTO RAW.NCBI_BOOKSHELF
        (uid, nbk_id, title, pubdate, abstract, url, query_term, full_data, source_file)
        SELECT %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s
        WHERE NOT EXISTS (SELECT 1 FROM RAW.NCBI_BOOKSHELF WHERE uid=%s)
    """
    batch = []
    count = 0
    for f in raw_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            payload = data.get("data", {})
            books = payload.get("books", [])
            query_term = payload.get("query_term", "")
            for book in books:
                uid = str(book.get("uid", ""))
                if not uid:
                    continue
                try:
                    batch.append((
                        uid[:50],
                        str(book.get("nbk_id", ""))[:50],
                        str(book.get("title", ""))[:1000],
                        str(book.get("pubdate", ""))[:50],
                        str(book.get("abstract", ""))[:100000],
                        str(book.get("url", ""))[:500],
                        query_term[:200],
                        json.dumps(book),
                        f.name,
                        uid[:50],
                    ))
                    if len(batch) >= BATCH_SIZE:
                        cursor.executemany(sql, batch)
                        count += len(batch)
                        batch = []
                except Exception as e:
                    print(f"  Skip book {uid}: {e}")
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
    if batch:
        cursor.executemany(sql, batch)
        count += len(batch)
    return count


def load_openstax(cursor) -> int:
    """Load OpenStax books from data/raw/openstax/extracted/. Single-row inserts to avoid Snowflake 252001."""
    print(">>> Step 9/12: Loading OpenStax books...", flush=True)
    extracted_dir = PROJECT_ROOT / "data" / "raw" / "openstax" / "extracted"
    if not extracted_dir.exists():
        return 0
    sql = """
        INSERT INTO RAW.OPENSTAX_BOOKS
        (book_slug, page_number, content, title, source_url, license, full_data, source_file)
        SELECT %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s
        WHERE NOT EXISTS (
            SELECT 1 FROM RAW.OPENSTAX_BOOKS 
            WHERE book_slug=%s AND page_number=%s
        )
    """
    count = 0
    for f in extracted_dir.glob("*.json"):
        try:
            with open(f) as fp:
                data = json.load(fp)
            payload = data.get("data", {})
            metadata = payload.get("metadata", {})
            chapters = payload.get("chapters", [])
            book_slug = metadata.get("book_slug", "")
            title = metadata.get("title", "")
            for chapter in chapters:
                page_num = chapter.get("page", 0)
                content = str(chapter.get("content", ""))
                if not book_slug and not title:
                    continue
                try:
                    row = (
                        book_slug[:100],
                        page_num,
                        content[:100000],
                        title[:500],
                        metadata.get("source_url", "")[:500],
                        metadata.get("license", "CC BY 4.0")[:50],
                        json.dumps(chapter),
                        f.name,
                        book_slug[:100],
                        page_num,
                    )
                    cursor.execute(sql, row)
                    count += 1
                    if count % 500 == 0:
                        print(f"  OpenStax pages: {count}...", flush=True)
                except Exception as e:
                    print(f"  Skip page {page_num} in {book_slug}: {e}")
        except Exception as e:
            print(f"  Skip {f.name}: {e}")
    return count


def load_orphanet(cursor) -> int:
    """Load Orphanet data from data/raw/orphanet/ (product1=genes, product4=phenotypes, product6=diseases)."""
    print(">>> Step 10/12: Loading Orphanet data...", flush=True)
    raw_dir = PROJECT_ROOT / "data" / "raw" / "orphanet"
    if not raw_dir.exists():
        return 0
    sql_phenotypes = """
        INSERT INTO RAW.ORPHANET_PHENOTYPES
        (phenotype_id, orpha_code, disease_name, hpo_id, phenotype_name, frequency, full_data, source_file)
        SELECT %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s
        WHERE NOT EXISTS (SELECT 1 FROM RAW.ORPHANET_PHENOTYPES WHERE phenotype_id=%s)
    """
    sql_diseases = """
        INSERT INTO RAW.ORPHANET_DISEASES
        (orpha_code, disease_name, definition, prevalence, inheritance, age_of_onset, full_data, source_file)
        SELECT %s, %s, %s, %s, %s, %s, PARSE_JSON(%s), %s
        WHERE NOT EXISTS (SELECT 1 FROM RAW.ORPHANET_DISEASES WHERE orpha_code=%s)
    """
    sql_genes = """
        INSERT INTO RAW.ORPHANET_GENES
        (gene_id, gene_symbol, gene_name, orpha_code, disease_name, full_data, source_file)
        SELECT %s, %s, %s, %s, %s, PARSE_JSON(%s), %s
        WHERE NOT EXISTS (SELECT 1 FROM RAW.ORPHANET_GENES WHERE gene_id=%s)
    """
    count = 0

    # Map dataset -> (subdirs to try, sql, handler)
    def _phenotype_rows(payload, fname):
        rows = []
        # Product4: HPODisorderSetStatusList.HPODisorderSetStatus[].Disorder
        hpo_list = payload.get("HPODisorderSetStatusList", {})
        statuses = hpo_list.get("HPODisorderSetStatus", [])
        if not isinstance(statuses, list):
            statuses = [statuses] if statuses else []
        for status in statuses:
            disorder = status.get("Disorder", {}) if isinstance(status, dict) else {}
            orpha = str(disorder.get("OrphaCode", disorder.get("OrphaNumber", "")))
            disease_name = str(disorder.get("Name", ""))
            assoc_list = disorder.get("HPODisorderAssociationList", {})
            if isinstance(assoc_list, dict):
                hpo_assoc = assoc_list.get("HPODisorderAssociation", [])
                if not isinstance(hpo_assoc, list):
                    hpo_assoc = [hpo_assoc] if hpo_assoc else []
                for idx, hpo in enumerate(hpo_assoc):
                    hpo_obj = hpo.get("HPO", {}) if isinstance(hpo, dict) else {}
                    freq_obj = hpo.get("HPOFrequency", {}) if isinstance(hpo, dict) else {}
                    hpo_id = str(hpo_obj.get("HPOId", ""))
                    phenotype_name = str(hpo_obj.get("HPOTerm", ""))
                    freq = str(freq_obj.get("Name", ""))
                    phenotype_id = f"{orpha}_{hpo_id}" if hpo_id else f"{orpha}_{idx}"
                    rows.append((phenotype_id[:50], orpha[:20], disease_name[:500], hpo_id[:50], phenotype_name[:500], freq[:100], json.dumps(hpo), fname, phenotype_id[:50]))
        # Legacy: DisorderList.Disorder[].HPODisorderAssociationList
        if not rows:
            disorder_list = payload.get("DisorderList", {})
            if isinstance(disorder_list, dict):
                disorders = disorder_list.get("Disorder", [])
                if not isinstance(disorders, list):
                    disorders = [disorders]
                for disorder in disorders:
                    orpha = str(disorder.get("OrphaNumber", disorder.get("OrphaCode", "")))
                    disease_name = str(disorder.get("Name", ""))
                    phenotype_list = disorder.get("HPODisorderAssociationList", {})
                    if isinstance(phenotype_list, dict):
                        hpo_list = phenotype_list.get("HPODisorderAssociation", [])
                        if not isinstance(hpo_list, list):
                            hpo_list = [hpo_list]
                        for idx, hpo in enumerate(hpo_list):
                            hpo_id = str(hpo.get("HPO", {}).get("HPOId", ""))
                            phenotype_name = str(hpo.get("HPO", {}).get("HPOTerm", ""))
                            freq = str(hpo.get("HPOFrequency", {}).get("Name", ""))
                            phenotype_id = f"{orpha}_{hpo_id}" if hpo_id else f"{orpha}_{idx}"
                            rows.append((phenotype_id[:50], orpha[:20], disease_name[:500], hpo_id[:50], phenotype_name[:500], freq[:100], json.dumps(hpo), fname, phenotype_id[:50]))
        return rows

    def _disease_rows(payload, fname):
        rows = []
        disorder_list = payload.get("DisorderList", {})
        if not isinstance(disorder_list, dict):
            return rows
        disorders = disorder_list.get("Disorder", [])
        if not isinstance(disorders, list):
            disorders = [disorders]
        for disorder in disorders:
            orpha = str(disorder.get("OrphaNumber", disorder.get("OrphaCode", "")))
            if not orpha:
                continue
            prev = disorder.get("PrevalenceList", {}) or {}
            prev = prev.get("Prevalence", {}) if isinstance(prev, dict) else {}
            prev = prev.get("PrevalenceClass", {}) if isinstance(prev, dict) else {}
            prev_name = prev.get("Name", "") if isinstance(prev, dict) else ""
            rows.append((
                orpha[:20],
                str(disorder.get("Name", ""))[:500],
                str(disorder.get("Definition", ""))[:10000],
                str(prev_name)[:200],
                str((disorder.get("DisorderType") or {}).get("Name", ""))[:200],
                str((disorder.get("AverageAgeOfOnset") or {}).get("Name", ""))[:200],
                json.dumps(disorder),
                fname,
                orpha[:20],
            ))
        return rows

    def _gene_rows(payload, fname):
        rows = []
        disorder_list = payload.get("DisorderList", {})
        if not isinstance(disorder_list, dict):
            return rows
        disorders = disorder_list.get("Disorder", [])
        if not isinstance(disorders, list):
            disorders = [disorders]
        for disorder in disorders:
            orpha = str(disorder.get("OrphaNumber", disorder.get("OrphaCode", "")))
            disease_name = str(disorder.get("Name", ""))
            gene_list = disorder.get("DisorderGeneList", {})
            if not isinstance(gene_list, dict):
                continue
            genes = gene_list.get("DisorderGeneAssociation", [])
            if not isinstance(genes, list):
                genes = [genes]
            for gene_assoc in genes:
                gene = gene_assoc.get("Gene", {}) if isinstance(gene_assoc, dict) else {}
                gene_id = str(gene.get("Symbol", ""))
                if not gene_id:
                    continue
                rows.append((
                    gene_id[:50],
                    gene_id[:100],
                    str(gene.get("Name", ""))[:500],
                    orpha[:20],
                    disease_name[:500],
                    json.dumps(gene_assoc),
                    fname,
                    gene_id[:50],
                ))
        return rows

    # Subdirs to try per dataset: legacy name first, then product dir
    dataset_config = [
        ("phenotypes", ["phenotypes", "product4"], sql_phenotypes, _phenotype_rows),
        ("diseases", ["diseases", "product6"], sql_diseases, _disease_rows),
        ("genes", ["genes", "product1"], sql_genes, _gene_rows),
    ]
    for dataset_name, subdirs, sql, row_fn in dataset_config:
        for subdir in subdirs:
            dataset_path = raw_dir / subdir
            if not dataset_path.exists():
                continue
            print(f"  Orphanet: loading {dataset_name} from {subdir}...", flush=True)
            batch = []
            for f in sorted(dataset_path.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    with open(f) as fp:
                        data = json.load(fp)
                    payload = data.get("data", data)
                    if not isinstance(payload, dict):
                        continue
                    for row in row_fn(payload, f.name):
                        batch.append(row)
                        if len(batch) >= BATCH_SIZE:
                            cursor.executemany(sql, batch)
                            count += len(batch)
                            print(f"  Orphanet {dataset_name}: {count}...", flush=True)
                            batch = []
                except Exception as e:
                    print(f"  Skip {f.name}: {e}")
            if batch:
                cursor.executemany(sql, batch)
                count += len(batch)
            break  # one subdir per dataset is enough
    return count


def load_orphanet_web(cursor) -> int:
    """Load Orpha.net web crawl from data/raw/orphanet/web/ (.md + .metadata.json)."""
    print(">>> Step 10b/12: Loading Orphanet web pages...", flush=True)
    web_dir = PROJECT_ROOT / "data" / "raw" / "orphanet" / "web"
    if not web_dir.exists():
        return 0
    sql = """
        INSERT INTO RAW.ORPHANET_WEB_PAGES
        (orpha_code, url, title, content, fetched_at, source_file)
        SELECT %s, %s, %s, %s, %s, %s
        WHERE NOT EXISTS (SELECT 1 FROM RAW.ORPHANET_WEB_PAGES WHERE orpha_code=%s)
    """
    count = 0
    # Iterate .md files; pair with .metadata.json when present
    for md_path in sorted(web_dir.glob("*.md")):
        if md_path.name.startswith("."):
            continue
        meta_path = web_dir / (md_path.stem + ".metadata.json")
        try:
            content = md_path.read_text(encoding="utf-8")
            url = ""
            orpha_code = md_path.stem
            fetched_at = ""
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                url = str(meta.get("url", ""))[:500]
                orpha_code = str(meta.get("orpha_code", orpha_code))[:20]
                fetched_at = str(meta.get("fetched_at", ""))[:50]
            # Optional: try .json for title (crawl --format json)
            json_path = web_dir / (md_path.stem + ".json")
            title = ""
            if json_path.exists():
                try:
                    doc = json.loads(json_path.read_text(encoding="utf-8"))
                    title = str(doc.get("title", ""))[:1000]
                except Exception:
                    pass
            row = (orpha_code[:20], url, title[:1000], content[:1000000], fetched_at, md_path.name, orpha_code[:20])
            cursor.execute(sql, row)
            count += 1
            if count % 500 == 0:
                print(f"  Orphanet web: {count} pages...", flush=True)
        except Exception as e:
            print(f"  Skip {md_path.name}: {e}")
    return count


def load_rag_chunks(cursor) -> int:
    """Load RAG chunks from ChromaDB into Snowflake VECTORS.RAG_CHUNKS."""
    print(">>> Step 11/12: Loading RAG chunks...", flush=True)
    try:
        import chromadb
    except ImportError:
        print("  chromadb not installed, skipping RAG chunks")
        return 0

    persist_dir = PROJECT_ROOT / "data" / "vectors"
    if not persist_dir.exists():
        return 0
    try:
        client = chromadb.PersistentClient(path=str(persist_dir))
        coll = client.get_collection("medassist_rag")
    except Exception:
        return 0

    # Fetch all with embeddings
    data = coll.get(include=["documents", "metadatas", "embeddings"])
    if not data or not data["ids"]:
        return 0

    sql = """
        INSERT INTO VECTORS.RAG_CHUNKS (chunk_id, source, document_text, metadata, embedding)
        SELECT %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s)::VECTOR(FLOAT, 384)
        WHERE NOT EXISTS (SELECT 1 FROM VECTORS.RAG_CHUNKS WHERE chunk_id=%s)
    """
    count = 0
    for i, cid in enumerate(data["ids"]):
        # Access with bounds checks; data["documents"]/["metadatas"]/["embeddings"] may be lists or arrays
        docs = data.get("documents") or []
        metas = data.get("metadatas") or []
        embs = data.get("embeddings")

        doc = docs[i] if i < len(docs) else ""
        meta = metas[i] if i < len(metas) else {}
        emb = embs[i] if embs is not None and i < len(embs) else []
        if len(emb) != 384:
            continue  # all-MiniLM-L6-v2 produces 384-dim vectors
        try:
            meta_json = json.dumps(meta) if meta else "{}"
            emb_str = "[" + ",".join(str(float(x)) for x in emb) + "]"
            row = (cid, meta.get("source", "unknown"), doc[:100000], meta_json, emb_str, cid)
            # Single-row insert to avoid Snowflake 252001 multi-row rewrite issues
            cursor.execute(sql, row)
            count += 1
            if count % 200 == 0:
                print(f"  RAG chunks: {count}...", flush=True)
        except Exception as e:
            print(f"  Skip chunk {cid}: {e}")
    return count


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load MedAssist data into Snowflake")
    parser.add_argument("--manifests", action="store_true", help="Load fetch manifests")
    parser.add_argument("--symptoms", action="store_true", help="Load symptom index")
    parser.add_argument("--pubmed", action="store_true", help="Load PubMed articles")
    parser.add_argument("--pmc", action="store_true", help="Load PMC articles")
    parser.add_argument("--openfda", action="store_true", help="Load OpenFDA data")
    parser.add_argument("--rxnorm", action="store_true", help="Load RxNorm data")
    parser.add_argument("--who", action="store_true", help="Load WHO documents")
    parser.add_argument("--ncbi-bookshelf", action="store_true", dest="ncbi_bookshelf", help="Load NCBI Bookshelf data")
    parser.add_argument("--openstax", action="store_true", help="Load OpenStax books")
    parser.add_argument("--orphanet", action="store_true", help="Load Orphanet data")
    parser.add_argument("--orphanet-web", action="store_true", dest="orphanet_web", help="Load Orphanet web crawl (orpha.net pages)")
    parser.add_argument("--rag", action="store_true", help="Load RAG chunks from ChromaDB")
    parser.add_argument("--all", action="store_true", help="Load everything")
    parser.add_argument("--skip-loaded", action="store_true", help="Skip any step whose table(s) meet minimum row thresholds")
    parser.add_argument("--strict", action="store_true", help="Fail if post-load table coverage is below minimum thresholds")
    args = parser.parse_args()

    load_all = args.all or not any([
        args.manifests, args.symptoms, args.pubmed, args.pmc, args.openfda,
        args.rxnorm, args.who, args.ncbi_bookshelf, args.openstax, args.orphanet, args.orphanet_web, args.rag
    ])

    conn = get_connection()
    cursor = conn.cursor()

    def should_skip(step_key: str) -> bool:
        if not args.skip_loaded:
            return False
        for key, tables in SKIP_LOADED_CHECK:
            if key == step_key:
                loaded, total, incomplete = _step_already_loaded(cursor, tables)
                if loaded:
                    print(f"  (Step already loaded: {total:,} rows and thresholds met, skipping)", flush=True)
                elif total > 0 and incomplete:
                    print(f"  (Step partially loaded; re-running because {', '.join(incomplete)})", flush=True)
                return loaded
        return False

    try:
        warehouse = os.environ.get("SNOWFLAKE_WAREHOUSE", "MEDASSIST_WH")
        database = os.environ.get("SNOWFLAKE_DATABASE", "MEDASSIST_DB")
        cursor.execute(f"USE WAREHOUSE {warehouse}")
        cursor.execute(f"USE DATABASE {database}")

        total_start = time.time()
        if load_all:
            print(f"Started at {time.strftime('%H:%M:%S')} — loading all sources (--skip-loaded=%s)." % args.skip_loaded, flush=True)

        if load_all or args.manifests:
            if should_skip("manifests"):
                pass
            else:
                t0 = time.time()
                n = load_manifests(cursor)
                print(f"Loaded {n} manifests ({time.time() - t0:.1f}s)", flush=True)
        if load_all or args.symptoms:
            if should_skip("symptoms"):
                pass
            else:
                t0 = time.time()
                n = load_symptom_index(cursor)
                print(f"Loaded {n} symptom-disease rows ({time.time() - t0:.1f}s)", flush=True)
        if load_all or args.pubmed:
            if should_skip("pubmed"):
                pass
            else:
                t0 = time.time()
                n = load_pubmed(cursor)
                print(f"Loaded {n} PubMed articles ({time.time() - t0:.1f}s)", flush=True)
        if load_all or args.pmc:
            if should_skip("pmc"):
                pass
            else:
                t0 = time.time()
                n = load_pmc(cursor)
                print(f"Loaded {n} PMC articles ({time.time() - t0:.1f}s)", flush=True)
        if load_all or args.openfda:
            if should_skip("openfda"):
                pass
            else:
                t0 = time.time()
                n = load_openfda(cursor)
                print(f"Loaded {n} OpenFDA records ({time.time() - t0:.1f}s)", flush=True)
        if load_all or args.rxnorm:
            if should_skip("rxnorm"):
                pass
            else:
                t0 = time.time()
                n = load_rxnorm(cursor)
                print(f"Loaded {n} RxNorm drugs ({time.time() - t0:.1f}s)", flush=True)
        if load_all or args.who:
            if should_skip("who"):
                pass
            else:
                t0 = time.time()
                n = load_who(cursor)
                print(f"Loaded {n} WHO documents ({time.time() - t0:.1f}s)", flush=True)
        if load_all or args.ncbi_bookshelf:
            if should_skip("ncbi_bookshelf"):
                pass
            else:
                t0 = time.time()
                n = load_ncbi_bookshelf(cursor)
                print(f"Loaded {n} NCBI Bookshelf entries ({time.time() - t0:.1f}s)", flush=True)
        if load_all or args.openstax:
            if should_skip("openstax"):
                pass
            else:
                t0 = time.time()
                n = load_openstax(cursor)
                print(f"Loaded {n} OpenStax pages ({time.time() - t0:.1f}s)", flush=True)
        if load_all or args.orphanet:
            if should_skip("orphanet"):
                pass
            else:
                t0 = time.time()
                n = load_orphanet(cursor)
                print(f"Loaded {n} Orphanet records ({time.time() - t0:.1f}s)", flush=True)
        if load_all or args.orphanet_web:
            if should_skip("orphanet_web"):
                pass
            else:
                t0 = time.time()
                n = load_orphanet_web(cursor)
                print(f"Loaded {n} Orphanet web pages ({time.time() - t0:.1f}s)", flush=True)
        if load_all or args.rag:
            if should_skip("rag"):
                pass
            else:
                t0 = time.time()
                n = load_rag_chunks(cursor)
                print(f"Loaded {n} RAG chunks ({time.time() - t0:.1f}s)", flush=True)

        if args.strict:
            failures = []
            for table, minimum in TABLE_MIN_ROWS.items():
                cnt = _table_row_count(cursor, table)
                if cnt < minimum:
                    failures.append(f"{table}={cnt}<{minimum}")
            if failures:
                raise RuntimeError("Strict coverage failed: " + ", ".join(failures))

        conn.commit()
        elapsed = time.time() - total_start
        print(f"Done. Total time: {elapsed / 60:.1f} min ({elapsed:.0f}s)", flush=True)
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    main()
