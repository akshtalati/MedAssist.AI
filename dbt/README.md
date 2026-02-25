# MedAssist.AI dbt Project

This dbt project transforms raw medical data from Snowflake into analytics-ready datasets.

## Setup

1. **Install dbt-snowflake** (already in requirements.txt):
   ```bash
   pip install dbt-snowflake
   ```

2. **Configure environment variables** in `.env`:
   - `SNOWFLAKE_ACCOUNT`
   - `SNOWFLAKE_USER`
   - `SNOWFLAKE_PASSWORD`
   - `SNOWFLAKE_ROLE` (should be `MEDASSIST_TEAM_ROLE`)
   - `SNOWFLAKE_WAREHOUSE` (default: `MEDASSIST_WH`)
   - `SNOWFLAKE_DATABASE` (default: `MEDASSIST_DB`)

   **Note:** dbt profiles.yml uses environment variables. Make sure `.env` is loaded (or export vars manually).

3. **Create analytics schema** in Snowflake:
   ```sql
   USE ROLE MEDASSIST_TEAM_ROLE;
   CREATE SCHEMA IF NOT EXISTS MEDASSIST_DB.ANALYTICS;
   GRANT USAGE ON SCHEMA MEDASSIST_DB.ANALYTICS TO ROLE MEDASSIST_TEAM_ROLE;
   ```

4. **Test connection**:
   ```bash
   cd dbt
   dbt debug
   ```

## Running dbt

**Build all models**:
```bash
cd dbt
dbt run
```

**Build specific models**:
```bash
dbt run --select staging.*          # Only staging models
dbt run --select marts.*            # Only mart models
dbt run --select int_disease_symptoms  # Specific model
```

**Generate documentation**:
```bash
dbt docs generate
dbt docs serve
```

**Run tests** (if tests are added):
```bash
dbt test
```

## Project Structure

```
dbt/
├── dbt_project.yml       # Project configuration
├── profiles.yml           # Connection profile (uses env vars)
├── models/
│   ├── staging/          # Clean raw data (views)
│   ├── intermediate/     # Join and transform (views)
│   └── marts/            # Final datasets (tables)
└── macros/               # Reusable SQL macros
```

## Models

### Staging Models
- `stg_pubmed` - Cleaned PubMed articles
- `stg_pmc` - Cleaned PMC articles
- `stg_openfda_labels` - Flattened OpenFDA labels
- `stg_openfda_events` - Flattened OpenFDA events
- `stg_symptom_disease_map` - Symptom-disease mappings
- `stg_orphanet_diseases` - Orphanet disease data
- `stg_orphanet_phenotypes` - Orphanet phenotype data

### Intermediate Models
- `int_disease_symptoms` - Diseases joined with symptoms
- `int_drug_disease` - Drug-disease links (placeholder for NLP)
- `int_article_disease` - Article-disease links (placeholder for NLP)

### Mart Models
- `mart_disease_catalog` - Unified disease catalog with symptoms
- `mart_drug_reference` - Unified drug reference (RxNorm + OpenFDA)
- `mart_medical_literature` - Unified article catalog (PubMed + PMC + NCBI + OpenStax)

## Next Steps

1. Add data quality tests using dbt tests
2. Implement NLP-based linking in intermediate models
3. Add incremental models for large tables
4. Create additional marts for specific use cases
