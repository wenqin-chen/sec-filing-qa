-- secqa DuckDB schema (SPEC 4.2). Executed by DuckDBStore.init_schema with {dim} replaced by
-- the embedder dimension; never hard-code a width here. Every statement is idempotent so that
-- re-opening an existing index is a no-op.

CREATE TABLE IF NOT EXISTS documents (
    doc_name      VARCHAR PRIMARY KEY,
    ticker        VARCHAR,
    cik           VARCHAR,
    company       VARCHAR NOT NULL,
    form          VARCHAR NOT NULL,
    fiscal_year   INTEGER,
    period_end    DATE,
    source_kind   VARCHAR NOT NULL,
    source_url    VARCHAR NOT NULL,
    source_sha256 VARCHAR NOT NULL,
    n_pages       INTEGER NOT NULL,
    ingested_at   TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    doc_name VARCHAR NOT NULL,
    page_num INTEGER NOT NULL,
    text     VARCHAR NOT NULL,
    PRIMARY KEY (doc_name, page_num)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id  VARCHAR PRIMARY KEY,
    doc_name  VARCHAR NOT NULL,
    page_num  INTEGER NOT NULL,
    chunk_idx INTEGER NOT NULL,
    section   VARCHAR,
    text      VARCHAR NOT NULL,
    n_tokens  INTEGER NOT NULL,
    embedding FLOAT[{dim}] NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_doc_page_idx ON chunks (doc_name, page_num);

CREATE TABLE IF NOT EXISTS xbrl_facts (
    cik        VARCHAR NOT NULL,
    ticker     VARCHAR NOT NULL,
    taxonomy   VARCHAR NOT NULL,
    tag        VARCHAR NOT NULL,
    unit       VARCHAR NOT NULL,
    fy         INTEGER,
    fp         VARCHAR,
    form       VARCHAR,
    start_date DATE,
    end_date   DATE,
    val        DOUBLE NOT NULL,
    accn       VARCHAR NOT NULL,
    filed      DATE,
    frame      VARCHAR
);

CREATE INDEX IF NOT EXISTS xbrl_facts_lookup_idx ON xbrl_facts (ticker, tag, fy);

-- Baseline `financials` view so the name always resolves (the SQL tool allows it): one row per
-- (company, concept, unit, period) for annual (fp = 'FY') facts, latest filing wins, which drops
-- restated duplicates that appear in later 10-Ks. `fiscal_year` is derived from the period end
-- because the companyfacts `fy` column is the *filing's* fiscal year, so prior-year comparatives
-- in a 10-K carry the wrong `fy` for our purposes.
-- Ownership: `secqa.xbrl.create_financials_view` replaces this view (CREATE OR REPLACE) with the
-- curated per-(ticker, fy) metrics view built from tags.yaml; IF NOT EXISTS keeps that
-- replacement intact when an existing index is re-opened and init_schema runs again.
CREATE VIEW IF NOT EXISTS financials AS
SELECT
    cik,
    ticker,
    taxonomy,
    tag,
    unit,
    CAST(year(end_date) AS INTEGER)                                   AS fiscal_year,
    CASE WHEN start_date IS NULL THEN 'instant' ELSE 'duration' END   AS period_kind,
    start_date,
    end_date,
    val,
    fy,
    fp,
    form,
    accn,
    filed,
    frame
FROM xbrl_facts
WHERE fp = 'FY'
  AND end_date IS NOT NULL
  AND (start_date IS NULL OR date_diff('day', start_date, end_date) BETWEEN 350 AND 380)
QUALIFY row_number() OVER (
    PARTITION BY cik, taxonomy, tag, unit, start_date, end_date
    ORDER BY filed DESC NULLS LAST, accn DESC
) = 1;

CREATE TABLE IF NOT EXISTS index_manifest (
    key   VARCHAR PRIMARY KEY,
    value VARCHAR
);
