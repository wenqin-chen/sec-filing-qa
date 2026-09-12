"""secqa.indexing: batch ingestion orchestration (CLI-only; never imported by the API).

Public surface (see the module manifest and each submodule's docstring):

* :func:`ingest_document` -- pages + meta -> chunks, embeddings, store rows (idempotent).
* :func:`ingest_financebench_corpus` / :func:`download_financebench_pdfs` -- the benchmark
  corpus from ``data/raw/financebench/pdfs``.
* :func:`ingest_ticker` -- EDGAR 10-K/10-Q primary documents for one ticker.
* :func:`load_xbrl_for_companies` / :func:`load_companies` -- companyfacts for
  ``data/companies.yaml`` and the curated ``financials`` view.
* :func:`build_manifest`, :func:`pack_index`, :func:`fetch_index` -- provenance and the
  ``index-*.tar.zst`` release asset.
"""

from secqa.indexing.financebench_corpus import (
    PdfDownloadReport,
    download_financebench_pdfs,
    financebench_document_meta,
    financebench_pdf_url,
    ingest_financebench_corpus,
)
from secqa.indexing.manifest_build import (
    IndexPackError,
    build_manifest,
    fetch_index,
    hash_inputs,
    pack_index,
)
from secqa.indexing.pipeline import (
    DEFAULT_COMPANIES_PATH,
    Company,
    IngestReport,
    edgar_doc_name,
    ingest_document,
    ingest_ticker,
    load_companies,
    load_xbrl_for_companies,
    resolve_company,
)

__all__ = [
    "DEFAULT_COMPANIES_PATH",
    "Company",
    "IndexPackError",
    "IngestReport",
    "PdfDownloadReport",
    "build_manifest",
    "download_financebench_pdfs",
    "edgar_doc_name",
    "fetch_index",
    "financebench_document_meta",
    "financebench_pdf_url",
    "hash_inputs",
    "ingest_document",
    "ingest_financebench_corpus",
    "ingest_ticker",
    "load_companies",
    "load_xbrl_for_companies",
    "pack_index",
    "resolve_company",
]
