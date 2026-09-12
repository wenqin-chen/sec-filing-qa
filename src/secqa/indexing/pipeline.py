"""Batch ingestion orchestration: one document, one EDGAR ticker, XBRL for a company list.

This module is the *write path* of the index and is reached only from the CLI (``secqa ingest``,
``secqa data companies-xbrl``); the API never imports it (SPEC 8, ADR-07). It composes the
lower layers rather than re-implementing them:

* :mod:`secqa.ingest` turns bytes into 1-based :class:`~secqa.core.contracts.Page` records and
  page-bounded chunks with content-addressed ids;
* :mod:`secqa.embeddings` produces the vectors;
* :mod:`secqa.store` persists documents, pages and chunks and owns the FTS index;
* :mod:`secqa.edgar` is the only thing that talks to sec.gov;
* :mod:`secqa.xbrl` loads companyfacts and owns the ``financials`` view.

Two invariants worth defending:

1. **Idempotence per ``doc_name``.** Re-ingesting a document replaces its pages and chunks
   wholesale; chunk ids are content-addressed so an unchanged document yields exactly the same
   rows. Callers that want to skip unchanged sources compare ``DocumentMeta.source_sha256``.
2. **A ``documents`` row means a complete document.** :func:`ingest_document` computes chunks and
   embeddings before touching the store, clears the old rows, writes pages and chunks, and
   writes the ``documents`` row *last*. A crash mid-way therefore leaves no document row, and
   the next run re-ingests instead of skipping a half-written document as "unchanged".
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml
from pydantic import Field, field_validator

from secqa.core.contracts import DocumentMeta, Embedder, Frozen, Page
from secqa.core.errors import ConfigError
from secqa.core.ids import sha256_hex
from secqa.core.logging import get_logger
from secqa.edgar import EdgarClient, EdgarError, FilingRef, normalize_cik
from secqa.ingest import chunk_pages, extract_html_pages
from secqa.store import DuckDBStore
from secqa.xbrl import create_financials_view, load_companyfacts
from secqa.xbrl.load import normalize_ticker

log = get_logger(__name__)

DEFAULT_COMPANIES_PATH = Path("data/companies.yaml")
EDGAR_HTML_SUFFIXES = (".htm", ".html", ".xhtml", ".xml", ".txt")

_KEY_NORMALISE_RE = re.compile(r"[^A-Z0-9]")


def normalise_company_key(value: str) -> str:
    """Upper-case and strip everything but letters/digits: ``'Johnson & Johnson'`` ->
    ``'JOHNSONJOHNSON'``, ``'JOHNSON_JOHNSON'`` -> ``'JOHNSONJOHNSON'``."""
    return _KEY_NORMALISE_RE.sub("", value.upper())


# ---------------------------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------------------------


class Company(Frozen):
    """One row of ``data/companies.yaml``: benchmark company -> ticker -> CIK.

    ``financebench_aliases`` are the company prefixes of FinanceBench ``doc_name`` values
    (``'JOHNSON_JOHNSON'`` in ``'JOHNSON_JOHNSON_2022_10K'``); :meth:`matches` also accepts the
    ticker and the display name so the mapping degrades gracefully when an alias is missing.
    """

    name: str
    ticker: str
    cik: str  # 10-digit zero-padded
    financebench_aliases: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("company name must not be empty")
        return value

    @field_validator("ticker")
    @classmethod
    def _ticker_shape(cls, value: str) -> str:
        return normalize_ticker(value)

    @field_validator("cik", mode="before")
    @classmethod
    def _cik_padded(cls, value: object) -> str:
        if isinstance(value, bool) or not isinstance(value, int | str):
            raise ValueError(f"cik must be a string or int, got {type(value).__name__}")
        return normalize_cik(value)  # an unquoted YAML CIK arrives as an int

    @field_validator("financebench_aliases")
    @classmethod
    def _aliases_clean(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for alias in value:
            alias = str(alias).strip()
            if not alias:
                raise ValueError("financebench_aliases must not contain blank entries")
            key = normalise_company_key(alias)
            if key in seen:
                raise ValueError(f"duplicate alias {alias!r}")
            seen.add(key)
            cleaned.append(alias)
        return cleaned

    def keys(self) -> set[str]:
        """Every normalised key this company answers to (aliases, ticker, name)."""
        return {normalise_company_key(a) for a in self.financebench_aliases} | {
            normalise_company_key(self.ticker),
            normalise_company_key(self.name),
        }

    def matches(self, key: str) -> bool:
        """True if ``key`` (alias, ticker or name, case/punctuation-insensitive) is this company."""
        return normalise_company_key(key) in self.keys()


class IngestReport(Frozen):
    """Outcome of one corpus build.

    ``documents`` / ``pages`` / ``chunks`` count what was (re-)ingested in this run;
    ``unchanged`` lists documents skipped because the source bytes were already indexed;
    ``skipped`` lists documents that could not be ingested (missing or unreadable source).
    """

    documents: int
    pages: int
    chunks: int
    skipped: list[str]
    seconds: float
    unchanged: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# companies.yaml
# ---------------------------------------------------------------------------------------------


def load_companies(path: Path = DEFAULT_COMPANIES_PATH) -> list[Company]:
    """Read and validate ``data/companies.yaml`` (``companies: [{name, ticker, cik, ...}]``).

    Raises:
        ConfigError: when the file is missing or malformed, a CIK/ticker is invalid, or two
            companies share a ticker, a CIK or a lookup key (alias / ticker / name).
    """
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read companies from {path}: {exc}") from exc
    entries = raw.get("companies") if isinstance(raw, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ConfigError(f"{path}: expected a non-empty 'companies' list")
    companies: list[Company] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigError(f"{path}: companies[{index}] must be a mapping")
        try:
            companies.append(Company.model_validate(entry))
        except ValueError as exc:
            raise ConfigError(f"{path}: companies[{index}]: {exc}") from exc
    _check_unique(path, companies)
    log.info("companies_loaded", path=str(path), n=len(companies))
    return companies


def _check_unique(path: Path, companies: list[Company]) -> None:
    tickers: dict[str, str] = {}
    ciks: dict[str, str] = {}
    keys: dict[str, str] = {}
    for company in companies:
        for table, value, label in (
            (tickers, company.ticker, "ticker"),
            (ciks, company.cik, "cik"),
        ):
            if value in table:
                raise ConfigError(
                    f"{path}: {label} {value!r} is used by both {table[value]!r} "
                    f"and {company.name!r}"
                )
            table[value] = company.name
        for key in company.keys():
            if key in keys and keys[key] != company.name:
                raise ConfigError(
                    f"{path}: lookup key {key!r} is ambiguous between {keys[key]!r} "
                    f"and {company.name!r}"
                )
            keys[key] = company.name


def resolve_company(companies: Iterable[Company], key: str) -> Company | None:
    """Find the company for a FinanceBench doc-name prefix, ticker or name; ``None`` if unknown.

    Explicit aliases win over ticker / name matches so a curated mapping is never overridden
    by a coincidental name collision.
    """
    wanted = normalise_company_key(key)
    if not wanted:
        return None
    candidates = list(companies)
    for company in candidates:
        if any(normalise_company_key(a) == wanted for a in company.financebench_aliases):
            return company
    for company in candidates:
        if wanted in company.keys():
            return company
    return None


# ---------------------------------------------------------------------------------------------
# one document
# ---------------------------------------------------------------------------------------------


def ingest_document(
    store: DuckDBStore,
    embedder: Embedder,
    pages: list[Page],
    meta: DocumentMeta,
    batch_size: int = 64,
) -> int:
    """Chunk, embed and store one document; returns the number of chunks written.

    Idempotent per ``meta.doc_name``: existing pages, chunks and the document row are replaced.
    Embeddings are computed *before* any write and the ``documents`` row is written *last*, so
    an interrupted run never leaves a document that looks complete (see module docstring).
    Documents whose pages carry no text are stored with zero chunks (and a warning) so page
    numbering stays physical and the document is still listed.

    Raises:
        ConfigError: on a read-only store.
        ValueError: when a page belongs to another document or ``meta.n_pages`` disagrees with
            ``len(pages)``.
    """
    if store.read_only:
        raise ConfigError("ingest_document is not allowed on a read-only store")
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    foreign = sorted({page.doc_name for page in pages if page.doc_name != meta.doc_name})
    if foreign:
        raise ValueError(f"pages of {foreign} passed while ingesting {meta.doc_name!r}")
    if meta.n_pages != len(pages):
        raise ValueError(f"meta.n_pages={meta.n_pages} but {len(pages)} pages were given")

    started = time.perf_counter()
    chunks = chunk_pages(pages)
    if chunks:
        embeddings = embedder.embed(
            [chunk.text for chunk in chunks], batch_size=batch_size, kind="passage"
        )
    else:
        embeddings = np.zeros((0, store.dim), dtype=np.float32)
        log.warning("document_has_no_text", doc_name=meta.doc_name, n_pages=len(pages))
    embed_ms = (time.perf_counter() - started) * 1000.0

    store.delete_document(meta.doc_name)
    store.add_pages(pages)
    if chunks:
        store.add_chunks(chunks, embeddings)
    store.upsert_document(meta)
    log.info(
        "document_ingested",
        doc_name=meta.doc_name,
        source_kind=meta.source_kind,
        ticker=meta.ticker,
        n_pages=len(pages),
        n_chunks=len(chunks),
        embed_ms=round(embed_ms, 1),
        total_ms=round((time.perf_counter() - started) * 1000.0, 1),
    )
    return len(chunks)


def existing_documents(store: DuckDBStore) -> dict[str, DocumentMeta]:
    """``{doc_name: DocumentMeta}`` for every document in the store (one query)."""
    return {doc.doc_name: doc for doc in store.list_documents()}


def is_unchanged(existing: DocumentMeta | None, source_sha256: str, source_kind: str) -> bool:
    """True when the store already holds this exact source (same bytes, same kind)."""
    return (
        existing is not None
        and existing.source_sha256 == source_sha256
        and existing.source_kind == source_kind
    )


# ---------------------------------------------------------------------------------------------
# EDGAR: one ticker
# ---------------------------------------------------------------------------------------------


def edgar_doc_name(ticker: str, ref: FilingRef) -> str:
    """``'<TICKER>_<YEAR>[Q<n>]_<FORM>'`` for an EDGAR filing, e.g. ``'AAPL_2023_10-K'``.

    Annual forms use the period year alone; every other form (10-Q, 8-K ...) adds the calendar
    quarter of the period end (``'AAPL_2023Q2_10-Q'``) so filings of one year do not collide.
    Amendments drop the slash (``'10-K/A'`` -> ``'10-KA'``) because the name is used as a file
    stem; :func:`secqa.ingest.doc_meta_from_name` maps it back to ``'10-K/A'``.
    """
    symbol = normalize_ticker(ticker)
    form = ref.form.upper()
    period = ref.report_date or ref.filing_date
    year = ref.period_year
    if form.startswith("10-K"):
        stamp = str(year)
    else:
        stamp = f"{year}Q{(period.month - 1) // 3 + 1}"
    return f"{symbol}_{stamp}_{form.replace('/', '')}"


def ingest_ticker(
    store: DuckDBStore,
    embedder: Embedder,
    edgar: EdgarClient,
    ticker: str,
    forms: tuple[str, ...] = ("10-K",),
    years: Iterable[int] = range(2020, 2026),
    *,
    skip_unchanged: bool = True,
) -> list[str]:
    """Ingest a company's EDGAR filings (primary HTML documents) into the store.

    Resolves the ticker to a CIK, lists filings matching ``forms`` / ``years`` (period-of-report
    year), downloads each primary document through the cached, rate-limited client, converts
    it to pseudo-pages and ingests it. Returns the ``doc_name`` of every filing now present in
    the store for this request, in filing-date order (both newly ingested and unchanged ones).
    Filings whose primary document is not HTML/text are skipped with a warning; when two
    filings map to the same name (e.g. two amendments in a year) the latest filed wins.

    Raises:
        ConfigError: on a read-only store.
        TickerNotFound / EdgarError: propagated from the client (a missing ticker or a failed
            download is not something this function can decide for the caller).
    """
    if store.read_only:
        raise ConfigError("ingest_ticker is not allowed on a read-only store")
    symbol = normalize_ticker(ticker)
    started = time.perf_counter()
    cik = edgar.cik_for_ticker(symbol)
    years_list = list(years)
    refs = edgar.list_filings(cik, forms=forms, years=years_list)
    company_name = str(edgar.submissions(cik).get("name") or symbol).strip() or symbol
    existing = existing_documents(store)

    by_name: dict[str, FilingRef] = {}
    for ref in sorted(refs, key=lambda r: (r.filing_date, r.accession)):
        name = edgar_doc_name(symbol, ref)
        if name in by_name:
            log.warning(
                "edgar_doc_name_collision",
                doc_name=name,
                dropped_accession=by_name[name].accession,
                kept_accession=ref.accession,
            )
        by_name[name] = ref

    present: list[str] = []
    n_ingested = 0
    for doc_name, ref in by_name.items():
        if not ref.primary_doc.lower().endswith(EDGAR_HTML_SUFFIXES):
            log.warning(
                "edgar_filing_skipped",
                doc_name=doc_name,
                primary_doc=ref.primary_doc,
                reason="primary document is not HTML/text",
            )
            continue
        body = edgar.fetch_primary_document(ref)
        source_sha256 = sha256_hex(body)
        if skip_unchanged and is_unchanged(existing.get(doc_name), source_sha256, "edgar_html"):
            log.info("edgar_filing_unchanged", doc_name=doc_name, accession=ref.accession)
            present.append(doc_name)
            continue
        pages = extract_html_pages(body, doc_name)
        meta = DocumentMeta(
            doc_name=doc_name,
            company=company_name,
            ticker=symbol,
            cik=cik,
            form=ref.form,
            fiscal_year=ref.period_year,
            period_end=ref.report_date,
            source_kind="edgar_html",
            source_url=ref.url,
            source_sha256=source_sha256,
            n_pages=len(pages),
            ingested_at=datetime.now(UTC),
        )
        ingest_document(store, embedder, pages, meta)
        n_ingested += 1
        present.append(doc_name)

    if n_ingested:
        store.rebuild_fts()
    log.info(
        "ticker_ingested",
        ticker=symbol,
        cik=cik,
        forms=list(forms),
        years=years_list,
        n_filings=len(refs),
        n_ingested=n_ingested,
        n_present=len(present),
        seconds=round(time.perf_counter() - started, 2),
    )
    return present


# ---------------------------------------------------------------------------------------------
# XBRL for the company list
# ---------------------------------------------------------------------------------------------


def load_xbrl_for_companies(
    store: DuckDBStore, edgar: EdgarClient, companies: list[Company]
) -> int:
    """Load ``companyfacts`` for every company and (re)create the ``financials`` view.

    Each company is loaded in its own transaction by :func:`secqa.xbrl.load_companyfacts`
    (delete-by-CIK + bulk insert, manifest ``n_facts`` refreshed). A company whose facts cannot
    be fetched (:class:`EdgarError`, e.g. 404 for an entity with no XBRL) is logged and skipped
    so one bad CIK does not abort a 40-company run; the view is created once at the end.
    Returns the total number of fact rows loaded in this call.

    Raises:
        ConfigError: on a read-only store, or when *every* company failed (nothing was loaded,
            which almost always means a wrong User-Agent or no network rather than bad data).
    """
    if store.read_only:
        raise ConfigError("load_xbrl_for_companies is not allowed on a read-only store")
    if not companies:
        raise ValueError("companies must not be empty")
    started = time.perf_counter()
    total = 0
    failed: list[str] = []
    for company in companies:
        try:
            facts = edgar.companyfacts(company.cik)
        except EdgarError as exc:
            failed.append(company.ticker)
            log.error(
                "companyfacts_fetch_failed",
                ticker=company.ticker,
                cik=company.cik,
                status=exc.status_code,
                error=str(exc),
            )
            continue
        total += load_companyfacts(store, facts, company.ticker)
    if failed and len(failed) == len(companies):
        raise ConfigError(
            f"companyfacts could not be fetched for any of {len(companies)} companies "
            f"({', '.join(failed)}); check SEC_USER_AGENT and network access"
        )
    create_financials_view(store)
    log.info(
        "xbrl_loaded",
        n_companies=len(companies),
        n_failed=len(failed),
        failed=failed,
        n_facts=total,
        seconds=round(time.perf_counter() - started, 2),
    )
    return total


__all__ = [
    "DEFAULT_COMPANIES_PATH",
    "EDGAR_HTML_SUFFIXES",
    "Company",
    "IngestReport",
    "edgar_doc_name",
    "existing_documents",
    "ingest_document",
    "ingest_ticker",
    "is_unchanged",
    "load_companies",
    "load_xbrl_for_companies",
    "normalise_company_key",
    "resolve_company",
]
