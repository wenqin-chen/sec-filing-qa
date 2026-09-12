"""FinanceBench corpus builder: PDF download (only when invoked) and corpus ingestion.

The ~80 filings referenced by the FinanceBench open set live in the ``patronus-ai/financebench``
GitHub repository under ``pdfs/<doc_name>.pdf``. This module

* downloads them into ``data/raw/financebench/pdfs/`` with a ``%PDF`` magic-byte check (a
  GitHub 404 page or an LFS pointer saved as ``.pdf`` must fail loudly, not index as zero
  pages), a per-question ``doc_link`` fallback through the cached EDGAR client, and a
  ``MANIFEST.json`` recording the upstream commit and the SHA-256 of every file;
* turns ``doc_name`` (``'3M_2022_10K'``) plus ``data/companies.yaml`` into a
  :class:`~secqa.core.contracts.DocumentMeta` and ingests each PDF through
  :func:`secqa.indexing.pipeline.ingest_document`.

Nothing here runs in the test suite against the network: the download function takes an
injectable ``httpx.Client`` and is exercised with ``respx``. The PDFs themselves are never
committed (FinanceBench is CC-BY-NC-4.0; SPEC 9).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx
from pydantic import Field

from secqa.core.contracts import DocumentMeta, Embedder, Frozen
from secqa.core.errors import ConfigError
from secqa.core.ids import file_sha256, sha256_hex
from secqa.core.logging import get_logger
from secqa.edgar import EdgarClient, EdgarError
from secqa.indexing.pipeline import (
    Company,
    IngestReport,
    existing_documents,
    ingest_document,
    is_unchanged,
    resolve_company,
)
from secqa.ingest import doc_meta_from_name, extract_pdf_pages
from secqa.ingest.pdf import is_pdf_bytes
from secqa.store import DuckDBStore

log = get_logger(__name__)

FINANCEBENCH_REPO = "patronus-ai/financebench"
FINANCEBENCH_PDF_BASE = f"https://raw.githubusercontent.com/{FINANCEBENCH_REPO}/main/pdfs/"
FINANCEBENCH_COMMIT_URL = f"https://api.github.com/repos/{FINANCEBENCH_REPO}/commits/main"
DOWNLOAD_USER_AGENT = "sec-filing-qa/0.1 (+https://github.com/wenqinchen/sec-filing-qa)"
PDF_MANIFEST_NAME = "MANIFEST.json"
PDF_MANIFEST_FORMAT = "secqa-financebench-pdfs/1"
SOURCE_KIND: Literal["financebench_pdf"] = "financebench_pdf"


class PdfDownloadReport(Frozen):
    """Outcome of :func:`download_financebench_pdfs`.

    ``downloaded`` came from GitHub in this run, ``fallback`` from the question's ``doc_link``,
    ``cached`` were already on disk as valid PDFs, ``missing`` could not be obtained anywhere.
    """

    downloaded: list[str]
    cached: list[str]
    fallback: list[str]
    missing: list[str]
    upstream_commit: str | None
    seconds: float
    errors: dict[str, str] = Field(default_factory=dict)


def financebench_pdf_url(doc_name: str) -> str:
    """Raw GitHub URL of a FinanceBench PDF (``doc_name`` is case-sensitive upstream)."""
    if not doc_name or "/" in doc_name or doc_name in (".", ".."):
        raise ValueError(f"invalid FinanceBench doc_name {doc_name!r}")
    return f"{FINANCEBENCH_PDF_BASE}{doc_name}.pdf"


def pdf_path(pdf_dir: Path, doc_name: str) -> Path:
    """``<pdf_dir>/<doc_name>.pdf``."""
    return Path(pdf_dir) / f"{doc_name}.pdf"


def is_pdf_file(path: Path) -> bool:
    """True when ``path`` exists and starts with the ``%PDF`` marker."""
    try:
        with Path(path).open("rb") as fh:
            return is_pdf_bytes(fh.read(4))
    except OSError:
        return False


# ---------------------------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------------------------


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _read_pdf_manifest(out_dir: Path) -> dict[str, Any]:
    path = out_dir / PDF_MANIFEST_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"format": PDF_MANIFEST_FORMAT, "upstream_commit": None, "files": {}}
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"corrupt {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("files"), dict):
        raise ConfigError(f"{path} does not look like a PDF manifest")
    return raw


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _resolve_upstream_commit(client: httpx.Client) -> str | None:
    """SHA of the upstream ``main`` (provenance only; ``None`` when GitHub's API is unreachable)."""
    try:
        response = client.get(FINANCEBENCH_COMMIT_URL)
        response.raise_for_status()
        sha = response.json().get("sha")
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        log.warning("financebench_commit_unavailable", error=str(exc)[:200])
        return None
    return str(sha) if sha else None


def _get_pdf(client: httpx.Client, url: str) -> tuple[bytes | None, str | None]:
    """GET ``url``; returns ``(pdf_bytes, None)`` or ``(None, reason)``."""
    try:
        response = client.get(url)
    except httpx.HTTPError as exc:
        return None, f"network error: {exc}"
    if response.status_code == 404:
        return None, "HTTP 404"
    if response.is_error:
        return None, f"HTTP {response.status_code}"
    if not is_pdf_bytes(response.content[:4]):
        return None, "response is not a PDF (no %PDF marker)"
    return response.content, None


def download_financebench_pdfs(
    doc_names: Iterable[str],
    out_dir: Path,
    edgar: EdgarClient | None = None,
    doc_links: dict[str, str] | None = None,
    *,
    http: httpx.Client | None = None,
    timeout_s: float = 60.0,
) -> PdfDownloadReport:
    """Fetch FinanceBench PDFs into ``out_dir`` (skipping valid ones already there).

    Order of attempts per document: existing valid file -> raw GitHub URL -> ``doc_links``
    entry through ``edgar`` (SEC or issuer site; cached and rate-limited) -> ``missing``.
    ``MANIFEST.json`` in ``out_dir`` is updated with the upstream commit SHA and, per file, the
    source URL, SHA-256, size and fetch time. Network failures for one document never abort
    the batch; they are listed in ``errors`` and the document in ``missing`` so a re-run
    completes the set.
    """
    started = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = _unique(doc_names)
    links = doc_links or {}
    manifest = _read_pdf_manifest(out_dir)
    files: dict[str, Any] = manifest["files"]

    downloaded: list[str] = []
    cached: list[str] = []
    fallback: list[str] = []
    missing: list[str] = []
    errors: dict[str, str] = {}

    own_client = http is None
    client = http or httpx.Client(
        timeout=httpx.Timeout(timeout_s),
        follow_redirects=True,
        headers={"User-Agent": DOWNLOAD_USER_AGENT, "Accept": "application/pdf, */*"},
    )
    try:
        upstream_commit = _resolve_upstream_commit(client) if names else None
        for doc_name in names:
            target = pdf_path(out_dir, doc_name)
            if is_pdf_file(target):
                cached.append(doc_name)
                continue
            url = financebench_pdf_url(doc_name)
            body, reason = _get_pdf(client, url)
            source = url
            via_fallback = False
            if body is None:
                log.warning(
                    "financebench_pdf_unavailable", doc_name=doc_name, url=url, reason=reason
                )
                link = links.get(doc_name)
                if link and edgar is not None:
                    try:
                        candidate = edgar.get_bytes(link)
                    except EdgarError as exc:
                        candidate = None
                        reason = f"{reason}; fallback {link}: {exc}"
                    if candidate is not None and is_pdf_bytes(candidate[:4]):
                        body, source, via_fallback = candidate, link, True
                    elif candidate is not None:
                        reason = f"{reason}; fallback {link}: not a PDF"
                elif link:
                    reason = f"{reason}; doc_link fallback needs an EdgarClient"
            if body is None:
                missing.append(doc_name)
                errors[doc_name] = reason or "unknown"
                continue
            _write_atomic(target, body)
            files[doc_name] = {
                "url": source,
                "sha256": sha256_hex(body),
                "size": len(body),
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "fallback": via_fallback,
            }
            (fallback if via_fallback else downloaded).append(doc_name)
            log.info(
                "financebench_pdf_downloaded",
                doc_name=doc_name,
                url=source,
                bytes=len(body),
                fallback=via_fallback,
            )
    finally:
        if own_client:
            client.close()

    if upstream_commit:
        manifest["upstream_commit"] = upstream_commit
    manifest["format"] = PDF_MANIFEST_FORMAT
    manifest["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    _write_atomic(
        out_dir / PDF_MANIFEST_NAME,
        json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    report = PdfDownloadReport(
        downloaded=downloaded,
        cached=cached,
        fallback=fallback,
        missing=missing,
        upstream_commit=manifest.get("upstream_commit"),
        seconds=time.perf_counter() - started,
        errors=errors,
    )
    log.info(
        "financebench_pdfs_fetched",
        n_requested=len(names),
        n_downloaded=len(downloaded),
        n_cached=len(cached),
        n_fallback=len(fallback),
        n_missing=len(missing),
        upstream_commit=report.upstream_commit,
        seconds=round(report.seconds, 2),
    )
    return report


# ---------------------------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------------------------


def financebench_document_meta(
    doc_name: str,
    companies: Iterable[Company],
    *,
    source_sha256: str,
    n_pages: int,
    source_url: str | None = None,
    ingested_at: datetime | None = None,
) -> DocumentMeta:
    """Build the ``DocumentMeta`` for a FinanceBench PDF from its name and the company table.

    ``'3M_2022_10K'`` -> company ``3M Company`` (ticker MMM, CIK from ``companies.yaml``),
    fiscal year 2022, form ``10-K``. An unknown company prefix still yields a valid record
    (``ticker``/``cik`` ``None``, company = the prefix) with a warning, so retrieval works and
    only the ticker filter misses it; ``period_end`` stays ``None`` because the name does not
    carry it.
    """
    company_key, fiscal_year, form = doc_meta_from_name(doc_name)
    company = resolve_company(companies, company_key)
    if company is None:
        log.warning("financebench_company_unresolved", doc_name=doc_name, key=company_key)
    return DocumentMeta(
        doc_name=doc_name,
        company=company.name if company else company_key,
        ticker=company.ticker if company else None,
        cik=company.cik if company else None,
        form=form,
        fiscal_year=fiscal_year,
        period_end=None,
        source_kind=SOURCE_KIND,
        source_url=source_url or financebench_pdf_url(doc_name),
        source_sha256=source_sha256,
        n_pages=n_pages,
        ingested_at=ingested_at or datetime.now(UTC),
    )


def ingest_financebench_corpus(
    store: DuckDBStore,
    embedder: Embedder,
    pdf_dir: Path,
    doc_names: list[str],
    companies: list[Company],
    *,
    skip_unchanged: bool = True,
) -> IngestReport:
    """Ingest ``<pdf_dir>/<doc_name>.pdf`` for every requested document.

    Documents whose PDF is missing or unreadable are listed in ``IngestReport.skipped`` (with
    an error log line) and the rest of the corpus still builds. With ``skip_unchanged`` (the
    default) a document whose file bytes are already indexed is left alone and listed under
    ``unchanged`` -- a resumed 20-30 minute bge build only embeds what changed. The FTS index
    and manifest counts are rebuilt once at the end when anything was ingested.

    Raises:
        ConfigError: on a read-only store.
    """
    if store.read_only:
        raise ConfigError("ingest_financebench_corpus is not allowed on a read-only store")
    started = time.perf_counter()
    pdf_dir = Path(pdf_dir)
    existing = existing_documents(store)
    n_documents = n_pages = n_chunks = 0
    skipped: list[str] = []
    unchanged: list[str] = []

    for doc_name in _unique(doc_names):
        path = pdf_path(pdf_dir, doc_name)
        if not path.is_file():
            log.warning("financebench_pdf_missing", doc_name=doc_name, path=str(path))
            skipped.append(doc_name)
            continue
        source_sha256 = file_sha256(path)
        if skip_unchanged and is_unchanged(existing.get(doc_name), source_sha256, SOURCE_KIND):
            log.info("financebench_document_unchanged", doc_name=doc_name)
            unchanged.append(doc_name)
            continue
        try:
            pages = extract_pdf_pages(path, doc_name)
        except ValueError as exc:
            log.error("financebench_pdf_unreadable", doc_name=doc_name, error=str(exc))
            skipped.append(doc_name)
            continue
        meta = financebench_document_meta(
            doc_name, companies, source_sha256=source_sha256, n_pages=len(pages)
        )
        n_chunks += ingest_document(store, embedder, pages, meta)
        n_pages += len(pages)
        n_documents += 1

    if n_documents:
        store.rebuild_fts()
    report = IngestReport(
        documents=n_documents,
        pages=n_pages,
        chunks=n_chunks,
        skipped=skipped,
        seconds=time.perf_counter() - started,
        unchanged=unchanged,
    )
    log.info(
        "financebench_corpus_ingested",
        pdf_dir=str(pdf_dir),
        n_requested=len(_unique(doc_names)),
        n_documents=report.documents,
        n_pages=report.pages,
        n_chunks=report.chunks,
        n_unchanged=len(unchanged),
        n_skipped=len(skipped),
        skipped=skipped,
        seconds=round(report.seconds, 2),
    )
    return report


__all__ = [
    "DOWNLOAD_USER_AGENT",
    "FINANCEBENCH_COMMIT_URL",
    "FINANCEBENCH_PDF_BASE",
    "PDF_MANIFEST_NAME",
    "SOURCE_KIND",
    "PdfDownloadReport",
    "download_financebench_pdfs",
    "financebench_document_meta",
    "financebench_pdf_url",
    "ingest_financebench_corpus",
    "is_pdf_file",
    "pdf_path",
]
