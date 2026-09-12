"""ingest_document and the FinanceBench corpus builder on reportlab PDFs (offline)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from secqa.core.contracts import DocumentMeta, Page
from secqa.core.errors import ConfigError
from secqa.core.ids import file_sha256
from secqa.embeddings import HashingEmbedder
from secqa.indexing import (
    Company,
    IngestReport,
    financebench_document_meta,
    financebench_pdf_url,
    ingest_document,
    ingest_financebench_corpus,
)
from secqa.ingest import extract_pdf_pages
from secqa.retrieval import Retriever
from secqa.store import DuckDBStore
from tests.indexing.conftest import CountingEmbedder

DOC = "FIXTURE_2023_10K"


def _meta(doc_name: str, n_pages: int, sha: str = "0" * 64) -> DocumentMeta:
    return DocumentMeta(
        doc_name=doc_name,
        company="Fixture Corp",
        ticker="FIXT",
        cik="0001234567",
        form="10-K",
        fiscal_year=2023,
        period_end=None,
        source_kind="fixture",
        source_url=f"https://example.invalid/{doc_name}.pdf",
        source_sha256=sha,
        n_pages=n_pages,
        ingested_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
    )


def _chunk_ids(store: DuckDBStore, doc_name: str) -> list[str]:
    rows = store.conn.execute(
        "SELECT chunk_id FROM chunks WHERE doc_name = ? ORDER BY page_num, chunk_idx", [doc_name]
    ).fetchall()
    return [row[0] for row in rows]


# ---- ingest_document ----------------------------------------------------------------------


def test_ingest_document_writes_pages_chunks_and_document(
    store: DuckDBStore, embedder: HashingEmbedder, fixture_pdf: Path
) -> None:
    pages = extract_pdf_pages(fixture_pdf, DOC)
    n = ingest_document(store, embedder, pages, _meta(DOC, len(pages)))
    assert n > 0
    counts = store.counts()
    assert counts == {"documents": 1, "pages": 3, "chunks": n, "facts": 0}
    [doc] = store.list_documents()
    assert (doc.doc_name, doc.ticker, doc.cik, doc.n_pages) == (DOC, "FIXT", "0001234567", 3)
    assert [p.page_num for p in store.get_pages(DOC, [1, 2, 3])] == [1, 2, 3]
    assert all(c.doc_name == DOC for c in store.get_chunks(_chunk_ids(store, DOC)))


def test_ingest_document_is_idempotent(
    store: DuckDBStore, embedder: HashingEmbedder, fixture_pdf: Path
) -> None:
    pages = extract_pdf_pages(fixture_pdf, DOC)
    first = ingest_document(store, embedder, pages, _meta(DOC, len(pages)))
    ids_first = _chunk_ids(store, DOC)
    second = ingest_document(store, embedder, pages, _meta(DOC, len(pages)))
    assert first == second
    assert _chunk_ids(store, DOC) == ids_first
    assert store.counts() == {"documents": 1, "pages": 3, "chunks": first, "facts": 0}


def test_reingest_with_fewer_pages_drops_stale_rows(
    store: DuckDBStore, embedder: HashingEmbedder, fixture_pdf: Path
) -> None:
    pages = extract_pdf_pages(fixture_pdf, DOC)
    ingest_document(store, embedder, pages, _meta(DOC, len(pages)))
    n = ingest_document(store, embedder, pages[:1], _meta(DOC, 1, sha="1" * 64))
    counts = store.counts()
    assert counts["pages"] == 1
    assert counts["chunks"] == n
    assert all(c.page_num == 1 for c in store.get_chunks(_chunk_ids(store, DOC)))
    [doc] = store.list_documents()
    assert doc.n_pages == 1 and doc.source_sha256 == "1" * 64


def test_ingest_document_with_no_text_stores_zero_chunks(
    store: DuckDBStore, embedder: HashingEmbedder
) -> None:
    pages = [Page(doc_name=DOC, page_num=1, text=""), Page(doc_name=DOC, page_num=2, text="  ")]
    assert ingest_document(store, embedder, pages, _meta(DOC, 2)) == 0
    assert store.counts() == {"documents": 1, "pages": 2, "chunks": 0, "facts": 0}


def test_ingest_document_rejects_foreign_pages_and_bad_n_pages(
    store: DuckDBStore, embedder: HashingEmbedder
) -> None:
    pages = [Page(doc_name=DOC, page_num=1, text="hello world")]
    with pytest.raises(ValueError, match="OTHER_2022_10K"):
        ingest_document(
            store,
            embedder,
            [*pages, Page(doc_name="OTHER_2022_10K", page_num=1, text="x")],
            _meta(DOC, 2),
        )
    with pytest.raises(ValueError, match="n_pages"):
        ingest_document(store, embedder, pages, _meta(DOC, 5))
    with pytest.raises(ValueError, match="batch_size"):
        ingest_document(store, embedder, pages, _meta(DOC, 1), batch_size=0)
    assert store.counts()["documents"] == 0


def test_ingest_document_refuses_read_only_store(
    file_store_factory: Callable[[str], DuckDBStore], embedder: HashingEmbedder, tmp_path: Path
) -> None:
    writable = file_store_factory("ro.duckdb")
    writable.close()
    with DuckDBStore(tmp_path / "ro.duckdb", read_only=True) as ro:
        with pytest.raises(ConfigError, match="read-only"):
            ingest_document(ro, embedder, [Page(doc_name=DOC, page_num=1, text="x")], _meta(DOC, 1))


def test_failed_ingest_leaves_no_document_row(
    store: DuckDBStore,
    embedder: HashingEmbedder,
    fixture_pdf: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documents row is written last, so a crash cannot be mistaken for 'unchanged'."""
    pages = extract_pdf_pages(fixture_pdf, DOC)

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "add_chunks", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        ingest_document(store, embedder, pages, _meta(DOC, len(pages)))
    assert store.list_documents() == []
    monkeypatch.undo()
    assert ingest_document(store, embedder, pages, _meta(DOC, len(pages))) > 0
    assert len(store.list_documents()) == 1


# ---- financebench_document_meta ------------------------------------------------------------


def test_financebench_pdf_url_and_meta(companies: list[Company]) -> None:
    assert financebench_pdf_url("3M_2022_10K").endswith("/main/pdfs/3M_2022_10K.pdf")
    with pytest.raises(ValueError):
        financebench_pdf_url("../etc/passwd")
    meta = financebench_document_meta(
        "FIXTURE_2023Q2_10Q", companies, source_sha256="a" * 64, n_pages=4
    )
    assert (meta.company, meta.ticker, meta.cik) == ("Fixture Corp", "FIXT", "0001234567")
    assert (meta.form, meta.fiscal_year, meta.period_end) == ("10-Q", 2023, None)
    assert meta.source_kind == "financebench_pdf"
    assert meta.source_url == financebench_pdf_url("FIXTURE_2023Q2_10Q")
    assert meta.n_pages == 4 and meta.ingested_at.tzinfo is not None


def test_financebench_document_meta_unknown_company_keeps_prefix(
    companies: list[Company],
) -> None:
    meta = financebench_document_meta(
        "UNKNOWNCO_2020_10K", companies, source_sha256="b" * 64, n_pages=1
    )
    assert (meta.company, meta.ticker, meta.cik, meta.fiscal_year) == (
        "UNKNOWNCO",
        None,
        None,
        2020,
    )


# ---- ingest_financebench_corpus ------------------------------------------------------------


@pytest.fixture
def corpus_dir(fixture_pdf_factory: Callable[..., Path], tmp_path: Path) -> Path:
    fixture_pdf_factory(name="FIXTURE_2023_10K")
    fixture_pdf_factory(
        pages=[
            "ACME HOLDINGS FORM 10-K FISCAL 2022. Item 7. Revenue from widget subscriptions "
            "reached $900 million.",
            "Item 8. The zirconium flywheel amortisation schedule is reviewed annually.",
        ],
        name="ACME_2022_10K",
    )
    return tmp_path


def test_corpus_builds_report_counts_and_is_retrievable(
    store: DuckDBStore,
    counting_embedder: CountingEmbedder,
    corpus_dir: Path,
    companies: list[Company],
) -> None:
    names = ["FIXTURE_2023_10K", "ACME_2022_10K", "MISSING_2021_10K"]
    report = ingest_financebench_corpus(store, counting_embedder, corpus_dir, names, companies)
    assert isinstance(report, IngestReport)
    assert (report.documents, report.pages) == (2, 5)
    assert report.skipped == ["MISSING_2021_10K"]
    assert report.unchanged == []
    assert report.seconds >= 0.0
    counts = store.counts()
    assert counts["documents"] == 2 and counts["pages"] == 5 and counts["chunks"] == report.chunks
    assert counting_embedder.calls == 2 and counting_embedder.texts == report.chunks

    manifest = store.manifest()
    assert (manifest.n_documents, manifest.n_pages, manifest.n_chunks) == (2, 5, report.chunks)

    by_name = {d.doc_name: d for d in store.list_documents()}
    fixture, acme = by_name["FIXTURE_2023_10K"], by_name["ACME_2022_10K"]
    assert (fixture.ticker, fixture.cik, fixture.fiscal_year, fixture.form) == (
        "FIXT",
        "0001234567",
        2023,
        "10-K",
    )
    assert (acme.ticker, acme.cik, acme.fiscal_year) == ("ACME", "0007654321", 2022)
    assert fixture.source_kind == "financebench_pdf"
    assert fixture.source_sha256 == file_sha256(corpus_dir / "FIXTURE_2023_10K.pdf")

    retriever = Retriever(store, counting_embedder, strategy="hybrid", k=3)  # type: ignore[arg-type]
    hits = retriever.retrieve("zirconium flywheel amortisation schedule").hits
    assert hits and (hits[0].chunk.doc_name, hits[0].chunk.page_num) == ("ACME_2022_10K", 2)
    hits = retriever.retrieve("total net sales $1,577 million fiscal 2023").hits
    assert hits and hits[0].chunk.doc_name == "FIXTURE_2023_10K"


def test_corpus_second_run_skips_unchanged_documents(
    store: DuckDBStore,
    counting_embedder: CountingEmbedder,
    corpus_dir: Path,
    companies: list[Company],
) -> None:
    names = ["FIXTURE_2023_10K", "ACME_2022_10K"]
    first = ingest_financebench_corpus(store, counting_embedder, corpus_dir, names, companies)
    ids_before = _chunk_ids(store, "FIXTURE_2023_10K")
    embed_calls = counting_embedder.calls

    second = ingest_financebench_corpus(store, counting_embedder, corpus_dir, names, companies)
    assert (second.documents, second.pages, second.chunks) == (0, 0, 0)
    assert second.unchanged == names and second.skipped == []
    assert counting_embedder.calls == embed_calls  # nothing re-embedded
    assert store.counts()["chunks"] == first.chunks

    forced = ingest_financebench_corpus(
        store, counting_embedder, corpus_dir, names, companies, skip_unchanged=False
    )
    assert (forced.documents, forced.chunks) == (2, first.chunks)
    assert forced.unchanged == []
    assert _chunk_ids(store, "FIXTURE_2023_10K") == ids_before  # content-addressed ids
    assert store.counts()["chunks"] == first.chunks


def test_corpus_reingests_when_pdf_changes(
    store: DuckDBStore,
    embedder: HashingEmbedder,
    corpus_dir: Path,
    companies: list[Company],
    fixture_pdf_factory: Callable[..., Path],
) -> None:
    ingest_financebench_corpus(store, embedder, corpus_dir, ["ACME_2022_10K"], companies)
    ids_before = _chunk_ids(store, "ACME_2022_10K")
    fixture_pdf_factory(
        pages=["ACME HOLDINGS restated report. Revenue was $950 million."], name="ACME_2022_10K"
    )
    report = ingest_financebench_corpus(store, embedder, corpus_dir, ["ACME_2022_10K"], companies)
    assert report.documents == 1 and report.unchanged == []
    assert _chunk_ids(store, "ACME_2022_10K") != ids_before
    assert store.counts()["pages"] == 1


def test_corpus_skips_files_that_are_not_pdfs_and_dedupes_names(
    store: DuckDBStore, embedder: HashingEmbedder, corpus_dir: Path, companies: list[Company]
) -> None:
    (corpus_dir / "BOGUS_2020_10K.pdf").write_bytes(b"<html>404 not found</html>")
    report = ingest_financebench_corpus(
        store,
        embedder,
        corpus_dir,
        ["BOGUS_2020_10K", "FIXTURE_2023_10K", "FIXTURE_2023_10K"],
        companies,
    )
    assert report.skipped == ["BOGUS_2020_10K"]
    assert report.documents == 1
    assert store.counts()["documents"] == 1


def test_corpus_unresolved_company_is_still_ingested(
    store: DuckDBStore,
    embedder: HashingEmbedder,
    tmp_path: Path,
    fixture_pdf_factory: Callable[..., Path],
    companies: list[Company],
) -> None:
    fixture_pdf_factory(
        pages=["Unknown Co annual report. Net income was $5 million."], name="UNKNOWNCO_2020_10K"
    )
    report = ingest_financebench_corpus(
        store, embedder, tmp_path, ["UNKNOWNCO_2020_10K"], companies
    )
    assert report.documents == 1
    [doc] = store.list_documents()
    assert (doc.company, doc.ticker, doc.cik) == ("UNKNOWNCO", None, None)
    assert store.list_documents(ticker="FIXT") == []


def test_corpus_refuses_read_only_store(
    file_store_factory: Callable[[str], DuckDBStore],
    embedder: HashingEmbedder,
    tmp_path: Path,
    companies: list[Company],
) -> None:
    file_store_factory("ro.duckdb").close()
    with DuckDBStore(tmp_path / "ro.duckdb", read_only=True) as ro:
        with pytest.raises(ConfigError, match="read-only"):
            ingest_financebench_corpus(ro, embedder, tmp_path, ["FIXTURE_2023_10K"], companies)
