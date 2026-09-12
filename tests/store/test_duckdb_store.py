"""DuckDBStore: schema, writes, BM25 / dense / hybrid search, filters, read-only guard, export."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import pytest

from secqa.core.contracts import Chunk, Page
from secqa.core.errors import ConfigError, IndexMismatch, SqlRejected
from secqa.core.ids import chunk_id
from secqa.store import DuckDBStore, ReadOnlyConnection
from secqa.store.duckdb_store import _PythonBM25
from tests.store.conftest import (
    DIM,
    DOCS,
    EXACT_PHRASE,
    PARAPHRASE_QUERY,
    HashingEmbedder,
    embed_corpus,
    make_corpus,
    make_document,
)

# ---- schema / open ------------------------------------------------------------------------


def test_schema_creates_tables_and_embedding_width_from_dim() -> None:
    with DuckDBStore(":memory:") as store:
        store.init_schema("hashing", 12)
        tables = {
            row[0]
            for row in store.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
        }
        assert {"documents", "pages", "chunks", "xbrl_facts", "index_manifest", "financials"} <= (
            tables
        )
        width = store.conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_name = 'chunks' AND column_name = 'embedding'"
        ).fetchone()
        assert width == ("FLOAT[12]",)
        assert store.dim == 12
        assert store.counts() == {"documents": 0, "pages": 0, "chunks": 0, "facts": 0}


def test_index_mismatch_on_dim_change(tmp_duckdb_path: Path) -> None:
    store = DuckDBStore(tmp_duckdb_path, embed_dim=16)
    store.init_schema("hashing", 16)
    with pytest.raises(IndexMismatch, match="dim"):
        store.init_schema("hashing", 32)
    with pytest.raises(IndexMismatch, match="embedder"):
        store.init_schema("bge-small", 16)
    store.close()
    with pytest.raises(IndexMismatch, match="dim 16"):
        DuckDBStore(tmp_duckdb_path, embed_dim=32)
    # Re-opening without a dim reads it from the manifest.
    with DuckDBStore(tmp_duckdb_path) as reopened:
        assert reopened.dim == 16
        assert reopened.embedder_name == "hashing"
        with pytest.raises(IndexMismatch, match="shape"):
            reopened.search_dense(np.zeros(32, dtype=np.float32))


def test_open_errors() -> None:
    with pytest.raises(ConfigError, match="in-memory"):
        DuckDBStore(":memory:", read_only=True)
    with pytest.raises(ConfigError, match="does not exist"):
        DuckDBStore("/nonexistent/dir/index.duckdb", read_only=True)
    with pytest.raises(ValueError, match="embed_dim"):
        DuckDBStore(":memory:", embed_dim=0)


def test_read_only_open_requires_manifest(tmp_duckdb_path: Path) -> None:
    duckdb.connect(str(tmp_duckdb_path)).close()  # empty file, no schema
    with pytest.raises(ConfigError, match="no index manifest"):
        DuckDBStore(tmp_duckdb_path, read_only=True)


# ---- writes / reads -----------------------------------------------------------------------


def test_documents_pages_round_trip(populated_store: DuckDBStore) -> None:
    docs = populated_store.list_documents()
    assert [d.doc_name for d in docs] == sorted(DOCS)
    acme = populated_store.list_documents(ticker="acme")
    assert {d.doc_name for d in acme} == {"ACME_2022_10K", "ACME_2023_10K"}
    assert [d.doc_name for d in populated_store.list_documents(fiscal_year=2022)] == [
        "ACME_2022_10K"
    ]
    assert [
        d.doc_name for d in populated_store.list_documents(ticker="BOLT", fiscal_year=2023)
    ] == ["BOLT_2023_10K", "BOLT_2023_10Q"]
    assert populated_store.list_documents(doc_names=[]) == []
    assert [d.doc_name for d in populated_store.list_documents(doc_names=["BOLT_2023_10Q"])] == [
        "BOLT_2023_10Q"
    ]
    # Aware datetimes are stored as naive UTC wall-clock and come back naive.
    assert acme[0].ingested_at == datetime(2026, 9, 11, 12, 0, 0)
    assert acme[0].period_end.year == 2022

    pages = populated_store.get_pages("ACME_2022_10K", [3, 1, 3, 99])
    assert [(p.page_num, p.text) for p in pages] == [
        (1, "ACME_2022_10K full page 1 text"),
        (3, "ACME_2022_10K full page 3 text"),
    ]
    assert populated_store.get_pages("ACME_2022_10K", []) == []


def test_upsert_document_replaces_and_pages_are_idempotent(
    store_factory: Callable[..., DuckDBStore],
) -> None:
    store = store_factory()
    meta = make_document("ACME_2022_10K", n_pages=3)
    store.upsert_document(meta)
    store.upsert_document(meta.model_copy(update={"n_pages": 7}))
    assert store.list_documents()[0].n_pages == 7
    store.add_pages([Page(doc_name="ACME_2022_10K", page_num=1, text="v1")])
    store.add_pages([Page(doc_name="ACME_2022_10K", page_num=1, text="v2")])
    assert store.get_pages("ACME_2022_10K", [1])[0].text == "v2"
    assert store.counts()["pages"] == 1
    with pytest.raises(ValueError, match="1-based"):
        store.add_pages([Page(doc_name="ACME_2022_10K", page_num=0, text="x")])


def test_get_chunks_preserves_order_and_skips_unknown(
    populated_store: DuckDBStore, corpus: tuple[list[Chunk], dict[str, str]]
) -> None:
    chunks, _ = corpus
    wanted = [chunks[5].chunk_id, "missing", chunks[1].chunk_id, chunks[5].chunk_id]
    got = populated_store.get_chunks(wanted)
    assert got == [chunks[5], chunks[1]]
    assert populated_store.get_chunks([]) == []


def test_add_chunks_validation(store_factory: Callable[..., DuckDBStore]) -> None:
    store = store_factory()
    text = "some text"
    chunk = Chunk(
        chunk_id=chunk_id("D_2020_10K", 1, 0, text),
        doc_name="D_2020_10K",
        page_num=1,
        chunk_idx=0,
        text=text,
        n_tokens=2,
    )
    with pytest.raises(ValueError, match="shape"):
        store.add_chunks([chunk], np.zeros((1, DIM + 1), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        store.add_chunks([chunk], np.zeros(DIM, dtype=np.float32))
    with pytest.raises(ValueError, match="unique"):
        store.add_chunks([chunk, chunk], np.zeros((2, DIM), dtype=np.float32))
    bad = np.zeros((1, DIM), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        store.add_chunks([chunk], bad)
    store.add_chunks([], np.zeros((0, DIM), dtype=np.float32))  # no-op
    assert store.counts()["chunks"] == 0


def test_readding_a_document_is_idempotent(
    populated_store: DuckDBStore,
    corpus: tuple[list[Chunk], dict[str, str]],
    embedder: HashingEmbedder,
) -> None:
    chunks, seeded = corpus
    before = populated_store.counts()
    acme = [c for c in chunks if c.doc_name == "ACME_2022_10K"]
    vectors = embed_corpus(chunks, embedder, seeded)
    acme_vectors = np.stack(
        [vectors[i] for i, c in enumerate(chunks) if c.doc_name == acme[0].doc_name]
    )
    populated_store.add_chunks(acme, acme_vectors)
    populated_store.rebuild_fts()
    assert populated_store.counts() == before
    # Re-ingesting a document with fewer chunks drops the stale ones (replace, not append).
    populated_store.add_chunks(acme[:3], acme_vectors[:3])
    populated_store.rebuild_fts()
    assert populated_store.counts()["chunks"] == before["chunks"] - len(acme) + 3
    assert populated_store.manifest().n_chunks == populated_store.counts()["chunks"]
    assert populated_store.get_chunks([acme[3].chunk_id]) == []


def test_delete_document_removes_rows_and_invalidates_bm25(
    populated_store: DuckDBStore, corpus: tuple[list[Chunk], dict[str, str]]
) -> None:
    chunks, seeded = corpus
    exact_doc = next(c.doc_name for c in chunks if c.chunk_id == seeded["exact"])
    before = populated_store.counts()
    n_doc_chunks = sum(1 for c in chunks if c.doc_name == exact_doc)
    assert populated_store.search_bm25(EXACT_PHRASE, k=1)[0].chunk.chunk_id == seeded["exact"]

    assert populated_store.delete_document(exact_doc) is True
    after = populated_store.counts()
    assert after["documents"] == before["documents"] - 1
    assert after["pages"] == before["pages"] - 5
    assert after["chunks"] == before["chunks"] - n_doc_chunks
    assert populated_store.get_pages(exact_doc, [1]) == []
    assert populated_store.get_chunks([seeded["exact"]]) == []
    # The stale FTS snapshot is rebuilt before the next search, so the deleted chunk is gone.
    hits = populated_store.search_bm25(EXACT_PHRASE, k=5)
    assert all(hit.chunk.doc_name != exact_doc for hit in hits)
    assert populated_store.manifest().n_chunks == after["chunks"]
    # Facts are keyed by CIK, not by document, so they are untouched; a second call is a no-op.
    assert after["facts"] == before["facts"]
    assert populated_store.delete_document(exact_doc) is False
    assert populated_store.counts() == after


def test_delete_document_refused_on_read_only(tmp_duckdb_path: Path) -> None:
    with DuckDBStore(tmp_duckdb_path, embed_dim=DIM) as store:
        store.init_schema("hashing", DIM)
        store.upsert_document(make_document("ACME_2022_10K"))
    with DuckDBStore(tmp_duckdb_path, read_only=True) as store:
        with pytest.raises(ConfigError, match="read-only"):
            store.delete_document("ACME_2022_10K")


# ---- search -------------------------------------------------------------------------------


def test_bm25_finds_exact_phrase(
    populated_store: DuckDBStore, corpus: tuple[list[Chunk], dict[str, str]]
) -> None:
    _, seeded = corpus
    hits = populated_store.search_bm25(EXACT_PHRASE, k=5)
    assert hits and hits[0].chunk.chunk_id == seeded["exact"]
    assert hits[0].source == "bm25" and hits[0].rank == 1 and hits[0].bm25_rank == 1
    assert hits[0].dense_rank is None
    assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    assert populated_store.search_bm25("   ", k=5) == []
    assert populated_store.search_bm25("qwertyuiopnomatch", k=5) == []
    # Punctuation and quotes in the query are data, not syntax.
    assert populated_store.search_bm25('flywheel\'s "schedule" (zirconium) $1,577; --x', k=3)
    with pytest.raises(ValueError, match="k must be"):
        populated_store.search_bm25(EXACT_PHRASE, k=0)


def test_dense_finds_seeded_paraphrase(
    populated_store: DuckDBStore,
    corpus: tuple[list[Chunk], dict[str, str]],
    embedder: HashingEmbedder,
) -> None:
    _, seeded = corpus
    qvec = embedder.embed([PARAPHRASE_QUERY], kind="query")[0]
    hits = populated_store.search_dense(qvec, k=5)
    assert hits[0].chunk.chunk_id == seeded["paraphrase"]
    assert hits[0].score > 0.9
    assert hits[0].source == "dense" and hits[0].dense_rank == 1 and hits[0].bm25_rank is None
    # Lexical search cannot see the paraphrase (no shared content words).
    assert seeded["paraphrase"] not in {
        h.chunk.chunk_id for h in populated_store.search_bm25(PARAPHRASE_QUERY, k=20)
    }
    # A (1, dim) row vector is accepted too.
    assert (
        populated_store.search_dense(qvec[None, :], k=1)[0].chunk.chunk_id == (seeded["paraphrase"])
    )
    with pytest.raises(ValueError, match="NaN"):
        populated_store.search_dense(np.full(DIM, np.nan, dtype=np.float32), k=1)


def test_hybrid_contains_both_targets(
    populated_store: DuckDBStore,
    corpus: tuple[list[Chunk], dict[str, str]],
    embedder: HashingEmbedder,
) -> None:
    _, seeded = corpus
    # The text drives BM25 (only the seeded exact chunk matches) and the vector drives dense
    # (the planted paraphrase is rank 1), so each target enters the fusion from one side.
    qvec = embedder.embed([PARAPHRASE_QUERY], kind="query")[0]
    hits = populated_store.hybrid_search(EXACT_PHRASE, qvec, k=10, k_each=30)
    ids = [h.chunk.chunk_id for h in hits]
    assert seeded["exact"] in ids and seeded["paraphrase"] in ids
    assert len(hits) == 10 and len(set(ids)) == 10
    assert [h.rank for h in hits] == list(range(1, 11))
    assert all(h.source == "hybrid" for h in hits)
    assert all(hits[i].score >= hits[i + 1].score for i in range(len(hits) - 1))
    exact_hit = next(h for h in hits if h.chunk.chunk_id == seeded["exact"])
    para_hit = next(h for h in hits if h.chunk.chunk_id == seeded["paraphrase"])
    assert exact_hit.bm25_rank == 1 and exact_hit.dense_rank is None
    assert para_hit.dense_rank == 1 and para_hit.bm25_rank is None
    assert hits[0].score == pytest.approx(1 / 61) and hits[1].score == pytest.approx(1 / 61)
    assert {hits[0].chunk.chunk_id, hits[1].chunk.chunk_id} == {
        seeded["exact"],
        seeded["paraphrase"],
    }
    # A hit found by exactly one ranking scores 1/(rrf_k + rank).
    single = [h for h in hits if (h.bm25_rank is None) != (h.dense_rank is None)]
    for h in single:
        rank = h.bm25_rank if h.bm25_rank is not None else h.dense_rank
        assert h.score == pytest.approx(1 / (60 + rank))


def test_doc_filter_respected(
    populated_store: DuckDBStore,
    corpus: tuple[list[Chunk], dict[str, str]],
    embedder: HashingEmbedder,
) -> None:
    _, seeded = corpus
    exact_doc = populated_store.get_chunks([seeded["exact"]])[0].doc_name
    other = next(d for d in DOCS if d != exact_doc)
    qvec = embedder.embed([PARAPHRASE_QUERY], kind="query")[0]

    bm25 = populated_store.search_bm25("revenue income cash", k=20, doc_filter=[other])
    assert len(bm25) == 20 and all(h.chunk.doc_name == other for h in bm25)
    # The seeded phrase exists only in exact_doc: filtering it out yields nothing, not a leak.
    assert populated_store.search_bm25(EXACT_PHRASE, k=20, doc_filter=[other]) == []
    assert (
        populated_store.search_bm25(EXACT_PHRASE, k=5, doc_filter=[exact_doc])[0].chunk.chunk_id
        == (seeded["exact"])
    )

    dense = populated_store.search_dense(qvec, k=20, doc_filter=[other, other])
    assert len(dense) == 20 and all(h.chunk.doc_name == other for h in dense)

    hybrid = populated_store.hybrid_search("revenue income cash", qvec, k=10, doc_filter=[other])
    assert len(hybrid) == 10 and all(h.chunk.doc_name == other for h in hybrid)

    assert populated_store.search_bm25(EXACT_PHRASE, k=5, doc_filter=[]) == []
    assert populated_store.search_dense(qvec, k=5, doc_filter=[]) == []
    assert populated_store.hybrid_search(EXACT_PHRASE, qvec, k=5, doc_filter=[]) == []


def test_bm25_auto_rebuilds_after_add_chunks(
    store_factory: Callable[..., DuckDBStore],
    corpus: tuple[list[Chunk], dict[str, str]],
    embedder: HashingEmbedder,
) -> None:
    store = store_factory()
    chunks, seeded = corpus
    store.add_chunks(chunks, embed_corpus(chunks, embedder, seeded))
    # No explicit rebuild_fts: a writable store rebuilds lazily on the first BM25 query.
    assert store.search_bm25(EXACT_PHRASE, k=1)[0].chunk.chunk_id == seeded["exact"]
    assert store.manifest().n_chunks == len(chunks)


def test_python_bm25_fallback_when_fts_extension_missing(
    monkeypatch: pytest.MonkeyPatch,
    corpus: tuple[list[Chunk], dict[str, str]],
    embedder: HashingEmbedder,
) -> None:
    monkeypatch.setattr(DuckDBStore, "_load_fts_extension", lambda self: False)
    chunks, seeded = corpus
    with DuckDBStore(":memory:") as store:
        assert store.bm25_backend == "python"
        store.init_schema(embedder.name, DIM)
        store.add_chunks(chunks, embed_corpus(chunks, embedder, seeded))
        store.rebuild_fts()
        hits = store.search_bm25(EXACT_PHRASE, k=3)
        assert hits[0].chunk.chunk_id == seeded["exact"] and hits[0].source == "bm25"
        exact_doc = hits[0].chunk.doc_name
        other = next(d for d in DOCS if d != exact_doc)
        filtered = store.search_bm25(EXACT_PHRASE, k=3, doc_filter=[other])
        assert all(h.chunk.doc_name == other for h in filtered)
        assert store.search_bm25("nomatchterm", k=3) == []
        qvec = embedder.embed([PARAPHRASE_QUERY], kind="query")[0]
        ids = {h.chunk.chunk_id for h in store.hybrid_search(EXACT_PHRASE, qvec, k=10)}
        assert {seeded["exact"], seeded["paraphrase"]} <= ids


def test_python_bm25_scores_by_hand() -> None:
    index = _PythonBM25(
        [("a", "D", "revenue revenue growth"), ("b", "D", "growth outlook"), ("c", "E", "cash")],
        k1=1.2,
        b=0.75,
    )
    ranked = index.search("revenue", k=10)
    assert [cid for cid, _ in ranked] == ["a"]
    # idf(revenue) = ln(1 + (3 - 1 + 0.5) / (1 + 0.5)); tf = 2, len = 3, avg_len = 2.
    idf = np.log(1 + 2.5 / 1.5)
    norm = 1.2 * (1 - 0.75 + 0.75 * 3 / 2)
    assert ranked[0][1] == pytest.approx(idf * 2 * 2.2 / (2 + norm))
    assert [cid for cid, _ in index.search("growth", k=10)] == ["b", "a"]
    assert index.search("growth", k=10, allowed_docs={"E"}) == []
    assert index.search("the and of", k=10) == []
    assert _PythonBM25([]).search("anything", k=3) == []


# ---- read-only connection -----------------------------------------------------------------


def test_readonly_connection_rejects_writes_on_writable_store(populated_store: DuckDBStore) -> None:
    conn = populated_store.readonly_connection()
    assert isinstance(conn, ReadOnlyConnection)
    assert conn.engine_read_only is False
    result = conn.execute("SELECT count(*) AS n FROM chunks")
    assert result.columns == ["n"] and result.fetchone() == (200,)
    assert conn.execute("SELECT doc_name FROM documents ORDER BY 1 LIMIT 2").fetchall() == [
        ("ACME_2022_10K",),
        ("ACME_2023_10K",),
    ]
    rows = conn.execute("SELECT doc_name FROM documents WHERE fiscal_year = ?", [2022])
    assert list(rows) == [("ACME_2022_10K",)] and len(rows) == 1
    assert rows.description[0][0] == "doc_name"
    for bad in (
        "INSERT INTO index_manifest VALUES ('x', 'y')",
        "DELETE FROM chunks",
        "UPDATE documents SET n_pages = 0",
        "CREATE TABLE z AS SELECT 1",
        "DROP TABLE chunks",
        "ATTACH ':memory:' AS other",
        "SET threads = 1",
        "COPY chunks TO '/tmp/out.csv'",
        "BEGIN TRANSACTION",
        "COMMIT",
        "SELECT 1; SELECT 2",
        "PRAGMA create_fts_index('chunks', 'chunk_id', 'text', overwrite=1)",
        "  -- comment\n PRAGMA database_size",
        "CALL pragma_version()",
        "",
    ):
        with pytest.raises(SqlRejected):
            conn.execute(bad)
    # The connection keeps working after rejections and after a runtime error.
    with pytest.raises(duckdb.Error):
        conn.execute("SELECT no_such_column FROM chunks")
    assert conn.execute("SELECT 1").fetchall() == [(1,)]
    assert populated_store.counts()["chunks"] == 200
    conn.close()
    conn.close()  # idempotent
    with pytest.raises(ConfigError, match="closed"):
        conn.execute("SELECT 1")


def test_readonly_connection_engine_level_guard(
    tmp_duckdb_path: Path,
    corpus: tuple[list[Chunk], dict[str, str]],
    embedder: HashingEmbedder,
) -> None:
    """Even if the statement check were bypassed, the engine refuses writes."""
    chunks, seeded = corpus
    with DuckDBStore(tmp_duckdb_path, embed_dim=DIM) as store:
        store.init_schema(embedder.name, DIM)
        store.add_chunks(chunks[:60], embed_corpus(chunks, embedder, seeded)[:60])
        store.rebuild_fts()
    with DuckDBStore(tmp_duckdb_path, read_only=True) as ro_store:
        conn = ro_store.readonly_connection()
        assert conn.engine_read_only is True
        assert conn.execute("SELECT count(*) FROM chunks").fetchone() == (60,)
        # Read-only stores can still run BM25 against the persisted FTS index.
        assert ro_store.search_bm25(EXACT_PHRASE, k=1)[0].chunk.chunk_id == seeded["exact"]
        with pytest.raises(ConfigError, match="read-only"):
            ro_store.rebuild_fts()
        with pytest.raises(ConfigError, match="read-only"):
            ro_store.set_manifest(inputs_sha256="x")
        with pytest.raises(ConfigError, match="read-only"):
            ro_store.add_pages([Page(doc_name="D", page_num=1, text="x")])
        # Bypass the parser-level check on purpose: the read-only transaction still refuses.
        raw = conn._cursor
        raw.execute("BEGIN TRANSACTION READ ONLY")
        with pytest.raises(duckdb.Error, match="read-only"):
            raw.execute("INSERT INTO index_manifest VALUES ('x', 'y')")
        raw.execute("ROLLBACK")


def test_extension_autoload_is_off_on_every_store(populated_store: DuckDBStore) -> None:
    """Extension auto-install / auto-load are off even on writable stores.

    The sqlglot guard denies functions by name, so a scalar call to a function of an extension
    that is not loaded reached DuckDB's binder, which auto-installed the extension from
    extensions.duckdb.org (network egress, ~60 MB on disk and in RAM) before failing. Now the
    binder answers with a catalog error and nothing is installed or loaded.
    """
    assert populated_store.external_access is True
    raw = populated_store.readonly_connection()._cursor
    settings = dict(
        raw.execute(
            "SELECT name, value FROM duckdb_settings() WHERE name IN "
            "('autoinstall_known_extensions', 'autoload_known_extensions', "
            "'enable_external_access', 'lock_configuration')"
        ).fetchall()
    )
    assert settings == {
        "autoinstall_known_extensions": "false",
        "autoload_known_extensions": "false",
        "enable_external_access": "true",  # ingest registers frames and export_parquet COPYs
        "lock_configuration": "false",  # create_fts_index needs SET; see _harden_configuration
    }
    with pytest.raises(duckdb.CatalogException, match="spatial extension"):
        raw.execute("SELECT st_point(1, 2)")
    assert raw.execute(
        "SELECT count(*) FROM duckdb_functions() WHERE function_name = 'st_point'"
    ).fetchone() == (0,)
    assert raw.execute(
        "SELECT count(*) FROM duckdb_extensions() WHERE extension_name = 'spatial' AND loaded"
    ).fetchone() == (0,)
    # fts was loaded before the lock and keeps working.
    assert raw.execute("SELECT stem('running', 'porter')").fetchone() == ("run",)


def test_read_only_store_is_sandboxed_and_locked(
    tmp_duckdb_path: Path,
    tmp_path: Path,
    corpus: tuple[list[Chunk], dict[str, str]],
    embedder: HashingEmbedder,
) -> None:
    """The serving store (read_only=True) cannot reach the file system or the network from SQL,
    and no ``SET`` on any cursor can give that back."""
    chunks, seeded = corpus
    with DuckDBStore(tmp_duckdb_path, embed_dim=DIM) as store:
        store.init_schema(embedder.name, DIM)
        store.add_chunks(chunks[:60], embed_corpus(chunks, embedder, seeded)[:60])
        store.rebuild_fts()
    with DuckDBStore(tmp_duckdb_path, read_only=True) as ro_store:
        assert ro_store.external_access is False
        raw = ro_store.readonly_connection()._cursor
        assert raw.execute("SELECT current_setting('enable_external_access')").fetchone() == (
            False,
        )
        assert raw.execute("SELECT current_setting('lock_configuration')").fetchone() == (True,)
        with pytest.raises(duckdb.PermissionException, match="disabled by configuration"):
            raw.execute("SELECT * FROM read_csv('/etc/hosts')")
        with pytest.raises(duckdb.PermissionException, match="disabled by configuration"):
            ro_store.export_parquet(tmp_path / "parquet")
        for statement in (
            "SET enable_external_access = true",
            "SET autoload_known_extensions = true",
            "SET autoinstall_known_extensions = true",
            "SET lock_configuration = false",
        ):
            with pytest.raises(duckdb.InvalidInputException, match="locked"):
                raw.execute(statement)
        with pytest.raises(duckdb.CatalogException, match="spatial extension"):
            raw.execute("SELECT st_point(1, 2)")
        # Reads, per-thread cursors and the persisted FTS index are unaffected.
        assert ro_store.readonly_connection().execute("SELECT count(*) FROM chunks").fetchone() == (
            60,
        )
        assert ro_store.search_bm25(EXACT_PHRASE, k=1)[0].chunk.chunk_id == seeded["exact"]
    # `secqa export` opts back in to the file system explicitly; auto-load stays off and locked.
    with DuckDBStore(tmp_duckdb_path, read_only=True, external_access=True) as export_store:
        export_store.export_parquet(tmp_path / "parquet")
        raw = export_store.readonly_connection()._cursor
        assert raw.execute("SELECT current_setting('lock_configuration')").fetchone() == (True,)
        with pytest.raises(duckdb.CatalogException, match="spatial extension"):
            raw.execute("SELECT st_point(1, 2)")
    assert (tmp_path / "parquet" / "chunks.parquet").is_file()


def test_readonly_result_fetch_helpers(populated_store: DuckDBStore) -> None:
    conn = populated_store.readonly_connection()
    result = conn.execute("SELECT page_num FROM pages WHERE doc_name = 'ACME_2022_10K' ORDER BY 1")
    assert result.fetchmany(2) == [(1,), (2,)]
    assert result.fetchone() == (3,)
    assert result.fetchall() == [(4,), (5,)]
    assert result.fetchone() is None
    assert result.fetchall() == []
    with pytest.raises(ValueError):
        result.fetchmany(-1)
    with conn as managed:
        assert managed.execute("SELECT 2").fetchone() == (2,)


# ---- export / lifecycle -------------------------------------------------------------------


def test_export_parquet(populated_store: DuckDBStore, tmp_path: Path) -> None:
    out = tmp_path / "export it's here"
    populated_store.export_parquet(out)
    for table in ("documents", "pages", "chunks", "xbrl_facts", "index_manifest"):
        assert (out / f"{table}.parquet").is_file()
    check = duckdb.connect()
    path = str(out / "chunks.parquet").replace("'", "''")
    n, width = check.execute(
        f"SELECT count(*), len(embedding) FROM read_parquet('{path}') GROUP BY 2"
    ).fetchone()
    assert (n, width) == (200, DIM)
    check.close()
    with DuckDBStore(":memory:") as fresh:
        with pytest.raises(ConfigError, match="init_schema"):
            fresh.export_parquet(tmp_path / "nope")


def test_close_is_idempotent_and_blocks_use(store_factory: Callable[..., DuckDBStore]) -> None:
    store = store_factory()
    store.close()
    store.close()
    with pytest.raises(ConfigError, match="closed"):
        store.counts()


def test_financials_view_dedupes_restatements_and_keeps_annual_periods(
    store_factory: Callable[..., DuckDBStore],
) -> None:
    store = store_factory()

    def fact(
        fy: int,
        fp: str,
        form: str,
        start: str | None,
        end: str,
        val: float,
        accn: str,
        frame: str | None,
    ) -> tuple[object, ...]:
        filed = "2023-02-01" if accn == "a-2" else "2022-02-01" if accn == "a-1" else "2022-11-01"
        return ("0000000001", "ACME", fy, fp, form, start, end, val, accn, filed, frame)

    store.conn.executemany(
        "INSERT INTO xbrl_facts VALUES "
        "(?, ?, 'us-gaap', 'Revenues', 'USD', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # FY2021 revenue as first reported (10-K FY2021) and restated in the FY2022 10-K.
            fact(2021, "FY", "10-K", "2021-01-01", "2021-12-31", 100.0, "a-1", "CY2021"),
            fact(2022, "FY", "10-K", "2021-01-01", "2021-12-31", 101.0, "a-2", None),
            # FY2022 annual value.
            fact(2022, "FY", "10-K", "2022-01-01", "2022-12-31", 120.0, "a-2", "CY2022"),
            # A quarterly value from a 10-Q and a 9-month YTD from the 10-K are not annual.
            fact(2022, "Q3", "10-Q", "2022-07-01", "2022-09-30", 30.0, "q-1", None),
            fact(2022, "FY", "10-K", "2022-01-01", "2022-09-30", 90.0, "a-2", None),
            # An instant (balance-sheet style) fact at year end is kept as period_kind='instant'.
            fact(2022, "FY", "10-K", None, "2022-12-31", 500.0, "a-2", "CY2022Q4I"),
        ],
    )
    rows = store.conn.execute(
        "SELECT fiscal_year, period_kind, val, accn FROM financials "
        "ORDER BY fiscal_year, period_kind"
    ).fetchall()
    assert rows == [
        (2021, "duration", 101.0, "a-2"),
        (2022, "duration", 120.0, "a-2"),
        (2022, "instant", 500.0, "a-2"),
    ]
    assert store.counts()["facts"] == 6


def test_replaced_financials_view_survives_reopen(tmp_duckdb_path: Path) -> None:
    """secqa.xbrl owns the final `financials` definition; re-running init_schema must keep it."""
    store = DuckDBStore(tmp_duckdb_path, embed_dim=8)
    store.init_schema("hashing", 8)
    store.conn.execute(
        "CREATE OR REPLACE VIEW financials AS SELECT 'ACME' AS ticker, 1.0 AS revenue"
    )
    store.init_schema("hashing", 8)  # same process
    assert store.conn.execute("SELECT ticker, revenue FROM financials").fetchall() == [
        ("ACME", 1.0)
    ]
    store.close()

    reopened = DuckDBStore(tmp_duckdb_path, embed_dim=8)
    reopened.init_schema("hashing", 8)  # fresh open of the same file
    assert reopened.conn.execute("SELECT ticker FROM financials").fetchall() == [("ACME",)]
    reopened.close()


def test_corpus_fixture_is_the_expected_size(corpus: tuple[list[Chunk], dict[str, str]]) -> None:
    chunks, seeded = corpus
    assert len(chunks) == 200
    assert len({c.chunk_id for c in chunks}) == 200
    assert set(seeded) == {"exact", "paraphrase"}
    assert make_corpus() == corpus  # deterministic
    assert datetime.now(tz=UTC).tzinfo is UTC


# ---- thread safety ----------------------------------------------------------------------------


def test_conn_is_per_thread_and_concurrent_reads_do_not_cross(populated_store: DuckDBStore) -> None:
    """Eight threads read distinct documents through one store; every result matches its own
    filter. DuckDB connections are not thread-safe, so ``conn`` must hand each thread its own
    cursor (a shared connection returns another thread's rows a few times per thousand calls)."""
    n_threads, iterations = 8, 150
    conns: dict[int, list[int]] = {}
    failures: list[str] = []
    barrier = threading.Barrier(n_threads)

    def worker(doc_name: str) -> None:
        barrier.wait()
        conns[threading.get_ident()] = [id(populated_store.conn), id(populated_store.conn)]
        for _ in range(iterations):
            try:
                docs = [d.doc_name for d in populated_store.list_documents(doc_names=[doc_name])]
                pages = [p.doc_name for p in populated_store.get_pages(doc_name, [1])]
            except Exception as exc:  # noqa: BLE001 - the failure mode under test is any error
                failures.append(f"{doc_name}: {type(exc).__name__}: {exc}")
                return
            if docs != [doc_name] or pages != [doc_name]:
                failures.append(f"{doc_name}: got documents {docs} pages {pages}")
                return
        hits = populated_store.search_bm25("revenue income", k=5, doc_filter=[doc_name])
        if {h.chunk.doc_name for h in hits} != {doc_name}:
            failures.append(f"{doc_name}: bm25 hits from {[h.chunk.doc_name for h in hits]}")

    threads = [
        threading.Thread(target=worker, args=(DOCS[i % len(DOCS)],)) for i in range(n_threads)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len(conns) == n_threads
    # Each thread's connection is stable within the thread and distinct from every other one.
    assert all(pair[0] == pair[1] for pair in conns.values())
    assert len({pair[0] for pair in conns.values()}) == n_threads
    assert all(pair[0] != id(populated_store.conn) for pair in conns.values())


def test_close_releases_thread_cursors(store_factory: Callable[..., DuckDBStore]) -> None:
    store = store_factory()
    cursors: list[duckdb.DuckDBPyConnection] = []

    def worker() -> None:
        cursors.append(store.conn)
        assert store.conn is cursors[-1]

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert len(cursors) == 1 and cursors[0] is not store.conn
    store.close()
    with pytest.raises(ConfigError, match="closed"):
        store.conn.execute("SELECT 1")
    with pytest.raises(duckdb.Error):
        cursors[0].execute("SELECT 1")
    store.close()
