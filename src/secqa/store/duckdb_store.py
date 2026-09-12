"""Single-file DuckDB index: documents, pages, chunks (+ embeddings), XBRL facts, manifest.

Design notes (SPEC 4.2):

* The ``chunks.embedding`` column is ``FLOAT[dim]`` with ``dim`` taken from the embedder at
  :meth:`DuckDBStore.init_schema`; the value is recorded in ``index_manifest`` and re-checked on
  every open, so a query vector from a different model raises :class:`IndexMismatch` instead of
  silently returning garbage.
* BM25 uses DuckDB's ``fts`` extension (porter stemmer, English stopwords), rebuilt after every
  ingest batch because the FTS index is not maintained incrementally. When the extension cannot
  be loaded (no network to fetch it on first use) the store falls back to an in-process Okapi
  BM25 over the same texts, logs a warning, and reports ``bm25_backend='python'``.
* Dense search is a brute-force ``array_cosine_similarity`` scan (~40k rows, tens of ms).
* Hybrid search fuses the two rankings with reciprocal-rank fusion (:func:`secqa.store.fusion.rrf`).
* :meth:`DuckDBStore.readonly_connection` hands out a :class:`ReadOnlyConnection` for the SQL
  tool: every statement must parse as exactly one ``SELECT`` and runs inside its own
  ``BEGIN TRANSACTION READ ONLY``, which the engine enforces. DuckDB cannot open one file both
  read-write and read-only in one process, and ``enable_external_access`` is a global setting,
  so this per-statement transaction is the strongest per-connection guarantee available; the
  XBRL module adds a sqlglot allowlist (tables, no table functions) on top.
* The engine configuration is hardened once per store, right after ``fts`` is loaded
  (:meth:`DuckDBStore._harden_configuration`): extension auto-install / auto-load are off on
  every store, so a query naming a function of an extension that is not loaded is a catalog
  error rather than a download from ``extensions.duckdb.org`` plus a native library loaded into
  the process. Read-only (serving) stores additionally disable ``enable_external_access`` (no
  file-system or network access from SQL at all; ``secqa export`` opts back in with
  ``external_access=True`` for ``COPY TO``) and lock the configuration so no later ``SET`` can
  undo any of it. Writable stores are not locked: DuckDB's ``create_fts_index`` macro sets a
  session option internally and fails under ``lock_configuration``.

A DuckDB connection is not thread-safe (two threads interleaving ``execute`` and ``fetch`` on
one connection read each other's result sets), so :attr:`DuckDBStore.conn` is per thread: the
thread that opened the store gets the root connection and every other thread gets its own
cursor (an independent connection to the same database, created on first use and closed by
:meth:`DuckDBStore.close`). One store therefore serves a whole request threadpool; in-process
Python state (``_fts_stale``, the python BM25 fallback) is only mutated by the single-threaded
ingest path.
"""

from __future__ import annotations

import math
import re
import threading
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np

from secqa.core.contracts import Chunk, DocumentMeta, Hit, Page
from secqa.core.errors import ConfigError, IndexMismatch, SqlRejected
from secqa.core.logging import get_logger
from secqa.store.fusion import rrf
from secqa.store.manifest import (
    MANIFEST_KEYS,
    IndexManifest,
    resolve_git_sha,
    serialize_value,
    utc_now,
)

log = get_logger(__name__)

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")
_MEMORY = ":memory:"
_FTS_SCHEMA = "fts_main_chunks"
_EXPORT_TABLES: tuple[str, ...] = ("documents", "pages", "chunks", "xbrl_facts", "index_manifest")

_CHUNK_COLUMNS = "chunk_id, doc_name, page_num, chunk_idx, section, text, n_tokens"
_DOCUMENT_COLUMNS = (
    "doc_name, ticker, cik, company, form, fiscal_year, period_end, source_kind, "
    "source_url, source_sha256, n_pages, ingested_at"
)

# Engine options pinned on every store; read-only stores also disable ``enable_external_access``
# (unless opened with ``external_access=True``) and lock the configuration.
# See :meth:`DuckDBStore._harden_configuration`.
_HARDENED_SETTINGS: tuple[tuple[str, bool], ...] = (
    ("autoinstall_known_extensions", False),
    ("autoload_known_extensions", False),
)

# Leading keywords that DuckDB's parser classifies as SELECT but which are not plain reads.
_DENIED_LEADING_KEYWORDS = frozenset(
    {"PRAGMA", "CALL", "SET", "RESET", "EXPORT", "IMPORT", "CHECKPOINT", "FORCE", "VACUUM"}
)
_LEADING_COMMENT_RE = re.compile(r"^(?:\s*(?:--[^\n]*\n|/\*.*?\*/))*\s*", re.DOTALL)


# ---------------------------------------------------------------------------------------------
# Read-only connection for the SQL tool
# ---------------------------------------------------------------------------------------------


class ReadOnlyResult:
    """Materialised rows of one read-only statement (DB-API style ``fetch*`` accessors)."""

    def __init__(self, columns: list[str], rows: list[tuple[Any, ...]]):
        self.columns = columns
        self._rows = rows
        self._pos = 0

    @property
    def description(self) -> list[tuple[Any, ...]]:
        """DB-API ``description``: one ``(name, None, ...)`` tuple per column."""
        return [(name, None, None, None, None, None, None) for name in self.columns]

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Return every remaining row."""
        rows = self._rows[self._pos :]
        self._pos = len(self._rows)
        return rows

    def fetchone(self) -> tuple[Any, ...] | None:
        """Return the next row or ``None`` when exhausted."""
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchmany(self, size: int = 1) -> list[tuple[Any, ...]]:
        """Return up to ``size`` next rows."""
        if size < 0:
            raise ValueError("size must be non-negative")
        rows = self._rows[self._pos : self._pos + size]
        self._pos += len(rows)
        return rows

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        return iter(self._rows)

    def __len__(self) -> int:
        return len(self._rows)


class ReadOnlyConnection:
    """A DuckDB cursor that can only run single ``SELECT`` statements.

    Two layers: the statement is parsed with ``duckdb.extract_statements`` and rejected
    (:class:`SqlRejected`) unless it is exactly one SELECT-type statement whose leading keyword is
    not ``PRAGMA``/``CALL``/``SET``...; it then executes inside ``BEGIN TRANSACTION READ ONLY`` so
    the engine itself refuses any write that slipped through. Results are materialised before the
    transaction commits, so no transaction state outlives :meth:`execute`.
    """

    def __init__(self, cursor: duckdb.DuckDBPyConnection, *, engine_read_only: bool):
        self._cursor = cursor
        self.engine_read_only = engine_read_only  # True when the whole file is opened read-only
        self._closed = False

    def check_statement(self, sql: str) -> None:
        """Raise :class:`SqlRejected` unless ``sql`` is exactly one plain SELECT statement."""
        if not sql or not sql.strip():
            raise SqlRejected("empty statement")
        # Parse on this cursor: the module-level duckdb.extract_statements binds against the
        # default in-memory database, which does not know our tables.
        statements = self._cursor.extract_statements(sql)
        if len(statements) != 1:
            raise SqlRejected(f"expected exactly one statement, got {len(statements)}")
        statement_type = statements[0].type
        if statement_type != duckdb.StatementType.SELECT:
            raise SqlRejected(f"only SELECT statements are allowed, got {statement_type.name}")
        body = _LEADING_COMMENT_RE.sub("", sql, count=1)
        leading = re.match(r"[A-Za-z_]+", body)
        keyword = leading.group(0).upper() if leading else ""
        if keyword in _DENIED_LEADING_KEYWORDS:
            raise SqlRejected(f"{keyword} statements are not allowed on a read-only connection")

    def execute(self, sql: str, parameters: Sequence[Any] | None = None) -> ReadOnlyResult:
        """Run one SELECT inside a read-only transaction and return its materialised rows."""
        if self._closed:
            raise ConfigError("read-only connection is closed")
        self.check_statement(sql)
        cursor = self._cursor
        cursor.execute("BEGIN TRANSACTION READ ONLY")
        try:
            cursor.execute(sql, parameters if parameters is not None else [])
            description = cursor.description or []
            columns = [column[0] for column in description]
            rows = cursor.fetchall()
            cursor.execute("COMMIT")
        except BaseException:
            try:
                cursor.execute("ROLLBACK")
            except duckdb.Error:  # pragma: no cover - rollback of an already-dead transaction
                log.debug("rollback_failed")
            raise
        return ReadOnlyResult(columns, rows)

    def interrupt(self) -> None:
        """Interrupt the statement currently running on this cursor (from another thread)."""
        self._cursor.interrupt()

    def close(self) -> None:
        """Close the underlying cursor; idempotent."""
        if not self._closed:
            self._closed = True
            self._cursor.close()

    def __enter__(self) -> ReadOnlyConnection:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------------------------
# In-process BM25 fallback (only when the DuckDB fts extension cannot be loaded)
# ---------------------------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have if in into is it its of on or such that "
    "the their then there these they this to was were will with".split()
)


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS]


class _PythonBM25:
    """Okapi BM25 (k1=1.2, b=0.75, Lucene idf) over ``(chunk_id, doc_name, text)`` rows.

    No stemming, unlike DuckDB's porter-stemmed index; this is a degraded offline fallback and
    the store logs which backend answered a query.
    """

    def __init__(self, rows: Sequence[tuple[str, str, str]], k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.ids: list[str] = []
        self.doc_names: list[str] = []
        self._tf: list[dict[str, int]] = []
        self._len: list[int] = []
        df: dict[str, int] = {}
        for chunk_id, doc_name, text in rows:
            tokens = _tokenize(text)
            counts: dict[str, int] = {}
            for tok in tokens:
                counts[tok] = counts.get(tok, 0) + 1
            for tok in counts:
                df[tok] = df.get(tok, 0) + 1
            self.ids.append(chunk_id)
            self.doc_names.append(doc_name)
            self._tf.append(counts)
            self._len.append(len(tokens))
        n = len(self.ids)
        self._avg_len = (sum(self._len) / n) if n else 0.0
        self._idf = {tok: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for tok, d in df.items()}

    def search(
        self, query: str, k: int, allowed_docs: set[str] | None = None
    ) -> list[tuple[str, float]]:
        """Top-``k`` ``(chunk_id, score)`` for chunks sharing at least one query term."""
        terms = [tok for tok in _tokenize(query) if tok in self._idf]
        if not terms:
            return []
        scored: list[tuple[str, float]] = []
        for i, counts in enumerate(self._tf):
            if allowed_docs is not None and self.doc_names[i] not in allowed_docs:
                continue
            norm = self.k1 * (1.0 - self.b + self.b * self._len[i] / (self._avg_len or 1.0))
            score = 0.0
            for tok in terms:
                tf = counts.get(tok)
                if tf:
                    score += self._idf[tok] * tf * (self.k1 + 1.0) / (tf + norm)
            if score > 0.0:
                scored.append((self.ids[i], score))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]


# ---------------------------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------------------------


class DuckDBStore:
    """One DuckDB file (or ``':memory:'``) holding the whole index. See the module docstring."""

    def __init__(
        self,
        path: Path | str = _MEMORY,
        embed_dim: int | None = None,
        read_only: bool = False,
        external_access: bool | None = None,
    ):
        """Open (or create) the store.

        Args:
            path: DuckDB file path or ``':memory:'``. Parent directories are created.
            embed_dim: expected embedding width; checked against an existing manifest
                (:class:`IndexMismatch` on disagreement) and otherwise remembered until
                :meth:`init_schema` confirms it.
            read_only: open the file read-only (serving / tools). Requires an existing,
                initialised file.
            external_access: whether SQL on this store may reach the file system and the
                network (``COPY TO``, ``read_csv``, ``INSTALL`` ...). Defaults to
                ``not read_only``: ingest needs it, serving must not have it. ``secqa export``
                opens read-only with ``external_access=True`` because ``COPY TO`` writes files.
        """
        self.path: str = _MEMORY if str(path) == _MEMORY else str(Path(path))
        self.read_only = read_only
        self.external_access: bool = (not read_only) if external_access is None else external_access
        if embed_dim is not None and embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if self.path == _MEMORY:
            if read_only:
                raise ConfigError("an in-memory store cannot be opened read-only")
        else:
            file_path = Path(self.path)
            if read_only and not file_path.exists():
                raise ConfigError(f"cannot open {self.path} read-only: file does not exist")
            file_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: duckdb.DuckDBPyConnection | None = duckdb.connect(
            self.path, read_only=read_only
        )
        self._owner_thread = threading.get_ident()
        self._thread_local = threading.local()
        self._cursors: list[duckdb.DuckDBPyConnection] = []
        self._cursors_lock = threading.Lock()
        self._fts_loaded = self._load_fts_extension()
        try:
            self._harden_configuration()
        except duckdb.Error as exc:
            self.close()
            raise ConfigError(
                f"could not harden the DuckDB configuration of {self.path}: {exc}"
            ) from exc
        self._fts_stale = False
        self._py_bm25: _PythonBM25 | None = None
        self._dim: int | None = None
        self._embedder_name: str | None = None

        stored = self._read_manifest_rows() if self._table_exists("index_manifest") else {}
        stored_dim = stored.get("dim")
        if stored_dim is not None:
            self._dim = int(stored_dim)
            self._embedder_name = stored.get("embedder")
            if embed_dim is not None and embed_dim != self._dim:
                self.close()
                raise IndexMismatch(
                    f"{self.path} was built with embedding dim {self._dim} "
                    f"({self._embedder_name}); requested dim {embed_dim}"
                )
        elif embed_dim is not None:
            self._dim = embed_dim
        if read_only and stored_dim is None:
            self.close()
            raise ConfigError(f"{self.path} has no index manifest; build the index first")
        log.debug(
            "store_opened",
            path=self.path,
            read_only=read_only,
            dim=self._dim,
            bm25_backend=self.bm25_backend,
        )

    # ---- properties -------------------------------------------------------------------------

    @property
    def dim(self) -> int:
        """Embedding width of this store (raises :class:`ConfigError` before ``init_schema``)."""
        if self._dim is None:
            raise ConfigError("embedding dim unknown: call init_schema first")
        return self._dim

    @property
    def embedder_name(self) -> str | None:
        """Embedder recorded in the manifest, or ``None`` before ``init_schema``."""
        return self._embedder_name

    @property
    def bm25_backend(self) -> str:
        """``'duckdb_fts'`` when the fts extension is loaded, else ``'python'``."""
        return "duckdb_fts" if self._fts_loaded else "python"

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """The calling thread's connection (raises :class:`ConfigError` after :meth:`close`).

        The thread that opened the store gets the root connection; any other thread gets a
        cursor of its own, created on first use and reused for that thread's lifetime. DuckDB
        cursors are independent connections to the same database, so concurrent threads never
        share a result set (see the module docstring).
        """
        root = self._conn
        if root is None:
            raise ConfigError("store is closed")
        if threading.get_ident() == self._owner_thread:
            return root
        cursor: duckdb.DuckDBPyConnection | None = getattr(self._thread_local, "cursor", None)
        if cursor is None:
            cursor = root.cursor()
            self._thread_local.cursor = cursor
            with self._cursors_lock:
                self._cursors.append(cursor)
            log.debug("store_thread_cursor_opened", path=self.path, thread=threading.get_ident())
        return cursor

    # ---- schema / manifest ------------------------------------------------------------------

    def init_schema(self, embedder_name: str, embed_dim: int) -> None:
        """Create tables from ``schema.sql`` with ``{dim}`` substituted and write the manifest.

        Idempotent on an existing store built with the same embedder and dimension; raises
        :class:`IndexMismatch` when either differs and :class:`ConfigError` when read-only.
        """
        self._require_writable("init_schema")
        if not embedder_name or not embedder_name.strip():
            raise ValueError("embedder_name must not be empty")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if self._dim is not None and self._dim != embed_dim:
            raise IndexMismatch(
                f"store dim is {self._dim}; cannot initialise with embed_dim={embed_dim}"
            )
        existing = self._read_manifest_rows() if self._table_exists("index_manifest") else {}
        if existing.get("dim") is not None:
            stored_embedder = existing.get("embedder")
            stored_dim = int(existing["dim"] or 0)
            if stored_embedder != embedder_name or stored_dim != embed_dim:
                raise IndexMismatch(
                    f"{self.path} was built with embedder={stored_embedder!r} dim={stored_dim}; "
                    f"requested embedder={embedder_name!r} dim={embed_dim}"
                )
            self._run_schema(
                embed_dim
            )  # no-op CREATE IF NOT EXISTS (a replaced financials view survives)
            self._dim, self._embedder_name = embed_dim, embedder_name
            log.debug("schema_already_initialised", path=self.path, dim=embed_dim)
            return
        if self._table_exists("chunks"):
            actual = self._chunks_embedding_dim()
            if actual is not None and actual != embed_dim:
                raise IndexMismatch(
                    f"chunks.embedding is FLOAT[{actual}] but embed_dim={embed_dim} requested"
                )
        self._run_schema(embed_dim)
        manifest = IndexManifest(
            embedder=embedder_name,
            dim=embed_dim,
            git_sha=resolve_git_sha(),
            built_at=utc_now(),
            n_documents=0,
            n_pages=0,
            n_chunks=0,
            n_facts=0,
            inputs_sha256="",
            dataset_revision=None,
        )
        self.conn.executemany(
            "INSERT OR REPLACE INTO index_manifest (key, value) VALUES (?, ?)", manifest.to_rows()
        )
        self._dim, self._embedder_name = embed_dim, embedder_name
        log.info(
            "schema_initialised",
            path=self.path,
            embedder=embedder_name,
            dim=embed_dim,
            git_sha=manifest.git_sha,
            bm25_backend=self.bm25_backend,
        )

    def manifest(self) -> IndexManifest:
        """Read the manifest; raises :class:`ConfigError` if the store was never initialised."""
        if not self._table_exists("index_manifest"):
            raise ConfigError("index_manifest table missing: call init_schema first")
        try:
            return IndexManifest.from_rows(self._read_manifest_rows())
        except KeyError as exc:
            raise ConfigError(str(exc)) from exc

    def set_manifest(self, **kv: Any) -> None:
        """Update manifest keys (``git_sha``, ``inputs_sha256``, ``n_chunks`` ...).

        Only keys of :class:`IndexManifest` are accepted; ``embedder`` and ``dim`` cannot be
        changed here because they define the index (raise :class:`IndexMismatch`).
        """
        self._require_writable("set_manifest")
        if not self._table_exists("index_manifest"):
            raise ConfigError("index_manifest table missing: call init_schema first")
        rows: list[tuple[str, str | None]] = []
        for key, value in kv.items():
            if key not in MANIFEST_KEYS:
                raise ValueError(f"unknown manifest key {key!r}; expected one of {MANIFEST_KEYS}")
            if key in ("embedder", "dim"):
                current = self._embedder_name if key == "embedder" else self._dim
                if serialize_value(key, value) != serialize_value(key, current):
                    raise IndexMismatch(f"manifest {key} is fixed at init_schema (got {value!r})")
            rows.append((key, serialize_value(key, value)))
        if rows:
            self.conn.executemany(
                "INSERT OR REPLACE INTO index_manifest (key, value) VALUES (?, ?)", rows
            )

    def counts(self) -> dict[str, int]:
        """Live row counts: ``{'documents', 'pages', 'chunks', 'facts'}``."""
        return {
            "documents": self._count("documents"),
            "pages": self._count("pages"),
            "chunks": self._count("chunks"),
            "facts": self._count("xbrl_facts"),
        }

    # ---- writes -----------------------------------------------------------------------------

    def upsert_document(self, meta: DocumentMeta) -> None:
        """Insert or replace one ``documents`` row keyed by ``doc_name``."""
        self._require_writable("upsert_document")
        self.conn.execute(
            f"INSERT OR REPLACE INTO documents ({_DOCUMENT_COLUMNS}) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                meta.doc_name,
                meta.ticker,
                meta.cik,
                meta.company,
                meta.form,
                meta.fiscal_year,
                meta.period_end,
                meta.source_kind,
                meta.source_url,
                meta.source_sha256,
                meta.n_pages,
                _to_naive_utc(meta.ingested_at),
            ],
        )

    def add_pages(self, pages: list[Page]) -> None:
        """Insert or replace page texts (idempotent on ``(doc_name, page_num)``)."""
        self._require_writable("add_pages")
        if not pages:
            return
        for page in pages:
            if page.page_num < 1:
                raise ValueError(f"page_num is 1-based, got {page.page_num} in {page.doc_name}")
        self.conn.executemany(
            "INSERT OR REPLACE INTO pages (doc_name, page_num, text) VALUES (?, ?, ?)",
            [(page.doc_name, page.page_num, page.text) for page in pages],
        )

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Replace every chunk of the documents present in ``chunks`` with the given batch.

        ``embeddings`` must be shaped ``(len(chunks), dim)``; rows are cast to float32. The
        delete + insert runs in one transaction, and the FTS index is marked stale until
        :meth:`rebuild_fts` (called automatically by the next BM25 search on a writable store).
        """
        self._require_writable("add_chunks")
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape != (len(chunks), self.dim):
            raise ValueError(
                f"embeddings must have shape ({len(chunks)}, {self.dim}), got {matrix.shape}"
            )
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk_ids in one batch must be unique")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("embeddings contain NaN or inf")
        import pandas as pd  # declared dependency; imported lazily (slow import, bulk path only)

        doc_names = sorted({chunk.doc_name for chunk in chunks})
        frame = pd.DataFrame(
            {
                "chunk_id": ids,
                "doc_name": [chunk.doc_name for chunk in chunks],
                "page_num": [chunk.page_num for chunk in chunks],
                "chunk_idx": [chunk.chunk_idx for chunk in chunks],
                "section": [chunk.section for chunk in chunks],
                "text": [chunk.text for chunk in chunks],
                "n_tokens": [chunk.n_tokens for chunk in chunks],
                "embedding": list(matrix),
            }
        )
        conn = self.conn
        conn.register("_secqa_chunk_batch", frame)
        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute(
                f"DELETE FROM chunks WHERE doc_name IN ({_placeholders(len(doc_names))})",
                doc_names,
            )
            conn.execute(
                f"INSERT INTO chunks ({_CHUNK_COLUMNS}, embedding) "
                f"SELECT {_CHUNK_COLUMNS}, embedding::FLOAT[{self.dim}] FROM _secqa_chunk_batch"
            )
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.unregister("_secqa_chunk_batch")
        self._fts_stale = True
        self._py_bm25 = None
        log.info("chunks_added", n_chunks=len(chunks), n_documents=len(doc_names))

    def delete_document(self, doc_name: str) -> bool:
        """Remove one document with all its pages and chunks in one transaction.

        Returns ``True`` when a ``documents`` row existed. Pages and chunks are deleted even
        when the document row is missing (an interrupted ingest writes the row last), and the
        BM25 indexes are marked stale so deleted chunks can never be returned by a search.
        XBRL facts are keyed by CIK, not by document, and are untouched.
        """
        self._require_writable("delete_document")
        conn = self.conn
        conn.execute("BEGIN TRANSACTION")
        try:
            chunk_rows = conn.execute(
                "DELETE FROM chunks WHERE doc_name = ? RETURNING chunk_id", [doc_name]
            ).fetchall()
            conn.execute("DELETE FROM pages WHERE doc_name = ?", [doc_name])
            doc_rows = conn.execute(
                "DELETE FROM documents WHERE doc_name = ? RETURNING doc_name", [doc_name]
            ).fetchall()
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        if chunk_rows:
            self._fts_stale = True
            self._py_bm25 = None
        log.info(
            "document_deleted",
            doc_name=doc_name,
            existed=bool(doc_rows),
            n_chunks=len(chunk_rows),
        )
        return bool(doc_rows)

    def rebuild_fts(self) -> None:
        """Rebuild the BM25 index over ``chunks.text`` and refresh manifest counts.

        DuckDB's FTS index is a snapshot, so this must run after every ingest batch (SPEC 4.2).
        """
        self._require_writable("rebuild_fts")
        started = time.perf_counter()
        if self._fts_loaded:
            self.conn.execute(
                "PRAGMA create_fts_index('chunks', 'chunk_id', 'text', "
                "stemmer='porter', stopwords='english', overwrite=1)"
            )
            self._py_bm25 = None
        else:
            self._py_bm25 = self._build_python_bm25()
        self._fts_stale = False
        counts = self.counts()
        self.set_manifest(
            n_documents=counts["documents"],
            n_pages=counts["pages"],
            n_chunks=counts["chunks"],
            n_facts=counts["facts"],
        )
        log.info(
            "fts_rebuilt",
            backend=self.bm25_backend,
            n_chunks=counts["chunks"],
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    # ---- search -----------------------------------------------------------------------------

    def search_bm25(
        self, query: str, k: int = 20, doc_filter: list[str] | None = None
    ) -> list[Hit]:
        """Top-``k`` chunks by BM25; ``doc_filter`` restricts to those ``doc_name`` values."""
        _check_k(k)
        if not query or not query.strip() or doc_filter == []:
            return []
        started = time.perf_counter()
        if self._fts_loaded:
            self._ensure_fts_index()
            where, params = _doc_filter_clause(doc_filter)
            rows = self.conn.execute(
                f"SELECT {_prefixed(_CHUNK_COLUMNS, 'c')}, s.score "
                "FROM chunks c JOIN ("
                f"    SELECT chunk_id, {_FTS_SCHEMA}.match_bm25(chunk_id, ?) AS score FROM chunks"
                ") s USING (chunk_id) "
                f"WHERE s.score IS NOT NULL{where} "
                "ORDER BY s.score DESC, c.chunk_id LIMIT ?",
                [query, *params, k],
            ).fetchall()
            hits = [
                Hit(
                    chunk=_row_to_chunk(row),
                    score=float(row[7]),
                    rank=i,
                    source="bm25",
                    bm25_rank=i,
                )
                for i, row in enumerate(rows, start=1)
            ]
        else:
            index = self._python_bm25()
            allowed = set(doc_filter) if doc_filter is not None else None
            scored = index.search(query, k, allowed)
            chunk_by_id = {c.chunk_id: c for c in self.get_chunks([cid for cid, _ in scored])}
            hits = [
                Hit(chunk=chunk_by_id[cid], score=score, rank=i, source="bm25", bm25_rank=i)
                for i, (cid, score) in enumerate(scored, start=1)
                if cid in chunk_by_id
            ]
        log.debug(
            "search_bm25",
            backend=self.bm25_backend,
            k=k,
            n_hits=len(hits),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return hits

    def search_dense(
        self, qvec: np.ndarray, k: int = 20, doc_filter: list[str] | None = None
    ) -> list[Hit]:
        """Top-``k`` chunks by cosine similarity to ``qvec`` (shape ``(dim,)`` or ``(1, dim)``)."""
        _check_k(k)
        if doc_filter == []:
            return []
        vector = np.asarray(qvec, dtype=np.float32)
        if vector.ndim == 2 and vector.shape[0] == 1:
            vector = vector[0]
        if vector.shape != (self.dim,):
            raise IndexMismatch(f"query vector has shape {vector.shape}; store dim is {self.dim}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("query vector contains NaN or inf")
        started = time.perf_counter()
        where, params = _doc_filter_clause(doc_filter)
        rows = self.conn.execute(
            f"SELECT {_CHUNK_COLUMNS}, array_cosine_similarity(embedding, ?::FLOAT[{self.dim}]) "
            f"AS score FROM chunks WHERE score IS NOT NULL{where} "
            "ORDER BY score DESC, chunk_id LIMIT ?",
            [vector.tolist(), *params, k],
        ).fetchall()
        hits = [
            Hit(chunk=_row_to_chunk(row), score=float(row[7]), rank=i, source="dense", dense_rank=i)
            for i, row in enumerate(rows, start=1)
        ]
        log.debug(
            "search_dense",
            k=k,
            n_hits=len(hits),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return hits

    def hybrid_search(
        self,
        query: str,
        qvec: np.ndarray,
        k: int = 10,
        k_each: int = 30,
        rrf_k: int = 60,
        doc_filter: list[str] | None = None,
    ) -> list[Hit]:
        """Reciprocal-rank fusion of BM25 and dense top-``k_each`` lists, truncated to ``k``.

        ``Hit.score`` is the RRF score; ``bm25_rank`` / ``dense_rank`` record where the chunk
        appeared in each sub-ranking (``None`` when absent from one of them).
        """
        _check_k(k)
        _check_k(k_each)
        bm25_hits = self.search_bm25(query, k=k_each, doc_filter=doc_filter)
        dense_hits = self.search_dense(qvec, k=k_each, doc_filter=doc_filter)
        chunks: dict[str, Chunk] = {}
        bm25_rank: dict[str, int] = {}
        dense_rank: dict[str, int] = {}
        for hit in bm25_hits:
            chunks[hit.chunk.chunk_id] = hit.chunk
            bm25_rank[hit.chunk.chunk_id] = hit.rank
        for hit in dense_hits:
            chunks[hit.chunk.chunk_id] = hit.chunk
            dense_rank[hit.chunk.chunk_id] = hit.rank
        fused = rrf([list(bm25_rank), list(dense_rank)], k=rrf_k)[:k]
        return [
            Hit(
                chunk=chunks[chunk_id],
                score=score,
                rank=i,
                source="hybrid",
                bm25_rank=bm25_rank.get(chunk_id),
                dense_rank=dense_rank.get(chunk_id),
            )
            for i, (chunk_id, score) in enumerate(fused, start=1)
        ]

    # ---- reads ------------------------------------------------------------------------------

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """Chunks for the given ids in the order requested; unknown ids are skipped."""
        unique = list(dict.fromkeys(chunk_ids))
        if not unique:
            return []
        rows = self.conn.execute(
            f"SELECT {_CHUNK_COLUMNS} FROM chunks WHERE chunk_id IN ({_placeholders(len(unique))})",
            unique,
        ).fetchall()
        by_id = {row[0]: _row_to_chunk(row) for row in rows}
        return [by_id[cid] for cid in unique if cid in by_id]

    def get_pages(self, doc_name: str, pages: list[int]) -> list[Page]:
        """Pages of ``doc_name`` with the given 1-based numbers, ordered by page number."""
        unique = sorted(set(pages))
        if not unique:
            return []
        rows = self.conn.execute(
            "SELECT doc_name, page_num, text FROM pages "
            f"WHERE doc_name = ? AND page_num IN ({_placeholders(len(unique))}) ORDER BY page_num",
            [doc_name, *unique],
        ).fetchall()
        return [Page(doc_name=row[0], page_num=row[1], text=row[2]) for row in rows]

    def list_documents(
        self,
        ticker: str | None = None,
        doc_names: list[str] | None = None,
        fiscal_year: int | None = None,
    ) -> list[DocumentMeta]:
        """Documents matching every given filter (ticker is compared case-insensitively)."""
        clauses: list[str] = []
        params: list[Any] = []
        if ticker is not None:
            clauses.append("upper(ticker) = upper(?)")
            params.append(ticker)
        if doc_names is not None:
            if not doc_names:
                return []
            clauses.append(f"doc_name IN ({_placeholders(len(doc_names))})")
            params.extend(doc_names)
        if fiscal_year is not None:
            clauses.append("fiscal_year = ?")
            params.append(fiscal_year)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents{where} ORDER BY doc_name", params
        ).fetchall()
        return [_row_to_document(row) for row in rows]

    # ---- connections / export / lifecycle ---------------------------------------------------

    def readonly_connection(self) -> ReadOnlyConnection:
        """A guarded cursor for the SQL tool (see :class:`ReadOnlyConnection`).

        Each call returns a fresh cursor, so one per thread is safe. When the store itself was
        opened ``read_only`` the file is additionally protected by the engine's access mode, and
        the cursor inherits the locked engine configuration (no extension auto-loading; no
        external access on read-only stores), so a statement cannot widen what it may reach.
        """
        cursor = self.conn.cursor()
        if self._fts_loaded:
            try:
                cursor.execute("LOAD fts")
            except duckdb.Error:  # pragma: no cover - extension already loaded database-wide
                log.debug("fts_load_on_cursor_skipped")
        return ReadOnlyConnection(cursor, engine_read_only=self.read_only)

    def export_parquet(self, out_dir: Path) -> None:
        """Write every table as ``<out_dir>/<table>.parquet`` (DuckDB-independent portability)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for table in _EXPORT_TABLES:
            if not self._table_exists(table):
                raise ConfigError(f"table {table} missing: call init_schema first")
            target = str(out_dir / f"{table}.parquet").replace("'", "''")
            self.conn.execute(f"COPY (SELECT * FROM {table}) TO '{target}' (FORMAT PARQUET)")
        log.info("parquet_exported", out_dir=str(out_dir), tables=list(_EXPORT_TABLES))

    def close(self) -> None:
        """Close the root connection and every per-thread cursor; idempotent.

        Cursors are closed first: closing the root invalidates them anyway, and releasing them
        explicitly does not depend on their threads still being alive.
        """
        with self._cursors_lock:
            cursors, self._cursors = self._cursors, []
        for cursor in cursors:
            cursor.close()
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> DuckDBStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- internals --------------------------------------------------------------------------

    def _require_writable(self, operation: str) -> None:
        if self.read_only:
            raise ConfigError(f"{operation} is not allowed on a read-only store")

    def _run_schema(self, dim: int) -> None:
        sql = _SCHEMA_PATH.read_text(encoding="utf-8").replace("{dim}", str(dim))
        self.conn.execute(sql)

    def _load_fts_extension(self) -> bool:
        """``LOAD fts``, installing it first if needed; ``False`` (with a warning) if impossible."""
        conn = self.conn
        try:
            conn.execute("LOAD fts")
            return True
        except duckdb.Error:
            pass
        try:
            conn.execute("INSTALL fts")
            conn.execute("LOAD fts")
            return True
        except (duckdb.Error, OSError) as exc:
            log.warning(
                "fts_extension_unavailable",
                error=str(exc).splitlines()[0][:200],
                fallback="python BM25 (no stemming)",
            )
            return False

    def _harden_configuration(self) -> None:
        """Pin the engine configuration so model-driven SQL cannot widen it; runs once per store.

        Must run after :meth:`_load_fts_extension`: ``INSTALL fts`` / ``LOAD fts`` need external
        access. Every option is global to the database instance, so per-thread cursors and
        :class:`ReadOnlyConnection` inherit it. Read-only stores are then locked; writable ones
        are not because ``PRAGMA create_fts_index`` sets a session option internally and fails
        under ``lock_configuration`` (and ingest never runs model-driven SQL). When the instance
        is already locked (a second read-only store on the same file in one process) the values
        are verified instead of set, and a disagreement is a :class:`ConfigError` because the
        hardening cannot be guaranteed.
        """
        conn = self.conn
        wanted = dict(_HARDENED_SETTINGS)
        wanted["enable_external_access"] = self.external_access
        locked = self._current_setting("lock_configuration")
        for name, value in wanted.items():
            current = self._current_setting(name)
            if current == value:
                continue
            if locked:
                raise ConfigError(
                    f"DuckDB configuration of {self.path} is locked with {name}={current!r}; "
                    f"expected {value!r}"
                )
            conn.execute(f"SET {name} = {'true' if value else 'false'}")
        if self.read_only and not locked:
            conn.execute("SET lock_configuration = true")
        log.debug("duckdb_configuration_hardened", locked=self.read_only or locked, **wanted)

    def _current_setting(self, name: str) -> bool:
        row = self.conn.execute("SELECT current_setting(?)", [name]).fetchone()
        if row is None:  # pragma: no cover - current_setting always yields one row
            raise ConfigError(f"DuckDB did not report the value of {name}")
        value = row[0]
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() == "true"

    def _ensure_fts_index(self) -> None:
        if self._fts_stale or not self._fts_index_exists():
            if self.read_only:
                raise ConfigError("FTS index missing: rebuild_fts on a writable store first")
            log.info("fts_index_missing_or_stale", action="rebuild")
            self.rebuild_fts()

    def _fts_index_exists(self) -> bool:
        row = self.conn.execute(
            "SELECT count(*) FROM information_schema.schemata WHERE schema_name = ?", [_FTS_SCHEMA]
        ).fetchone()
        return bool(row and row[0])

    def _python_bm25(self) -> _PythonBM25:
        if self._py_bm25 is None or self._fts_stale:
            self._py_bm25 = self._build_python_bm25()
            self._fts_stale = False
        return self._py_bm25

    def _build_python_bm25(self) -> _PythonBM25:
        rows = self.conn.execute("SELECT chunk_id, doc_name, text FROM chunks").fetchall()
        return _PythonBM25(rows)

    def _table_exists(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_schema = 'main' AND table_name = ?",
            [name],
        ).fetchone()
        return bool(row and row[0])

    def _chunks_embedding_dim(self) -> int | None:
        row = self.conn.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema = 'main' AND table_name = 'chunks' AND column_name = 'embedding'"
        ).fetchone()
        if not row:
            return None
        match = re.fullmatch(r"FLOAT\[(\d+)\]", str(row[0]))
        return int(match.group(1)) if match else None

    def _read_manifest_rows(self) -> dict[str, str | None]:
        rows = self.conn.execute("SELECT key, value FROM index_manifest").fetchall()
        return {str(key): value for key, value in rows}

    def _count(self, table: str) -> int:
        row = self.conn.execute(f"SELECT count(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _placeholders(n: int) -> str:
    return ", ".join("?" for _ in range(n))


def _prefixed(columns: str, alias: str) -> str:
    return ", ".join(f"{alias}.{col.strip()}" for col in columns.split(","))


def _doc_filter_clause(doc_filter: list[str] | None) -> tuple[str, list[str]]:
    """``(' AND doc_name IN (?, ?)', params)`` or ``('', [])`` when unfiltered."""
    if doc_filter is None:
        return "", []
    unique = list(dict.fromkeys(doc_filter))
    return f" AND doc_name IN ({_placeholders(len(unique))})", unique


def _row_to_chunk(row: tuple[Any, ...]) -> Chunk:
    return Chunk(
        chunk_id=row[0],
        doc_name=row[1],
        page_num=row[2],
        chunk_idx=row[3],
        section=row[4],
        text=row[5],
        n_tokens=row[6],
    )


def _row_to_document(row: tuple[Any, ...]) -> DocumentMeta:
    return DocumentMeta(
        doc_name=row[0],
        ticker=row[1],
        cik=row[2],
        company=row[3],
        form=row[4],
        fiscal_year=row[5],
        period_end=row[6],
        source_kind=row[7],
        source_url=row[8],
        source_sha256=row[9],
        n_pages=row[10],
        ingested_at=row[11],
    )


def _to_naive_utc(value: datetime) -> datetime:
    """DuckDB TIMESTAMP is naive; store aware datetimes as UTC wall-clock, naive ones as-is."""
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


__all__ = ["DuckDBStore", "ReadOnlyConnection", "ReadOnlyResult"]
