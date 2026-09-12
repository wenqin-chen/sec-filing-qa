"""secqa.store: the single-file DuckDB index (documents, pages, chunks, XBRL facts, manifest).

Public surface: :class:`DuckDBStore`, :class:`IndexManifest`, :func:`rrf` and the guarded
:class:`ReadOnlyConnection` returned by :meth:`DuckDBStore.readonly_connection`.
"""

from secqa.store.duckdb_store import DuckDBStore, ReadOnlyConnection, ReadOnlyResult
from secqa.store.fusion import rrf
from secqa.store.manifest import IndexManifest, resolve_git_sha

__all__ = [
    "DuckDBStore",
    "IndexManifest",
    "ReadOnlyConnection",
    "ReadOnlyResult",
    "resolve_git_sha",
    "rrf",
]
