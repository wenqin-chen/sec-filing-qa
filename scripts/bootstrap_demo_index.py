#!/usr/bin/env python
"""Make sure an index exists before the API starts (container entrypoint helper).

Order of preference:

1. ``--duckdb-path`` already exists -> do nothing (``--force`` rebuilds / re-downloads).
2. ``--index-url`` given -> :func:`secqa.indexing.fetch_index` (https://, gs://, file://; a
   ``pack_index`` tarball or a bare DuckDB file, integrity-checked), then a read-only open to
   report the manifest and warn when its embedder differs from ``--embedder``.
3. Otherwise -> build the bundled two-document synthetic fixture corpus
   (``tests/fixtures/eval_fixture_pages.json``) with the configured embedder, so the container is
   healthy with zero configuration and the deploy smoke can ask a real question.

Exit status is 0 on success and 1 on any failure (the entrypoint stops before uvicorn).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from secqa.core.errors import SecqaError
from secqa.core.logging import configure_logging, get_logger
from secqa.embeddings import get_embedder
from secqa.embeddings.local import DEFAULT_LOCAL_MODEL
from secqa.embeddings.openai_embed import DEFAULT_OPENAI_MODEL
from secqa.embeddings.registry import DEFAULT_DIM, parse_spec
from secqa.eval.runner import build_fixture_index
from secqa.indexing import build_manifest, fetch_index
from secqa.store import DuckDBStore
from secqa.store.manifest import resolve_git_sha

log = get_logger("secqa.bootstrap")

DEFAULT_FIXTURE_PAGES = Path("tests/fixtures/eval_fixture_pages.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments (environment variables supply the defaults)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--duckdb-path",
        type=Path,
        default=Path(os.environ.get("SECQA_DUCKDB_PATH") or "data/index.duckdb"),
        help="index file to create or verify (default: $SECQA_DUCKDB_PATH or data/index.duckdb)",
    )
    parser.add_argument(
        "--index-url",
        default=os.environ.get("SECQA_INDEX_URL") or None,
        help="https://, gs:// or file:// index to download when the file is absent",
    )
    parser.add_argument(
        "--embedder",
        default=os.environ.get("SECQA_EMBEDDER") or "hashing",
        help="embedder spec for the fixture index (default: $SECQA_EMBEDDER or 'hashing')",
    )
    parser.add_argument(
        "--fixture-pages",
        type=Path,
        default=DEFAULT_FIXTURE_PAGES,
        help="fixture corpus JSON used when no index URL is given",
    )
    parser.add_argument(
        "--force", action="store_true", help="rebuild / re-download even if the file exists"
    )
    parser.add_argument(
        "--log-json",
        action="store_true",
        default=os.environ.get("SECQA_LOG_JSON", "").lower() in {"1", "true", "yes"},
        help="emit JSON log lines (default: $SECQA_LOG_JSON)",
    )
    return parser.parse_args(argv)


def build_fixture(duckdb_path: Path, embedder_spec: str, pages_path: Path) -> dict[str, int]:
    """Build the synthetic fixture index at ``duckdb_path``; returns the store's row counts."""
    embedder = get_embedder(embedder_spec)
    store = DuckDBStore(duckdb_path, embed_dim=embedder.dim)
    try:
        store.init_schema(embedder.name, embedder.dim)
        n_chunks = build_fixture_index(store, embedder, pages_path)
        manifest = build_manifest(store, resolve_git_sha(), [pages_path], None)
        counts = store.counts()
    finally:
        store.close()
    log.info(
        "fixture_index_ready",
        path=str(duckdb_path),
        embedder=manifest.embedder,
        dim=manifest.dim,
        n_chunks=n_chunks,
        **counts,
    )
    return counts


def expected_embedder_name(spec: str) -> str:
    """The ``index_manifest.embedder`` value an index built with ``spec`` records.

    Mirrors the naming of the three embedders without instantiating them (the local one loads
    torch, which the API is about to do anyway): ``hashing[:dim]`` -> ``hashing-<dim>``,
    ``local[:model]`` -> the model's base name, ``openai[:model]`` -> the model id.
    """
    kind, arg = parse_spec(spec)
    if kind == "hashing":
        return f"hashing-{int(arg) if arg else DEFAULT_DIM}"
    if kind == "local":
        return (arg or DEFAULT_LOCAL_MODEL).rstrip("/").rsplit("/", 1)[-1]
    return arg or DEFAULT_OPENAI_MODEL


def fetch(duckdb_path: Path, url: str, embedder_spec: str) -> dict[str, int]:
    """Download the index at ``url`` to ``duckdb_path`` and report its manifest."""
    fetch_index(url, duckdb_path)
    store = DuckDBStore(duckdb_path, read_only=True)
    try:
        manifest = store.manifest()
        counts = store.counts()
    finally:
        store.close()
    expected = expected_embedder_name(embedder_spec)
    if expected != manifest.embedder:
        log.warning(
            "embedder_mismatch",
            index_embedder=manifest.embedder,
            configured=embedder_spec,
            expected_name=expected,
            effect="dense/hybrid retrieval will fail readiness; set SECQA_EMBEDDER to match",
        )
    log.info(
        "index_fetched_ready",
        path=str(duckdb_path),
        url=url,
        embedder=manifest.embedder,
        dim=manifest.dim,
        git_sha=manifest.git_sha,
        **counts,
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit status."""
    args = parse_args(argv)
    configure_logging(json=args.log_json)
    duckdb_path: Path = args.duckdb_path
    if duckdb_path.is_file() and not args.force:
        log.info("index_present", path=str(duckdb_path), action="skip")
        return 0
    if args.force and duckdb_path.exists():
        duckdb_path.unlink()
        wal = duckdb_path.with_name(duckdb_path.name + ".wal")
        wal.unlink(missing_ok=True)
    try:
        if args.index_url:
            fetch(duckdb_path, args.index_url, args.embedder)
        else:
            if not args.fixture_pages.is_file():
                log.error("fixture_pages_missing", path=str(args.fixture_pages))
                return 1
            build_fixture(duckdb_path, args.embedder, args.fixture_pages)
    except (SecqaError, OSError, ValueError) as exc:
        log.error("bootstrap_failed", error=f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
