"""Print the fixture corpus rankings twice per strategy so CI can show whether they are stable.

Run from the repo root (``uv run --no-sync python scripts/check_ranking_determinism.py``). Every
line is emitted as a GitHub Actions ``::notice::`` annotation so the result is readable from the
public checks API. Exit status is 1 when any strategy's order differs between two identical calls
or when a k=2 result is not the prefix of the k=4 result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.retrieval.conftest import (  # noqa: E402
    DIM,
    DOCS,
    HashingEmbedder,
    make_corpus,
    make_document,
)
from tests.retrieval.test_retriever import QUESTION  # noqa: E402

from secqa.core.contracts import Page  # noqa: E402
from secqa.retrieval.retriever import Retriever  # noqa: E402
from secqa.store import DuckDBStore  # noqa: E402


def main() -> int:
    embedder = HashingEmbedder(dim=DIM)
    corpus = make_corpus()
    store = DuckDBStore(":memory:", embed_dim=DIM)
    store.init_schema(embedder.name, embedder.dim)
    for doc_name in DOCS:
        store.upsert_document(make_document(doc_name))
    store.add_pages(
        [
            Page(doc_name=d, page_num=p, text=f"{d} full page {p} text")
            for d in DOCS
            for p in range(1, 4)
        ]
    )
    store.add_chunks(corpus, embedder.embed([c.text for c in corpus]))
    store.rebuild_fts()
    threads = store.conn.execute("SELECT current_setting('threads')").fetchone()[0]
    print(
        f"::notice title=ranking::duckdb={duckdb.__version__} threads={threads} "
        f"bm25_backend={store.bm25_backend}"
    )
    ok = True
    for strategy in ("bm25", "dense", "hybrid"):
        r4a = Retriever(store, embedder, strategy=strategy, k=4).retrieve(QUESTION)  # type: ignore[arg-type]
        r4b = Retriever(store, embedder, strategy=strategy, k=4).retrieve(QUESTION)  # type: ignore[arg-type]
        r2 = Retriever(store, embedder, strategy=strategy, k=2).retrieve(QUESTION)  # type: ignore[arg-type]
        fmt = lambda res: " ".join(f"{h.chunk.chunk_id[:8]}:{h.score:.4f}" for h in res.hits)  # noqa: E731
        stable = [h.chunk.chunk_id for h in r4a.hits] == [h.chunk.chunk_id for h in r4b.hits]
        prefix = [h.chunk.chunk_id for h in r2.hits] == [h.chunk.chunk_id for h in r4a.hits[:2]]
        ok = ok and stable and prefix
        print(
            f"::notice title=ranking {strategy}::stable={stable} prefix={prefix} "
            f"k4a=[{fmt(r4a)}] k4b=[{fmt(r4b)}] k2=[{fmt(r2)}]"
        )
    if store.bm25_backend == "duckdb_fts":
        rows = store.conn.execute(
            "SELECT chunk_id, fts_main_chunks.match_bm25(chunk_id, ?) AS s FROM chunks "
            "WHERE s IS NOT NULL ORDER BY s DESC, chunk_id LIMIT 6",
            [QUESTION],
        ).fetchall()
        raw = " ".join(f"{cid[:8]}:{s:.4f}" for cid, s in rows)
        print("::notice title=ranking raw fts::" + raw)
    store.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
