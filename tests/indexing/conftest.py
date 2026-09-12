"""Fixtures for the indexing tests: in-memory store, hashing embedder, fixture companies.

Everything is offline. PDFs come from the root ``fixture_pdf_factory`` (reportlab), EDGAR from
``respx`` routes over hand-made JSON, HTML from ``tests/fixtures/indexing_edgar_10k.html``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from secqa.core.contracts import Embedder
from secqa.edgar import EdgarClient
from secqa.embeddings import HashingEmbedder
from secqa.indexing import Company
from secqa.store import DuckDBStore

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[2]
DIM = 64
TEST_UA = "Test Runner test@example.com"
FIXTURE_CIK = "0001234567"


class CountingEmbedder:
    """Wraps an embedder and counts ``embed`` calls / texts (to prove skips do not embed)."""

    def __init__(self, inner: Embedder) -> None:
        self._inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self.calls = 0
        self.texts = 0

    def embed(self, texts: list[str], *, batch_size: int = 64, kind: str = "passage") -> np.ndarray:
        self.calls += 1
        self.texts += len(texts)
        return self._inner.embed(texts, batch_size=batch_size, kind=kind)  # type: ignore[arg-type]


@pytest.fixture(scope="session")
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=DIM)


@pytest.fixture
def counting_embedder(embedder: HashingEmbedder) -> CountingEmbedder:
    return CountingEmbedder(embedder)


@pytest.fixture
def store(embedder: HashingEmbedder) -> Iterator[DuckDBStore]:
    """Initialised, empty in-memory store."""
    store = DuckDBStore(":memory:", embed_dim=DIM)
    store.init_schema(embedder.name, embedder.dim)
    yield store
    store.close()


@pytest.fixture
def file_store_factory(
    tmp_path: Path, embedder: HashingEmbedder
) -> Iterator[Callable[[str], DuckDBStore]]:
    """Initialised file-backed stores under ``tmp_path`` (closed at teardown)."""
    opened: list[DuckDBStore] = []

    def _make(name: str = "index.duckdb") -> DuckDBStore:
        store = DuckDBStore(tmp_path / name, embed_dim=DIM)
        store.init_schema(embedder.name, embedder.dim)
        opened.append(store)
        return store

    yield _make
    for store in opened:
        store.close()


@pytest.fixture
def companies() -> list[Company]:
    return [
        Company(
            name="Fixture Corp",
            ticker="FIXT",
            cik="1234567",
            financebench_aliases=["FIXTURE"],
        ),
        Company(name="Acme Holdings", ticker="ACME", cik="7654321", financebench_aliases=["ACME"]),
    ]


@pytest.fixture
def edgar_html() -> bytes:
    return (FIXTURES / "indexing_edgar_10k.html").read_bytes()


@pytest.fixture
def companyfacts_json() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(
        (FIXTURES / "indexing_companyfacts_small.json").read_text(encoding="utf-8")
    )
    return data


@pytest.fixture
def edgar_client(tmp_path: Path) -> Iterator[EdgarClient]:
    """EDGAR client with a per-test cache directory and a no-op sleep (no real waiting)."""
    client = EdgarClient(TEST_UA, cache_dir=tmp_path / "edgar-cache", sleep=lambda _s: None)
    yield client
    client.close()
