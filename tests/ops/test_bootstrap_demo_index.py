"""scripts/bootstrap_demo_index.py: fixture build, idempotence, file:// fetch, and the API
serving what it built (the same path the container smoke exercises)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient

from secqa.api import create_app
from secqa.core.settings import Settings
from secqa.store import DuckDBStore
from tests.ops.conftest import FIXTURES

PAGES = FIXTURES / "eval_fixture_pages.json"


@pytest.fixture
def bootstrap(script: Callable[[str], ModuleType]) -> ModuleType:
    return script("bootstrap_demo_index")


def test_builds_fixture_index_then_skips(bootstrap: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "index.duckdb"
    assert bootstrap.main(["--duckdb-path", str(path), "--fixture-pages", str(PAGES)]) == 0
    assert path.is_file()
    store = DuckDBStore(path, read_only=True)
    try:
        counts = store.counts()
        manifest = store.manifest()
    finally:
        store.close()
    assert counts["documents"] == 2 and counts["chunks"] >= 2
    assert manifest.embedder == "hashing-384" and manifest.dim == 384
    mtime = path.stat().st_mtime
    assert bootstrap.main(["--duckdb-path", str(path), "--fixture-pages", str(PAGES)]) == 0
    assert path.stat().st_mtime == mtime, "second run must not touch an existing index"


def test_force_rebuilds(bootstrap: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "index.duckdb"
    assert bootstrap.main(["--duckdb-path", str(path), "--fixture-pages", str(PAGES)]) == 0
    assert (
        bootstrap.main(["--duckdb-path", str(path), "--fixture-pages", str(PAGES), "--force"]) == 0
    )
    assert path.is_file()


def test_missing_fixture_pages_is_an_error(bootstrap: ModuleType, tmp_path: Path) -> None:
    code = bootstrap.main(
        [
            "--duckdb-path",
            str(tmp_path / "i.duckdb"),
            "--fixture-pages",
            str(tmp_path / "nope.json"),
        ]
    )
    assert code == 1
    assert not (tmp_path / "i.duckdb").exists()


def test_fetches_index_from_file_url(bootstrap: ModuleType, tmp_path: Path) -> None:
    source = tmp_path / "src.duckdb"
    assert bootstrap.main(["--duckdb-path", str(source), "--fixture-pages", str(PAGES)]) == 0
    dest = tmp_path / "served" / "index.duckdb"
    code = bootstrap.main(
        ["--duckdb-path", str(dest), "--index-url", source.as_uri(), "--embedder", "hashing"]
    )
    assert code == 0
    assert dest.is_file()


def test_bad_index_url_fails_cleanly(bootstrap: ModuleType, tmp_path: Path) -> None:
    dest = tmp_path / "index.duckdb"
    code = bootstrap.main(["--duckdb-path", str(dest), "--index-url", "ftp://example.invalid/x"])
    assert code == 1
    assert not dest.exists()


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("hashing", "hashing-384"),
        ("hashing:64", "hashing-64"),
        ("local", "bge-small-en-v1.5"),
        ("local:BAAI/bge-base-en-v1.5", "bge-base-en-v1.5"),
        ("openai", "text-embedding-3-small"),
        ("openai:text-embedding-3-large", "text-embedding-3-large"),
    ],
)
def test_expected_embedder_name(bootstrap: ModuleType, spec: str, expected: str) -> None:
    assert bootstrap.expected_embedder_name(spec) == expected


def test_api_serves_the_bootstrapped_index(bootstrap: ModuleType, tmp_path: Path) -> None:
    """End to end: bootstrap -> create_app -> /readyz 200 -> mock /v1/ask with a citation."""
    path = tmp_path / "index.duckdb"
    assert bootstrap.main(["--duckdb-path", str(path), "--fixture-pages", str(PAGES)]) == 0
    settings = Settings(_env_file=None, duckdb_path=path, provider="mock", embedder="hashing")
    app = create_app(settings)
    with TestClient(app) as client:
        ready = client.get("/readyz")
        assert ready.status_code == 200, ready.text
        assert ready.json()["documents"] == 2
        answer = client.post(
            "/v1/ask",
            json={
                "question": "What were total net sales in fiscal 2023?",
                "provider": "mock",
                "k": 4,
            },
        )
        assert answer.status_code == 200, answer.text
        body = answer.json()
        assert body["citations"], "mock provider must cite the top passage"
        assert any(c["verified"] for c in body["citations"])
