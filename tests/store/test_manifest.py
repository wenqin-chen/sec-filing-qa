"""IndexManifest serialisation, git sha resolution and the store-level manifest API."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from secqa.core.errors import ConfigError, IndexMismatch
from secqa.store import DuckDBStore, IndexManifest, resolve_git_sha
from secqa.store.manifest import MANIFEST_KEYS, parse_value, serialize_value, utc_now


def _sample() -> IndexManifest:
    return IndexManifest(
        embedder="hashing",
        dim=384,
        git_sha="a" * 40,
        built_at=datetime(2026, 9, 11, 8, 30, tzinfo=UTC),
        n_documents=2,
        n_pages=6,
        n_chunks=40,
        n_facts=0,
        inputs_sha256="b" * 64,
        dataset_revision=None,
    )


def test_round_trip_through_rows() -> None:
    manifest = _sample()
    rows = manifest.to_rows()
    assert [key for key, _ in rows] == list(MANIFEST_KEYS)
    assert dict(rows)["dim"] == "384"
    assert dict(rows)["dataset_revision"] is None
    assert IndexManifest.from_rows(dict(rows)) == manifest


def test_from_rows_reports_missing_required_keys() -> None:
    with pytest.raises(KeyError, match="git_sha"):
        IndexManifest.from_rows({"embedder": "hashing", "dim": "384"})


def test_dataset_revision_is_optional_in_rows() -> None:
    rows = dict(_sample().to_rows())
    del rows["dataset_revision"]
    assert IndexManifest.from_rows(rows).dataset_revision is None


def test_validation_rejects_bad_dim_and_counts() -> None:
    with pytest.raises(ValueError, match="dim must be positive"):
        IndexManifest.model_validate({**_sample().model_dump(), "dim": 0})
    with pytest.raises(ValueError, match="non-negative"):
        IndexManifest.model_validate({**_sample().model_dump(), "n_chunks": -1})


def test_serialize_and_parse_helpers() -> None:
    assert serialize_value("n_chunks", 7) == "7"
    assert parse_value("n_chunks", "7") == 7
    assert parse_value("built_at", "2026-09-11T08:30:00+00:00") == datetime(
        2026, 9, 11, 8, 30, tzinfo=UTC
    )
    assert parse_value("inputs_sha256", None) is None
    with pytest.raises(ValueError, match="unknown manifest key"):
        serialize_value("nope", 1)
    with pytest.raises(ValueError, match="unknown manifest key"):
        parse_value("nope", "1")
    with pytest.raises(TypeError):
        serialize_value("n_chunks", True)
    assert utc_now().tzinfo is UTC


def test_resolve_git_sha_prefers_env_then_git_then_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SECQA_GIT_SHA", "deadbeef")
    assert resolve_git_sha() == "deadbeef"
    monkeypatch.delenv("SECQA_GIT_SHA")
    # A directory that is not a git repository resolves to 'unknown' without raising.
    assert resolve_git_sha(repo_root=tmp_path) == "unknown"
    sha = resolve_git_sha()
    assert sha == "unknown" or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))


def test_store_writes_and_reads_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SECQA_GIT_SHA", "cafebabe")
    with DuckDBStore(":memory:") as store:
        with pytest.raises(ConfigError, match="init_schema"):
            store.manifest()
        with pytest.raises(ConfigError, match="dim unknown"):
            _ = store.dim
        store.init_schema("hashing", 16)
        manifest = store.manifest()
        assert (manifest.embedder, manifest.dim, manifest.git_sha) == ("hashing", 16, "cafebabe")
        assert manifest.built_at.tzinfo is not None
        assert (manifest.n_documents, manifest.n_chunks) == (0, 0)
        assert store.embedder_name == "hashing"
        store.set_manifest(inputs_sha256="f" * 64, dataset_revision="rev-1", n_facts=3)
        updated = store.manifest()
        assert updated.inputs_sha256 == "f" * 64
        assert updated.dataset_revision == "rev-1"
        assert updated.n_facts == 3
        assert updated.built_at == manifest.built_at  # untouched
        with pytest.raises(ValueError, match="unknown manifest key"):
            store.set_manifest(bogus=1)
        with pytest.raises(IndexMismatch):
            store.set_manifest(dim=32)
        with pytest.raises(IndexMismatch):
            store.set_manifest(embedder="other")
        store.set_manifest(embedder="hashing", dim=16)  # same values are accepted


def test_init_schema_is_idempotent_and_keeps_built_at() -> None:
    with DuckDBStore(":memory:") as store:
        store.init_schema("hashing", 8)
        first = store.manifest()
        store.init_schema("hashing", 8)
        assert store.manifest() == first
        with pytest.raises(ValueError, match="embedder_name"):
            store.init_schema("  ", 8)
        with pytest.raises(ValueError, match="embed_dim"):
            store.init_schema("hashing", 0)
