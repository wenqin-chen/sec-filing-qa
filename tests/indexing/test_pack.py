"""build_manifest, pack_index and fetch_index round-trips (file://, https via respx, gs:// stub)."""

from __future__ import annotations

import io
import json
import shutil
import sys
import tarfile
import types
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import respx
import zstandard

from secqa.core.contracts import DocumentMeta
from secqa.core.errors import ConfigError
from secqa.core.ids import sha256_hex
from secqa.embeddings import HashingEmbedder
from secqa.indexing import (
    IndexPackError,
    build_manifest,
    fetch_index,
    hash_inputs,
    ingest_document,
    pack_index,
)
from secqa.indexing.manifest_build import (
    INDEX_FILE_NAME,
    INDEX_MANIFEST_FILE_NAME,
    PACK_FORMAT,
    ZSTD_MAGIC,
    is_duckdb_file,
)
from secqa.ingest import extract_pdf_pages
from secqa.store import DuckDBStore
from tests.indexing.conftest import DIM

GIT_SHA = "f" * 40


# ---- hash_inputs / build_manifest -----------------------------------------------------------


def test_hash_inputs_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    a = tmp_path / "a.pdf"
    b = tmp_path / "sub" / "b.pdf"
    b.parent.mkdir()
    a.write_bytes(b"%PDF-a")
    b.write_bytes(b"%PDF-b")
    digest = hash_inputs([a, b])
    assert len(digest) == 64
    assert hash_inputs([b, a]) == digest  # order-independent
    assert hash_inputs([tmp_path]) == digest  # directories expand recursively
    a.write_bytes(b"%PDF-a2")
    assert hash_inputs([a, b]) != digest
    assert hash_inputs([]) == sha256_hex("")
    with pytest.raises(FileNotFoundError):
        hash_inputs([tmp_path / "missing.pdf"])


def test_build_manifest_records_counts_inputs_and_revision(
    file_store_factory: Callable[[str], DuckDBStore],
    embedder: HashingEmbedder,
    fixture_pdf: Path,
) -> None:
    store = file_store_factory("index.duckdb")
    pages = extract_pdf_pages(fixture_pdf, "FIXTURE_2023_10K")
    n_chunks = ingest_document(store, embedder, pages, _meta(len(pages)))
    manifest = build_manifest(store, GIT_SHA, [fixture_pdf], dataset_revision="abc123")
    assert (manifest.n_documents, manifest.n_pages, manifest.n_chunks, manifest.n_facts) == (
        1,
        3,
        n_chunks,
        0,
    )
    assert manifest.git_sha == GIT_SHA
    assert manifest.inputs_sha256 == hash_inputs([fixture_pdf])
    assert manifest.dataset_revision == "abc123"
    assert manifest.built_at.tzinfo is not None
    assert (manifest.embedder, manifest.dim) == (embedder.name, DIM)
    assert store.manifest() == manifest
    with pytest.raises(ValueError, match="git_sha"):
        build_manifest(store, "  ", [], None)


def test_build_manifest_refuses_read_only(
    file_store_factory: Callable[[str], DuckDBStore], tmp_path: Path
) -> None:
    file_store_factory("ro.duckdb").close()
    with DuckDBStore(tmp_path / "ro.duckdb", read_only=True) as ro:
        with pytest.raises(ConfigError, match="read-only"):
            build_manifest(ro, GIT_SHA, [], None)


# ---- pack / fetch ---------------------------------------------------------------------------


def _meta(n_pages: int) -> DocumentMeta:
    from datetime import UTC, datetime

    return DocumentMeta(
        doc_name="FIXTURE_2023_10K",
        company="Fixture Corp",
        ticker="FIXT",
        cik="0001234567",
        form="10-K",
        fiscal_year=2023,
        source_kind="fixture",
        source_url="https://example.invalid/FIXTURE_2023_10K.pdf",
        source_sha256="0" * 64,
        n_pages=n_pages,
        ingested_at=datetime(2026, 9, 11, tzinfo=UTC),
    )


@pytest.fixture
def built_index(
    file_store_factory: Callable[[str], DuckDBStore],
    embedder: HashingEmbedder,
    fixture_pdf: Path,
    tmp_path: Path,
) -> Path:
    """A closed, populated ``index.duckdb`` with a built manifest."""
    store = file_store_factory("built/index.duckdb")
    pages = extract_pdf_pages(fixture_pdf, "FIXTURE_2023_10K")
    ingest_document(store, embedder, pages, _meta(len(pages)))
    store.rebuild_fts()
    build_manifest(store, GIT_SHA, [fixture_pdf], dataset_revision=None)
    store.close()
    return tmp_path / "built" / "index.duckdb"


def _open_and_check(path: Path, expected_chunks: int) -> None:
    with DuckDBStore(path, read_only=True) as store:
        manifest = store.manifest()
        assert manifest.git_sha == GIT_SHA and manifest.n_chunks == expected_chunks
        assert store.counts()["chunks"] == expected_chunks
        assert store.get_pages("FIXTURE_2023_10K", [1])[0].text.startswith("FIXTURE CORP")


def test_pack_then_fetch_file_url_round_trips(built_index: Path, tmp_path: Path) -> None:
    with DuckDBStore(built_index, read_only=True) as store:
        n_chunks = store.counts()["chunks"]
    tarball = pack_index(built_index, tmp_path / "dist" / "index-v0.1.tar.zst")
    assert tarball.is_file() and tarball.read_bytes()[:4] == ZSTD_MAGIC
    assert built_index.is_file()  # source untouched

    with tarball.open("rb") as raw, zstandard.ZstdDecompressor().stream_reader(raw) as zst:
        with tarfile.open(fileobj=zst, mode="r|") as tar:
            names = [m.name for m in tar]
    assert sorted(names) == sorted([INDEX_FILE_NAME, INDEX_MANIFEST_FILE_NAME])

    dest = tmp_path / "serve" / "index.duckdb"
    assert fetch_index(tarball.as_uri(), dest) == dest
    assert is_duckdb_file(dest)
    sidecar = json.loads(dest.with_name("index.duckdb.manifest.json").read_text())
    assert sidecar["format"] == PACK_FORMAT
    assert sidecar["manifest"]["git_sha"] == GIT_SHA
    assert sidecar["index_bytes"] == dest.stat().st_size
    assert not list(dest.parent.glob(".*"))  # no temp files left behind
    _open_and_check(dest, n_chunks)


def test_fetch_bare_duckdb_file(built_index: Path, tmp_path: Path) -> None:
    dest = tmp_path / "bare" / "index.duckdb"
    fetch_index(built_index.as_uri(), dest)
    with DuckDBStore(dest, read_only=True) as store:
        assert store.manifest().git_sha == GIT_SHA


def test_fetch_https_via_respx(
    built_index: Path, tmp_path: Path, respx_router: respx.MockRouter
) -> None:
    tarball = pack_index(built_index, tmp_path / "index.tar.zst")
    url = "https://github.com/example/sec-filing-qa/releases/download/v0.1.0/index-v0.1.tar.zst"
    redirect = "https://objects.githubusercontent.com/blob/index-v0.1.tar.zst"
    respx_router.get(url).mock(return_value=httpx.Response(302, headers={"location": redirect}))
    respx_router.get(redirect).mock(return_value=httpx.Response(200, content=tarball.read_bytes()))
    dest = tmp_path / "https" / "index.duckdb"
    fetch_index(url, dest)
    with DuckDBStore(dest, read_only=True) as store:
        assert store.manifest().git_sha == GIT_SHA

    respx_router.get("https://example.invalid/missing.tar.zst").mock(
        return_value=httpx.Response(404)
    )
    with pytest.raises(IndexPackError, match="HTTP 404"):
        fetch_index("https://example.invalid/missing.tar.zst", tmp_path / "x" / "index.duckdb")
    assert not (tmp_path / "x" / "index.duckdb").exists()


def test_fetch_gs_uses_optional_client(
    built_index: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tarball = pack_index(built_index, tmp_path / "index.tar.zst")
    seen: dict[str, str] = {}

    class _Blob:
        def __init__(self, name: str) -> None:
            seen["blob"] = name

        def download_to_filename(self, filename: str) -> None:
            shutil.copyfile(tarball, filename)

    class _Bucket:
        def __init__(self, name: str) -> None:
            seen["bucket"] = name

        def blob(self, name: str) -> _Blob:
            return _Blob(name)

    class _Client:
        def bucket(self, name: str) -> _Bucket:
            return _Bucket(name)

    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    storage = types.ModuleType("google.cloud.storage")
    storage.Client = _Client  # type: ignore[attr-defined]
    cloud.storage = storage  # type: ignore[attr-defined]
    google.cloud = cloud  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage)

    dest = tmp_path / "gs" / "index.duckdb"
    fetch_index("gs://my-bucket/indexes/index-v0.1.tar.zst", dest)
    assert seen == {"bucket": "my-bucket", "blob": "indexes/index-v0.1.tar.zst"}
    assert is_duckdb_file(dest)
    with pytest.raises(IndexPackError, match="gs://<bucket>/<object>"):
        fetch_index("gs://only-bucket", tmp_path / "gs2" / "index.duckdb")


def test_fetch_gs_without_extra_is_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "google.cloud.storage", None)
    with pytest.raises(ConfigError, match="google-cloud-storage"):
        fetch_index("gs://bucket/obj.tar.zst", tmp_path / "index.duckdb")


def test_fetch_rejects_unsupported_scheme_and_garbage(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unsupported index URL"):
        fetch_index("http://example.invalid/index.tar.zst", tmp_path / "index.duckdb")
    with pytest.raises(ConfigError, match="unsupported index URL"):
        fetch_index("s3://bucket/index.tar.zst", tmp_path / "index.duckdb")
    junk = tmp_path / "junk.bin"
    junk.write_bytes(b"this is neither a tarball nor a duckdb file")
    with pytest.raises(IndexPackError, match="neither"):
        fetch_index(junk.as_uri(), tmp_path / "out" / "index.duckdb")
    with pytest.raises(IndexPackError, match="file not found"):
        fetch_index((tmp_path / "nope.tar.zst").as_uri(), tmp_path / "out" / "index.duckdb")
    assert not (tmp_path / "out" / "index.duckdb").exists()


def _tarball(tmp_path: Path, members: dict[str, bytes]) -> Path:
    out = tmp_path / "crafted.tar.zst"
    with out.open("wb") as raw:
        with zstandard.ZstdCompressor().stream_writer(raw, closefd=False) as zst:
            with tarfile.open(fileobj=zst, mode="w|") as tar:
                for name, data in members.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
    return out


def test_fetch_verifies_tarball_contents(built_index: Path, tmp_path: Path) -> None:
    index_bytes = built_index.read_bytes()
    good_meta = {"index_sha256": "0" * 64, "format": PACK_FORMAT}
    dest = tmp_path / "out" / "index.duckdb"

    tampered = _tarball(
        tmp_path,
        {INDEX_MANIFEST_FILE_NAME: json.dumps(good_meta).encode(), INDEX_FILE_NAME: index_bytes},
    )
    with pytest.raises(IndexPackError, match="sha256 mismatch"):
        fetch_index(tampered.as_uri(), dest)
    assert not dest.exists() and not list(dest.parent.glob(".*"))

    no_index = _tarball(tmp_path, {INDEX_MANIFEST_FILE_NAME: b"{}", "other.bin": b"x"})
    with pytest.raises(IndexPackError, match="contains no index.duckdb"):
        fetch_index(no_index.as_uri(), dest)

    no_manifest = _tarball(tmp_path, {INDEX_FILE_NAME: index_bytes, "../evil": b"x"})
    with pytest.raises(IndexPackError, match="contains no manifest.json"):
        fetch_index(no_manifest.as_uri(), dest)
    assert not (tmp_path / "evil").exists()


def test_pack_index_errors(
    tmp_path: Path, file_store_factory: Callable[[str], DuckDBStore]
) -> None:
    with pytest.raises(FileNotFoundError):
        pack_index(tmp_path / "missing.duckdb", tmp_path / "out.tar.zst")
    # A DuckDB file that was never initialised has no manifest.
    import duckdb

    raw = tmp_path / "raw.duckdb"
    duckdb.connect(str(raw)).close()
    with pytest.raises(ConfigError):
        pack_index(raw, tmp_path / "out.tar.zst")
    assert not list(tmp_path.glob(".out.tar.zst.*"))
