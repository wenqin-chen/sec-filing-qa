"""Index provenance and portability: manifest, ``index-*.tar.zst`` packing, fetching.

* :func:`build_manifest` stamps a built store with the git SHA, row counts, a SHA-256 over
  every input file and the dataset revision, so ``/version`` and every ``EvalRecord`` can say
  exactly which index answered (SPEC 4.2, 9).
* :func:`pack_index` writes ``index.duckdb`` + ``manifest.json`` into a zstd-compressed tarball
  (the GitHub Release asset); ``manifest.json`` carries the SHA-256 of the DuckDB file so a
  fetch can verify what it unpacked.
* :func:`fetch_index` downloads that tarball (or a bare ``.duckdb``) over ``https://``,
  ``gs://`` (``google-cloud-storage`` optional extra, imported lazily) or ``file://`` (tests,
  local mirrors), verifies it and places ``index.duckdb`` where the container expects it.
  Only two members are ever extracted, by name, through ``extractfile`` -- no path from the
  archive touches the filesystem.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import httpx

from secqa.core.errors import ConfigError, SecqaError
from secqa.core.ids import file_sha256, sha256_hex
from secqa.core.logging import get_logger
from secqa.store import DuckDBStore, IndexManifest
from secqa.store.manifest import utc_now

log = get_logger(__name__)

INDEX_FILE_NAME = "index.duckdb"
INDEX_MANIFEST_FILE_NAME = "manifest.json"
PACK_FORMAT = "secqa-index/1"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
DUCKDB_MAGIC = b"DUCK"  # at byte offset 8 of every DuckDB file
DUCKDB_MAGIC_OFFSET = 8
DEFAULT_ZSTD_LEVEL = 10
DOWNLOAD_USER_AGENT = "sec-filing-qa/0.1 (+https://github.com/wenqinchen/sec-filing-qa)"
_COPY_CHUNK = 1 << 20


class IndexPackError(SecqaError):
    """A tarball could not be built, downloaded, verified or unpacked."""


# ---------------------------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------------------------


def _expand_inputs(inputs: list[Path]) -> list[Path]:
    """Files from ``inputs`` (directories recursively), deduplicated and sorted."""
    files: set[Path] = set()
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files.update(p for p in path.rglob("*") if p.is_file())
        elif path.is_file():
            files.add(path)
        else:
            raise FileNotFoundError(f"index input not found: {path}")
    return sorted(files)


def hash_inputs(inputs: list[Path]) -> str:
    """SHA-256 over ``'<name>\\t<sha256>\\n'`` lines of every input file, sorted by path.

    ``name`` is the file name (not the absolute path) so the hash is the same on every
    machine that holds the same files; an empty input list hashes the empty string.
    """
    lines = [f"{path.name}\t{file_sha256(path)}\n" for path in _expand_inputs(inputs)]
    return sha256_hex("".join(lines))


def build_manifest(
    store: DuckDBStore,
    git_sha: str,
    inputs: list[Path],
    dataset_revision: str | None,
) -> IndexManifest:
    """Refresh the store's manifest after a build and return it.

    Writes ``git_sha``, ``built_at`` (now, UTC), the live row counts, ``inputs_sha256`` (see
    :func:`hash_inputs`) and ``dataset_revision``; ``embedder`` / ``dim`` are fixed at
    ``init_schema`` and untouched.

    Raises:
        ConfigError: on a read-only or uninitialised store.
        FileNotFoundError: when an input path does not exist.
    """
    if store.read_only:
        raise ConfigError("build_manifest is not allowed on a read-only store")
    if not git_sha or not git_sha.strip():
        raise ValueError("git_sha must not be empty (use secqa.store.resolve_git_sha())")
    counts = store.counts()
    store.set_manifest(
        git_sha=git_sha.strip(),
        built_at=utc_now(),
        n_documents=counts["documents"],
        n_pages=counts["pages"],
        n_chunks=counts["chunks"],
        n_facts=counts["facts"],
        inputs_sha256=hash_inputs(inputs),
        dataset_revision=dataset_revision,
    )
    manifest = store.manifest()
    log.info(
        "index_manifest_built",
        embedder=manifest.embedder,
        dim=manifest.dim,
        git_sha=manifest.git_sha,
        n_documents=manifest.n_documents,
        n_pages=manifest.n_pages,
        n_chunks=manifest.n_chunks,
        n_facts=manifest.n_facts,
        inputs_sha256=manifest.inputs_sha256,
        dataset_revision=manifest.dataset_revision,
    )
    return manifest


# ---------------------------------------------------------------------------------------------
# pack
# ---------------------------------------------------------------------------------------------


def _checkpoint_and_read_manifest(store_path: Path) -> IndexManifest:
    """Open the file read-write so DuckDB folds the WAL into it, then read its manifest."""
    with DuckDBStore(store_path) as store:
        manifest = store.manifest()
        store.conn.execute("CHECKPOINT")
    return manifest


def pack_index(store_path: Path, out: Path, *, level: int = DEFAULT_ZSTD_LEVEL) -> Path:
    """Write ``index.duckdb`` + ``manifest.json`` from ``store_path`` into the tarball ``out``.

    The store must be closed by the caller (the file is opened briefly to checkpoint the WAL
    and read the manifest). ``out`` is written atomically via a sibling temp file. Returns
    ``out``.

    Raises:
        FileNotFoundError: when ``store_path`` does not exist.
        ConfigError: when the file has no index manifest (never initialised).
    """
    import zstandard

    store_path = Path(store_path)
    out = Path(out)
    if not store_path.is_file():
        raise FileNotFoundError(f"index not found: {store_path}")
    started = time.perf_counter()
    manifest = _checkpoint_and_read_manifest(store_path)
    index_sha = file_sha256(store_path)
    index_bytes = store_path.stat().st_size
    meta = {
        "format": PACK_FORMAT,
        "index_file": INDEX_FILE_NAME,
        "index_sha256": index_sha,
        "index_bytes": index_bytes,
        "packed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "manifest": manifest.model_dump(mode="json"),
    }
    meta_bytes = json.dumps(meta, indent=2, sort_keys=True).encode("utf-8") + b"\n"

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{out.name}.", suffix=".tmp", dir=out.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            compressor = zstandard.ZstdCompressor(level=level)
            with compressor.stream_writer(raw, closefd=False) as zst:
                with tarfile.open(fileobj=zst, mode="w|") as tar:
                    info = tarfile.TarInfo(INDEX_MANIFEST_FILE_NAME)
                    info.size = len(meta_bytes)
                    info.mtime = int(time.time())
                    tar.addfile(info, io.BytesIO(meta_bytes))
                    tar.add(str(store_path), arcname=INDEX_FILE_NAME, recursive=False)
        os.replace(tmp_name, out)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
    log.info(
        "index_packed",
        store_path=str(store_path),
        out=str(out),
        index_bytes=index_bytes,
        tarball_bytes=out.stat().st_size,
        index_sha256=index_sha,
        n_chunks=manifest.n_chunks,
        seconds=round(time.perf_counter() - started, 2),
    )
    return out


# ---------------------------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------------------------


def _download_https(url: str, target: Path, timeout_s: float) -> None:
    with httpx.Client(
        timeout=httpx.Timeout(timeout_s),
        follow_redirects=True,
        headers={"User-Agent": DOWNLOAD_USER_AGENT},
    ) as client:
        try:
            with client.stream("GET", url) as response:
                if response.is_error:
                    raise IndexPackError(f"HTTP {response.status_code} fetching {url}")
                with target.open("wb") as fh:
                    for block in response.iter_bytes():
                        fh.write(block)
        except httpx.HTTPError as exc:
            raise IndexPackError(f"network error fetching {url}: {exc}") from exc


def _download_gcs(url: str, target: Path) -> None:
    try:
        from google.cloud import storage  # optional extra [gcs]; imported lazily
    except ImportError as exc:
        raise ConfigError(
            "fetching gs:// URLs needs google-cloud-storage: `uv sync --extra gcs`"
        ) from exc
    parsed = urlparse(url)
    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")
    if not bucket_name or not blob_name:
        raise IndexPackError(f"gs:// URL must be gs://<bucket>/<object>, got {url!r}")
    try:
        client = storage.Client()
        client.bucket(bucket_name).blob(blob_name).download_to_filename(str(target))
    except Exception as exc:  # the GCS SDK raises many types; none are ours
        raise IndexPackError(f"could not download {url}: {exc}") from exc


def _copy_file_url(url: str, target: Path) -> None:
    parsed = urlparse(url)
    source = Path(url2pathname(parsed.path))
    if parsed.netloc not in ("", "localhost"):
        raise IndexPackError(f"file:// URL must be local, got {url!r}")
    if not source.is_file():
        raise IndexPackError(f"file not found: {source}")
    shutil.copyfile(source, target)


def _download(url: str, target: Path, timeout_s: float) -> None:
    scheme = urlparse(url).scheme.lower()
    if scheme == "https":
        _download_https(url, target, timeout_s)
    elif scheme == "gs":
        _download_gcs(url, target)
    elif scheme == "file":
        _copy_file_url(url, target)
    else:
        raise ConfigError(
            f"unsupported index URL {url!r}: expected https://, gs:// or file:// "
            "(plain http is refused so an index is never fetched over an unauthenticated link)"
        )


def _head(path: Path, n: int) -> bytes:
    with path.open("rb") as fh:
        return fh.read(n)


def is_duckdb_file(path: Path) -> bool:
    """True when ``path`` carries the ``DUCK`` marker at byte offset 8."""
    head = _head(path, DUCKDB_MAGIC_OFFSET + len(DUCKDB_MAGIC))
    return head[DUCKDB_MAGIC_OFFSET:] == DUCKDB_MAGIC


def _unpack_tarball(archive: Path, dest: Path) -> dict[str, Any]:
    """Extract ``index.duckdb`` to ``dest`` and return the parsed ``manifest.json``.

    Members are read by name through ``extractfile``; nothing else in the archive is touched.
    """
    import zstandard

    meta: dict[str, Any] | None = None
    index_tmp = dest.with_name(f".{dest.name}.unpack")
    found_index = False
    with archive.open("rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as zst:
            with tarfile.open(fileobj=zst, mode="r|") as tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    if member.name == INDEX_MANIFEST_FILE_NAME:
                        fh = tar.extractfile(member)
                        meta = json.loads((fh.read() if fh else b"{}").decode("utf-8"))
                    elif member.name == INDEX_FILE_NAME:
                        fh = tar.extractfile(member)
                        if fh is None:
                            raise IndexPackError("index.duckdb member is not a regular file")
                        with index_tmp.open("wb") as out:
                            shutil.copyfileobj(fh, out, _COPY_CHUNK)
                        found_index = True
    if not found_index:
        index_tmp.unlink(missing_ok=True)
        raise IndexPackError(f"{archive.name} contains no {INDEX_FILE_NAME}")
    if not isinstance(meta, dict):
        index_tmp.unlink(missing_ok=True)
        raise IndexPackError(f"{archive.name} contains no {INDEX_MANIFEST_FILE_NAME}")
    expected = meta.get("index_sha256")
    actual = file_sha256(index_tmp)
    if expected != actual:
        index_tmp.unlink(missing_ok=True)
        raise IndexPackError(
            f"index.duckdb sha256 mismatch: manifest says {expected}, file is {actual}"
        )
    os.replace(index_tmp, dest)
    return meta


def fetch_index(url: str, dest: Path, *, timeout_s: float = 600.0) -> Path:
    """Download an index to ``dest`` (the ``.duckdb`` path) and return ``dest``.

    Accepts a tarball made by :func:`pack_index` (verified against its ``manifest.json``,
    which is also written next to ``dest`` as ``<dest>.manifest.json``) or a bare DuckDB file
    (magic-byte checked). Supported schemes: ``https://``, ``gs://``, ``file://``. Downloads
    go to a temp file in ``dest``'s directory and ``dest`` only appears once verified.

    Raises:
        ConfigError: unsupported scheme or missing optional dependency.
        IndexPackError: download, integrity or format failure.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".download", dir=dest.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        _download(url, tmp, timeout_s)
        head = _head(tmp, DUCKDB_MAGIC_OFFSET + len(DUCKDB_MAGIC))
        if head.startswith(ZSTD_MAGIC):
            meta = _unpack_tarball(tmp, dest)
            manifest_path = dest.with_name(f"{dest.name}.manifest.json")
            manifest_path.write_text(
                json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            kind = "tarball"
        elif head[DUCKDB_MAGIC_OFFSET:] == DUCKDB_MAGIC:
            os.replace(tmp, dest)
            kind = "duckdb"
        else:
            raise IndexPackError(
                f"{url} is neither a zstd tarball nor a DuckDB file (head={head[:4]!r})"
            )
    finally:
        tmp.unlink(missing_ok=True)
    log.info(
        "index_fetched",
        url=url,
        dest=str(dest),
        kind=kind,
        bytes=dest.stat().st_size,
        seconds=round(time.perf_counter() - started, 2),
    )
    return dest


__all__ = [
    "DUCKDB_MAGIC",
    "INDEX_FILE_NAME",
    "INDEX_MANIFEST_FILE_NAME",
    "PACK_FORMAT",
    "ZSTD_MAGIC",
    "IndexPackError",
    "build_manifest",
    "fetch_index",
    "hash_inputs",
    "is_duckdb_file",
    "pack_index",
]
