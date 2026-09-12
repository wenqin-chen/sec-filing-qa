"""On-disk response cache keyed by ``sha256(url)``.

Every successful EDGAR response body is stored once under ``<cache_dir>/<sha256>.bin`` with a
sidecar ``<sha256>.json`` (url, content type, size, fetch time) so the directory is inspectable
and a cache hit never touches the network or the rate limiter. Writes go through a temporary
file and ``os.replace`` so a crash mid-write cannot leave a truncated body behind.

Only success bodies are cached; errors and retries never write to disk.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from secqa.core.ids import sha256_hex


@dataclass(frozen=True)
class CacheEntry:
    """Metadata written next to every cached body."""

    url: str
    sha256: str
    size: int
    content_type: str | None
    fetched_at: str  # ISO-8601 UTC


class DiskCache:
    """Content cache for HTTP responses, keyed by the SHA-256 of the request URL."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @staticmethod
    def key(url: str) -> str:
        """Cache key for a URL: hex SHA-256 of the exact URL string."""
        return sha256_hex(url)

    def body_path(self, url: str) -> Path:
        """Path of the cached body for ``url`` (may not exist)."""
        return self.root / f"{self.key(url)}.bin"

    def meta_path(self, url: str) -> Path:
        """Path of the metadata sidecar for ``url`` (may not exist)."""
        return self.root / f"{self.key(url)}.json"

    def has(self, url: str) -> bool:
        """True if a body for ``url`` is cached."""
        return self.body_path(url).is_file()

    def get(self, url: str) -> bytes | None:
        """Return the cached body for ``url`` or ``None`` on a miss."""
        path = self.body_path(url)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return None

    def get_meta(self, url: str) -> CacheEntry | None:
        """Return the metadata sidecar for ``url`` or ``None`` if absent."""
        path = self.meta_path(url)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        return CacheEntry(
            url=str(raw["url"]),
            sha256=str(raw["sha256"]),
            size=int(raw["size"]),
            content_type=raw.get("content_type"),
            fetched_at=str(raw["fetched_at"]),
        )

    def put(self, url: str, body: bytes, content_type: str | None = None) -> Path:
        """Atomically store ``body`` for ``url``; returns the body path."""
        self.root.mkdir(parents=True, exist_ok=True)
        body_path = self.body_path(url)
        entry = CacheEntry(
            url=url,
            sha256=self.key(url),
            size=len(body),
            content_type=content_type,
            fetched_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        self._atomic_write(body_path, body)
        self._atomic_write(
            self.meta_path(url),
            json.dumps(entry.__dict__, indent=2, sort_keys=True).encode("utf-8"),
        )
        return body_path

    def delete(self, url: str) -> bool:
        """Remove the cached body and sidecar for ``url``; True if a body existed."""
        existed = False
        for path in (self.body_path(url), self.meta_path(url)):
            try:
                path.unlink()
                existed = existed or path.suffix == ".bin"
            except FileNotFoundError:
                pass
        return existed

    def _atomic_write(self, path: Path, data: bytes) -> None:
        """Write via a sibling temp file + ``os.replace`` so readers never see partial content."""
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except FileNotFoundError:
                pass
            raise


__all__ = ["CacheEntry", "DiskCache"]
