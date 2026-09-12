"""Tests for secqa.edgar.cache.DiskCache."""

from __future__ import annotations

import hashlib
from pathlib import Path

from secqa.edgar.cache import DiskCache


def test_key_is_sha256_of_url(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    url = "https://data.sec.gov/submissions/CIK0001234567.json"
    assert cache.key(url) == hashlib.sha256(url.encode()).hexdigest()
    assert cache.body_path(url) == tmp_path / f"{cache.key(url)}.bin"


def test_miss_then_put_then_hit(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "nested" / "dir")  # created on first put
    url = "https://example.invalid/a"
    assert cache.get(url) is None
    assert not cache.has(url)
    cache.put(url, b"payload", content_type="application/json")
    assert cache.has(url)
    assert cache.get(url) == b"payload"
    meta = cache.get_meta(url)
    assert meta is not None
    assert meta.url == url
    assert meta.size == 7
    assert meta.content_type == "application/json"
    assert meta.sha256 == cache.key(url)


def test_put_overwrites_and_delete_removes_both_files(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    url = "https://example.invalid/b"
    cache.put(url, b"one")
    cache.put(url, b"two")
    assert cache.get(url) == b"two"
    assert cache.delete(url) is True
    assert cache.get(url) is None
    assert cache.get_meta(url) is None
    assert cache.delete(url) is False
    # No temp files left behind by the atomic writes.
    assert list(tmp_path.iterdir()) == []


def test_different_urls_do_not_collide(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put("https://example.invalid/x?page=1", b"1")
    cache.put("https://example.invalid/x?page=2", b"2")
    assert cache.get("https://example.invalid/x?page=1") == b"1"
    assert cache.get("https://example.invalid/x?page=2") == b"2"
