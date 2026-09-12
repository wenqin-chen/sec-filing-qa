"""Tests for secqa.core.ids."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from secqa.core.ids import chunk_id, file_sha256, request_id, sha256_hex


def test_chunk_id_is_deterministic_sha1_of_payload() -> None:
    cid = chunk_id("3M_2022_10K", 12, 3, "Net sales were $34.2 billion.")
    expected = hashlib.sha1(b"3M_2022_10K|12|3|Net sales were $34.2 billion.").hexdigest()
    assert cid == expected
    assert len(cid) == 40
    assert cid == chunk_id("3M_2022_10K", 12, 3, "Net sales were $34.2 billion.")


def test_chunk_id_changes_with_any_component() -> None:
    base = chunk_id("D", 1, 0, "text")
    assert chunk_id("E", 1, 0, "text") != base
    assert chunk_id("D", 2, 0, "text") != base
    assert chunk_id("D", 1, 1, "text") != base
    assert chunk_id("D", 1, 0, "text ") != base


def test_chunk_id_rejects_bad_indices() -> None:
    with pytest.raises(ValueError, match="1-based"):
        chunk_id("D", 0, 0, "text")
    with pytest.raises(ValueError, match="0-based"):
        chunk_id("D", 1, -1, "text")


def test_request_id_is_uuid4_hex_and_unique() -> None:
    a, b = request_id(), request_id()
    assert a != b
    assert len(a) == 32
    int(a, 16)  # hex


def test_sha256_hex_text_and_bytes_agree() -> None:
    assert sha256_hex("abc") == sha256_hex(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_file_sha256(tmp_path: Path) -> None:
    p = tmp_path / "blob.bin"
    data = b"x" * (3 * 1024 * 1024 + 17)
    p.write_bytes(data)
    assert file_sha256(p) == hashlib.sha256(data).hexdigest()
    assert file_sha256(p, chunk_size=1000) == hashlib.sha256(data).hexdigest()
