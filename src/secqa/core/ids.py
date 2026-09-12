"""Stable identifiers and content hashes.

Chunk ids are content-addressed so re-ingesting the same document yields the same ids (idempotent
indexing) while any text change produces a new id (no stale citations).
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path


def chunk_id(doc_name: str, page_num: int, chunk_idx: int, text: str) -> str:
    """SHA-1 hex digest of ``'<doc_name>|<page_num>|<chunk_idx>|<text>'``."""
    if page_num < 1:
        raise ValueError(f"page_num is 1-based, got {page_num}")
    if chunk_idx < 0:
        raise ValueError(f"chunk_idx is 0-based and non-negative, got {chunk_idx}")
    payload = f"{doc_name}|{page_num}|{chunk_idx}|{text}".encode()
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()


def request_id() -> str:
    """Random 32-char hex request id (uuid4)."""
    return uuid.uuid4().hex


def sha256_hex(data: bytes | str) -> str:
    """SHA-256 hex digest of bytes or UTF-8 text (used for prompt hashes and cassette keys)."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (source PDFs, index inputs)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = ["chunk_id", "file_sha256", "request_id", "sha256_hex"]
