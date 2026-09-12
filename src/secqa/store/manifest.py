"""Index manifest: what an ``index.duckdb`` file was built with.

The manifest lives in the ``index_manifest(key, value)`` table as strings and is the single
source of truth for the embedder name and dimension. Opening a store with a different embedder
or dimension raises :class:`~secqa.core.errors.IndexMismatch` so a query vector can never be
compared against vectors from another model.
"""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import field_validator

from secqa.core.contracts import Frozen

# Manifest keys, in the order they are written. Every key maps to an ``IndexManifest`` field.
MANIFEST_KEYS: tuple[str, ...] = (
    "embedder",
    "dim",
    "git_sha",
    "built_at",
    "n_documents",
    "n_pages",
    "n_chunks",
    "n_facts",
    "inputs_sha256",
    "dataset_revision",
)

_INT_KEYS = frozenset({"dim", "n_documents", "n_pages", "n_chunks", "n_facts"})
_UNKNOWN_SHA = "unknown"


class IndexManifest(Frozen):
    """Provenance of a built index (see SPEC 4.2)."""

    embedder: str
    dim: int
    git_sha: str
    built_at: datetime
    n_documents: int
    n_pages: int
    n_chunks: int
    n_facts: int
    inputs_sha256: str
    dataset_revision: str | None

    @field_validator("dim")
    @classmethod
    def _dim_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f"dim must be positive, got {value}")
        return value

    @field_validator("n_documents", "n_pages", "n_chunks", "n_facts")
    @classmethod
    def _counts_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError(f"counts must be non-negative, got {value}")
        return value

    def to_rows(self) -> list[tuple[str, str | None]]:
        """Serialise to ``(key, value)`` string rows for the ``index_manifest`` table."""
        return [(key, serialize_value(key, getattr(self, key))) for key in MANIFEST_KEYS]

    @classmethod
    def from_rows(cls, rows: dict[str, str | None]) -> IndexManifest:
        """Build a manifest from ``{key: value}`` strings read back from the table.

        Raises ``KeyError`` when a required key is absent (a store whose schema was created but
        never initialised through :meth:`DuckDBStore.init_schema`).
        """
        missing = [key for key in MANIFEST_KEYS if key not in rows and key != "dataset_revision"]
        if missing:
            raise KeyError(f"index_manifest is missing keys: {', '.join(missing)}")
        values: dict[str, Any] = {key: parse_value(key, rows.get(key)) for key in MANIFEST_KEYS}
        return cls(**values)


def serialize_value(key: str, value: Any) -> str | None:
    """Convert a manifest field to its stored string form (``None`` stays NULL)."""
    if key not in MANIFEST_KEYS:
        raise ValueError(f"unknown manifest key {key!r}; expected one of {MANIFEST_KEYS}")
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, bool):  # bool is an int subclass; reject explicitly for count keys
        raise TypeError(f"manifest key {key!r} cannot be a bool")
    return str(value)


def parse_value(key: str, raw: str | None) -> Any:
    """Inverse of :func:`serialize_value` for one key."""
    if key not in MANIFEST_KEYS:
        raise ValueError(f"unknown manifest key {key!r}; expected one of {MANIFEST_KEYS}")
    if raw is None:
        return None
    if key in _INT_KEYS:
        return int(raw)
    if key == "built_at":
        return datetime.fromisoformat(raw)
    return raw


def utc_now() -> datetime:
    """Timezone-aware current time; manifests always record UTC."""
    return datetime.now(tz=UTC)


def resolve_git_sha(repo_root: Path | None = None, timeout_s: float = 2.0) -> str:
    """Best-effort git commit for provenance.

    Order: ``SECQA_GIT_SHA`` environment variable (CI / Docker builds where ``.git`` is absent),
    then ``git rev-parse HEAD`` in ``repo_root`` (defaults to this package's repository), then
    ``'unknown'``. Never raises.
    """
    env_sha = os.environ.get("SECQA_GIT_SHA", "").strip()
    if env_sha:
        return env_sha
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return _UNKNOWN_SHA
    sha = completed.stdout.strip()
    if completed.returncode != 0 or len(sha) != 40:
        return _UNKNOWN_SHA
    return sha


__all__ = [
    "MANIFEST_KEYS",
    "IndexManifest",
    "parse_value",
    "resolve_git_sha",
    "serialize_value",
    "utc_now",
]
