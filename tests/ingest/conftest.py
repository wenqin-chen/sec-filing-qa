"""Fixtures shared by the ingest tests (all offline)."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def mini_10k_html() -> bytes:
    """The synthetic inline-XBRL-style mini 10-K in ``tests/fixtures/mini_10k.html``."""
    return (FIXTURES / "mini_10k.html").read_bytes()
