"""Fixtures for the embeddings tests (all offline; the OpenAI backend is respx-mocked)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def openai_embeddings_payload() -> dict[str, Any]:
    """Hand-written ``POST /v1/embeddings`` body: 3 vectors of width 4, ``data`` out of order."""
    with (FIXTURES / "embeddings_openai_response.json").open(encoding="utf-8") as fh:
        payload: dict[str, Any] = json.load(fh)
    return copy.deepcopy(payload)
