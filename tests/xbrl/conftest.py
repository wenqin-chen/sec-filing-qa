"""XBRL test fixtures: the synthetic companyfacts document and stores loaded from it.

``tests/fixtures/xbrl_companyfacts_small.json`` is hand-made (ticker FIXT, CIK 1234567) and
deliberately contains the awkward cases companyfacts has in the wild: an exact duplicate entry,
a prior-year value restated in a later 10-K, a concept switch between years, a Q4 duration
inside a 10-K, 10-Q rows, a ``dei`` cover-date instant and a period that only ever appears as a
comparative. Expected numbers used across the tests are collected here so they are stated once.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from secqa.store import DuckDBStore
from secqa.xbrl import create_financials_view, load_companyfacts

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "xbrl_companyfacts_small.json"

TICKER = "FIXT"
CIK = "0001234567"
ACCN_FY2021 = "0001234567-22-000010"  # FY2021 10-K, filed 2022-02-15
ACCN_FY2022 = "0001234567-23-000010"  # FY2022 10-K, filed 2023-02-15
ACCN_FY2023 = "0001234567-24-000010"  # FY2023 10-K, filed 2024-02-15
ACCN_Q2_2023 = "0001234567-23-000050"  # 10-Q
ACCN_Q1_2024 = "0001234567-24-000030"  # 10-Q carrying the 2023-12-31 balance sheet comparative

N_FIXTURE_ENTRIES = 21  # raw entries in the JSON
N_FIXTURE_ROWS = 20  # after dropping the one exact duplicate

EMBED_DIM = 8


@pytest.fixture(scope="session")
def companyfacts_doc() -> dict[str, Any]:
    """The fixture document (session-scoped; tests copy before mutating)."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def companyfacts(companyfacts_doc: dict[str, Any]) -> dict[str, Any]:
    """A deep copy of the fixture document that a test may mutate freely."""
    return copy.deepcopy(companyfacts_doc)


@pytest.fixture
def store() -> Iterator[DuckDBStore]:
    """Empty, initialised in-memory store."""
    with DuckDBStore(":memory:", embed_dim=EMBED_DIM) as store:
        store.init_schema("hashing-test", EMBED_DIM)
        yield store


@pytest.fixture
def loaded_store(store: DuckDBStore, companyfacts: dict[str, Any]) -> DuckDBStore:
    """Store with the fixture facts loaded and the curated ``financials`` view created."""
    load_companyfacts(store, companyfacts, TICKER)
    create_financials_view(store)
    return store
