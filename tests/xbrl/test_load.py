"""flatten_companyfacts / load_companyfacts: row counts, dtypes, dedup, idempotency, errors."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from secqa.core.contracts import FactRow
from secqa.core.errors import ConfigError
from secqa.store import DuckDBStore
from secqa.xbrl import flatten_companyfacts, load_companyfacts
from secqa.xbrl.load import normalize_ticker, pad_cik
from tests.xbrl.conftest import (
    ACCN_FY2021,
    ACCN_FY2023,
    CIK,
    EMBED_DIM,
    N_FIXTURE_ENTRIES,
    N_FIXTURE_ROWS,
    TICKER,
)

# ---- flatten --------------------------------------------------------------------------------


def _count_entries(doc: dict[str, Any]) -> int:
    return sum(
        len(entries)
        for taxonomy in doc["facts"].values()
        for concept in taxonomy.values()
        for entries in concept["units"].values()
    )


def test_flatten_row_count_drops_exact_duplicate(companyfacts: dict[str, Any]) -> None:
    assert _count_entries(companyfacts) == N_FIXTURE_ENTRIES
    rows = flatten_companyfacts(companyfacts, "fixt")
    assert len(rows) == N_FIXTURE_ROWS
    assert all(isinstance(row, FactRow) for row in rows)
    keys = {(r.taxonomy, r.tag, r.unit, r.start_date, r.end_date, r.fy, r.fp, r.accn) for r in rows}
    assert len(keys) == len(rows)


def test_flatten_dtypes_and_normalisation(companyfacts: dict[str, Any]) -> None:
    rows = flatten_companyfacts(companyfacts, " fixt ")
    assert {row.cik for row in rows} == {CIK}
    assert {row.ticker for row in rows} == {TICKER}
    assert {row.taxonomy for row in rows} == {"us-gaap", "dei"}
    for row in rows:
        assert isinstance(row.val, float)
        assert isinstance(row.end_date, date)
        assert row.start_date is None or isinstance(row.start_date, date)
        assert isinstance(row.filed, date)
        assert isinstance(row.fy, int)
        assert row.concept_used is None
    eps = [r for r in rows if r.tag == "EarningsPerShareDiluted"]
    assert len(eps) == 1 and eps[0].unit == "USD/shares" and eps[0].val == 2.45
    shares = [r for r in rows if r.taxonomy == "dei"]
    assert {s.unit for s in shares} == {"shares"} and {s.start_date for s in shares} == {None}
    # ``frame`` is optional in the source and stays None when absent.
    restated = next(r for r in rows if r.tag == "Revenues" and r.accn == ACCN_FY2023)
    original = next(r for r in rows if r.tag == "Revenues" and r.accn == ACCN_FY2021)
    assert restated.frame == "CY2021" and restated.val == 1_210_000_000.0
    assert original.frame is None and original.val == 1_200_000_000.0
    assert original.ref == f"xbrl:Revenues|FY2021|{ACCN_FY2021}"


def test_flatten_keeps_distinct_filings_and_periods(companyfacts: dict[str, Any]) -> None:
    """Different accessions for the same period are distinct facts; so are Q4 and annual."""
    rows = flatten_companyfacts(companyfacts, TICKER)
    fy2021_revenue = [r for r in rows if r.tag == "Revenues" and r.end_date == date(2021, 12, 31)]
    assert sorted(r.accn for r in fy2021_revenue) == sorted(
        ["0001234567-22-000010", "0001234567-23-000010", ACCN_FY2023]
    )
    q4 = [r for r in rows if r.tag == "Revenues" and r.start_date == date(2022, 10, 1)]
    assert len(q4) == 1 and q4[0].val == 380_000_000.0
    tenq = [r for r in rows if r.form == "10-Q"]
    assert len(tenq) == 2 and {r.fp for r in tenq} == {"Q2", "Q1"}


def test_flatten_is_sorted_deterministically(companyfacts: dict[str, Any]) -> None:
    rows = flatten_companyfacts(companyfacts, TICKER)
    keys = [
        (r.taxonomy, r.tag, r.unit, r.end_date, r.start_date or date.min, r.filed, r.accn)
        for r in rows
    ]
    assert keys == sorted(keys)


def test_flatten_skips_unusable_entries() -> None:
    doc = {
        "cik": "CIK0000000042",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {"end": "2023-12-31", "val": 10, "accn": "a", "fy": 2023, "fp": "FY"},
                            {"end": "2023-12-31", "accn": "b", "fy": 2023, "fp": "FY"},  # no val
                            {"end": "2023-12-31", "val": "n/a", "accn": "c"},  # non-numeric
                            {"end": "not-a-date", "val": 1, "accn": "d"},
                            {"val": 1, "accn": "e"},  # no end
                            {"end": "2023-12-31", "val": float("nan"), "accn": "f"},
                            "garbage",
                        ]
                    }
                },
                "Broken": "not a concept",
            },
            "junk": [],
        },
    }
    rows = flatten_companyfacts(doc, "abc")
    assert [(r.cik, r.tag, r.val, r.accn, r.fy, r.fp) for r in rows] == [
        ("0000000042", "Assets", 10.0, "a", 2023, "FY")
    ]


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ({"facts": {}}, "no 'cik'"),
        ({"cik": 1}, "no 'facts'"),
        ({"cik": "abc", "facts": {}}, "invalid CIK"),
        ({"cik": 0, "facts": {}}, "invalid CIK"),
        ([], "JSON object"),
    ],
)
def test_flatten_rejects_malformed_document(doc: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        flatten_companyfacts(doc, TICKER)


@pytest.mark.parametrize("ticker", ["", "   ", "toolongticker", "bad ticker", "a;b"])
def test_flatten_rejects_bad_ticker(companyfacts: dict[str, Any], ticker: str) -> None:
    with pytest.raises(ValueError, match="invalid ticker"):
        flatten_companyfacts(companyfacts, ticker)


def test_helpers() -> None:
    assert pad_cik(320193) == "0000320193"
    assert pad_cik("CIK0000320193") == "0000320193"
    assert normalize_ticker(" brk.b ") == "BRK.B"


# ---- load -----------------------------------------------------------------------------------


def test_load_round_trips_known_values(store: DuckDBStore, companyfacts: dict[str, Any]) -> None:
    assert load_companyfacts(store, companyfacts, "fixt") == N_FIXTURE_ROWS
    assert store.counts()["facts"] == N_FIXTURE_ROWS
    assert store.manifest().n_facts == N_FIXTURE_ROWS
    row = store.conn.execute(
        "SELECT cik, ticker, taxonomy, unit, fy, fp, form, start_date, end_date, val, filed, frame "
        "FROM xbrl_facts WHERE tag = 'Revenues' AND accn = ?",
        [ACCN_FY2023],
    ).fetchall()
    assert len(row) == 2  # FY2021 restated comparative + FY2022 comparative
    restated = next(r for r in row if r[8] == date(2021, 12, 31))
    assert restated == (
        CIK,
        TICKER,
        "us-gaap",
        "USD",
        2023,
        "FY",
        "10-K",
        date(2021, 1, 1),
        date(2021, 12, 31),
        1_210_000_000.0,
        date(2024, 2, 15),
        "CY2021",
    )
    types = dict(
        store.conn.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name = 'xbrl_facts'"
        ).fetchall()
    )
    assert types["val"] == "DOUBLE" and types["fy"] == "INTEGER" and types["end_date"] == "DATE"
    # Instants store NULL start dates; the dei frame-less rows store NULL frames.
    assert store.conn.execute(
        "SELECT count(*) FROM xbrl_facts WHERE start_date IS NULL"
    ).fetchone() == (6,)
    assert store.conn.execute("SELECT count(*) FROM xbrl_facts WHERE frame IS NULL").fetchone() == (
        11,
    )


def test_load_is_idempotent_and_replaces_per_cik(
    store: DuckDBStore, companyfacts: dict[str, Any]
) -> None:
    load_companyfacts(store, companyfacts, TICKER)
    load_companyfacts(store, companyfacts, TICKER)
    assert store.counts()["facts"] == N_FIXTURE_ROWS
    # A second company is untouched when the first is reloaded with fewer rows.
    other = {
        "cik": 99,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2023-12-31",
                                "val": 5,
                                "accn": "x",
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                            }
                        ]
                    }
                }
            }
        },
    }
    assert load_companyfacts(store, other, "OTHR") == 1
    trimmed = dict(companyfacts)
    trimmed["facts"] = {"us-gaap": {"Assets": companyfacts["facts"]["us-gaap"]["Assets"]}}
    assert load_companyfacts(store, trimmed, TICKER) == 4
    counts = dict(
        store.conn.execute("SELECT ticker, count(*) FROM xbrl_facts GROUP BY ticker").fetchall()
    )
    assert counts == {TICKER: 4, "OTHR": 1}
    assert store.manifest().n_facts == 5


def test_load_empty_document_clears_company(
    store: DuckDBStore, companyfacts: dict[str, Any]
) -> None:
    load_companyfacts(store, companyfacts, TICKER)
    assert load_companyfacts(store, {"cik": 1234567, "facts": {}}, TICKER) == 0
    assert store.counts()["facts"] == 0
    assert store.manifest().n_facts == 0


def test_load_rejects_read_only_store(tmp_path: Path, companyfacts: dict[str, Any]) -> None:
    path = tmp_path / "index.duckdb"
    with DuckDBStore(path, embed_dim=EMBED_DIM) as writable:
        writable.init_schema("hashing-test", EMBED_DIM)
    with DuckDBStore(path, read_only=True) as ro_store:
        with pytest.raises(ConfigError, match="read-only"):
            load_companyfacts(ro_store, companyfacts, TICKER)
