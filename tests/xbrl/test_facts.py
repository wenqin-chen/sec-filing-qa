"""lookup_fact: alias chain, derived fiscal year, accession numbers, escape hatch, errors."""

from __future__ import annotations

from datetime import date

import pytest

from secqa.core.contracts import FactRow
from secqa.grounding import parse_ref
from secqa.store import DuckDBStore
from secqa.xbrl import CURATED_METRICS, lookup_fact, resolve_metric
from tests.xbrl.conftest import ACCN_FY2021, ACCN_FY2023, CIK, TICKER


def test_lookup_returns_accession_and_ref(loaded_store: DuckDBStore) -> None:
    rows = lookup_fact(loaded_store, "fixt", "revenue", 2023)
    assert len(rows) == 1
    fact = rows[0]
    assert isinstance(fact, FactRow)
    assert fact.accn == ACCN_FY2023
    assert fact.filed == date(2024, 2, 15)
    assert fact.form == "10-K"
    assert fact.val == 1_577_000_000.0 and fact.unit == "USD"
    assert fact.cik == CIK and fact.ticker == TICKER
    assert fact.fy == 2023 and fact.fp == "FY"
    assert fact.start_date == date(2023, 1, 1) and fact.end_date == date(2023, 12, 31)
    # alias chain: no `Revenues` in FY2023 -> second alias resolved
    assert fact.tag == "RevenueFromContractWithCustomerExcludingAssessedTax"
    assert fact.concept_used == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    assert fact.ref == (
        f"xbrl:RevenueFromContractWithCustomerExcludingAssessedTax|FY2023|{ACCN_FY2023}"
    )
    assert parse_ref(fact.ref)[0] == "xbrl"


def test_lookup_uses_latest_filing_and_derived_fiscal_year(loaded_store: DuckDBStore) -> None:
    (fact,) = lookup_fact(loaded_store, TICKER, "revenue", 2021)
    assert fact.tag == "Revenues" and fact.concept_used == "us-gaap:Revenues"
    assert fact.val == 1_210_000_000.0  # restated value from the FY2023 10-K wins
    assert fact.accn == ACCN_FY2023 and fact.frame == "CY2021"
    assert fact.fy == 2021  # the year the value belongs to, not the filing's fy=2023
    (older,) = lookup_fact(loaded_store, TICKER, "net_income", 2019)
    assert older.val == 150_000_000.0 and older.accn == ACCN_FY2021 and older.fy == 2019


def test_lookup_instants_and_dei(loaded_store: DuckDBStore) -> None:
    (assets,) = lookup_fact(loaded_store, TICKER, "total_assets", 2023)
    assert assets.val == 5_600_000_000.0
    assert assets.accn == ACCN_FY2023  # the 10-K row, not the 10-Q comparative
    assert assets.start_date is None and assets.end_date == date(2023, 12, 31)
    (shares,) = lookup_fact(loaded_store, TICKER, "shares_outstanding", 2023)
    assert shares.taxonomy == "dei" and shares.unit == "shares"
    assert shares.val == 100_000_000.0 and shares.end_date == date(2024, 1, 31)
    assert shares.concept_used == "dei:EntityCommonStockSharesOutstanding"
    (eps,) = lookup_fact(loaded_store, TICKER, "EPS_DILUTED", 2023)  # case-insensitive
    assert eps.unit == "USD/shares" and eps.val == 2.45


def test_lookup_misses_return_empty(loaded_store: DuckDBStore) -> None:
    assert lookup_fact(loaded_store, TICKER, "gross_profit", 2023) == []
    assert lookup_fact(loaded_store, TICKER, "revenue", 2030) == []
    assert lookup_fact(loaded_store, "OTHR", "revenue", 2023) == []
    # A Q2 10-Q value never counts as an annual fact.
    assert lookup_fact(loaded_store, TICKER, "revenue", 2023)[0].val != 350_000_000.0


def test_lookup_raw_concept_escape_hatch(loaded_store: DuckDBStore) -> None:
    (fact,) = lookup_fact(loaded_store, TICKER, "Assets", 2022)
    assert fact.val == 5_000_000_000.0 and fact.concept_used == "us-gaap:Assets"
    (fact,) = lookup_fact(loaded_store, TICKER, "dei:EntityCommonStockSharesOutstanding", 2022)
    assert fact.val == 102_000_000.0
    assert lookup_fact(loaded_store, TICKER, "NoSuchConceptAnywhere", 2022) == []


def test_lookup_returns_one_row_per_unit_for_raw_concepts(store: DuckDBStore) -> None:
    from secqa.xbrl import load_companyfacts

    doc = {
        "cik": 5,
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 100,
                                "accn": "a",
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-01",
                            }
                        ],
                        "EUR": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 90,
                                "accn": "a",
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-01",
                            }
                        ],
                    }
                }
            }
        },
    }
    load_companyfacts(store, doc, "MULT")
    raw = lookup_fact(store, "MULT", "Revenues", 2023)
    assert [(r.unit, r.val) for r in raw] == [("EUR", 90.0), ("USD", 100.0)]
    curated = lookup_fact(store, "MULT", "revenue", 2023)
    assert [(r.unit, r.val) for r in curated] == [("USD", 100.0)]  # unit enforced


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("revenue", ("revenue", ("us-gaap", "Revenues"), "USD")),
        ("Net_Income", ("net_income", ("us-gaap", "NetIncomeLoss"), "USD")),
        ("Assets", ("us-gaap:Assets", ("us-gaap", "Assets"), None)),
        ("dei:Thing", ("dei:Thing", ("dei", "Thing"), None)),
    ],
)
def test_resolve_metric(metric: str, expected: tuple[str, tuple[str, str], str | None]) -> None:
    name, chain, unit = resolve_metric(metric)
    assert (name, chain[0], unit) == expected
    if name in CURATED_METRICS:
        assert len(chain) == len(CURATED_METRICS[name])


@pytest.mark.parametrize("metric", ["", "  ", "net income", "revenue; drop", "us-gaap:"])
def test_resolve_metric_rejects(metric: str) -> None:
    with pytest.raises(ValueError, match="metric"):
        resolve_metric(metric)


def test_lookup_argument_validation(loaded_store: DuckDBStore) -> None:
    with pytest.raises(ValueError, match="invalid ticker"):
        lookup_fact(loaded_store, "", "revenue", 2023)
    with pytest.raises(ValueError, match="unknown metric"):
        lookup_fact(loaded_store, TICKER, "not a metric", 2023)
    with pytest.raises(ValueError, match="fiscal_year"):
        lookup_fact(loaded_store, TICKER, "revenue", "2023")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fiscal_year"):
        lookup_fact(loaded_store, TICKER, "revenue", True)
