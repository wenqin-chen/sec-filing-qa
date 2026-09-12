"""tags.yaml loading, the curated ``financials`` view and its fiscal-year / alias semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from secqa.core.errors import ConfigError
from secqa.store import DuckDBStore
from secqa.xbrl import (
    CURATED_METRICS,
    CURATED_UNITS,
    create_financials_view,
    financials_columns,
    load_tags,
    split_alias,
)
from secqa.xbrl.financials import (
    ALLOWED_UNITS,
    FINANCIALS_FIXED_COLUMNS,
    TAGS_PATH,
    financials_view_sql,
)
from tests.xbrl.conftest import CIK, EMBED_DIM, TICKER

# ---- tags.yaml ------------------------------------------------------------------------------


def test_curated_metrics_cover_the_contract_list() -> None:
    expected = {
        "revenue",
        "cost_of_revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "total_assets",
        "total_liabilities",
        "equity",
        "cash",
        "cfo",
        "capex",
        "dividends",
        "shares_outstanding",
    }
    assert expected <= set(CURATED_METRICS)
    assert len(CURATED_METRICS) >= 25
    assert set(CURATED_UNITS) == set(CURATED_METRICS)
    assert set(CURATED_UNITS.values()) <= ALLOWED_UNITS
    for metric, aliases in CURATED_METRICS.items():
        assert aliases, metric
        assert len(set(aliases)) == len(aliases), metric
        for alias in aliases:
            taxonomy, tag = split_alias(alias)
            assert taxonomy in {"us-gaap", "dei"}
            assert tag[0].isupper() and tag.isalnum()
    assert CURATED_METRICS["revenue"][0] == "Revenues"
    assert CURATED_METRICS["shares_outstanding"][0] == "dei:EntityCommonStockSharesOutstanding"
    assert CURATED_UNITS["eps_diluted"] == "USD/shares"


def test_load_tags_reads_the_shipped_file() -> None:
    metrics, units = load_tags(TAGS_PATH)
    assert metrics == CURATED_METRICS and units == CURATED_UNITS


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ("metrics: {}", "non-empty"),
        ("nothing: 1", "non-empty"),
        ("metrics:\n  Bad-Name:\n    unit: USD\n    tags: [Assets]\n", "invalid metric name"),
        ("metrics:\n  cik:\n    unit: USD\n    tags: [Assets]\n", "invalid metric name"),
        ("metrics:\n  x:\n    unit: EUR\n    tags: [Assets]\n", "unit"),
        ("metrics:\n  x:\n    unit: USD\n    tags: []\n", "non-empty 'tags'"),
        ("metrics:\n  x:\n    unit: USD\n    tags: ['Assets; DROP']\n", "invalid concept alias"),
        ("metrics:\n  x:\n    unit: USD\n    tags: [Assets, Assets]\n", "twice"),
        ("metrics:\n  x: [Assets]\n", "must map"),
        ("metrics: [\n", "cannot read"),
    ],
)
def test_load_tags_validates(tmp_path: Path, body: str, match: str) -> None:
    path = tmp_path / "tags.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ConfigError, match=match):
        load_tags(path)


def test_load_tags_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_tags(tmp_path / "absent.yaml")


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("Revenues", ("us-gaap", "Revenues")),
        ("dei:EntityCommonStockSharesOutstanding", ("dei", "EntityCommonStockSharesOutstanding")),
        (" ifrs-full:Revenue ", ("ifrs-full", "Revenue")),
    ],
)
def test_split_alias(alias: str, expected: tuple[str, str]) -> None:
    assert split_alias(alias) == expected


@pytest.mark.parametrize(
    "alias", ["", "revenues", "Rev enues", "us-gaap:", "x'--", "US-GAAP:Assets"]
)
def test_split_alias_rejects(alias: str) -> None:
    with pytest.raises(ValueError, match="invalid concept alias"):
        split_alias(alias)


# ---- the view -------------------------------------------------------------------------------


def _view_rows(store: DuckDBStore, *columns: str) -> dict[int, tuple[Any, ...]]:
    sql = f"SELECT fiscal_year, {', '.join(columns)} FROM financials ORDER BY fiscal_year"
    return {row[0]: row[1:] for row in store.conn.execute(sql).fetchall()}


def test_view_columns_and_replacement_of_baseline(store: DuckDBStore) -> None:
    baseline = [c[0] for c in store.conn.execute("SELECT * FROM financials LIMIT 0").description]
    assert "revenue" not in baseline  # the store's baseline view is a different shape
    create_financials_view(store)
    columns = [c[0] for c in store.conn.execute("SELECT * FROM financials LIMIT 0").description]
    assert columns == financials_columns() == [*FINANCIALS_FIXED_COLUMNS, *CURATED_METRICS]
    assert store.conn.execute("SELECT count(*) FROM financials").fetchone() == (0,)
    create_financials_view(store)  # idempotent
    # init_schema on an existing store runs CREATE VIEW IF NOT EXISTS -> curated view survives.
    store.init_schema("hashing-test", EMBED_DIM)
    columns_after = [
        c[0] for c in store.conn.execute("SELECT * FROM financials LIMIT 0").description
    ]
    assert columns_after == columns


def test_view_values_per_fiscal_year(loaded_store: DuckDBStore) -> None:
    rows = _view_rows(
        loaded_store,
        "revenue",
        "net_income",
        "total_assets",
        "shares_outstanding",
        "eps_diluted",
        "gross_profit",
    )
    assert rows == {
        # comparative-only period: no original filing in the data -> calendar year of the end
        2019: (None, 150_000_000.0, None, None, None, None),
        # restated in the FY2023 10-K: latest filing wins (1,210 not 1,200)
        2021: (1_210_000_000.0, 190_000_000.0, None, None, None, None),
        # Q4 duration inside the 10-K (380M) is not the annual value; the dei cover-date instant
        # (2023-01-31, fy 2022) is attributed to fiscal 2022 through the filing's fiscal focus
        2022: (1_400_000_000.0, 210_000_000.0, 5_000_000_000.0, 102_000_000.0, None, None),
        # alias fallback: no `Revenues` for 2023 -> RevenueFromContractWithCustomer...; the
        # 2023-12-31 balance sheet comes from the 10-K, not the later 10-Q comparative
        2023: (1_577_000_000.0, 250_000_000.0, 5_600_000_000.0, 100_000_000.0, 2.45, None),
    }
    ids = loaded_store.conn.execute("SELECT DISTINCT cik, ticker FROM financials").fetchall()
    assert ids == [(CIK, TICKER)]


def test_view_alias_priority_prefers_first_present(store: DuckDBStore) -> None:
    """When two aliases both have a value, the earlier one in tags.yaml wins."""
    doc = {
        "cik": 7,
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 999,
                                "accn": "a",
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-01",
                            },
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 1000,
                                "accn": "a",
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-01",
                            },
                            # FY2022 in USD (kept); the EUR row for the same period below is ignored
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "val": 5,
                                "accn": "a",
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-01",
                            },
                        ],
                        "EUR": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "val": 4,
                                "accn": "a",
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-01",
                            },
                        ],
                    }
                },
            }
        },
    }
    from secqa.xbrl import load_companyfacts

    load_companyfacts(store, doc, "ALIA")
    create_financials_view(store)
    assert _view_rows(store, "revenue") == {2022: (5.0,), 2023: (1000.0,)}


def test_view_ignores_non_annual_and_non_10k_rows(store: DuckDBStore) -> None:
    doc = {
        "cik": 8,
        "facts": {
            "us-gaap": {
                "Assets": {
                    "units": {
                        "USD": [
                            # only a 10-Q reports this instant -> not in the view
                            {
                                "end": "2023-12-31",
                                "val": 1,
                                "accn": "q",
                                "fy": 2024,
                                "fp": "Q1",
                                "form": "10-Q",
                                "filed": "2024-05-01",
                                "frame": "CY2023Q4I",
                            },
                            # 10-K/A is not '10-K'
                            {
                                "end": "2022-12-31",
                                "val": 2,
                                "accn": "ka",
                                "fy": 2022,
                                "fp": "FY",
                                "form": "10-K/A",
                                "filed": "2023-06-01",
                            },
                        ]
                    }
                },
                "Revenues": {
                    "units": {
                        "USD": [
                            # a 10-K row whose duration is not a year (Q4 in the 10-K)
                            {
                                "start": "2023-10-01",
                                "end": "2023-12-31",
                                "val": 3,
                                "accn": "k",
                                "fy": 2023,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2024-02-01",
                            },
                            # missing end date is unusable
                        ]
                    }
                },
            }
        },
    }
    from secqa.xbrl import load_companyfacts

    load_companyfacts(store, doc, "NOPE")
    create_financials_view(store)
    assert store.conn.execute("SELECT count(*) FROM financials").fetchone() == (0,)


def test_view_sql_builder_validates_inputs() -> None:
    sql = financials_view_sql({"m1": ["Assets", "dei:Thing"]}, {"m1": "USD"})
    assert sql.startswith("CREATE OR REPLACE VIEW financials AS")
    assert "('m1', 'us-gaap', 'Assets', 'USD', 0)" in sql
    assert "('m1', 'dei', 'Thing', 'USD', 1)" in sql
    assert "AS m1" in sql
    with pytest.raises(ValueError, match="at least one"):
        financials_view_sql({}, {})
    with pytest.raises(ValueError, match="invalid metric name"):
        financials_view_sql({"Bad Name": ["Assets"]}, {"Bad Name": "USD"})
    with pytest.raises(ValueError, match="invalid metric name"):
        financials_view_sql({"ticker": ["Assets"]}, {"ticker": "USD"})
    with pytest.raises(ValueError, match="no valid unit"):
        financials_view_sql({"m1": ["Assets"]}, {})
    with pytest.raises(ValueError, match="invalid concept alias"):
        financials_view_sql({"m1": ["Assets'); DROP TABLE x; --"]}, {"m1": "USD"})


def test_create_view_requires_writable_store(tmp_path: Path) -> None:
    path = tmp_path / "index.duckdb"
    with DuckDBStore(path, embed_dim=EMBED_DIM) as writable:
        writable.init_schema("hashing-test", EMBED_DIM)
    with DuckDBStore(path, read_only=True) as ro_store:
        with pytest.raises(ConfigError, match="read-only"):
            create_financials_view(ro_store)
