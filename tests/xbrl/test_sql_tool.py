"""run_readonly_sql: results, truncation, JSON-native values, timeout interrupt, error mapping."""

from __future__ import annotations

import time
from datetime import date, datetime
from decimal import Decimal

import duckdb
import pytest

from secqa.core.contracts import SqlResult
from secqa.core.errors import SqlRejected
from secqa.store import DuckDBStore
from secqa.xbrl import MAX_ROWS, SqlExecutionError, SqlTimeout, run_readonly_sql
from secqa.xbrl.sql_tool import jsonable
from tests.xbrl.conftest import ACCN_FY2023, N_FIXTURE_ROWS


def test_simple_query_returns_json_native_rows(loaded_store: DuckDBStore) -> None:
    result = run_readonly_sql(
        loaded_store,
        "SELECT tag, val, end_date, filed, frame FROM xbrl_facts "
        "WHERE tag = 'Revenues' AND accn = '0001234567-24-000010' ORDER BY end_date",
    )
    assert isinstance(result, SqlResult)
    assert result.columns == ["tag", "val", "end_date", "filed", "frame"]
    assert result.rows == [
        ["Revenues", 1_210_000_000.0, "2021-12-31", "2024-02-15", "CY2021"],
        ["Revenues", 1_400_000_000.0, "2022-12-31", "2024-02-15", "CY2022"],
    ]
    assert result.row_count == 2 and result.truncated is False
    assert result.sql.endswith(f"LIMIT {MAX_ROWS}")
    assert result.model_dump_json()  # serialisable as-is


def test_query_over_curated_view(loaded_store: DuckDBStore) -> None:
    result = run_readonly_sql(
        loaded_store,
        "SELECT fiscal_year, revenue, net_income / revenue AS margin FROM financials "
        "WHERE ticker = 'FIXT' AND revenue IS NOT NULL ORDER BY fiscal_year",
    )
    assert [row[0] for row in result.rows] == [2021, 2022, 2023]
    assert result.rows[-1][1] == 1_577_000_000.0
    assert result.rows[-1][2] == pytest.approx(250 / 1577)


def test_unloaded_extension_function_is_a_catalog_error_not_a_download(
    loaded_store: DuckDBStore,
) -> None:
    """A scalar call the guard does not know by name must not make DuckDB fetch an extension.

    ``st_point`` lives in the ``spatial`` extension, which is never installed here. With DuckDB's
    defaults the binder would auto-install it from extensions.duckdb.org (network egress from
    the API pod, ~60 MB, multi-second stall) before reporting the error; the store locks
    auto-install / auto-load off, so the failure is an immediate catalog error.
    """
    with pytest.raises(SqlExecutionError, match="spatial extension") as info:
        run_readonly_sql(loaded_store, "SELECT st_point(1, 2) AS p FROM xbrl_facts")
    assert isinstance(info.value.__cause__, duckdb.CatalogException)
    raw = loaded_store.readonly_connection()._cursor
    assert raw.execute(
        "SELECT count(*) FROM duckdb_extensions() WHERE extension_name = 'spatial' AND loaded"
    ).fetchone() == (0,)
    assert raw.execute("SELECT current_setting('autoinstall_known_extensions')").fetchone() == (
        False,
    )


def test_truncated_flag_is_exact(loaded_store: DuckDBStore) -> None:
    assert N_FIXTURE_ROWS**2 > MAX_ROWS
    result = run_readonly_sql(loaded_store, "SELECT a.tag FROM xbrl_facts a, xbrl_facts b")
    assert result.row_count == MAX_ROWS and len(result.rows) == MAX_ROWS
    assert result.truncated is True
    assert result.sql.endswith(f"LIMIT {MAX_ROWS}")
    # Exactly MAX_ROWS rows available -> not truncated.
    exact = run_readonly_sql(
        loaded_store, f"SELECT a.tag FROM xbrl_facts a, xbrl_facts b LIMIT {MAX_ROWS}"
    )
    assert exact.row_count == MAX_ROWS and exact.truncated is False
    # The caller's own smaller LIMIT is respected and is not "truncation".
    small = run_readonly_sql(loaded_store, "SELECT tag FROM xbrl_facts LIMIT 3")
    assert small.row_count == 3 and small.truncated is False and small.sql.endswith("LIMIT 3")


def test_empty_result(loaded_store: DuckDBStore) -> None:
    result = run_readonly_sql(loaded_store, "SELECT tag FROM xbrl_facts WHERE tag = 'Nope'")
    assert result.rows == [] and result.row_count == 0 and result.truncated is False
    assert result.columns == ["tag"]


def test_guard_rejections_propagate(loaded_store: DuckDBStore) -> None:
    with pytest.raises(SqlRejected, match="got DROP"):
        run_readonly_sql(loaded_store, "DROP TABLE xbrl_facts")
    with pytest.raises(SqlRejected, match="table 'chunks'"):
        run_readonly_sql(loaded_store, "SELECT * FROM chunks")
    assert loaded_store.counts()["facts"] == N_FIXTURE_ROWS


def test_engine_errors_become_sql_execution_error(loaded_store: DuckDBStore) -> None:
    with pytest.raises(SqlExecutionError, match="nosuchcol") as info:
        run_readonly_sql(loaded_store, "SELECT nosuchcol FROM xbrl_facts")
    assert isinstance(info.value, SqlRejected)
    assert info.value.reason.startswith("query failed: ")
    # The store is still usable afterwards.
    assert run_readonly_sql(loaded_store, "SELECT count(*) FROM xbrl_facts").rows == [
        [N_FIXTURE_ROWS]
    ]


def test_timeout_interrupts_cross_join(loaded_store: DuckDBStore) -> None:
    """A 7-way self cross join (20^7 rows) cannot finish in 0.2 s; it must be interrupted."""
    sql = (
        "SELECT count(*) FROM xbrl_facts a, xbrl_facts b, xbrl_facts c, xbrl_facts d, "
        "xbrl_facts e, xbrl_facts f, xbrl_facts g "
        "WHERE a.val + b.val + c.val + d.val + e.val + f.val + g.val < -1"
    )
    started = time.perf_counter()
    with pytest.raises(SqlTimeout, match="exceeded 0.2 s") as info:
        run_readonly_sql(loaded_store, sql, timeout_s=0.2)
    elapsed = time.perf_counter() - started
    assert isinstance(info.value, SqlRejected)
    assert elapsed < 5.0, f"interrupt did not take effect promptly ({elapsed:.1f}s)"
    # The interrupted cursor is gone and the store still answers.
    assert loaded_store.counts()["facts"] == N_FIXTURE_ROWS
    quick = run_readonly_sql(loaded_store, "SELECT accn FROM xbrl_facts WHERE tag = 'Assets'")
    assert ACCN_FY2023 in {row[0] for row in quick.rows}


def test_invalid_timeout(loaded_store: DuckDBStore) -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        run_readonly_sql(loaded_store, "SELECT 1", timeout_s=0)


def test_readonly_connection_second_gate(loaded_store: DuckDBStore) -> None:
    """Even a statement that passed the parser gate is executed READ ONLY by the store."""
    conn = loaded_store.readonly_connection()
    try:
        with pytest.raises(SqlRejected):
            conn.execute("INSERT INTO xbrl_facts SELECT * FROM xbrl_facts")
    finally:
        conn.close()
    assert loaded_store.counts()["facts"] == N_FIXTURE_ROWS


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (3, 3),
        ("x", "x"),
        (1.5, 1.5),
        (float("nan"), None),
        (float("inf"), None),
        (Decimal("12.50"), 12.5),
        (date(2023, 12, 31), "2023-12-31"),
        (datetime(2023, 12, 31, 8, 30), "2023-12-31T08:30:00"),
        (b"\x00\xff", "00ff"),
        ([1, Decimal("2"), date(2020, 1, 1)], [1, 2.0, "2020-01-01"]),
        ({"a": Decimal("1")}, {"a": 1.0}),
        ((1, 2), [1, 2]),
    ],
)
def test_jsonable(value: object, expected: object) -> None:
    assert jsonable(value) == expected
