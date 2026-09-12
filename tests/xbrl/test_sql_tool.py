"""run_readonly_sql: results, truncation, JSON-native values, timeout interrupt, error mapping."""

from __future__ import annotations

import threading
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import duckdb
import pytest

from secqa.core.contracts import SqlResult
from secqa.core.errors import SqlRejected
from secqa.store import DuckDBStore, ReadOnlyResult
from secqa.xbrl import (
    MAX_ROWS,
    SqlExecutionError,
    SqlTimeout,
    SqlToolUnavailable,
    create_financials_view,
    load_companyfacts,
    run_readonly_sql,
    sql_tool,
)
from secqa.xbrl.sql_tool import (
    MAX_CELL_CHARS,
    MAX_LEAKED_WORKERS,
    TRUNCATION_MARKER,
    jsonable,
    leaked_worker_count,
)
from tests.xbrl.conftest import ACCN_FY2023, EMBED_DIM, N_FIXTURE_ROWS, TICKER


def _doubling(levels: int) -> str:
    """A subquery whose single string column doubles ``levels`` times (2**levels chars)."""
    sql = "SELECT 'a' AS s"
    for _ in range(levels):
        sql = f"SELECT s || s AS s FROM ({sql})"
    return sql


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


def test_generators_are_refused_before_execution(loaded_store: DuckDBStore) -> None:
    """``repeat`` / ``range`` / ``list_resize`` allocate outside DuckDB's memory_limit and ignore
    the interrupt (a 600 MB repeat took the process to 1.3 GB); the guard stops them cold."""
    for sql in (
        "SELECT length(repeat('a', 600000000)) AS n",
        "SELECT range(1000000000) AS r",
        "SELECT list_resize([], 1000000000) AS l",
        "SELECT rpad(tag, 1000000000, 'x') FROM xbrl_facts",
    ):
        with pytest.raises(SqlRejected, match="unbounded by the data"):
            run_readonly_sql(loaded_store, sql)
    assert loaded_store.counts()["facts"] == N_FIXTURE_ROWS


def test_memory_limit_turns_oversized_query_into_execution_error(
    companyfacts: dict[str, Any],
) -> None:
    """Under the store's memory_limit an allocation the buffer manager cannot satisfy is an
    OutOfMemoryException -> SqlExecutionError (400), not a growing process."""
    with DuckDBStore(":memory:", embed_dim=EMBED_DIM, memory_limit="64MB", threads=1) as store:
        store.init_schema("hashing-test", EMBED_DIM)
        load_companyfacts(store, companyfacts, TICKER)
        create_financials_view(store)
        with pytest.raises(SqlExecutionError, match="Out of Memory") as info:
            run_readonly_sql(store, f"SELECT length(s) FROM ({_doubling(30)})")
        assert isinstance(info.value.__cause__, duckdb.OutOfMemoryException)
        # The store is still usable afterwards.
        assert run_readonly_sql(store, "SELECT count(*) FROM xbrl_facts").rows == [[N_FIXTURE_ROWS]]


def test_long_cells_are_truncated(loaded_store: DuckDBStore) -> None:
    """A 16 KB string cell, and list / struct cells whose JSON is that long, are cut to
    MAX_CELL_CHARS and marked; small nested values keep their type."""
    result = run_readonly_sql(
        loaded_store,
        f"SELECT s, [s, s] AS l, {{'k': s}} AS st, length(s) AS n FROM ({_doubling(14)})",
    )
    text, as_list, as_struct, length = result.rows[0]
    assert length == 2**14
    for cell in (text, as_list, as_struct):
        assert isinstance(cell, str)
        assert len(cell) == MAX_CELL_CHARS + len(TRUNCATION_MARKER)
        assert cell.endswith(TRUNCATION_MARKER)
    assert (
        text.startswith("aaaa")
        and as_list.startswith('["aaa')
        and as_struct.startswith('{"k":"aaa')
    )
    assert len(result.model_dump_json()) < 4 * (MAX_CELL_CHARS + 100)
    small = run_readonly_sql(
        loaded_store, "SELECT list(val ORDER BY val) AS vals FROM xbrl_facts WHERE tag = 'Assets'"
    )
    assert isinstance(small.rows[0][0], list) and len(small.rows[0][0]) > 1


class _StuckConnection:
    """A cursor whose statement blocks until released and whose interrupt is ignored, like a
    scalar allocation DuckDB does not check for interrupts."""

    def __init__(self, release: threading.Event):
        self.release = release
        self.interrupts = 0
        self.closed = False

    def execute(self, sql: str) -> ReadOnlyResult:
        self.release.wait()
        return ReadOnlyResult(["x"], [(1,)])

    def interrupt(self) -> None:
        self.interrupts += 1

    def close(self) -> None:
        self.closed = True


class _StuckStore:
    def __init__(self) -> None:
        self.release = threading.Event()
        self.connections: list[_StuckConnection] = []

    def readonly_connection(self) -> _StuckConnection:
        conn = _StuckConnection(self.release)
        self.connections.append(conn)
        return conn


def test_workers_that_ignore_the_interrupt_are_counted_and_capped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An abandoned worker is not closed underneath DuckDB: it is counted, new SQL is refused at
    MAX_LEAKED_WORKERS, and the worker closes its own cursor when it finally finishes."""
    monkeypatch.setattr(sql_tool, "INTERRUPT_GRACE_S", 0.05)
    monkeypatch.setattr(sql_tool, "_leaked_workers", [])
    store = _StuckStore()
    for expected in range(1, MAX_LEAKED_WORKERS + 1):
        with pytest.raises(SqlTimeout):
            run_readonly_sql(store, "SELECT 1", timeout_s=0.05)  # type: ignore[arg-type]
        assert leaked_worker_count() == expected
    assert [conn.interrupts for conn in store.connections] == [1] * MAX_LEAKED_WORKERS
    assert not any(conn.closed for conn in store.connections)
    with pytest.raises(SqlToolUnavailable, match=f"{MAX_LEAKED_WORKERS} earlier queries") as info:
        run_readonly_sql(store, "SELECT 1", timeout_s=0.05)  # type: ignore[arg-type]
    assert isinstance(info.value, SqlRejected)
    assert len(store.connections) == MAX_LEAKED_WORKERS  # refused before opening a cursor
    store.release.set()
    for worker in list(sql_tool._leaked_workers):
        worker.join(2.0)
    assert all(conn.closed for conn in store.connections)
    assert leaked_worker_count() == 0
    result = run_readonly_sql(store, "SELECT 1", timeout_s=1.0)  # type: ignore[arg-type]
    assert result.rows == [[1]] and store.connections[-1].closed


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


def test_jsonable_caps_cell_size() -> None:
    long_text = "x" * (MAX_CELL_CHARS + 1)
    assert jsonable(long_text) == "x" * MAX_CELL_CHARS + TRUNCATION_MARKER
    assert jsonable("x" * MAX_CELL_CHARS) == "x" * MAX_CELL_CHARS
    long_json = "[" + ",".join(str(i) for i in range(3000)) + "]"
    assert jsonable(list(range(3000))) == long_json[:MAX_CELL_CHARS] + TRUNCATION_MARKER
    assert jsonable([1, 2]) == [1, 2]
    assert jsonable({"k": "v" * 10}, max_chars=8) == '{"k":"vv' + TRUNCATION_MARKER
    assert jsonable(b"\x00" * 5000) == "00" * (MAX_CELL_CHARS // 2) + TRUNCATION_MARKER
    with pytest.raises(ValueError, match="max_chars"):
        jsonable("x", max_chars=0)
