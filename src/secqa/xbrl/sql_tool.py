"""``run_readonly_sql``: execute a guarded query with a hard wall-clock limit.

Flow: :func:`secqa.xbrl.sql_guard.validate_sql` (AST allowlist, ``LIMIT 200``) -> a fresh
:class:`secqa.store.ReadOnlyConnection` (DuckDB re-parse, ``READ ONLY`` transaction) -> execution
on a worker thread. If the worker has not finished after ``timeout_s`` the main thread calls
``conn.interrupt()``, which makes DuckDB abort the running statement, and raises
:class:`SqlTimeout`.

The statement actually executed asks for ``MAX_ROWS + 1`` rows (or the caller's own smaller
``LIMIT``), so ``truncated`` is exact: it is true only when the query had more rows than the
cap. Values are converted to JSON-native Python (``Decimal`` -> float, dates -> ISO strings,
non-finite floats -> ``None``) because the result is shown to a model and serialised by the
API; ``SqlResult.sql`` records the validated statement, not the probe.
"""

from __future__ import annotations

import math
import threading
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import duckdb

from secqa.core.contracts import SqlResult
from secqa.core.errors import SqlRejected
from secqa.core.logging import get_logger
from secqa.store import DuckDBStore, ReadOnlyResult
from secqa.xbrl.sql_guard import MAX_ROWS, guard_sql, validate_sql

log = get_logger(__name__)

DEFAULT_TIMEOUT_S = 5.0
# How long to wait for DuckDB to honour an interrupt before giving up on the worker thread.
INTERRUPT_GRACE_S = 5.0


class SqlTimeout(SqlRejected):
    """The query exceeded ``timeout_s`` and was interrupted (a ``SqlRejected`` so callers that
    already map the guard's refusals to a 400 / tool error handle it the same way)."""


class SqlExecutionError(SqlRejected):
    """DuckDB refused the validated statement (unknown column, type error, engine guard ...)."""


def run_readonly_sql(
    store: DuckDBStore, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S
) -> SqlResult:
    """Validate ``sql`` and run it read-only with at most ``timeout_s`` seconds of wall clock.

    Returns at most ``MAX_ROWS`` rows; ``truncated`` is true when more were available.

    Raises:
        SqlRejected: the guard refused the statement (see :mod:`secqa.xbrl.sql_guard`).
        SqlTimeout: the statement ran past ``timeout_s`` and was interrupted.
        SqlExecutionError: DuckDB raised while executing the validated statement.
        ValueError: ``timeout_s`` is not positive.
    """
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")
    validated = validate_sql(sql)
    probe = guard_sql(sql, max_rows=MAX_ROWS + 1)
    conn = store.readonly_connection()
    outcome: dict[str, Any] = {}

    def _worker() -> None:
        try:
            outcome["result"] = conn.execute(probe)
        except BaseException as exc:  # stored, re-raised on the caller's thread
            outcome["error"] = exc

    started = time.perf_counter()
    worker = threading.Thread(target=_worker, name="secqa-xbrl-sql", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        conn.interrupt()
        worker.join(INTERRUPT_GRACE_S)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        if worker.is_alive():
            # The cursor is still executing; closing it underneath DuckDB is unsafe, so leak it.
            log.error("xbrl_sql_interrupt_ignored", timeout_s=timeout_s, latency_ms=elapsed_ms)
        else:
            conn.close()
            log.warning("xbrl_sql_timeout", timeout_s=timeout_s, latency_ms=elapsed_ms)
        raise SqlTimeout(f"query exceeded {timeout_s:g} s and was interrupted")
    conn.close()
    elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
    if "error" in outcome:
        error = outcome["error"]
        if isinstance(error, SqlRejected):
            raise error
        if isinstance(error, duckdb.Error):
            reason = str(error).strip().splitlines()[0][:300] if str(error).strip() else "error"
            log.info("xbrl_sql_failed", reason=reason, latency_ms=elapsed_ms)
            raise SqlExecutionError(f"query failed: {reason}") from error
        raise error
    result: ReadOnlyResult = outcome["result"]
    rows = result.fetchall()
    truncated = len(rows) > MAX_ROWS
    kept = rows[:MAX_ROWS]
    log.info(
        "xbrl_sql",
        row_count=len(kept),
        truncated=truncated,
        n_columns=len(result.columns),
        latency_ms=elapsed_ms,
    )
    return SqlResult(
        columns=list(result.columns),
        rows=[[jsonable(value) for value in row] for row in kept],
        row_count=len(kept),
        truncated=truncated,
        sql=validated,
    )


def jsonable(value: Any) -> Any:
    """Convert a DuckDB cell to a JSON-native Python value (lists / structs recursively)."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, bytes | bytearray):
        return bytes(value).hex()
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    return str(value)


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "INTERRUPT_GRACE_S",
    "SqlExecutionError",
    "SqlTimeout",
    "jsonable",
    "run_readonly_sql",
]
