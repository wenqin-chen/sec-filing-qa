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

Resource bounds, and what each one covers:

* Memory: the serving store pins DuckDB's ``memory_limit`` (:mod:`secqa.store.duckdb_store`),
  so an aggregate, join, sort or recursive CTE that outgrows it fails with an
  ``OutOfMemoryException`` (a :class:`SqlExecutionError` here) instead of growing the process;
  the guard refuses the scalar generators DuckDB allocates outside that limit (``repeat``,
  ``range``, ``list_resize`` ...). Neither bounds a string that doubles through a deep chain of
  ``s || s`` subqueries and is then copied per joined row: that path is interruptible and
  needs a crafted query, but its allocations are untracked, so the container memory limit
  remains the last line of defence.
* Result size: each cell is capped at :data:`MAX_CELL_CHARS` characters (strings directly,
  lists / structs by their JSON form) and marked with :data:`TRUNCATION_MARKER`, so a
  200-row result is at most a few hundred kilobytes however wide its cells are.
* Time: ``timeout_s`` then ``conn.interrupt()``. A statement that ignores the interrupt for
  :data:`INTERRUPT_GRACE_S` more seconds is abandoned: its thread and cursor cannot be closed
  underneath DuckDB, so they are left to finish (the worker closes the cursor itself when it
  does) and counted; once :data:`MAX_LEAKED_WORKERS` such workers are still running, new
  statements are refused with :class:`SqlToolUnavailable` (503 at the API) rather than piling
  more work onto a process that is already over its limits.
"""

from __future__ import annotations

import json
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
from secqa.store import DuckDBStore, ReadOnlyConnection, ReadOnlyResult
from secqa.xbrl.sql_guard import MAX_ROWS, guard_sql, validate_sql

log = get_logger(__name__)

DEFAULT_TIMEOUT_S = 5.0
# How long to wait for DuckDB to honour an interrupt before giving up on the worker thread.
INTERRUPT_GRACE_S = 5.0
# Abandoned workers (interrupt ignored) still running before new statements are refused.
MAX_LEAKED_WORKERS = 2
# Longest cell returned, in characters of the string or of the JSON form of a list / struct.
MAX_CELL_CHARS = 4096
TRUNCATION_MARKER = "…[truncated]"

_leaked_workers: list[threading.Thread] = []
_leaked_lock = threading.Lock()


class SqlTimeout(SqlRejected):
    """The query exceeded ``timeout_s`` and was interrupted (a ``SqlRejected`` so callers that
    already map the guard's refusals to a 400 / tool error handle it the same way)."""


class SqlExecutionError(SqlRejected):
    """DuckDB refused the validated statement (unknown column, type error, engine guard ...)."""


class SqlToolUnavailable(SqlRejected):
    """Refused before execution: too many earlier statements ignored their interrupt and are
    still running (:data:`MAX_LEAKED_WORKERS`). Transient; retry once they finish."""


def run_readonly_sql(
    store: DuckDBStore, sql: str, timeout_s: float = DEFAULT_TIMEOUT_S
) -> SqlResult:
    """Validate ``sql`` and run it read-only with at most ``timeout_s`` seconds of wall clock.

    Returns at most ``MAX_ROWS`` rows; ``truncated`` is true when more were available. Cells
    longer than :data:`MAX_CELL_CHARS` are cut and end in :data:`TRUNCATION_MARKER`.

    Raises:
        SqlRejected: the guard refused the statement (see :mod:`secqa.xbrl.sql_guard`).
        SqlToolUnavailable: :data:`MAX_LEAKED_WORKERS` earlier statements that ignored their
            interrupt are still running; nothing was executed.
        SqlTimeout: the statement ran past ``timeout_s`` and was interrupted.
        SqlExecutionError: DuckDB raised while executing the validated statement (including
            ``OutOfMemoryException`` under the store's ``memory_limit``).
        ValueError: ``timeout_s`` is not positive.
    """
    if timeout_s <= 0:
        raise ValueError(f"timeout_s must be positive, got {timeout_s}")
    validated = validate_sql(sql)
    probe = guard_sql(sql, max_rows=MAX_ROWS + 1)
    leaked = leaked_worker_count()
    if leaked >= MAX_LEAKED_WORKERS:
        log.warning("xbrl_sql_refused_leaked_workers", leaked=leaked)
        raise SqlToolUnavailable(
            f"SQL tool is busy: {leaked} earlier queries are still running after being "
            "interrupted; retry shortly"
        )
    conn = store.readonly_connection()
    outcome: dict[str, Any] = {}
    handoff = threading.Lock()  # guards ``done`` / ``abandoned``: exactly one side closes conn
    state = {"done": False, "abandoned": False}

    def _worker() -> None:
        try:
            outcome["result"] = conn.execute(probe)
        except BaseException as exc:  # stored, re-raised on the caller's thread
            outcome["error"] = exc
        finally:
            with handoff:
                state["done"] = True
                if state["abandoned"]:
                    _close_quietly(conn)
                    log.info("xbrl_sql_abandoned_worker_finished")

    started = time.perf_counter()
    worker = threading.Thread(target=_worker, name="secqa-xbrl-sql", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        conn.interrupt()
        worker.join(INTERRUPT_GRACE_S)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
        with handoff:
            finished = state["done"]
            state["abandoned"] = not finished
        if finished:
            conn.close()
            log.warning("xbrl_sql_timeout", timeout_s=timeout_s, latency_ms=elapsed_ms)
        else:
            # Still executing: closing the cursor underneath DuckDB is unsafe, so the worker
            # closes it when it finishes; until then it counts against MAX_LEAKED_WORKERS.
            with _leaked_lock:
                _leaked_workers.append(worker)
                leaked = len(_leaked_workers)
            log.error(
                "xbrl_sql_interrupt_ignored",
                timeout_s=timeout_s,
                latency_ms=elapsed_ms,
                leaked_workers=leaked,
            )
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


def leaked_worker_count() -> int:
    """Number of abandoned workers (interrupt ignored) that are still running.

    Finished ones are dropped from the registry here, so the count is current at every call.
    """
    with _leaked_lock:
        _leaked_workers[:] = [worker for worker in _leaked_workers if worker.is_alive()]
        return len(_leaked_workers)


def jsonable(value: Any, *, max_chars: int = MAX_CELL_CHARS) -> Any:
    """Convert a DuckDB cell to a JSON-native Python value (lists / structs recursively).

    A string longer than ``max_chars`` is cut to ``max_chars`` and suffixed with
    :data:`TRUNCATION_MARKER`; a list or struct whose compact JSON form is longer than
    ``max_chars`` is replaced by that JSON text, cut the same way, so no cell serialises to
    more than ``max_chars + len(TRUNCATION_MARKER)`` characters.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be positive, got {max_chars}")
    converted = _to_json_native(value)
    if isinstance(converted, str):
        return _cap(converted, max_chars)
    if isinstance(converted, list | dict):
        text = json.dumps(converted, separators=(",", ":"), ensure_ascii=False)
        if len(text) > max_chars:
            return _cap(text, max_chars)
    return converted


def _to_json_native(value: Any) -> Any:
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
        return [_to_json_native(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_json_native(item) for key, item in value.items()}
    return str(value)


def _cap(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_MARKER


def _close_quietly(conn: ReadOnlyConnection) -> None:
    try:
        conn.close()
    except duckdb.Error as exc:  # pragma: no cover - cursor already invalidated by store.close()
        log.debug("xbrl_sql_abandoned_cursor_close_failed", error=str(exc)[:200])


__all__ = [
    "DEFAULT_TIMEOUT_S",
    "INTERRUPT_GRACE_S",
    "MAX_CELL_CHARS",
    "MAX_LEAKED_WORKERS",
    "TRUNCATION_MARKER",
    "SqlExecutionError",
    "SqlTimeout",
    "SqlToolUnavailable",
    "jsonable",
    "leaked_worker_count",
    "run_readonly_sql",
]
