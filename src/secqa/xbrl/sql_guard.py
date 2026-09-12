"""Read-only SQL guard for the ``query_xbrl`` tool and ``POST /v1/xbrl/query``.

The statement is parsed with sqlglot (DuckDB dialect) and accepted only if *every* rule holds;
anything unexpected is rejected, never "cleaned up". This is the first of two gates: the second
is :class:`secqa.store.ReadOnlyConnection`, which re-classifies the statement with DuckDB's own
parser and runs it inside ``BEGIN TRANSACTION READ ONLY``. The guard adds what the engine cannot
express per connection: which tables may be read and that no file / system function runs.

Rules (each has a test in ``tests/xbrl/test_sql_guard.py``):

1. Exactly one statement, and it must be a ``SELECT`` (``WITH ... SELECT``, ``UNION`` and other
   set operations included). DDL, DML, ``COPY``, ``PRAGMA``, ``ATTACH``, ``INSTALL``, ``LOAD``,
   ``SET``, ``DESCRIBE``, ``SUMMARIZE``, ``SELECT ... INTO`` and unparsable input are rejected.
2. Every table reference is unqualified and is either one of ``ALLOWED_TABLES`` or a CTE defined
   in the query. CTE aliases may not shadow a store table (otherwise ``WITH chunks AS (SELECT *
   FROM chunks)`` would read the base table).
3. No table functions (``read_csv``, ``glob``, ``range`` ...) and no file / system functions
   anywhere in the tree (``read_text``, ``sqlite_scan``, ``pragma_*`` ...).
4. No bind parameters (``?``, ``$1``) - values are inlined by the model.
5. ``LIMIT`` is forced: absent, non-literal, percentage or above ``MAX_ROWS`` -> ``LIMIT 200``.

The validated statement is re-rendered from the AST, so what runs is what was checked.
"""

from __future__ import annotations

import logging

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from secqa.core.errors import SqlRejected
from secqa.core.logging import get_logger

log = get_logger(__name__)

DIALECT = "duckdb"
MAX_ROWS = 200
ALLOWED_TABLES = frozenset({"xbrl_facts", "financials", "documents"})
# Every table the store creates in the main schema; CTE aliases may not shadow any of them.
STORE_TABLES = frozenset(
    {"documents", "pages", "chunks", "xbrl_facts", "financials", "index_manifest"}
)

_DENIED_FUNCTION_PREFIXES: tuple[str, ...] = (
    "read_",  # read_csv, read_parquet, read_json, read_text, read_blob, read_ndjson ...
    "parquet_",  # parquet_scan, parquet_metadata, parquet_schema
    "sqlite_",
    "postgres_",
    "mysql_",
    "arrow_",
    "pragma_",  # pragma_database_list(), pragma_storage_info() ...
    "duckdb_",  # duckdb_settings(), duckdb_extensions() ...
    "icu_",
)
_DENIED_FUNCTIONS = frozenset(
    {
        "glob",
        "query",
        "query_table",
        "sniff_csv",
        "checkpoint",
        "force_checkpoint",
        "getenv",
        "shell",
        "load_extension",
        "from_substrait",
        "get_substrait",
    }
)
# ``exp.DDL`` / ``exp.DML`` are mixins rather than ``Expression`` subclasses, hence ``type``.
_DENIED_NODE_TYPES: tuple[type, ...] = (
    exp.Into,
    exp.Command,
    exp.DDL,
    exp.DML,
    exp.Pragma,
    exp.Set,
    exp.Copy,
    exp.Attach,
    exp.Detach,
    exp.Install,
    exp.Describe,
    exp.Summarize,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
    exp.Use,
    exp.Placeholder,
    exp.Parameter,
)


class _DropCommandFallbackWarning(logging.Filter):
    """Silence sqlglot's "falling back to parsing as a 'Command'" warning.

    sqlglot logs it for every statement it cannot model (``LOAD httpfs``, ``INSTALL ...``,
    prompt-injection attempts ...). The guard rejects every ``Command`` node anyway, so the
    warning carries no information here and would only add a noisy line to the JSON log per
    rejected statement. The filter is attached to the ``sqlglot`` logger, which is the logger
    the parser writes to, and matches on the message text so genuine sqlglot warnings still pass.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "Falling back to parsing as a 'Command'" not in record.getMessage()


_SQLGLOT_LOGGER = logging.getLogger("sqlglot")
_COMMAND_FALLBACK_FILTER = _DropCommandFallbackWarning()
if not any(isinstance(f, _DropCommandFallbackWarning) for f in _SQLGLOT_LOGGER.filters):
    _SQLGLOT_LOGGER.addFilter(_COMMAND_FALLBACK_FILTER)


def validate_sql(sql: str) -> str:
    """Return the canonical read-only form of ``sql`` (with ``LIMIT 200`` forced).

    Raises:
        SqlRejected: with a model-safe ``reason`` when any rule above fails.
    """
    return guard_sql(sql, max_rows=MAX_ROWS)


def guard_sql(sql: str, *, max_rows: int = MAX_ROWS) -> str:
    """:func:`validate_sql` with a configurable row cap (the tool probes with ``MAX_ROWS + 1``)."""
    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")
    query = _parse_single_query(sql)
    _check_tree(query)
    _check_tables(query)
    _force_limit(query, max_rows)
    rendered = query.sql(dialect=DIALECT, comments=False)
    log.debug("sql_validated", max_rows=max_rows, sql=rendered[:500])
    return rendered


def _parse_single_query(sql: str) -> exp.Query:
    if not isinstance(sql, str) or not sql.strip():
        raise SqlRejected("empty SQL statement")
    try:
        parsed = sqlglot.parse(sql, read=DIALECT)
    except SqlglotError as exc:
        raise SqlRejected(f"could not parse SQL: {_first_line(str(exc))}") from exc
    statements = [statement for statement in parsed if statement is not None]
    if len(statements) != 1:
        raise SqlRejected(f"expected exactly one statement, got {len(statements)}")
    root = statements[0]
    if not isinstance(root, exp.Select | exp.SetOperation):
        raise SqlRejected(
            f"only a single SELECT / WITH query is allowed, got {type(root).__name__.upper()}"
        )
    return root


def _check_tree(query: exp.Query) -> None:
    for node in query.walk():
        if isinstance(node, _DENIED_NODE_TYPES):
            raise SqlRejected(f"{_describe(node)} is not allowed in a read-only query")
        if isinstance(node, exp.Func):
            name = _function_name(node)
            if name in _DENIED_FUNCTIONS or name.startswith(_DENIED_FUNCTION_PREFIXES):
                raise SqlRejected(f"function {name}() is not allowed in a read-only query")


def _check_tables(query: exp.Query) -> None:
    cte_names = {cte.alias_or_name.lower() for cte in query.find_all(exp.CTE)}
    shadowed = sorted(cte_names & STORE_TABLES)
    if shadowed:
        raise SqlRejected(f"CTE alias {shadowed[0]!r} shadows a table; choose another name")
    for table in query.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise SqlRejected(
                f"table functions are not allowed (got {_describe(table.this)}); "
                f"query {', '.join(sorted(ALLOWED_TABLES))} instead"
            )
        if table.db or table.catalog:
            raise SqlRejected(
                f"qualified table name {table.sql(dialect=DIALECT)!r} is not allowed; "
                "use an unqualified name"
            )
        name = table.name.lower()
        if name in cte_names or name in ALLOWED_TABLES:
            continue
        raise SqlRejected(
            f"table {table.name!r} is not allowed; only {', '.join(sorted(ALLOWED_TABLES))} "
            "may be queried"
        )


def _force_limit(query: exp.Query, max_rows: int) -> None:
    limit = query.args.get("limit")
    if isinstance(limit, exp.Limit):
        literal = limit.expression
        options = limit.args.get("limit_options")
        is_percent = bool(options is not None and options.args.get("percent"))
        if (
            isinstance(literal, exp.Literal)
            and literal.is_int
            and not is_percent
            and int(literal.this) <= max_rows
        ):
            return
    query.limit(max_rows, copy=False)


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def _describe(node: object) -> str:  # sqlglot types Table.this loosely
    if isinstance(node, exp.Func):
        return f"function {_function_name(node)}()"
    if isinstance(node, exp.Placeholder | exp.Parameter):
        return "a bind parameter"
    return type(node).__name__.upper()


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0][:200] if text.strip() else "syntax error"


__all__ = ["ALLOWED_TABLES", "DIALECT", "MAX_ROWS", "STORE_TABLES", "guard_sql", "validate_sql"]
