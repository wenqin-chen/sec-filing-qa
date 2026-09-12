"""validate_sql: every rule in the guard's docstring, rejected and accepted forms."""

from __future__ import annotations

import logging

import pytest

from secqa.core.errors import SqlRejected
from secqa.xbrl import ALLOWED_TABLES, MAX_ROWS, guard_sql, validate_sql

# ---- rejected -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "match"),
    [
        ("DROP TABLE xbrl_facts", "got DROP"),
        ("SELECT 1; SELECT 2", "exactly one statement"),
        ("SELECT 1; DROP TABLE xbrl_facts", "exactly one statement"),
        ("COPY (SELECT * FROM xbrl_facts) TO '/tmp/out.csv'", "got COPY"),
        ("COPY xbrl_facts FROM '/etc/passwd'", "got COPY"),
        ("PRAGMA database_list", "got PRAGMA"),
        ("PRAGMA create_fts_index('chunks', 'chunk_id', 'text')", "got PRAGMA"),
        ("INSTALL httpfs", "got INSTALL"),
        ("LOAD httpfs", "got COMMAND"),
        ("ATTACH '/tmp/other.duckdb' AS other", "got ATTACH"),
        ("SET threads = 1", "got SET"),
        ("INSERT INTO xbrl_facts SELECT * FROM xbrl_facts", "got INSERT"),
        ("UPDATE xbrl_facts SET val = 0", "got UPDATE"),
        ("DELETE FROM xbrl_facts", "got DELETE"),
        ("CREATE TABLE t AS SELECT * FROM xbrl_facts", "got CREATE"),
        ("DESCRIBE xbrl_facts", "got DESCRIBE"),
        ("SUMMARIZE xbrl_facts", "got SUMMARIZE"),
        ("BEGIN TRANSACTION", "got TRANSACTION"),
        ("SELECT * FROM read_csv('/etc/passwd')", "read_csv"),
        ("SELECT * FROM read_csv_auto('/etc/passwd')", "read_csv_auto"),
        ("SELECT * FROM read_parquet('s3://bucket/x.parquet')", "read_parquet"),
        ("SELECT * FROM read_json('/etc/x.json')", "read_json"),
        ("SELECT * FROM glob('*')", "glob"),
        ("SELECT read_text('/etc/passwd')", "read_text"),
        ("SELECT read_blob('/etc/passwd') AS b", "read_blob"),
        ("SELECT * FROM sqlite_scan('x.db', 't')", "sqlite_scan"),
        ("SELECT * FROM pragma_database_list()", "pragma_database_list"),
        ("SELECT * FROM duckdb_settings()", "duckdb_settings"),
        ("SELECT current_setting('home_directory') FROM documents", "current_setting"),
        ("SELECT * FROM xbrl_facts WHERE tag = current_setting('x')", "current_setting"),
        ("SELECT * FROM range(10)", "table functions are not allowed"),
        # Rule 4: generators whose result size is an argument (allocated outside memory_limit).
        ("SELECT length(repeat('a', 600000000)) AS n", "repeat"),
        ("SELECT repeat(tag, 100000000) FROM xbrl_facts", "repeat"),
        ("SELECT range(1000000000) AS r", "range"),
        ("SELECT range(1, 1000000000, 1) AS r", "range"),
        ("SELECT generate_series(1, 1000000000) AS r", "generate_series"),
        ("SELECT unnest(range(1000000000)) AS r", "range"),
        ("SELECT list_transform(range(1000000000), x -> x) AS r", "range"),
        ("SELECT list_resize([], 1000000000) AS l", "list_resize"),
        ("SELECT array_resize([], 1000000000) AS l", "array_resize"),
        ("SELECT rpad(tag, 1000000000, 'x') FROM xbrl_facts", "rpad"),
        ("SELECT lpad('a', 1000000000, 'x') AS s", "lpad"),
        ("SELECT printf('%1000000000s', 'a') AS s", "printf"),
        ("SELECT format('{:>1000000000}', 'a') AS s", "format"),
        ("SELECT bitstring('1', 1000000000) AS b", "bitstring"),
        ("SELECT length(x) FROM (SELECT repeat('a', 10) AS x)", "repeat"),
        ("WITH g AS (SELECT repeat('a', 10) AS x) SELECT * FROM g", "repeat"),
        ("SELECT tag FROM xbrl_facts WHERE tag = repeat('a', 10)", "repeat"),
        ("SELECT * FROM '/tmp/x.parquet'", "not allowed"),
        ("SELECT * FROM chunks", "table 'chunks' is not allowed"),
        ("SELECT text FROM pages", "table 'pages' is not allowed"),
        ("SELECT * FROM index_manifest", "table 'index_manifest' is not allowed"),
        ("SELECT * FROM information_schema.tables", "qualified table name"),
        ("SELECT * FROM main.xbrl_facts", "qualified table name"),
        ("SELECT * FROM fts_main_chunks.docs", "qualified table name"),
        ("SELECT * FROM xbrl_facts JOIN chunks USING (doc_name)", "table 'chunks'"),
        ("SELECT * FROM xbrl_facts WHERE tag IN (SELECT chunk_id FROM chunks)", "table 'chunks'"),
        ("WITH x AS (SELECT * FROM chunks) SELECT * FROM x", "table 'chunks'"),
        ("WITH chunks AS (SELECT * FROM chunks) SELECT * FROM chunks", "shadows a table"),
        ("WITH financials AS (SELECT 1) SELECT * FROM financials", "shadows a table"),
        ("SELECT 1 INTO t", "INTO is not allowed"),
        ("SELECT * FROM xbrl_facts WHERE tag = ?", "bind parameter"),
        ("SELECT * FROM xbrl_facts WHERE tag = $1", "bind parameter"),
        ("", "empty"),
        ("   \n", "empty"),
        ("SELECT ) FROM xbrl_facts", "could not parse"),
        ("(SELECT * FROM xbrl_facts)", "got SUBQUERY"),
    ],
)
def test_rejects(sql: str, match: str) -> None:
    with pytest.raises(SqlRejected, match=match) as info:
        validate_sql(sql)
    assert info.value.reason == str(info.value)


def test_command_fallback_warning_is_silenced(caplog: pytest.LogCaptureFixture) -> None:
    """sqlglot's per-statement 'falling back to Command' warning must not reach the log."""
    with caplog.at_level(logging.WARNING, logger="sqlglot"):
        with pytest.raises(SqlRejected, match="COMMAND"):
            validate_sql("LOAD httpfs")
    assert not [r for r in caplog.records if "Falling back to parsing" in r.getMessage()]


def test_rejects_non_string() -> None:
    with pytest.raises(SqlRejected, match="empty"):
        validate_sql(None)  # type: ignore[arg-type]


# ---- accepted -------------------------------------------------------------------------------


def test_generator_rejection_names_the_rule() -> None:
    with pytest.raises(SqlRejected, match="result size is unbounded by the data") as info:
        validate_sql("SELECT repeat('a', 600000000)")
    assert "repeat()" in info.value.reason


def test_accepts_size_bounded_string_and_list_functions() -> None:
    """Functions whose output is a function of their input stay available to the model."""
    sql = (
        "SELECT upper(tag), left(tag, 3), substring(tag, 1, 5), tag || '-' || accn, "
        "concat(tag, accn), string_agg(tag, ','), list(val), len(list(val)), "
        "round(sum(val), 2), strftime(end_date, '%Y'), list_sort(list(val)) "
        "FROM xbrl_facts GROUP BY tag, accn"
    )
    assert validate_sql(sql).endswith(f"LIMIT {MAX_ROWS}")


def test_accepts_cte_and_appends_limit() -> None:
    sql = (
        "WITH latest AS (SELECT ticker, tag, val FROM xbrl_facts WHERE fp = 'FY') "
        "SELECT ticker, tag, val FROM latest WHERE tag = 'Revenues'"
    )
    out = validate_sql(sql)
    assert out.startswith("WITH latest AS (")
    assert out.endswith(f"LIMIT {MAX_ROWS}")


def test_accepts_all_allowed_tables_and_joins() -> None:
    out = validate_sql(
        "SELECT d.doc_name, f.revenue FROM documents d "
        "JOIN financials f ON upper(d.ticker) = f.ticker AND d.fiscal_year = f.fiscal_year "
        "JOIN xbrl_facts x ON x.ticker = f.ticker"
    )
    assert out.endswith("LIMIT 200")
    assert ALLOWED_TABLES == {"xbrl_facts", "financials", "documents"}


def test_accepts_union_qualify_and_window_functions() -> None:
    out = validate_sql("SELECT tag FROM xbrl_facts UNION ALL SELECT ticker FROM financials")
    assert out == "SELECT tag FROM xbrl_facts UNION ALL SELECT ticker FROM financials LIMIT 200"
    out = validate_sql(
        "SELECT tag, val FROM xbrl_facts "
        "QUALIFY row_number() OVER (PARTITION BY tag ORDER BY filed DESC) = 1"
    )
    assert "QUALIFY ROW_NUMBER() OVER (PARTITION BY tag ORDER BY filed DESC) = 1" in out
    assert out.endswith("LIMIT 200")


def test_accepts_cte_alias_that_is_not_a_store_table() -> None:
    out = validate_sql("WITH t AS (SELECT * FROM xbrl_facts) SELECT count(*) FROM t")
    assert out.endswith("LIMIT 200")


def test_accepts_comments_and_trailing_semicolon() -> None:
    out = validate_sql("-- top comment\nSELECT tag FROM xbrl_facts; ")
    assert out == "SELECT tag FROM xbrl_facts LIMIT 200"
    out = validate_sql("SELECT tag FROM xbrl_facts -- ; DROP TABLE xbrl_facts")
    assert out == "SELECT tag FROM xbrl_facts LIMIT 200"


@pytest.mark.parametrize(
    ("sql", "expected_tail"),
    [
        ("SELECT * FROM xbrl_facts", "LIMIT 200"),
        ("SELECT * FROM xbrl_facts LIMIT 1000", "LIMIT 200"),
        ("SELECT * FROM xbrl_facts LIMIT 200", "LIMIT 200"),
        ("SELECT * FROM xbrl_facts LIMIT 5", "LIMIT 5"),
        ("SELECT * FROM xbrl_facts LIMIT 0", "LIMIT 0"),
        ("SELECT * FROM xbrl_facts LIMIT 10%", "LIMIT 200"),
        ("SELECT * FROM xbrl_facts LIMIT (SELECT 1)", "LIMIT 200"),
        ("SELECT * FROM xbrl_facts LIMIT -1", "LIMIT 200"),
        ("SELECT * FROM xbrl_facts ORDER BY val LIMIT 5 OFFSET 2", "LIMIT 5 OFFSET 2"),
        ("SELECT * FROM xbrl_facts ORDER BY val LIMIT 999 OFFSET 2", "LIMIT 200 OFFSET 2"),
    ],
)
def test_forces_limit(sql: str, expected_tail: str) -> None:
    assert validate_sql(sql).endswith(expected_tail)


def test_guard_sql_custom_cap() -> None:
    assert guard_sql("SELECT * FROM xbrl_facts", max_rows=201).endswith("LIMIT 201")
    assert guard_sql("SELECT * FROM xbrl_facts LIMIT 5", max_rows=201).endswith("LIMIT 5")
    assert guard_sql("SELECT * FROM xbrl_facts LIMIT 500", max_rows=201).endswith("LIMIT 201")
    with pytest.raises(ValueError, match="max_rows"):
        guard_sql("SELECT 1", max_rows=0)


def test_output_is_idempotent() -> None:
    once = validate_sql("select tag,val from xbrl_facts where fy=2023")
    assert validate_sql(once) == once
