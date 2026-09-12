"""secqa.xbrl: companyfacts loader, curated ``financials`` view, read-only SQL guard and tools.

Public surface (see each module's docstring for the reasoning):

* :func:`flatten_companyfacts` / :func:`load_companyfacts` - companyfacts JSON -> ``xbrl_facts``.
* :func:`create_financials_view` + :data:`CURATED_METRICS` - per-(company, fiscal year) metrics.
* :func:`lookup_fact` - curated metric -> fact row(s) with accession number (agent tool).
* :func:`validate_sql` / :func:`run_readonly_sql` - the guarded ``query_xbrl`` tool.
"""

from secqa.xbrl.facts import lookup_fact, resolve_metric
from secqa.xbrl.financials import (
    CURATED_METRICS,
    CURATED_UNITS,
    create_financials_view,
    financials_columns,
    load_tags,
    split_alias,
)
from secqa.xbrl.load import flatten_companyfacts, load_companyfacts
from secqa.xbrl.sql_guard import ALLOWED_TABLES, MAX_ROWS, guard_sql, validate_sql
from secqa.xbrl.sql_tool import SqlExecutionError, SqlTimeout, run_readonly_sql

__all__ = [
    "ALLOWED_TABLES",
    "CURATED_METRICS",
    "CURATED_UNITS",
    "MAX_ROWS",
    "SqlExecutionError",
    "SqlTimeout",
    "create_financials_view",
    "financials_columns",
    "flatten_companyfacts",
    "guard_sql",
    "load_companyfacts",
    "load_tags",
    "lookup_fact",
    "resolve_metric",
    "run_readonly_sql",
    "split_alias",
    "validate_sql",
]
