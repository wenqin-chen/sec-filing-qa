"""Curated ``financials`` view: one row per (company, fiscal year) with ~30 named metrics.

Why a curated view exists at all: the raw ``xbrl_facts`` table is faithful to companyfacts, and
companyfacts is awkward to query directly:

* ``fy``/``fp`` describe the *filing*, so a FY2023 10-K's FY2021 comparative has ``fy=2023``.
  The view derives ``fiscal_year`` from the fact's own period: for every ``(cik, end_date)`` the
  smallest ``fy`` among 10-K rows is the fiscal-year focus of the filing that first reported that
  period, which is exactly the company's own name for that year (Home Depot's year ending
  January 2024 is "fiscal 2023"; Walmart's is "fiscal 2024"). When no original filing is in the
  data (that value only ever appears as a comparative), we fall back to the calendar year of the
  period end. The sanity bound ``fy in [year(end) - 1, year(end)]`` decides which case we are in.
* Each value is repeated in every later 10-K that carries it as a comparative, sometimes
  restated. The view keeps the latest-filed 10-K row per (concept, unit, fiscal year), so a
  restatement wins over the original figure and the cited accession is the filing that last
  reported it. The SEC's ``frame`` marker is *not* used for this: frames migrate to the next
  10-Q for year-end instants, which would silently drop the newest balance-sheet values.
* Companies switch concept names over time (``Revenues`` -> ``RevenueFromContractWithCustomer...``),
  so each metric in ``tags.yaml`` is an ordered alias chain and the first alias with an annual
  value wins, per (company, fiscal year).

Only 10-K facts with ``fp = 'FY'`` are considered: instants (balance sheet) as-is, durations only
when they span ~1 year (350-380 days), which drops the Q4 columns some 10-Ks include. The same
SQL feeds :func:`secqa.xbrl.facts.lookup_fact`, so a value in the view is the value the lookup
tool returns with its accession number.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from secqa.core.errors import ConfigError
from secqa.core.logging import get_logger
from secqa.store import DuckDBStore

log = get_logger(__name__)

TAGS_PATH = Path(__file__).with_name("tags.yaml")
DEFAULT_TAXONOMY = "us-gaap"
ALLOWED_UNITS = frozenset({"USD", "shares", "USD/shares", "pure"})
FINANCIALS_FIXED_COLUMNS: tuple[str, ...] = ("cik", "ticker", "fiscal_year")

_METRIC_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_ALIAS_RE = re.compile(
    r"^(?:(?P<taxonomy>[a-z][a-z0-9\-]{0,15}):)?(?P<tag>[A-Z][A-Za-z0-9]{0,254})$"
)

# Annual 10-K facts with a derived ``fiscal_year``, deduplicated to the latest-filed row per
# (company, concept, unit, fiscal year). ``{extra_where}`` lets callers push a ticker filter into
# the first scan; it must be a trusted SQL fragment (this module only ever injects a
# parameterised ``AND upper(ticker) = upper(?)``).
ANNUAL_FACTS_SQL = """
WITH tenk AS (
    SELECT cik, ticker, taxonomy, tag, unit, fy, fp, form, start_date, end_date, val, accn,
           filed, frame
    FROM xbrl_facts
    WHERE form = '10-K'
      AND fp = 'FY'
      AND end_date IS NOT NULL
      AND (start_date IS NULL OR date_diff('day', start_date, end_date) BETWEEN 350 AND 380)
      {extra_where}
),
labelled AS (
    SELECT *,
           CAST(
               CASE WHEN first_fy BETWEEN year(end_date) - 1 AND year(end_date) THEN first_fy
                    ELSE year(end_date)
               END AS INTEGER
           ) AS fiscal_year
    FROM (SELECT *, min(fy) OVER (PARTITION BY cik, end_date) AS first_fy FROM tenk)
)
SELECT cik, ticker, taxonomy, tag, unit, fy, fp, form, start_date, end_date, val, accn, filed,
       frame, fiscal_year
FROM labelled
QUALIFY row_number() OVER (
    PARTITION BY cik, taxonomy, tag, unit, fiscal_year
    ORDER BY filed DESC NULLS LAST, accn DESC, end_date DESC
) = 1
"""


def split_alias(alias: str) -> tuple[str, str]:
    """``'Revenues'`` -> ``('us-gaap', 'Revenues')``; ``'dei:EntityCommonStockSharesOutstanding'``
    -> ``('dei', 'EntityCommonStockSharesOutstanding')``.

    Raises ``ValueError`` when the alias is not a taxonomy-qualified CamelCase identifier (the
    strings are spliced into SQL literals, so the shape is enforced, not trusted).
    """
    match = _ALIAS_RE.match(alias.strip())
    if not match:
        raise ValueError(
            f"invalid concept alias {alias!r}: expected 'Tag' or 'taxonomy:Tag' identifiers"
        )
    return match.group("taxonomy") or DEFAULT_TAXONOMY, match.group("tag")


def load_tags(path: Path = TAGS_PATH) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Read and validate ``tags.yaml``; returns ``(metric -> aliases, metric -> unit)``.

    Raises:
        ConfigError: if the file is missing, malformed, or any name fails validation.
    """
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read curated metrics from {path}: {exc}") from exc
    metrics_raw = raw.get("metrics") if isinstance(raw, dict) else None
    if not isinstance(metrics_raw, dict) or not metrics_raw:
        raise ConfigError(f"{path}: expected a non-empty 'metrics' mapping")
    metrics: dict[str, list[str]] = {}
    units: dict[str, str] = {}
    for name, spec in metrics_raw.items():
        metric = str(name)
        if not _METRIC_RE.match(metric) or metric in FINANCIALS_FIXED_COLUMNS:
            raise ConfigError(f"{path}: invalid metric name {metric!r}")
        if not isinstance(spec, dict):
            raise ConfigError(f"{path}: metric {metric!r} must map to {{unit, tags}}")
        unit = str(spec.get("unit", ""))
        if unit not in ALLOWED_UNITS:
            raise ConfigError(
                f"{path}: metric {metric!r} unit {unit!r} not in {sorted(ALLOWED_UNITS)}"
            )
        aliases = spec.get("tags")
        if not isinstance(aliases, list) or not aliases:
            raise ConfigError(f"{path}: metric {metric!r} needs a non-empty 'tags' list")
        cleaned: list[str] = []
        for alias in aliases:
            try:
                taxonomy, tag = split_alias(str(alias))
            except ValueError as exc:
                raise ConfigError(f"{path}: metric {metric!r}: {exc}") from exc
            canonical = tag if taxonomy == DEFAULT_TAXONOMY else f"{taxonomy}:{tag}"
            if canonical in cleaned:
                raise ConfigError(f"{path}: metric {metric!r} lists {canonical!r} twice")
            cleaned.append(canonical)
        metrics[metric] = cleaned
        units[metric] = unit
    return metrics, units


CURATED_METRICS: dict[str, list[str]]
CURATED_UNITS: dict[str, str]
CURATED_METRICS, CURATED_UNITS = load_tags()


def financials_view_sql(
    metrics: dict[str, list[str]] | None = None, units: dict[str, str] | None = None
) -> str:
    """The ``CREATE OR REPLACE VIEW financials`` statement for the given metric chains."""
    metrics = CURATED_METRICS if metrics is None else metrics
    units = CURATED_UNITS if units is None else units
    if not metrics:
        raise ValueError("at least one metric is required")
    alias_rows: list[str] = []
    for metric, aliases in metrics.items():
        if not _METRIC_RE.match(metric) or metric in FINANCIALS_FIXED_COLUMNS:
            raise ValueError(f"invalid metric name {metric!r}")
        unit = units.get(metric)
        if unit not in ALLOWED_UNITS:
            raise ValueError(f"metric {metric!r} has no valid unit")
        for priority, alias in enumerate(aliases):
            taxonomy, tag = split_alias(alias)  # validates the identifier shape
            alias_rows.append(f"('{metric}', '{taxonomy}', '{tag}', '{unit}', {priority})")
    metric_columns = ",\n       ".join(
        f"max(CASE WHEN metric = '{metric}' THEN val END) AS {metric}" for metric in metrics
    )
    annual = ANNUAL_FACTS_SQL.format(extra_where="")
    return (
        "CREATE OR REPLACE VIEW financials AS\n"
        f"WITH annual AS ({annual}),\n"
        "aliases AS (\n"
        "    SELECT * FROM (VALUES\n        "
        + ",\n        ".join(alias_rows)
        + "\n    ) AS v(metric, taxonomy, tag, unit, priority)\n"
        "),\n"
        "picked AS (\n"
        "    SELECT a.cik, a.ticker, a.fiscal_year, al.metric, a.val\n"
        "    FROM annual a\n"
        "    JOIN aliases al ON a.taxonomy = al.taxonomy AND a.tag = al.tag AND a.unit = al.unit\n"
        "    QUALIFY row_number() OVER (\n"
        "        PARTITION BY a.cik, a.fiscal_year, al.metric ORDER BY al.priority\n"
        "    ) = 1\n"
        ")\n"
        "SELECT cik, ticker, fiscal_year,\n"
        f"       {metric_columns}\n"
        "FROM picked\n"
        "GROUP BY cik, ticker, fiscal_year"
    )


def create_financials_view(store: DuckDBStore) -> None:
    """``CREATE OR REPLACE VIEW financials`` over ``xbrl_facts`` (replaces the store's baseline).

    Columns: ``cik, ticker, fiscal_year`` then one DOUBLE column per metric in ``tags.yaml``
    order (NULL when no alias has an annual value). Safe to call repeatedly.

    Raises:
        ConfigError: if the store is read-only or its schema was never initialised.
    """
    if store.read_only:
        raise ConfigError("create_financials_view is not allowed on a read-only store")
    store.conn.execute(financials_view_sql())
    log.info("financials_view_created", n_metrics=len(CURATED_METRICS))


def financials_columns() -> list[str]:
    """Column names of the curated view, in order."""
    return [*FINANCIALS_FIXED_COLUMNS, *CURATED_METRICS]


def annual_facts_sql(*, ticker_filter: bool) -> str:
    """The annual-facts query, optionally with a ``upper(ticker) = upper(?)`` parameter slot."""
    extra: Any = "AND upper(ticker) = upper(?)" if ticker_filter else ""
    return ANNUAL_FACTS_SQL.format(extra_where=extra)


__all__ = [
    "ALLOWED_UNITS",
    "ANNUAL_FACTS_SQL",
    "CURATED_METRICS",
    "CURATED_UNITS",
    "DEFAULT_TAXONOMY",
    "FINANCIALS_FIXED_COLUMNS",
    "TAGS_PATH",
    "annual_facts_sql",
    "create_financials_view",
    "financials_columns",
    "financials_view_sql",
    "load_tags",
    "split_alias",
]
