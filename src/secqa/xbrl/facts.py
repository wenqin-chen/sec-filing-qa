"""``lookup_fact``: curated metric -> the annual XBRL fact row(s) with an accession number.

This is the agent's ``lookup_fact(ticker, metric, fiscal_year)`` tool. It walks the metric's
alias chain from ``tags.yaml`` (``revenue`` -> ``Revenues``, then
``RevenueFromContractWithCustomerExcludingAssessedTax`` ...) over the same annual-facts query
that builds the ``financials`` view, so the number the tool cites is the number the view shows.

``metric`` may also be a raw concept name (``Assets`` or ``dei:EntityCommonStockSharesOutstanding``)
as an escape hatch for anything not curated; in that case every unit the concept is reported in
is returned (one row per unit), which is why the return type is a list.

The returned :class:`FactRow` carries ``fy`` = the derived fiscal year the value belongs to (and
``fp = 'FY'``), not the raw filing focus stored in ``xbrl_facts``; ``accn``/``filed`` identify
the 10-K that last reported the value, and ``ref`` (``xbrl:<tag>|FY<fy>|<accn>``) is the
citation key the verifier expects.
"""

from __future__ import annotations

from typing import Any

from secqa.core.contracts import FactRow
from secqa.core.logging import get_logger
from secqa.store import DuckDBStore
from secqa.xbrl.financials import CURATED_METRICS, CURATED_UNITS, annual_facts_sql, split_alias
from secqa.xbrl.load import normalize_ticker

log = get_logger(__name__)

_LOOKUP_SQL = (
    "SELECT cik, ticker, taxonomy, tag, unit, fy, fp, form, start_date, end_date, val, accn, "
    "filed, frame, fiscal_year "
    f"FROM ({annual_facts_sql(ticker_filter=True)}) "
    "WHERE fiscal_year = ? ORDER BY taxonomy, tag, unit"
)


def resolve_metric(metric: str) -> tuple[str, list[tuple[str, str]], str | None]:
    """``metric`` -> ``(canonical name, [(taxonomy, tag), ...] alias chain, required unit)``.

    Curated names are matched case-insensitively; anything else is treated as a raw concept
    (``Tag`` or ``taxonomy:Tag``) with no unit restriction.

    Raises:
        ValueError: when ``metric`` is neither a curated metric nor a concept-shaped name.
    """
    candidate = (metric or "").strip()
    if not candidate:
        raise ValueError("metric must not be empty")
    key = candidate.lower()
    if key in CURATED_METRICS:
        chain = [split_alias(alias) for alias in CURATED_METRICS[key]]
        return key, chain, CURATED_UNITS[key]
    try:
        taxonomy, tag = split_alias(candidate)
    except ValueError as exc:
        raise ValueError(
            f"unknown metric {metric!r}; use one of {', '.join(CURATED_METRICS)} "
            "or a concept name such as 'Assets'"
        ) from exc
    return f"{taxonomy}:{tag}", [(taxonomy, tag)], None


def lookup_fact(store: DuckDBStore, ticker: str, metric: str, fiscal_year: int) -> list[FactRow]:
    """Annual value(s) of ``metric`` for ``ticker`` in ``fiscal_year`` from the latest 10-K.

    Returns the rows of the first alias in the chain that has a value (one per unit; a curated
    metric is restricted to its unit, so normally exactly one row), or ``[]`` when nothing
    matches. ``concept_used`` is set to ``'<taxonomy>:<tag>'`` of the alias that resolved.

    Raises:
        ValueError: invalid ticker, unknown metric, or a non-integer fiscal year.
    """
    symbol = normalize_ticker(ticker)
    if isinstance(fiscal_year, bool) or not isinstance(fiscal_year, int):
        raise ValueError(f"fiscal_year must be an integer, got {fiscal_year!r}")
    name, chain, unit = resolve_metric(metric)
    rows = store.conn.execute(_LOOKUP_SQL, [symbol, fiscal_year]).fetchall()
    by_concept: dict[tuple[str, str], list[tuple[Any, ...]]] = {}
    for row in rows:
        if unit is not None and row[4] != unit:
            continue
        by_concept.setdefault((row[2], row[3]), []).append(row)
    for taxonomy, tag in chain:
        matched = by_concept.get((taxonomy, tag))
        if not matched:
            continue
        facts = [_row_to_fact(row, concept_used=f"{taxonomy}:{tag}") for row in matched]
        log.debug(
            "lookup_fact",
            ticker=symbol,
            metric=name,
            fiscal_year=fiscal_year,
            concept_used=f"{taxonomy}:{tag}",
            n_rows=len(facts),
        )
        return facts
    log.debug("lookup_fact_miss", ticker=symbol, metric=name, fiscal_year=fiscal_year)
    return []


def _row_to_fact(row: tuple[Any, ...], *, concept_used: str) -> FactRow:
    return FactRow(
        cik=row[0],
        ticker=row[1],
        taxonomy=row[2],
        tag=row[3],
        unit=row[4],
        fy=int(row[14]),
        fp="FY",
        form=row[7],
        start_date=row[8],
        end_date=row[9],
        val=float(row[10]),
        accn=row[11],
        filed=row[12],
        frame=row[13],
        concept_used=concept_used,
    )


__all__ = ["lookup_fact", "resolve_metric"]
