"""Flatten SEC ``companyfacts`` JSON into :class:`FactRow` objects and load them into the store.

The companyfacts document (``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json``)
nests values as ``facts[taxonomy][tag]["units"][unit] -> [entry, ...]`` where each entry has
``end``, ``val``, ``accn``, ``fy``, ``fp``, ``form``, ``filed`` and optionally ``start`` (durations)
and ``frame`` (set on the single fact the SEC frames API picks for a calendar period).

Two facts about the raw data drive the design of the whole ``xbrl`` package:

* ``fy``/``fp`` are the *filing's* fiscal-year and period focus, not the fact's. A FY2023 10-K
  carries its FY2021 and FY2022 comparatives with ``fy=2023, fp='FY'``. This module stores the
  raw values unchanged (so the table is a faithful copy of the source); the fiscal year a value
  belongs to is derived later by :mod:`secqa.xbrl.financials`.
* The same value is repeated once per filing that reports it (original plus every later 10-K's
  comparative), so exact duplicates are common. :func:`flatten_companyfacts` drops entries that
  repeat the same ``(taxonomy, tag, unit, start, end, fy, fp, accn)`` key; distinct filings of
  the same period are kept because their ``accn`` differs and either may be cited.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from secqa.core.contracts import FactRow
from secqa.core.errors import ConfigError
from secqa.core.logging import get_logger
from secqa.store import DuckDBStore

log = get_logger(__name__)

_FACT_COLUMNS = (
    "cik, ticker, taxonomy, tag, unit, fy, fp, form, start_date, end_date, val, accn, filed, frame"
)
_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,10}$")
_REQUIRED_ENTRY_KEYS = ("end", "val", "accn")


def normalize_ticker(ticker: str) -> str:
    """Upper-case and strip a ticker; raises ``ValueError`` when it is not ticker-shaped."""
    candidate = (ticker or "").strip().upper()
    if not _TICKER_RE.match(candidate):
        raise ValueError(f"invalid ticker {ticker!r}: expected 1-10 characters [A-Z0-9.-]")
    return candidate


def pad_cik(value: Any) -> str:
    """10-digit zero-padded CIK from the integer or string companyfacts ``cik`` field."""
    text = str(value).strip()
    if text.upper().startswith("CIK"):
        text = text[3:]
    if not text.isdigit() or int(text) <= 0:
        raise ValueError(f"invalid CIK {value!r} in companyfacts document")
    return f"{int(text):010d}"


def flatten_companyfacts(facts: dict[str, Any], ticker: str) -> list[FactRow]:
    """Flatten a companyfacts document into :class:`FactRow` objects (raw ``fy``/``fp`` kept).

    Entries missing ``end``, ``val`` or ``accn`` (or with a non-numeric ``val``) are skipped and
    counted in the log line; exact duplicates (same taxonomy, tag, unit, period, fy, fp and
    accession) are dropped, first occurrence wins. Rows are returned in a stable order
    (taxonomy, tag, unit, end date, start date, filed, accession).

    Raises:
        ValueError: on a malformed document (no ``facts`` mapping, missing/invalid ``cik``) or an
            invalid ticker.
    """
    if not isinstance(facts, dict):
        raise ValueError("companyfacts must be a JSON object")
    if "cik" not in facts:
        raise ValueError("companyfacts document has no 'cik' field")
    cik = pad_cik(facts["cik"])
    symbol = normalize_ticker(ticker)
    taxonomies = facts.get("facts")
    if not isinstance(taxonomies, dict):
        raise ValueError("companyfacts document has no 'facts' mapping")

    rows: list[FactRow] = []
    seen: set[tuple[Any, ...]] = set()
    n_skipped = 0
    n_duplicates = 0
    for taxonomy, tags in taxonomies.items():
        if not isinstance(tags, dict):
            continue
        for tag, concept in tags.items():
            units = concept.get("units") if isinstance(concept, dict) else None
            if not isinstance(units, dict):
                continue
            for unit, entries in units.items():
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    row = _entry_to_row(entry, cik=cik, ticker=symbol, taxonomy=taxonomy, tag=tag)
                    if row is None:
                        n_skipped += 1
                        continue
                    row = row.model_copy(update={"unit": str(unit)})
                    key = (
                        row.taxonomy,
                        row.tag,
                        row.unit,
                        row.start_date,
                        row.end_date,
                        row.fy,
                        row.fp,
                        row.accn,
                    )
                    if key in seen:
                        n_duplicates += 1
                        continue
                    seen.add(key)
                    rows.append(row)
    rows.sort(
        key=lambda r: (
            r.taxonomy,
            r.tag,
            r.unit,
            r.end_date or date.min,
            r.start_date or date.min,
            r.filed or date.min,
            r.accn,
        )
    )
    log.info(
        "companyfacts_flattened",
        ticker=symbol,
        cik=cik,
        n_rows=len(rows),
        n_duplicates=n_duplicates,
        n_skipped=n_skipped,
    )
    return rows


def load_companyfacts(store: DuckDBStore, facts: dict[str, Any], ticker: str) -> int:
    """Replace every ``xbrl_facts`` row of this company with the flattened document.

    Idempotent: rows with the same ``cik`` are deleted first, then the new rows are inserted in
    one transaction, and the manifest's ``n_facts`` is refreshed (there is no ``store.add_facts``,
    so this module owns the write). Returns the number of rows inserted.

    Raises:
        ConfigError: if the store is read-only or not initialised.
        ValueError: see :func:`flatten_companyfacts`.
    """
    if store.read_only:
        raise ConfigError("load_companyfacts is not allowed on a read-only store")
    rows = flatten_companyfacts(facts, ticker)
    cik = pad_cik(facts["cik"])
    conn = store.conn
    conn.execute("BEGIN TRANSACTION")
    try:
        conn.execute("DELETE FROM xbrl_facts WHERE cik = ?", [cik])
        if rows:
            _insert_rows(store, rows)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    store.set_manifest(n_facts=store.counts()["facts"])
    log.info("companyfacts_loaded", ticker=normalize_ticker(ticker), cik=cik, n_rows=len(rows))
    return len(rows)


def _insert_rows(store: DuckDBStore, rows: list[FactRow]) -> None:
    """Bulk insert through a registered DataFrame (executemany is ~100x slower in DuckDB)."""
    import pandas as pd  # declared dependency; imported lazily (slow import, bulk path only)

    frame = pd.DataFrame(
        {
            "cik": [r.cik for r in rows],
            "ticker": [r.ticker for r in rows],
            "taxonomy": [r.taxonomy for r in rows],
            "tag": [r.tag for r in rows],
            "unit": [r.unit for r in rows],
            "fy": pd.array([r.fy for r in rows], dtype="Int64"),
            "fp": [r.fp for r in rows],
            "form": [r.form for r in rows],
            "start_date": [_iso(r.start_date) for r in rows],
            "end_date": [_iso(r.end_date) for r in rows],
            "val": [r.val for r in rows],
            "accn": [r.accn for r in rows],
            "filed": [_iso(r.filed) for r in rows],
            "frame": [r.frame for r in rows],
        }
    )
    for column in ("fp", "form", "start_date", "end_date", "filed", "frame"):
        frame[column] = frame[column].astype(object)
    conn = store.conn
    conn.register("_secqa_xbrl_batch", frame)
    try:
        conn.execute(
            f"INSERT INTO xbrl_facts ({_FACT_COLUMNS}) "
            "SELECT cik, ticker, taxonomy, tag, unit, CAST(fy AS INTEGER), fp, form, "
            "CAST(start_date AS DATE), CAST(end_date AS DATE), CAST(val AS DOUBLE), accn, "
            "CAST(filed AS DATE), frame FROM _secqa_xbrl_batch"
        )
    finally:
        conn.unregister("_secqa_xbrl_batch")


def _entry_to_row(entry: Any, *, cik: str, ticker: str, taxonomy: str, tag: str) -> FactRow | None:
    """One companyfacts entry -> ``FactRow`` (``unit`` set by the caller); None if unusable."""
    if not isinstance(entry, dict) or any(entry.get(key) is None for key in _REQUIRED_ENTRY_KEYS):
        return None
    try:
        val = float(entry["val"])
        end_date = date.fromisoformat(str(entry["end"]))
        start_date = _optional_date(entry.get("start"))
        filed = _optional_date(entry.get("filed"))
        fy = int(entry["fy"]) if entry.get("fy") is not None else None
    except (TypeError, ValueError):
        return None
    if val != val or val in (float("inf"), float("-inf")):  # NaN / inf are not facts
        return None
    return FactRow(
        cik=cik,
        ticker=ticker,
        taxonomy=str(taxonomy),
        tag=str(tag),
        unit="",  # replaced by the caller
        fy=fy,
        fp=_optional_str(entry.get("fp")),
        form=_optional_str(entry.get("form")),
        start_date=start_date,
        end_date=end_date,
        val=val,
        accn=str(entry["accn"]),
        filed=filed,
        frame=_optional_str(entry.get("frame")),
    )


def _optional_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    return date.fromisoformat(str(value))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _iso(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


__all__ = ["flatten_companyfacts", "load_companyfacts", "normalize_ticker", "pad_cik"]
