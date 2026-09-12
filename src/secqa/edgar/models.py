"""EDGAR data models and URL helpers.

Only the pieces of the EDGAR surface that secqa uses are modelled: the ticker map, the
per-company submissions index (which yields :class:`FilingRef`), primary filing documents and
the XBRL ``companyfacts`` JSON. CIKs are always carried as 10-digit zero-padded strings (the
``DocumentMeta.cik`` convention); archive URLs use the un-padded integer form EDGAR expects.
"""

from __future__ import annotations

import re
from datetime import date

from secqa.core.contracts import Frozen

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_BASE = "https://data.sec.gov/submissions/"
COMPANYFACTS_BASE = "https://data.sec.gov/api/xbrl/companyfacts/"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/"

_CIK_RE = re.compile(r"^(?:CIK)?0*(\d{1,10})$", re.IGNORECASE)
_ACCESSION_RE = re.compile(r"^(\d{10})-?(\d{2})-?(\d{6})$")


def normalize_cik(value: str | int) -> str:
    """Return a 10-digit zero-padded CIK from an int, digit string or ``CIK##########`` form."""
    text = str(value).strip()
    match = _CIK_RE.match(text)
    if not match:
        raise ValueError(f"invalid CIK {value!r}: expected up to 10 digits")
    number = int(match.group(1))
    if number <= 0:
        raise ValueError(f"invalid CIK {value!r}: must be positive")
    return f"{number:010d}"


def accession_nodash(accession: str) -> str:
    """``'0000320193-23-000106'`` -> ``'000032019323000106'`` (also accepts the dashless form)."""
    match = _ACCESSION_RE.match(accession.strip())
    if not match:
        raise ValueError(f"invalid accession number {accession!r}")
    return "".join(match.groups())


def accession_dashed(accession: str) -> str:
    """``'000032019323000106'`` -> ``'0000320193-23-000106'`` (also accepts the dashed form)."""
    match = _ACCESSION_RE.match(accession.strip())
    if not match:
        raise ValueError(f"invalid accession number {accession!r}")
    return "-".join(match.groups())


def submissions_url(cik: str | int) -> str:
    """``https://data.sec.gov/submissions/CIK##########.json``."""
    return f"{SUBMISSIONS_BASE}CIK{normalize_cik(cik)}.json"


def submissions_page_url(name: str) -> str:
    """URL of an older-filings page listed under ``filings.files[].name`` in submissions."""
    return f"{SUBMISSIONS_BASE}{name}"


def companyfacts_url(cik: str | int) -> str:
    """``https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json``."""
    return f"{COMPANYFACTS_BASE}CIK{normalize_cik(cik)}.json"


def archive_url(cik: str | int, accession: str, primary_doc: str) -> str:
    """``https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{primary_doc}``."""
    cik_int = int(normalize_cik(cik))
    return f"{ARCHIVES_BASE}{cik_int}/{accession_nodash(accession)}/{primary_doc}"


class FilingRef(Frozen):
    """One filing from a company's submissions index, resolved to its primary-document URL."""

    cik: str  # 10-digit zero-padded
    accession: str  # dashed form, e.g. '0000320193-23-000106'
    form: str  # '10-K', '10-Q', '10-K/A', '8-K', ...
    filing_date: date
    report_date: date | None  # period of report; None when EDGAR leaves it blank
    primary_doc: str
    url: str

    @property
    def period_year(self) -> int:
        """Year the filing reports on: ``report_date.year`` when known, else the filing year."""
        return (self.report_date or self.filing_date).year


__all__ = [
    "ARCHIVES_BASE",
    "COMPANYFACTS_BASE",
    "SUBMISSIONS_BASE",
    "TICKERS_URL",
    "FilingRef",
    "accession_dashed",
    "accession_nodash",
    "archive_url",
    "companyfacts_url",
    "normalize_cik",
    "submissions_page_url",
    "submissions_url",
]
