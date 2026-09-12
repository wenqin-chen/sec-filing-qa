"""Polite, cached, rate-limited access to SEC EDGAR (the only module that talks to sec.gov).

Public surface: :class:`EdgarClient` (ticker -> CIK, submissions, filing lists, primary
documents, XBRL companyfacts), :class:`FilingRef`, :class:`TokenBucket` and :class:`DiskCache`.
"""

from secqa.edgar.cache import CacheEntry, DiskCache
from secqa.edgar.client import EdgarClient, EdgarError, TickerNotFound
from secqa.edgar.models import (
    FilingRef,
    accession_dashed,
    accession_nodash,
    archive_url,
    companyfacts_url,
    normalize_cik,
    submissions_url,
)
from secqa.edgar.ratelimit import TokenBucket

__all__ = [
    "CacheEntry",
    "DiskCache",
    "EdgarClient",
    "EdgarError",
    "FilingRef",
    "TickerNotFound",
    "TokenBucket",
    "accession_dashed",
    "accession_nodash",
    "archive_url",
    "companyfacts_url",
    "normalize_cik",
    "submissions_url",
]
