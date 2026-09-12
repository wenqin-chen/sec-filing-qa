"""Polite, cached, rate-limited SEC EDGAR client — the only module that talks to sec.gov.

Every request carries the declared ``User-Agent`` (SEC fair-access policy: ``"Name email"``) and
``Accept-Encoding: gzip``, passes through a :class:`~secqa.edgar.ratelimit.TokenBucket`
(default 8 req/s, under the 10 req/s limit) and is retried with tenacity (5 attempts,
exponential backoff 1-30 s with jitter) on HTTP 429/503 and transport errors only. Successful
bodies are cached on disk by ``sha256(url)`` so a re-run of an ingest never re-downloads and a
cache hit neither touches the network nor consumes a rate-limit token.

EDGAR is contacted only from CLI ingest paths; nothing at answer time imports this module.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from secqa.core.errors import ConfigError, SecqaError
from secqa.core.logging import get_logger
from secqa.core.settings import EMAIL_RE, Settings
from secqa.edgar.cache import DiskCache
from secqa.edgar.models import (
    TICKERS_URL,
    FilingRef,
    archive_url,
    companyfacts_url,
    normalize_cik,
    submissions_page_url,
    submissions_url,
)
from secqa.edgar.ratelimit import TokenBucket

log = get_logger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache/edgar")
RETRY_ATTEMPTS = 5
RETRY_WAIT_INITIAL_S = 1.0
RETRY_WAIT_MAX_S = 30.0
_RETRYABLE_STATUS = frozenset({429, 503})


class EdgarError(SecqaError):
    """An EDGAR request failed after retries (or immediately, for non-retryable statuses)."""

    def __init__(
        self,
        message: str,
        *,
        url: str,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.url = url
        self.status_code = status_code
        self.retryable = retryable


class TickerNotFound(EdgarError, LookupError):
    """The ticker is absent from ``company_tickers.json``."""


def _is_retryable(exc: BaseException) -> bool:
    """Retry policy: 429/503 and transport-level failures only; every other error is final."""
    if isinstance(exc, EdgarError):
        return exc.retryable
    return isinstance(exc, httpx.TransportError)


def _parse_date(value: Any) -> date | None:
    """Parse an EDGAR ``YYYY-MM-DD`` string; blank or malformed values become ``None``."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        return None


class EdgarClient:
    """Synchronous EDGAR client with rate limiting, retries and an on-disk cache.

    ``sleep`` and ``clock`` are injectable so tests can run the rate limiter and the retry
    backoff against a fake clock; production code leaves the defaults.
    """

    def __init__(
        self,
        user_agent: str,
        max_rps: float = 8.0,
        cache_dir: Path = DEFAULT_CACHE_DIR,
        timeout_s: float = 30.0,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        user_agent = user_agent.strip()
        if not EMAIL_RE.search(user_agent):
            raise ConfigError(
                "EDGAR User-Agent must contain a contact email, e.g. 'Jane Doe jane@example.com'"
            )
        if max_rps <= 0:
            raise ValueError(f"max_rps must be positive, got {max_rps}")
        self.user_agent = user_agent
        self.max_rps = float(max_rps)
        self.timeout_s = float(timeout_s)
        self.cache = DiskCache(Path(cache_dir))
        self.bucket = TokenBucket(rate=self.max_rps, burst=1, clock=clock, sleep=sleep)
        self._sleep = sleep
        self._http = httpx.Client(
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json, text/html, */*",
            },
            timeout=httpx.Timeout(self.timeout_s),
            follow_redirects=True,
        )

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> EdgarClient:
        """Build a client from :class:`~secqa.core.settings.Settings` (requires SEC_USER_AGENT)."""
        return cls(settings.require_sec_user_agent(), **overrides)

    # ---- lifecycle ----

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._http.close()

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---- transport ----

    def _request_once(self, url: str) -> httpx.Response:
        """One rate-limited GET; raises ``EdgarError`` (retryable for 429/503) on HTTP errors."""
        self.bucket.acquire()
        response = self._http.get(url)
        if response.status_code in _RETRYABLE_STATUS:
            raise EdgarError(
                f"EDGAR returned HTTP {response.status_code} for {url}",
                url=url,
                status_code=response.status_code,
                retryable=True,
            )
        if response.is_error:
            raise EdgarError(
                f"EDGAR returned HTTP {response.status_code} for {url}",
                url=url,
                status_code=response.status_code,
                retryable=False,
            )
        return response

    def _log_retry(self, state: RetryCallState) -> None:
        exc = state.outcome.exception() if state.outcome is not None else None
        log.warning(
            "edgar.retry",
            attempt=state.attempt_number,
            sleep_s=round(state.idle_for, 3),
            error=str(exc) if exc else None,
        )

    def _fetch(self, url: str) -> httpx.Response:
        """GET with retries on 429/503/network errors; other failures raise immediately."""
        retrying = Retrying(
            stop=stop_after_attempt(RETRY_ATTEMPTS),
            wait=wait_exponential_jitter(initial=RETRY_WAIT_INITIAL_S, max=RETRY_WAIT_MAX_S),
            retry=retry_if_exception(_is_retryable),
            sleep=self._sleep,
            before_sleep=self._log_retry,
            reraise=True,
        )
        try:
            return retrying(self._request_once, url)
        except httpx.TransportError as exc:
            raise EdgarError(
                f"network error contacting EDGAR for {url}: {exc}", url=url, retryable=True
            ) from exc

    # ---- public fetchers ----

    def get_bytes(self, url: str) -> bytes:
        """Return the response body for ``url``, served from the disk cache when present."""
        cached = self.cache.get(url)
        if cached is not None:
            log.debug("edgar.fetch", url=url, cache="hit", bytes=len(cached))
            return cached
        started = time.perf_counter()
        response = self._fetch(url)
        body = response.content
        self.cache.put(url, body, content_type=response.headers.get("content-type"))
        log.info(
            "edgar.fetch",
            url=url,
            cache="miss",
            status=response.status_code,
            bytes=len(body),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        return body

    def get_json(self, url: str) -> dict[str, Any]:
        """Return the JSON object at ``url`` (cached by URL like :meth:`get_bytes`)."""
        body = self.get_bytes(url)
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            # A corrupt cache entry must not poison every later call: evict it.
            self.cache.delete(url)
            raise EdgarError(f"EDGAR response at {url} is not valid JSON", url=url) from exc
        if not isinstance(parsed, dict):
            raise EdgarError(f"EDGAR response at {url} is not a JSON object", url=url)
        return parsed

    # ---- EDGAR endpoints ----

    def cik_for_ticker(self, ticker: str) -> str:
        """Resolve a ticker (case-insensitive) to its 10-digit zero-padded CIK."""
        wanted = ticker.strip().upper()
        if not wanted:
            raise ValueError("ticker must not be empty")
        table = self.get_json(TICKERS_URL)
        for entry in table.values():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("ticker", "")).upper() == wanted:
                return normalize_cik(entry["cik_str"])
        raise TickerNotFound(
            f"ticker {ticker!r} not found in EDGAR company_tickers.json", url=TICKERS_URL
        )

    def submissions(self, cik: str) -> dict[str, Any]:
        """Return the submissions index for ``cik`` (recent filings + pointers to older pages)."""
        return self.get_json(submissions_url(cik))

    def list_filings(
        self,
        cik: str,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
        years: Iterable[int] | None = None,
    ) -> list[FilingRef]:
        """List filings for ``cik`` filtered by form type and (optionally) report-period year.

        ``years`` matches :attr:`FilingRef.period_year` (the period of report, falling back to
        the filing date), so ``years=[2022]`` returns the FY2022 10-K filed in early 2023.
        Older filings that EDGAR moves to separate pages (``filings.files``) are fetched only
        when their date range could contain a requested year (all pages when ``years`` is None).
        """
        cik10 = normalize_cik(cik)
        wanted_forms = {form.strip().upper() for form in forms}
        wanted_years = None if years is None else {int(year) for year in years}
        index = self.submissions(cik10)
        filings = index.get("filings", {})
        columns: list[dict[str, Any]] = [filings.get("recent", {})]
        for page in filings.get("files", []) or []:
            if wanted_years is not None and not self._page_may_cover(page, wanted_years):
                continue
            columns.append(self.get_json(submissions_page_url(str(page["name"]))))
        out: list[FilingRef] = []
        for block in columns:
            out.extend(self._parse_filing_block(cik10, block, wanted_forms, wanted_years))
        log.info(
            "edgar.list_filings",
            cik=cik10,
            forms=sorted(wanted_forms),
            years=sorted(wanted_years) if wanted_years else None,
            n=len(out),
        )
        return out

    @staticmethod
    def _page_may_cover(page: dict[str, Any], years: set[int]) -> bool:
        """True unless the page's filing-date range ends before the earliest wanted year."""
        to = _parse_date(page.get("filingTo"))
        if to is None:
            return True
        # A filing for period year Y is never filed before Y, so a page ending before min(years)
        # cannot hold one; a page starting after max(years)+1 could still (late amendments).
        return to.year >= min(years)

    @staticmethod
    def _parse_filing_block(
        cik10: str,
        block: dict[str, Any],
        wanted_forms: set[str],
        wanted_years: set[int] | None,
    ) -> list[FilingRef]:
        """Turn EDGAR's columnar filing arrays into ``FilingRef`` rows matching the filters."""
        accessions = block.get("accessionNumber", []) or []
        form_col = block.get("form", []) or []
        filed_col = block.get("filingDate", []) or []
        report_col = block.get("reportDate", []) or []
        doc_col = block.get("primaryDocument", []) or []
        n = len(accessions)
        if not all(len(col) == n for col in (form_col, filed_col, report_col, doc_col)):
            raise EdgarError(
                "EDGAR submissions block has ragged columns", url=submissions_url(cik10)
            )
        refs: list[FilingRef] = []
        for i in range(n):
            form = str(form_col[i]).strip().upper()
            if form not in wanted_forms:
                continue
            filing_date = _parse_date(filed_col[i])
            if filing_date is None:
                log.warning("edgar.filing_skipped", accession=accessions[i], reason="no filingDate")
                continue
            primary_doc = str(doc_col[i] or "").strip()
            if not primary_doc:
                log.warning(
                    "edgar.filing_skipped", accession=accessions[i], reason="no primaryDocument"
                )
                continue
            ref = FilingRef(
                cik=cik10,
                accession=str(accessions[i]),
                form=form,
                filing_date=filing_date,
                report_date=_parse_date(report_col[i]),
                primary_doc=primary_doc,
                url=archive_url(cik10, str(accessions[i]), primary_doc),
            )
            if wanted_years is not None and ref.period_year not in wanted_years:
                continue
            refs.append(ref)
        return refs

    def fetch_primary_document(self, ref: FilingRef) -> bytes:
        """Download (or read from cache) the primary document of a filing."""
        return self.get_bytes(ref.url)

    def companyfacts(self, cik: str) -> dict[str, Any]:
        """Return the XBRL ``companyfacts`` JSON for ``cik``."""
        return self.get_json(companyfacts_url(cik))


__all__ = [
    "DEFAULT_CACHE_DIR",
    "RETRY_ATTEMPTS",
    "EdgarClient",
    "EdgarError",
    "TickerNotFound",
]
