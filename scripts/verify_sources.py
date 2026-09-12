#!/usr/bin/env python
"""Verify the external sources the benchmark depends on; exit non-zero if any check fails.

Checks (SPEC 4.1, Day 1 task):

1. **HF split** -- ``PatronusAI/financebench`` exposes a single ``train`` split with exactly 150
   rows (datasets-server API; no ``datasets`` download).
2. **PDF magic bytes** -- a sample of the ``doc_name`` PDFs at
   ``raw.githubusercontent.com/patronus-ai/financebench/main/pdfs/`` starts with ``%PDF`` (a 404
   page or an LFS pointer would otherwise index as zero pages).
3. **companies.yaml CIKs** -- every ticker in ``data/companies.yaml`` resolves to the recorded CIK
   in ``company_tickers.json``; delisted tickers fall back to the submissions endpoint, which
   must know the CIK.

Network only, no keys. SEC requests carry the declared ``SEC_USER_AGENT``. Nothing is cached so
the answer reflects the sources *now*; run it with ``--report`` to keep a JSON record.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from secqa.core.errors import ConfigError
from secqa.core.settings import EMAIL_RE, get_settings
from secqa.edgar.models import TICKERS_URL, normalize_cik, submissions_url
from secqa.indexing import financebench_pdf_url, load_companies
from secqa.indexing.financebench_corpus import DOWNLOAD_USER_AGENT
from secqa.indexing.pipeline import DEFAULT_COMPANIES_PATH
from secqa.ingest.pdf import is_pdf_bytes

HF_DATASET = "PatronusAI/financebench"
HF_API = "https://datasets-server.huggingface.co"
HF_SPLITS_URL = f"{HF_API}/splits?dataset=PatronusAI%2Ffinancebench"
HF_ROWS_URL = (
    f"{HF_API}/rows?dataset=PatronusAI%2Ffinancebench&config=default&split=train"
    "&offset={offset}&length={length}"
)
HF_PAGE = 100
EXPECTED_SPLIT = "train"
EXPECTED_ROWS = 150
DEFAULT_PDF_SAMPLE = 3
SEC_PAUSE_S = 0.15  # well under the 10 req/s fair-access limit


@dataclass(frozen=True)
class Check:
    """Outcome of one verification."""

    name: str
    ok: bool
    detail: str


# ---------------------------------------------------------------------------------------------
# 1. Hugging Face split
# ---------------------------------------------------------------------------------------------


def check_hf_split(client: httpx.Client) -> tuple[Check, list[str]]:
    """Confirm the single ``train`` split of 150 rows; also return the distinct ``doc_name``s.

    Uses the datasets-server ``/splits`` and ``/rows`` endpoints (paginated) so no dataset text
    is written anywhere; the doc names feed the PDF check.
    """
    try:
        splits_body = client.get(HF_SPLITS_URL).raise_for_status().json()
    except (httpx.HTTPError, ValueError) as exc:
        return Check("hf_split", False, f"splits endpoint failed: {exc}"), []
    splits = [
        str(s.get("split"))
        for s in splits_body.get("splits", [])
        if s.get("dataset") in (None, HF_DATASET)
    ]
    if splits != [EXPECTED_SPLIT]:
        return Check("hf_split", False, f"expected splits ['{EXPECTED_SPLIT}'], got {splits}"), []

    doc_names: list[str] = []
    total: int | None = None
    offset = 0
    while True:
        url = HF_ROWS_URL.format(offset=offset, length=HF_PAGE)
        try:
            body = client.get(url).raise_for_status().json()
        except (httpx.HTTPError, ValueError) as exc:
            return Check("hf_split", False, f"rows endpoint failed at offset {offset}: {exc}"), []
        total = int(body.get("num_rows_total", 0))
        rows = body.get("rows", [])
        for entry in rows:
            row = entry.get("row", {}) if isinstance(entry, dict) else {}
            name = row.get("doc_name")
            if name and name not in doc_names:
                doc_names.append(str(name))
        offset += len(rows)
        if not rows or offset >= total:
            break
    if total != EXPECTED_ROWS:
        return Check("hf_split", False, f"expected {EXPECTED_ROWS} rows, got {total}"), doc_names
    if offset != total:
        return Check("hf_split", False, f"paginated {offset} rows but total is {total}"), doc_names
    return (
        Check("hf_split", True, f"split '{EXPECTED_SPLIT}', {total} rows, {len(doc_names)} docs"),
        doc_names,
    )


# ---------------------------------------------------------------------------------------------
# 2. PDF magic bytes
# ---------------------------------------------------------------------------------------------


def check_pdf_magic(client: httpx.Client, doc_names: Iterable[str], sample: int) -> Check:
    """GET the first bytes of ``sample`` PDFs and require the ``%PDF`` marker on each."""
    names = list(dict.fromkeys(doc_names))[: max(sample, 0)]
    if not names:
        return Check("pdf_magic", False, "no doc_names to sample (HF check failed?)")
    bad: list[str] = []
    for name in names:
        url = financebench_pdf_url(name)
        try:
            with client.stream("GET", url) as response:
                if response.status_code != 200:
                    bad.append(f"{name}: HTTP {response.status_code}")
                    continue
                head = b""
                for chunk in response.iter_bytes(chunk_size=64):
                    head += chunk
                    if len(head) >= 5:
                        break
        except httpx.HTTPError as exc:
            bad.append(f"{name}: {exc}")
            continue
        if not is_pdf_bytes(head):
            bad.append(f"{name}: no %PDF marker (got {head[:5]!r})")
    if bad:
        return Check("pdf_magic", False, "; ".join(bad))
    return Check("pdf_magic", True, f"{len(names)} sampled PDFs start with %PDF: {names}")


# ---------------------------------------------------------------------------------------------
# 3. companies.yaml CIKs
# ---------------------------------------------------------------------------------------------


def _ticker_map(client: httpx.Client) -> dict[str, str]:
    """``{TICKER: 10-digit CIK}`` from ``company_tickers.json``."""
    body = client.get(TICKERS_URL).raise_for_status().json()
    out: dict[str, str] = {}
    for entry in body.values():
        ticker = str(entry.get("ticker", "")).strip().upper()
        cik = entry.get("cik_str")
        if ticker and cik is not None:
            out[ticker] = normalize_cik(cik)
    return out


def check_companies(
    client: httpx.Client,
    companies_path: Path = DEFAULT_COMPANIES_PATH,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> Check:
    """Every company's ticker -> CIK must agree with SEC; delisted tickers must still resolve."""
    try:
        companies = load_companies(companies_path)
    except ConfigError as exc:
        return Check("companies_ciks", False, str(exc))
    try:
        tickers = _ticker_map(client)
    except (httpx.HTTPError, ValueError, AttributeError) as exc:
        return Check("companies_ciks", False, f"company_tickers.json failed: {exc}")
    problems: list[str] = []
    delisted: list[str] = []
    for company in companies:
        listed = tickers.get(company.ticker)
        if listed is not None:
            if listed != company.cik:
                problems.append(f"{company.ticker}: yaml {company.cik} != SEC {listed}")
            continue
        sleep(SEC_PAUSE_S)
        try:
            body = client.get(submissions_url(company.cik)).raise_for_status().json()
        except (httpx.HTTPError, ValueError) as exc:
            problems.append(f"{company.ticker}: not listed and submissions failed: {exc}")
            continue
        if normalize_cik(body.get("cik", "")) != company.cik:
            problems.append(f"{company.ticker}: submissions CIK {body.get('cik')!r} mismatch")
            continue
        delisted.append(company.ticker)
    if problems:
        return Check("companies_ciks", False, "; ".join(problems))
    detail = f"{len(companies)} companies verified"
    if delisted:
        detail += f"; delisted but resolvable via submissions: {delisted}"
    return Check("companies_ciks", True, detail)


# ---------------------------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------------------------


def run_checks(
    client: httpx.Client,
    *,
    companies_path: Path = DEFAULT_COMPANIES_PATH,
    pdf_sample: int = DEFAULT_PDF_SAMPLE,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Check]:
    """Run every check and return the results in order (never raises on a failed source)."""
    hf, doc_names = check_hf_split(client)
    checks = [hf]
    checks.append(check_pdf_magic(client, doc_names, pdf_sample))
    checks.append(check_companies(client, companies_path, sleep=sleep))
    return checks


def make_client(user_agent: str, timeout_s: float = 30.0) -> httpx.Client:
    """An ``httpx.Client`` declaring the SEC User-Agent (GitHub/HF accept it too)."""
    return httpx.Client(
        timeout=httpx.Timeout(timeout_s),
        follow_redirects=True,
        headers={"User-Agent": f"{user_agent} {DOWNLOAD_USER_AGENT}"},
    )


def resolve_user_agent(explicit: str | None) -> str:
    """``--user-agent`` or ``SEC_USER_AGENT``; must contain an email (SEC fair-access policy)."""
    if explicit:
        if not EMAIL_RE.search(explicit):
            raise ConfigError("--user-agent must contain a contact email ('Name email')")
        return explicit.strip()
    return get_settings().require_sec_user_agent()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--companies", type=Path, default=DEFAULT_COMPANIES_PATH)
    parser.add_argument("--pdf-sample", type=int, default=DEFAULT_PDF_SAMPLE)
    parser.add_argument("--user-agent", default=None, help="overrides SEC_USER_AGENT")
    parser.add_argument("--report", type=Path, default=None, help="write a JSON report here")
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def write_report(path: Path, checks: list[Check]) -> None:
    """Persist the outcome (timestamped) as JSON; parent directories are created."""
    payload: dict[str, Any] = {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ok": all(c.ok for c in checks),
        "checks": [asdict(c) for c in checks],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point; 0 when every check passes, 1 otherwise (2 on a configuration error)."""
    args = parse_args(argv)
    try:
        user_agent = resolve_user_agent(args.user_agent)
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    with make_client(user_agent, args.timeout) as client:
        checks = run_checks(client, companies_path=args.companies, pdf_sample=args.pdf_sample)
    for check in checks:
        print(f"[{'ok' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
    if args.report is not None:
        write_report(args.report, checks)
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
