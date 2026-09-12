"""data/companies.yaml parses, CIKs are 10 digits, tickers unique, open-set prefixes resolve."""

from __future__ import annotations

from pathlib import Path

import pytest

from secqa.core.errors import ConfigError
from secqa.indexing import DEFAULT_COMPANIES_PATH, Company, load_companies, resolve_company
from tests.indexing.conftest import REPO_ROOT

COMPANIES_YAML = REPO_ROOT / DEFAULT_COMPANIES_PATH

# Company prefixes of the FinanceBench open-set doc_name values ('<PREFIX>_<YEAR>[Qn]_<FORM>').
# These are company identifiers, not dataset rows.
OPEN_SET_PREFIXES = (
    "3M",
    "ACTIVISIONBLIZZARD",
    "ADOBE",
    "AES",
    "AMAZON",
    "AMCOR",
    "AMD",
    "AMERICANEXPRESS",
    "AMERICANWATERWORKS",
    "BESTBUY",
    "BLOCK",
    "BOEING",
    "COCACOLA",
    "CORNING",
    "COSTCO",
    "CVSHEALTH",
    "FOOTLOCKER",
    "GENERALMILLS",
    "JOHNSON_JOHNSON",
    "JPMORGAN",
    "KRAFTHEINZ",
    "LOCKHEEDMARTIN",
    "MGMRESORTS",
    "MICROSOFT",
    "NETFLIX",
    "NIKE",
    "PAYPAL",
    "PEPSICO",
    "PFIZER",
    "Pfizer",
    "ULTABEAUTY",
    "VERIZON",
    "WALMART",
)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "companies.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# ---- the committed file -------------------------------------------------------------------


def test_committed_file_parses_with_valid_ciks_and_unique_tickers() -> None:
    companies = load_companies(COMPANIES_YAML)
    assert len(companies) >= 30
    for company in companies:
        assert len(company.cik) == 10 and company.cik.isdigit() and int(company.cik) > 0
        assert company.ticker == company.ticker.upper()
        assert company.name and company.financebench_aliases
    assert len({c.ticker for c in companies}) == len(companies)
    assert len({c.cik for c in companies}) == len(companies)
    assert len({c.name for c in companies}) == len(companies)


def test_every_open_set_prefix_resolves() -> None:
    companies = load_companies(COMPANIES_YAML)
    unresolved = [p for p in OPEN_SET_PREFIXES if resolve_company(companies, p) is None]
    assert unresolved == []
    assert resolve_company(companies, "JOHNSON_JOHNSON").ticker == "JNJ"  # type: ignore[union-attr]
    assert resolve_company(companies, "3M").cik == "0000066740"  # type: ignore[union-attr]


# ---- validation ----------------------------------------------------------------------------


def test_company_model_normalises_ticker_and_cik() -> None:
    company = Company(name=" Fixture Corp ", ticker="fixt", cik="1234567")
    assert (company.name, company.ticker, company.cik) == ("Fixture Corp", "FIXT", "0001234567")
    assert company.financebench_aliases == []
    assert company.matches("fixture corp") and company.matches("FIXT")
    with pytest.raises(ValueError):
        Company(name="X", ticker="FIXT", cik="not-a-cik")
    with pytest.raises(ValueError):
        Company(name="X", ticker="", cik="1")
    with pytest.raises(ValueError, match="duplicate alias"):
        Company(name="X", ticker="X", cik="1", financebench_aliases=["ACME", "acme"])


def test_missing_or_malformed_file_is_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_companies(tmp_path / "nope.yaml")
    with pytest.raises(ConfigError, match="'companies' list"):
        load_companies(_write(tmp_path, "companies: []\n"))
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_companies(_write(tmp_path, "companies:\n  - just-a-string\n"))
    with pytest.raises(ConfigError, match=r"companies\[0\]"):
        load_companies(_write(tmp_path, "companies:\n  - {name: A, ticker: A, cik: xyz}\n"))


def test_duplicates_are_config_errors(tmp_path: Path) -> None:
    dup_ticker = (
        "companies:\n  - {name: A, ticker: AAA, cik: 1}\n  - {name: B, ticker: AAA, cik: 2}\n"
    )
    with pytest.raises(ConfigError, match="ticker 'AAA'"):
        load_companies(_write(tmp_path, dup_ticker))
    dup_cik = "companies:\n  - {name: A, ticker: AAA, cik: 1}\n  - {name: B, ticker: BBB, cik: 1}\n"
    with pytest.raises(ConfigError, match="cik '0000000001'"):
        load_companies(_write(tmp_path, dup_cik))
    ambiguous = (
        "companies:\n"
        "  - {name: A, ticker: AAA, cik: 1, financebench_aliases: [SHARED]}\n"
        "  - {name: B, ticker: BBB, cik: 2, financebench_aliases: [shared]}\n"
    )
    with pytest.raises(ConfigError, match="ambiguous"):
        load_companies(_write(tmp_path, ambiguous))


def test_resolve_company_prefers_explicit_alias_over_name_match() -> None:
    companies = [
        Company(name="Block", ticker="XYZ", cik="1", financebench_aliases=["BLOCK"]),
        Company(name="Square Peg", ticker="SQP", cik="2", financebench_aliases=["BLOCKCHAIN"]),
    ]
    assert resolve_company(companies, "block").ticker == "XYZ"  # type: ignore[union-attr]
    assert resolve_company(companies, "BLOCK_CHAIN").ticker == "SQP"  # type: ignore[union-attr]
    assert resolve_company(companies, "sqp").ticker == "SQP"  # type: ignore[union-attr]
    assert resolve_company(companies, "") is None
    assert resolve_company(companies, "NOPE") is None
