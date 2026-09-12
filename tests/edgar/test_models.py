"""Tests for secqa.edgar.models: CIK/accession normalisation and URL builders."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from secqa.edgar.models import (
    FilingRef,
    accession_dashed,
    accession_nodash,
    archive_url,
    companyfacts_url,
    normalize_cik,
    submissions_url,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (320193, "0000320193"),
        ("320193", "0000320193"),
        ("0000320193", "0000320193"),
        ("CIK0000320193", "0000320193"),
        ("  1234567 ", "0001234567"),
    ],
)
def test_normalize_cik(raw: str | int, expected: str) -> None:
    assert normalize_cik(raw) == expected


@pytest.mark.parametrize("raw", ["", "abc", "12345678901", "0", "-5", "CIK"])
def test_normalize_cik_rejects_garbage(raw: str) -> None:
    with pytest.raises(ValueError):
        normalize_cik(raw)


def test_accession_forms_round_trip() -> None:
    dashed = "0001234567-24-000010"
    nodash = "000123456724000010"
    assert accession_nodash(dashed) == nodash
    assert accession_nodash(nodash) == nodash
    assert accession_dashed(nodash) == dashed
    assert accession_dashed(dashed) == dashed
    with pytest.raises(ValueError):
        accession_nodash("12-34")


def test_url_builders_follow_edgar_conventions() -> None:
    assert submissions_url("320193") == "https://data.sec.gov/submissions/CIK0000320193.json"
    assert (
        companyfacts_url(320193) == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    )
    # Archive paths use the *un-padded* CIK and the dashless accession number.
    assert (
        archive_url("0000320193", "0000320193-23-000106", "aapl-20230930.htm")
        == "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    )


def test_filing_ref_is_frozen_and_reports_period_year() -> None:
    ref = FilingRef(
        cik="0001234567",
        accession="0001234567-24-000010",
        form="10-K",
        filing_date=date(2024, 2, 15),
        report_date=date(2023, 12, 31),
        primary_doc="fixt-20231231.htm",
        url=archive_url("0001234567", "0001234567-24-000010", "fixt-20231231.htm"),
    )
    assert ref.period_year == 2023
    with pytest.raises(ValidationError):
        ref.form = "10-Q"  # type: ignore[misc]
    no_report = ref.model_copy(update={"report_date": None})
    assert no_report.period_year == 2024
