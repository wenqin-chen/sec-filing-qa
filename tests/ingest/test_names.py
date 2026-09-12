"""Tests for secqa.ingest.doc_meta_from_name / normalize_form."""

from __future__ import annotations

import pytest

from secqa.ingest import UNKNOWN_FORM, doc_meta_from_name, normalize_form


@pytest.mark.parametrize(
    ("doc_name", "expected"),
    [
        ("3M_2022_10K", ("3M", 2022, "10-K")),
        ("3M_2023Q2_10Q", ("3M", 2023, "10-Q")),
        ("ACTIVISIONBLIZZARD_2019_10K", ("ACTIVISIONBLIZZARD", 2019, "10-K")),
        ("JOHNSON_JOHNSON_2022_10K", ("JOHNSON_JOHNSON", 2022, "10-K")),
        ("Pfizer_2021_10K", ("Pfizer", 2021, "10-K")),
        ("BESTBUY_2024Q2_10Q", ("BESTBUY", 2024, "10-Q")),
        ("FOOTLOCKER_2022_8K_dated-2022-05-20", ("FOOTLOCKER", 2022, "8-K")),
        ("AMCOR_2023_10KA", ("AMCOR", 2023, "10-K/A")),
        # EDGAR-ingest names '<TICKER>_<FY>_<FORM>' with the SEC hyphenated form label
        ("AAPL_2023_10-K", ("AAPL", 2023, "10-K")),
        ("MSFT_2024_10-Q", ("MSFT", 2024, "10-Q")),
        ("  3M_2022_10K  ", ("3M", 2022, "10-K")),
    ],
)
def test_doc_meta_from_name_table(doc_name: str, expected: tuple[str, int | None, str]) -> None:
    assert doc_meta_from_name(doc_name) == expected


@pytest.mark.parametrize("doc_name", ["weird", "3M-2022-10K", "3M_22_10K", "", "3M_2022"])
def test_unparseable_names_are_not_guessed(doc_name: str) -> None:
    company, year, form = doc_meta_from_name(doc_name)
    assert company == doc_name
    assert year is None
    assert form == UNKNOWN_FORM


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10K", "10-K"),
        ("10-k", "10-K"),
        ("10-K", "10-K"),
        ("10-K/A", "10-K/A"),
        ("10KA", "10-K/A"),
        ("8K", "8-K"),
        ("20F", "20-F"),
        ("DEF 14A", "DEF 14A"),
        ("S-1", "S-1"),
        ("XYZ", "XYZ"),  # unknown labels pass through upper-cased, never mapped
    ],
)
def test_normalize_form(raw: str, expected: str) -> None:
    assert normalize_form(raw) == expected
