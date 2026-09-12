"""Tests for secqa.ingest.pdf on reportlab-generated PDFs (no third-party files)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from secqa.core.contracts import Page
from secqa.ingest import extract_pdf_pages
from secqa.ingest.pdf import is_pdf_bytes, normalize_extracted_text


def test_three_page_fixture_yields_one_based_pages(fixture_pdf: Path) -> None:
    pages = extract_pdf_pages(fixture_pdf, "FIXTURE_2023_10K")
    assert [p.page_num for p in pages] == [1, 2, 3]
    assert all(isinstance(p, Page) and p.doc_name == "FIXTURE_2023_10K" for p in pages)
    assert "Total net sales were $1,577 million in fiscal 2023" in pages[0].text
    assert "Item 8. Financial Statements" in pages[1].text
    assert "Long-term debt was $1,200 million" in pages[2].text
    # reportlab wraps the paragraph over several lines; the join must not lose or glue words.
    assert "an increase of 12% over 2022." in pages[0].text


def test_page_text_is_whitespace_collapsed(fixture_pdf: Path) -> None:
    for page in extract_pdf_pages(fixture_pdf, "FIXTURE_2023_10K"):
        assert "\n" not in page.text
        assert "  " not in page.text
        assert page.text == page.text.strip()


def test_blank_page_is_kept_so_numbering_stays_physical(
    fixture_pdf_factory: Callable[..., Path],
) -> None:
    path = fixture_pdf_factory(pages=["first page", "", "third page"], name="BLANK_2022_10K")
    pages = extract_pdf_pages(path, "BLANK_2022_10K")
    assert [(p.page_num, p.text) for p in pages] == [
        (1, "first page"),
        (2, ""),
        (3, "third page"),
    ]


def test_page_num_matches_pdfium_index_plus_one(
    fixture_pdf_factory: Callable[..., Path],
) -> None:
    texts = [f"marker page {i}" for i in range(1, 8)]
    path = fixture_pdf_factory(pages=texts, name="SEVEN_2022_10K")
    pages = extract_pdf_pages(path, "SEVEN_2022_10K")
    assert len(pages) == 7
    for index, page in enumerate(pages):
        assert page.page_num == index + 1
        assert page.text == f"marker page {index + 1}"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        extract_pdf_pages(tmp_path / "nope.pdf", "NOPE_2022_10K")


def test_non_pdf_bytes_raise_value_error(tmp_path: Path) -> None:
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"<html><body>404 Not Found</body></html>")
    with pytest.raises(ValueError, match="%PDF"):
        extract_pdf_pages(fake, "FAKE_2022_10K")


def test_is_pdf_bytes() -> None:
    assert is_pdf_bytes(b"%PDF-1.7\n")
    assert not is_pdf_bytes(b"<!DOCTYPE html>")
    assert not is_pdf_bytes(b"")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Manage-\r\nment discussion", "Management discussion"),  # lowercase continuation
        ("Coca-\nCola Company", "Coca-Cola Company"),  # capitalised continuation keeps hyphen
        ("Form 10-K for 2022", "Form 10-K for 2022"),  # inline hyphen untouched
        ("ﬁnancial ﬂows", "financial flows"),  # NFKC ligatures
        ("a\x02b \x0cc", "ab c"),  # control chars dropped, form feed is a break
        ("  lots   of \t whitespace \r\n here ", "lots of whitespace here"),
        ("", ""),
    ],
)
def test_normalize_extracted_text(raw: str, expected: str) -> None:
    assert normalize_extracted_text(raw) == expected


def test_normalize_keep_newlines_collapses_blank_lines() -> None:
    raw = "first  line\n\n\n  second line \r\nthird"
    assert normalize_extracted_text(raw, keep_newlines=True) == "first line\nsecond line\nthird"
