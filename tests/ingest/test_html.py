"""Tests for secqa.ingest.html on tests/fixtures/mini_10k.html (synthetic)."""

from __future__ import annotations

import warnings

import pytest

from secqa.ingest import extract_html_pages
from secqa.ingest.html import PAGE_BREAK, flatten_table, html_to_blocks

DOC = "FIXTURE_2023_10K"
NOISE = (
    "SCRIPT_TEXT_MUST_NOT_APPEAR",
    "HIDDEN_HEADER_MUST_NOT_APPEAR",
    "HIDDEN_ROW_MUST_NOT_APPEAR",
    "a comment that must not appear",
    "padding",
    "fixture-2023-10k.htm",  # <title>
)


def test_page_breaks_produce_one_based_pseudo_pages(mini_10k_html: bytes) -> None:
    pages = extract_html_pages(mini_10k_html, DOC)
    assert [p.page_num for p in pages] == [1, 2, 3]
    assert all(p.doc_name == DOC for p in pages)
    assert "FORM 10-K" in pages[0].text
    assert pages[1].text.startswith("Item 1. Business")
    assert pages[2].text.startswith("Item 7. Management's Discussion and Analysis")


def test_tables_are_flattened_row_wise(mini_10k_html: bytes) -> None:
    text = extract_html_pages(mini_10k_html, DOC)[2].text
    assert "2023 | 2022" in text
    assert "Net sales | $1,577 | $1,408" in text
    assert "Operating income | $245 | $201" in text
    assert "Net loss on disposal | $(12) | —" in text  # ')' cell glued back onto its number


def test_noise_is_removed(mini_10k_html: bytes) -> None:
    full = "\n".join(p.text for p in extract_html_pages(mini_10k_html, DOC))
    for needle in NOISE:
        assert needle not in full


def test_inline_markup_nbsp_and_br_are_normalised(mini_10k_html: bytes) -> None:
    full = "\n".join(p.text for p in extract_html_pages(mini_10k_html, DOC))
    assert "Fixture Corp annual report for the fiscal year ended December 31, 2023." in full
    assert "Cash and cash equivalents were $410 million at year end. Long-term debt" in full
    assert "\xa0" not in full


def test_target_chars_bounds_pages_and_preserves_order(mini_10k_html: bytes) -> None:
    blocks = [b for b in html_to_blocks(mini_10k_html) if b is not PAGE_BREAK]
    pages = extract_html_pages(mini_10k_html, DOC, target_chars=120)
    assert len(pages) > 3
    for page in pages:
        # a page is either within budget or a single block that could not be split
        assert len(page.text) <= 120 or "\n" not in page.text
    reassembled = [line for page in pages for line in page.text.split("\n")]
    assert reassembled == blocks


def test_page_break_before_variants_split_pages() -> None:
    html = (
        b"<html><body><p>one</p><div style='break-before: page'></div><p>two</p>"
        b"<p style='PAGE-BREAK-BEFORE:always'>three</p></body></html>"
    )
    pages = extract_html_pages(html, DOC)
    assert [p.text for p in pages] == ["one", "two", "three"]


def test_empty_and_whitespace_only_documents_yield_no_pages() -> None:
    assert extract_html_pages(b"", DOC) == []
    assert extract_html_pages(b"<html><body>   \n </body></html>", DOC) == []
    only_break = b"<html><body><p></p><hr style='page-break-after:always'/></body></html>"
    assert extract_html_pages(only_break, DOC) == []


def test_invalid_target_chars_rejected(mini_10k_html: bytes) -> None:
    with pytest.raises(ValueError, match="target_chars"):
        extract_html_pages(mini_10k_html, DOC, target_chars=0)


def test_xml_prolog_does_not_warn(mini_10k_html: bytes) -> None:
    assert mini_10k_html.startswith(b"<?xml")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        extract_html_pages(mini_10k_html, DOC)


def test_flatten_table_merges_marker_cells() -> None:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(
        "<table><tr><th>Metric</th><th>FY23</th></tr>"
        "<tr><td>Margin</td><td>12.5</td><td>%</td></tr>"
        "<tr><td>Revenue</td><td>$</td><td>1,577</td></tr>"
        "<tr><td></td><td>   </td></tr></table>",
        "lxml",
    )
    table = soup.find("table")
    assert table is not None
    assert flatten_table(table) == ["Metric | FY23", "Margin | 12.5%", "Revenue | $1,577"]


def test_nested_inline_elements_stay_in_one_block() -> None:
    html = b"<div><span>Net <b>sales</b> were <i>$1,577</i> million.</span></div><p>Next.</p>"
    assert html_to_blocks(html) == ["Net sales were $1,577 million.", "Next."]
