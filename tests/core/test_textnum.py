"""Table tests for secqa.core.textnum."""

from __future__ import annotations

import math

import pytest

from secqa.core.textnum import extract_numbers, normalize_text, numbers_equal, parse_number

PARSE_CASES: list[tuple[str, float | None]] = [
    ("$1.2 billion", 1.2e9),
    ("1,200 million", 1.2e9),
    ("$1.2B", 1.2e9),
    ("US$4bn", 4e9),
    ("USD 2.3 million", 2.3e6),
    ("$5M", 5e6),
    ("2 thousand", 2000.0),
    ("3.4 trillion", 3.4e12),
    ("(1,577)", -1577.0),
    ("(45)", -45.0),
    ("-45", -45.0),
    ("−12.5%", -0.125),  # unicode minus
    ("12%", 0.12),
    ("5 percent", 0.05),
    ("18 pct", 0.18),
    ("3 bps", 0.0003),
    ("4 basis points", 0.0004),
    ("1.5x", 1.5),
    ("1577.00", 1577.0),
    ("$ 1,234.56", 1234.56),
    (".5", 0.5),
    ("0.12", 0.12),
    ("  7  ", 7.0),
    ("$0", 0.0),
    ("(0)", 0.0),
    ("abc", None),
    ("", None),
    ("1.2.3", None),
    ("10-K", None),
    ("FY2022", None),
    ("5th", None),
    ("Revenue was $1.2 billion in 2022", None),  # two numbers -> not a single literal
    ("12 percentage points", None),
]


@pytest.mark.parametrize(("text", "expected"), PARSE_CASES)
def test_parse_number_table(text: str, expected: float | None) -> None:
    got = parse_number(text)
    if expected is None:
        assert got is None
    else:
        assert got is not None
        assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-15)


def test_parse_number_none_input() -> None:
    assert parse_number(None) is None  # type: ignore[arg-type]


def test_parse_number_exact_fraction() -> None:
    assert parse_number("3 bps") == 0.0003
    assert parse_number("12%") == 0.12


def test_extract_numbers_sentence() -> None:
    text = (
        "Revenue was $1.2 billion in FY2022, up 12% versus the 10-K filed on 5th March; "
        "spreads widened 3 bps, the loss was (1,577) and over 5 months Q3 2023 saw 1.5x growth "
        "of $5M."
    )
    assert extract_numbers(text) == [1.2e9, 0.12, 0.0003, -1577.0, 5.0, 2023.0, 1.5, 5e6]


def test_extract_numbers_skips_identifiers() -> None:
    # digits glued to letters (before or after) are identifiers, not quantities
    assert extract_numbers("Form 10-K, Form 10-Q, FY2022, Q3, 5th, ASC 842A") == []
    assert extract_numbers("ASC 842 and Note 12 apply") == [842.0, 12.0]
    assert extract_numbers("") == []
    assert extract_numbers("no digits here") == []


def test_extract_numbers_plural_scale_words() -> None:
    assert extract_numbers("in millions: 12 millions and 3 billions") == [12e6, 3e9]


def test_extract_numbers_scale_needs_boundary() -> None:
    # 'm' in 'months' is not a scale suffix; '3km' is a unit-glued token and is skipped entirely.
    assert extract_numbers("5 months and 3km and 2 kilos") == [5.0, 2.0]
    assert extract_numbers("$5 m and 5 M and 5mm") == [5e6, 5e6, 5e6]


EQUAL_CASES: list[tuple[float, float, bool]] = [
    (1.2e9, 1200e6, True),  # '$1.2B' vs '1,200 million'
    (0.12, 12.0, True),  # '12%' vs '0.12' (ratio/percent equivalence)
    (12.0, 0.12, True),
    (-45.0, -45.0, True),  # '(45)' vs '-45'
    (1.5, 1.6, False),
    (100.0, 100.9, True),  # within 1 %
    (100.0, 101.5, False),  # outside 1 %
    (0.0, 0.0, True),
    (0.0, 1e-9, False),
    (1577.0, 1577.004, True),
    (-1577.0, 1577.0, False),
    (0.0003, 0.03, True),  # bps written as a percent
]


@pytest.mark.parametrize(("a", "b", "expected"), EQUAL_CASES)
def test_numbers_equal_table(a: float, b: float, expected: bool) -> None:
    assert numbers_equal(a, b) is expected


def test_numbers_equal_custom_tolerance_and_nan() -> None:
    assert numbers_equal(100.0, 104.0, rel_tol=0.05) is True
    assert numbers_equal(100.0, 104.0, rel_tol=0.01) is False
    assert numbers_equal(float("nan"), 1.0) is False
    assert numbers_equal(1.0, None) is False  # type: ignore[arg-type]


def test_parse_then_equal_round_trip() -> None:
    a = parse_number("$1.2 billion")
    b = parse_number("1,200 million")
    assert a is not None and b is not None
    assert numbers_equal(a, b)


def test_normalize_text() -> None:
    assert normalize_text("  Net  Sales—“Total”\nwere $1,577.00 (2022). ﬁne") == (
        "net sales total were 1 577 00 2022 fine"
    )
    assert normalize_text("") == ""
    assert normalize_text("ABC") == normalize_text("abc")
    assert normalize_text("Straße") == "strasse"
