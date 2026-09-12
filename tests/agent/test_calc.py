"""safe_calculate: the AST whitelist, resource bounds and arithmetic results."""

from __future__ import annotations

import math
import time

import pytest

from secqa.agent import safe_calculate
from secqa.agent.calc import MAX_EXPONENT, MAX_EXPRESSION_CHARS, MAX_ROUND_DIGITS
from secqa.core.errors import CalcRejected

# ---- accepted arithmetic ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", 3.0),
        ("1.2e9 / 3", 4e8),
        ("(1577 - 1408) / 1408 * 100", 12.00284090909091),
        ("1577000000 / 1000000", 1577.0),
        ("-5 + 2", -3.0),
        ("--5", 5.0),
        ("2 ** 10", 1024.0),
        ("7 % 3", 1.0),
        ("1_577_000_000 * 0.12", 189240000.0),
        ("abs(-3.5)", 3.5),
        ("round(2.5678, 2)", 2.57),
        ("round(2.5678, 2.0)", 2.57),
        ("min(3, 1, 2)", 1.0),
        ("max(3, 1, 2)", 3.0),
        ("((2 + 3) * (4 - 1)) / 5", 3.0),
        ("  42  ", 42.0),
    ],
)
def test_accepts_whitelisted_arithmetic(expression: str, expected: float) -> None:
    result = safe_calculate(expression)
    assert isinstance(result, float)
    assert math.isclose(result, expected, rel_tol=1e-12)


def test_percentage_change_matches_hand_computation() -> None:
    growth = safe_calculate("(1577000000 - 1408000000) / 1408000000")
    assert math.isclose(growth, 169 / 1408)
    assert math.isclose(safe_calculate("round((1577 - 1408) / 1408 * 100, 1)"), 12.0)


# ---- rejected constructs ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("x + 1", "name 'x'"),
        ("__import__('os')", "function __import__()"),
        ("__import__('os').system('ls')", "attribute access .system"),
        ("(1).__class__", "attribute"),
        ("(1).real", "attribute"),
        ("[1, 2][0]", "subscripting"),
        ("[1, 2]", "collection literal"),
        ("(1, 2)", "collection literal"),
        ("1 < 2", "comparison"),
        ("1 and 2", "boolean logic"),
        ("lambda: 1", "lambda"),
        ("'abc'", "not a number"),
        ("True + 1", "not a number"),
        ("1 // 2", "FloorDiv"),
        ("1 << 2", "LShift"),
        ("~1", "unary operator"),
        ("+1", "unary operator"),
        ("pow(2, 3)", "function pow()"),
        ("sum(1, 2)", "function sum()"),
        ("eval('1')", "function eval()"),
        ("round(1.5, ndigits=1)", "keyword"),
        ("abs(1, 2)", "between 1 and 1"),
        ("min()", "between 1 and 64"),
        ("f'{1}'", "f-string"),
        ("1 +", "not a valid arithmetic expression"),
        ("$1,577", "not a valid arithmetic expression"),
        ("", "empty"),
        ("   ", "empty"),
    ],
)
def test_rejects_everything_outside_the_whitelist(expression: str, reason: str) -> None:
    with pytest.raises(CalcRejected) as excinfo:
        safe_calculate(expression)
    assert reason in excinfo.value.reason


def test_rejects_non_string_input() -> None:
    with pytest.raises(CalcRejected, match="empty"):
        safe_calculate(None)  # type: ignore[arg-type]


# ---- resource bounds and numeric edge cases -----------------------------------------------


def test_rejects_division_by_zero() -> None:
    with pytest.raises(CalcRejected, match="division by zero"):
        safe_calculate("1 / 0")
    with pytest.raises(CalcRejected, match="division by zero"):
        safe_calculate("1 % 0")
    with pytest.raises(CalcRejected, match="division by zero"):
        safe_calculate("1 / (2 - 2)")


def test_rejects_huge_exponents_and_overflow_quickly() -> None:
    with pytest.raises(CalcRejected, match="exponent"):
        safe_calculate(f"2 ** {MAX_EXPONENT + 1}")
    with pytest.raises(CalcRejected, match="exponent"):
        safe_calculate("9 ** 9 ** 9")
    with pytest.raises(CalcRejected, match="overflow"):
        safe_calculate("(10 ** 300) ** 300")
    with pytest.raises(CalcRejected, match="not a finite number"):
        safe_calculate("1e308 * 10")
    with pytest.raises(CalcRejected, match="overflow"):
        safe_calculate("10.0 ** 400")


def test_rejects_complex_and_non_finite_results() -> None:
    with pytest.raises(CalcRejected, match="not a real number"):
        safe_calculate("(-8) ** 0.5")
    with pytest.raises(CalcRejected):
        safe_calculate("0.0 ** -1")


def test_rejects_overlong_expressions() -> None:
    expression = "1+" * (MAX_EXPRESSION_CHARS // 2) + "1"
    assert len(expression) > MAX_EXPRESSION_CHARS
    with pytest.raises(CalcRejected, match="characters long"):
        safe_calculate(expression)


def test_round_with_fractional_digits_is_rejected() -> None:
    with pytest.raises(CalcRejected, match="digits must be an integer"):
        safe_calculate("round(2.5, 1.5)")


@pytest.mark.parametrize(
    "expression",
    [
        "round(5, -10**9)",  # CPython builds 10**(10**9) for an int operand: hours of CPU
        "round(5, -1000000)",
        "round(5, 10**9)",
        "round(5, 1e300)",  # a float that is integral but far too large to become ndigits
        "round(5.0, -1e300)",
        f"round(5, -{MAX_ROUND_DIGITS + 1})",
    ],
)
def test_round_with_huge_digits_is_rejected_fast(expression: str) -> None:
    started = time.perf_counter()
    with pytest.raises(CalcRejected, match="exceed the limit"):
        safe_calculate(expression)
    assert time.perf_counter() - started < 0.5


def test_round_digits_at_the_limit_are_accepted() -> None:
    assert safe_calculate(f"round(5, -{MAX_ROUND_DIGITS})") == 0.0
    assert safe_calculate(f"round(5, {MAX_ROUND_DIGITS})") == 5.0
    assert safe_calculate(f"round(2.5678, {MAX_ROUND_DIGITS})") == 2.5678


def test_result_is_always_a_plain_float() -> None:
    assert type(safe_calculate("2 ** 3")) is float
    assert type(safe_calculate("7")) is float
