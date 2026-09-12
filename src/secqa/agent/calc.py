"""``safe_calculate``: the agent's calculator, an AST whitelist instead of ``eval``.

The model sends arithmetic as text (``"(1577 - 1408) / 1408 * 100"``). Evaluating that with
``eval`` would hand the model a Python interpreter, so the expression is parsed with
:mod:`ast` and walked by hand; every node type that is not on the list below is rejected with
:class:`~secqa.core.errors.CalcRejected` and a reason the model can read.

Allowed: numeric literals (``1577``, ``1.2e9``, ``1_577_000``), the binary operators
``+ - * / ** %``, unary minus, parentheses, and calls to ``abs``, ``round``, ``min`` and ``max``.
Everything else (names, attributes, subscripts, comparisons, boolean logic, strings, tuples,
lambdas, ``//``, bit operations, other function names) is rejected.

Resource bounds: the expression is capped at :data:`MAX_EXPRESSION_CHARS`, exponents at
:data:`MAX_EXPONENT`, and every intermediate integer must fit in a ``float`` so that
``(10**300)**300`` cannot allocate a million-digit integer. The result is always a finite
``float``; division by zero, overflow, NaN and complex results are rejected.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from secqa.core.errors import CalcRejected
from secqa.core.logging import get_logger

log = get_logger(__name__)

MAX_EXPRESSION_CHARS = 500
"""Longest expression accepted; anything longer is not arithmetic a filing question needs."""

MAX_EXPONENT = 1000
"""Largest absolute exponent accepted by ``**`` (``2**1000`` is already ~1e301)."""

_MAX_INT_BITS = 1024  # an int wider than this cannot become a float (DBL_MAX ~ 2**1024)

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.USub: operator.neg,
}
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_FUNCTION_ARITY: dict[str, tuple[int, int]] = {
    "abs": (1, 1),
    "round": (1, 2),
    "min": (1, 64),
    "max": (1, 64),
}


def safe_calculate(expression: str) -> float:
    """Evaluate an arithmetic ``expression`` under the whitelist and return a finite float.

    Raises:
        CalcRejected: the expression is empty, too long, not valid Python syntax, uses a node
            or function outside the whitelist, divides by zero, overflows, or does not produce
            a finite real number.
    """
    if not isinstance(expression, str) or not expression.strip():
        raise CalcRejected("expression is empty")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalcRejected(
            f"expression is {len(expression)} characters long; the limit is {MAX_EXPRESSION_CHARS}"
        )
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalcRejected(
            f"not a valid arithmetic expression: {exc.msg}; write plain numbers without "
            "currency symbols or thousands separators (e.g. 1577000000 not $1,577,000,000)"
        ) from exc
    value = _evaluate(tree.body)
    try:
        result = float(value)
    except OverflowError as exc:
        raise CalcRejected("result is too large to represent") from exc
    if not math.isfinite(result):
        raise CalcRejected("result is not a finite number")
    log.debug("calculate", expression=expression.strip(), result=result)
    return result


def _evaluate(node: ast.expr) -> int | float:
    """Recursively evaluate one whitelisted node; every other node type is rejected."""
    if isinstance(node, ast.Constant):
        return _constant(node)
    if isinstance(node, ast.BinOp):
        return _binary(node)
    if isinstance(node, ast.UnaryOp):
        return _unary(node)
    if isinstance(node, ast.Call):
        return _call(node)
    raise CalcRejected(
        f"{_describe(node)} is not allowed; use numbers, + - * / ** % and abs/round/min/max"
    )


def _constant(node: ast.Constant) -> int | float:
    value = node.value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CalcRejected(f"literal {value!r} is not a number")
    return _bounded(value)


def _binary(node: ast.BinOp) -> int | float:
    apply = _BINARY_OPERATORS.get(type(node.op))
    if apply is None:
        raise CalcRejected(f"operator {type(node.op).__name__} is not allowed; use + - * / ** %")
    left = _evaluate(node.left)
    right = _evaluate(node.right)
    if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
        raise CalcRejected(f"exponent {right!r} exceeds the limit of {MAX_EXPONENT}")
    if isinstance(node.op, ast.Div | ast.Mod) and right == 0:
        raise CalcRejected("division by zero")
    try:
        result = apply(left, right)
    except OverflowError as exc:
        raise CalcRejected("intermediate result overflows") from exc
    except (ZeroDivisionError, ValueError) as exc:  # e.g. 0.0 ** -1
        raise CalcRejected(str(exc)) from exc
    return _bounded(result)


def _unary(node: ast.UnaryOp) -> int | float:
    apply = _UNARY_OPERATORS.get(type(node.op))
    if apply is None:
        raise CalcRejected(f"unary operator {type(node.op).__name__} is not allowed; only minus")
    return _bounded(apply(_evaluate(node.operand)))


def _call(node: ast.Call) -> int | float:
    if not isinstance(node.func, ast.Name):
        raise CalcRejected(f"calling {_describe(node.func)} is not allowed")
    name = node.func.id
    function = _FUNCTIONS.get(name)
    if function is None:
        raise CalcRejected(
            f"function {name}() is not allowed; only {', '.join(sorted(_FUNCTIONS))}"
        )
    if node.keywords:
        raise CalcRejected(f"{name}() does not accept keyword arguments")
    low, high = _FUNCTION_ARITY[name]
    if not low <= len(node.args) <= high:
        raise CalcRejected(f"{name}() takes between {low} and {high} arguments")
    arguments = [_evaluate(argument) for argument in node.args]
    if name == "round" and len(arguments) == 2:
        digits = arguments[1]
        if isinstance(digits, float) and not digits.is_integer():
            raise CalcRejected("round() digits must be an integer")
        arguments[1] = int(digits)
    return _bounded(function(*arguments))


def _bounded(value: Any) -> int | float:
    """Reject anything that is not a real number a float can hold (guards int blow-up)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CalcRejected(f"result {value!r} is not a real number")
    if isinstance(value, int) and value.bit_length() > _MAX_INT_BITS:
        raise CalcRejected("intermediate result overflows")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalcRejected("intermediate result is not a finite number")
    return value


def _describe(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return f"name {node.id!r}"
    if isinstance(node, ast.Attribute):
        return f"attribute access .{node.attr}"
    if isinstance(node, ast.Subscript):
        return "subscripting"
    if isinstance(node, ast.Compare):
        return "comparison"
    if isinstance(node, ast.BoolOp):
        return "boolean logic"
    if isinstance(node, ast.Lambda):
        return "lambda"
    if isinstance(node, ast.Tuple | ast.List | ast.Set | ast.Dict):
        return "a collection literal"
    if isinstance(node, ast.JoinedStr):
        return "an f-string"
    return type(node).__name__


__all__ = ["MAX_EXPONENT", "MAX_EXPRESSION_CHARS", "safe_calculate"]
