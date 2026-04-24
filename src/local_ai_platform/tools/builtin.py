"""Basic utility tools: math, datetime, calculator."""
from __future__ import annotations

import ast
import math
import operator
from datetime import datetime, timezone
from typing import Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..observability import emit


class MultiplyInput(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")


class CalculatorInput(BaseModel):
    expression: str = Field(
        ...,
        description=(
            "Math expression to evaluate, e.g. '2 + 3 * 4' or "
            "'sqrt(16) + sin(pi/2)'"
        ),
    )


def multiply_numbers(a: float, b: float) -> str:
    return f"{a} * {b} = {a * b}"


def utc_now() -> str:
    """Get the current date and time in UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

# Whitelisted named constants — the only ast.Name values the evaluator
# will resolve. Hand-rolled rather than pulling simpleeval/asteval to keep
# the repo dep-free, per ch 4 §4.14:
#   - simpleeval: https://github.com/danthedeckie/simpleeval (MIT)
#   - asteval:    https://pypi.org/project/asteval/
_SAFE_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

# Whitelisted callables — the only functions an ast.Call node is allowed
# to target. Each must take numeric args and return a number. Anything that
# touches attributes, env, or filesystem is prohibited by construction:
# this dict is the only escape hatch out of pure-AST arithmetic.
_SAFE_FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    # Builtin pow (not math.pow) so integer-only inputs stay int-typed.
    "pow": pow,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "min": min,
    "max": max,
}


class _UnsafeExpressionError(ValueError):
    """The AST walker rejected a disallowed construct.

    Split from plain ValueError so math-domain errors raised by the
    whitelisted functions themselves (e.g. ``math.sqrt(-1)`` → ValueError)
    don't pollute the ``UnsafeExpression`` observability signal used for
    weekly jailbreak-attempt review.
    """


def _safe_eval(node: ast.expr) -> int | float:
    # Numeric literals. Explicitly reject bool — it's an int subclass in
    # Python, so ``True`` would otherwise slip through as the number 1.
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            raise _UnsafeExpressionError("Boolean literals are not allowed")
        if isinstance(node.value, (int, float)):
            return node.value
        raise _UnsafeExpressionError(
            f"Only numeric literals are allowed (got {type(node.value).__name__})"
        )
    # Named constants: pi, e, tau.
    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise _UnsafeExpressionError(f"Unknown name: {node.id!r}")
    # Binary arithmetic: + - * / % **
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    # Unary minus.
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    # Function call: bare-name, whitelisted, positional-only.
    if isinstance(node, ast.Call):
        # Attribute access (``math.sqrt(4)``) is rejected here — func must
        # be a bare ast.Name, not an ast.Attribute / Subscript / Lambda.
        if not isinstance(node.func, ast.Name):
            raise _UnsafeExpressionError(
                "Function calls must use a bare name (no attribute access)"
            )
        if node.func.id not in _SAFE_FUNCS:
            raise _UnsafeExpressionError(
                f"Function {node.func.id!r} is not allowed"
            )
        if node.keywords:
            raise _UnsafeExpressionError(
                "Keyword arguments are not allowed in calculator calls"
            )
        args = [_safe_eval(a) for a in node.args]
        return _SAFE_FUNCS[node.func.id](*args)
    raise _UnsafeExpressionError(
        f"Unsupported expression node: {type(node).__name__}"
    )


def calculator(expression: str) -> str:
    """Safely evaluate a math expression.

    Supports:
      - arithmetic:         + - * / % ** and unary -
      - named constants:    pi, e, tau
      - functions:          sqrt, sin, cos, tan, log, log10, exp, abs,
                            pow, floor, ceil, round, min, max

    Rejects attribute access, imports, arbitrary names, lambdas, keyword
    arguments, and any function not on the whitelist. Parsing uses
    ``ast.parse`` with a hand-rolled AST walker — no ``eval`` / ``exec``.
    """
    expr = expression.strip()
    try:
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
    except _UnsafeExpressionError as exc:
        # Whitelist violation — either a feature miss or a jailbreak
        # attempt. The UnsafeExpression error_code powers the weekly
        # review query without being diluted by runtime math errors.
        emit(
            "tool", "calculator_eval", status="error",
            error_code="UnsafeExpression",
            error_message=str(exc)[:200],
            context={"expression_length": len(expr)},
        )
        return f"Error evaluating '{expression}': {exc}"
    except SyntaxError as exc:
        emit(
            "tool", "calculator_eval", status="error",
            error_code="SyntaxError",
            error_message=str(exc)[:200],
            context={"expression_length": len(expr)},
        )
        return f"Error evaluating '{expression}': {exc}"
    except Exception as exc:
        # Math-domain errors, ZeroDivisionError, OverflowError, TypeError
        # from calling sqrt() with no args, etc. These are user-input
        # errors, NOT jailbreak signals — keep them on their own code.
        emit(
            "tool", "calculator_eval", status="error",
            error_code="EvalError",
            error_message=str(exc)[:200],
            context={"expression_length": len(expr)},
        )
        return f"Error evaluating '{expression}': {exc}"

    emit(
        "tool", "calculator_eval", status="ok",
        context={"expression_length": len(expr)},
        perf={"result_type": type(result).__name__},
    )
    return f"{expression} = {result}"


def get_builtin_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=multiply_numbers,
            name="multiply_numbers",
            description="Multiply two numbers together.",
            args_schema=MultiplyInput,
        ),
        StructuredTool.from_function(
            func=utc_now,
            name="utc_now",
            description="Get the current date and time in UTC.",
        ),
        StructuredTool.from_function(
            func=calculator,
            name="calculator",
            description=(
                "Evaluate a mathematical expression safely. Supports "
                "arithmetic (+ - * / % **), named constants (pi, e, tau), "
                "and math functions (sqrt, sin, cos, tan, log, log10, exp, "
                "abs, pow, floor, ceil, round, min, max)."
            ),
            args_schema=CalculatorInput,
        ),
    ]
