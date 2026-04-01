"""Basic utility tools: math, datetime, calculator."""
from __future__ import annotations

import ast
import operator
from datetime import datetime, timezone

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="Math expression to evaluate, e.g. '2 + 3 * 4'")


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


def _safe_eval(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _safe_eval(tree.body)
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Error evaluating '{expression}': {exc}"


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
            description="Evaluate a mathematical expression safely. Supports +, -, *, /, **, %.",
            args_schema=CalculatorInput,
        ),
    ]
