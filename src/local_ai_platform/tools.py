from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class MultiplyInput(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")


def multiply_numbers(a: float, b: float) -> str:
    return f"{a} * {b} = {a * b}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_default_tools() -> list[StructuredTool]:
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
            description="Get the current UTC time in ISO format.",
        ),
    ]
