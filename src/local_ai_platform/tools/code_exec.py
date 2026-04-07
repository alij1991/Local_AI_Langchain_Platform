"""Sandboxed code execution tools."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

WORKSPACE_ROOT = Path(os.getenv("LOCAL_AI_WORKSPACE", "./workspace")).resolve()
MAX_OUTPUT = 4000  # characters


class RunPythonInput(BaseModel):
    code: str = Field(..., description="Python code to execute")
    timeout: int = Field(30, description="Timeout in seconds (max 60)")


class RunShellInput(BaseModel):
    command: str = Field(..., description="Shell command to execute")
    timeout: int = Field(15, description="Timeout in seconds (max 30)")


def run_python(code: str, timeout: int = 30) -> str:
    """Execute Python code in a sandboxed subprocess."""
    timeout = min(timeout, 60)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", dir=str(WORKSPACE_ROOT),
            delete=False, encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
            cwd=str(WORKSPACE_ROOT),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" if output else "") + result.stderr
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"

        if not output.strip():
            output = "(no output)"

        return output[:MAX_OUTPUT]

    except subprocess.TimeoutExpired:
        return f"Execution timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing Python code: {e}"
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def run_shell(command: str, timeout: int = 15) -> str:
    """Execute a shell command in a sandboxed subprocess."""
    timeout = min(timeout, 30)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            command, shell=True,
            capture_output=True, text=True, timeout=timeout,
            cwd=str(WORKSPACE_ROOT),
        )

        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" if output else "") + result.stderr
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"

        if not output.strip():
            output = "(no output)"

        return output[:MAX_OUTPUT]

    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {e}"


def get_code_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=run_python,
            name="run_python",
            description="Execute Python code in a sandboxed subprocess. Returns stdout/stderr. Use for calculations, data processing, or testing code.",
            args_schema=RunPythonInput,
            metadata={"dangerous": True},
        ),
        StructuredTool.from_function(
            func=run_shell,
            name="run_shell",
            description="Execute a shell command. Returns stdout/stderr. Use for system operations, file manipulation, or running programs.",
            args_schema=RunShellInput,
            metadata={"dangerous": True},
        ),
    ]
