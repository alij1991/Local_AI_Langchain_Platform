"""File operation tools: read, write, list, search files."""
from __future__ import annotations

import os
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..observability import emit

# Sandbox root — all file operations are restricted to this directory
WORKSPACE_ROOT = Path(os.getenv("LOCAL_AI_WORKSPACE", "./workspace")).resolve()


def _safe_path(user_path: str) -> Path:
    """Resolve a user-provided path and verify it's inside the workspace.

    Uses Path.relative_to() for strict containment. The previous
    str(resolved).startswith(str(WORKSPACE_ROOT)) check was vulnerable to
    sibling-prefix escape — e.g. WORKSPACE_ROOT=/home/a/workspace and
    resolved=/home/a/workspace_other/file satisfied startswith() but is
    plainly outside the sandbox. relative_to() raises ValueError whenever
    the resolved path is not an ancestor-descendant of WORKSPACE_ROOT,
    which is the actual containment semantics we want.
    """
    resolved = (WORKSPACE_ROOT / user_path).resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError:
        # Record every sandbox escape attempt so the weekly observability
        # review can flag repeated attacks or a misbehaving agent.
        emit(
            "tool",
            "file_ops.path_rejected",
            status="error",
            error_code="PathOutsideWorkspace",
            context={"user_path": user_path[:200]},
        )
        raise ValueError(f"Path '{user_path}' is outside the workspace directory")
    return resolved


class ReadFileInput(BaseModel):
    path: str = Field(..., description="File path relative to workspace")
    max_lines: int = Field(200, description="Maximum lines to read")


class WriteFileInput(BaseModel):
    path: str = Field(..., description="File path relative to workspace")
    content: str = Field(..., description="Content to write")


class ListDirectoryInput(BaseModel):
    path: str = Field(".", description="Directory path relative to workspace")
    pattern: str = Field("*", description="Glob pattern to filter entries")


class SearchFilesInput(BaseModel):
    pattern: str = Field(..., description="Glob pattern to search for, e.g. '**/*.py'")
    directory: str = Field(".", description="Directory to search in, relative to workspace")
    max_results: int = Field(20, description="Maximum files to return")


def read_file(path: str, max_lines: int = 200) -> str:
    """Read file contents from the workspace."""
    try:
        fp = _safe_path(path)
        if not fp.exists():
            return f"File not found: {path}"
        if not fp.is_file():
            return f"Not a file: {path}"
        lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
        if len(lines) > max_lines:
            return "\n".join(lines[:max_lines]) + f"\n\n... ({len(lines) - max_lines} more lines truncated)"
        return "\n".join(lines)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file in the workspace. Creates parent directories if needed."""
    try:
        fp = _safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error writing file: {e}"


def list_directory(path: str = ".", pattern: str = "*") -> str:
    """List files and directories in the workspace."""
    try:
        dp = _safe_path(path)
        if not dp.exists():
            return f"Directory not found: {path}"
        if not dp.is_dir():
            return f"Not a directory: {path}"
        entries = sorted(dp.glob(pattern))
        lines = []
        for e in entries[:100]:
            rel = e.relative_to(WORKSPACE_ROOT)
            suffix = "/" if e.is_dir() else f" ({e.stat().st_size} bytes)"
            lines.append(f"  {rel}{suffix}")
        header = f"Contents of {path}/ ({len(entries)} items):"
        if len(entries) > 100:
            lines.append(f"  ... and {len(entries) - 100} more")
        return header + "\n" + "\n".join(lines)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error listing directory: {e}"


def search_files(pattern: str, directory: str = ".", max_results: int = 20) -> str:
    """Search for files matching a glob pattern in the workspace."""
    try:
        dp = _safe_path(directory)
        if not dp.exists():
            return f"Directory not found: {directory}"
        matches = list(dp.glob(pattern))[:max_results]
        if not matches:
            return f"No files matching '{pattern}' in {directory}"
        lines = [str(m.relative_to(WORKSPACE_ROOT)) for m in matches]
        return f"Found {len(matches)} matches:\n" + "\n".join(lines)
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Error searching files: {e}"


def get_file_tools() -> list[StructuredTool]:
    # Ensure workspace exists
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)

    return [
        StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description="Read the contents of a file in the workspace.",
            args_schema=ReadFileInput,
        ),
        StructuredTool.from_function(
            func=write_file,
            name="write_file",
            description="Write content to a file in the workspace. Creates parent directories if needed.",
            args_schema=WriteFileInput,
        ),
        StructuredTool.from_function(
            func=list_directory,
            name="list_directory",
            description="List files and directories in the workspace.",
            args_schema=ListDirectoryInput,
        ),
        StructuredTool.from_function(
            func=search_files,
            name="search_files",
            description="Search for files matching a glob pattern in the workspace.",
            args_schema=SearchFilesInput,
        ),
    ]
