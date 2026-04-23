"""Sandbox-containment tests for tools/rag_tools._safe_path.

Regression coverage for the [IMPROVE-23] follow-up: rag_tools carried a
duplicate _safe_path with the same str(resolved).startswith(...) bug that
file_ops had. Mirrors tests/test_file_ops_security.py — keep them in sync
when either helper changes.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


def _load_with_workspace(tmp_root: Path):
    """Import tools.rag_tools with LOCAL_AI_WORKSPACE pointing at tmp_root."""
    os.environ["LOCAL_AI_WORKSPACE"] = str(tmp_root)
    mod_name = "local_ai_platform.tools.rag_tools"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


@pytest.fixture
def sandbox(tmp_path: Path):
    """Give each test its own workspace + a sibling dir to attack from."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Sibling that shares the 'workspace' prefix — the attack surface
    # that startswith() missed.
    (tmp_path / "workspace_other").mkdir()
    mod = _load_with_workspace(workspace)
    yield mod, workspace, tmp_path
    os.environ.pop("LOCAL_AI_WORKSPACE", None)


def test_direct_path_inside_workspace_allowed(sandbox):
    mod, workspace, _ = sandbox
    resolved = mod._safe_path("subdir/file.txt")
    assert resolved.is_relative_to(workspace)
    assert resolved == workspace / "subdir" / "file.txt"


def test_workspace_root_itself_allowed(sandbox):
    mod, workspace, _ = sandbox
    assert mod._safe_path(".") == workspace


def test_nested_traversal_that_lands_inside_is_allowed(sandbox):
    mod, workspace, _ = sandbox
    resolved = mod._safe_path("subdir/../file.txt")
    assert resolved == workspace / "file.txt"


def test_parent_traversal_blocked(sandbox):
    mod, _, _ = sandbox
    with pytest.raises(ValueError, match="outside the workspace"):
        mod._safe_path("../../../etc/passwd")


def test_absolute_path_blocked(sandbox):
    mod, _, _ = sandbox
    outside = "C:/Windows/System32/drivers/etc/hosts" if os.name == "nt" else "/etc/passwd"
    with pytest.raises(ValueError, match="outside the workspace"):
        mod._safe_path(outside)


def test_sibling_directory_with_shared_prefix_blocked(sandbox):
    """Regression: startswith() let /tmp/workspace_other/file slip through
    because it shares the /tmp/workspace prefix. relative_to() rejects it."""
    mod, _, _ = sandbox
    with pytest.raises(ValueError, match="outside the workspace"):
        mod._safe_path("../workspace_other/secret.txt")


def test_resolved_path_is_returned_on_success(sandbox):
    mod, workspace, _ = sandbox
    resolved = mod._safe_path("a/b/../c/./d.txt")
    assert resolved == workspace / "a" / "c" / "d.txt"
    assert resolved.is_absolute()
