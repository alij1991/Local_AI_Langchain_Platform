"""Sandbox-containment tests for tools/file_ops._safe_path.

Regression coverage for [IMPROVE-23]: the old implementation used
str(resolved).startswith(str(WORKSPACE_ROOT)), which accepted sibling
directories that happened to share a prefix (workspace vs workspace_other).
The fix swaps to Path.relative_to(), which enforces actual ancestry.

Each test instantiates a fresh WORKSPACE_ROOT under tmp_path and reloads
the module so the module-level constant picks up the override. Reloading
(rather than monkeypatching the constant after import) also exercises the
resolve() codepath that runs at import time.
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pytest


def _load_with_workspace(tmp_root: Path):
    """Import tools.file_ops with LOCAL_AI_WORKSPACE pointing at tmp_root.

    Post-IMPROVE-69 tools/file_ops resolves WORKSPACE_ROOT via
    ``AppSettings.local_ai_workspace`` instead of ``os.getenv``. The
    AppSettings singleton is process-scoped, so a stale cache from an
    earlier test would override our fresh env var — explicitly reset
    it here so the module reload below sees the value we just set.
    """
    os.environ["LOCAL_AI_WORKSPACE"] = str(tmp_root)
    # Invalidate the cached AppSettings so the module-level
    # ``get_settings().local_ai_workspace`` read picks up our env.
    from local_ai_platform.config import reset_settings_cache
    reset_settings_cache()
    mod_name = "local_ai_platform.tools.file_ops"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


@pytest.fixture
def sandbox(tmp_path: Path):
    """Give each test its own workspace + a sibling dir to attack from."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    # Sibling that shares the 'workspace' prefix — this is the attack surface
    # that startswith() missed.
    (tmp_path / "workspace_other").mkdir()
    mod = _load_with_workspace(workspace)
    yield mod, workspace, tmp_path
    # Clean up so later tests don't inherit the override
    os.environ.pop("LOCAL_AI_WORKSPACE", None)


def test_direct_path_inside_workspace_allowed(sandbox):
    mod, workspace, _ = sandbox
    resolved = mod._safe_path("subdir/file.txt")
    # Resolved path must be inside WORKSPACE_ROOT
    assert resolved.is_relative_to(workspace)
    assert resolved == workspace / "subdir" / "file.txt"


def test_workspace_root_itself_allowed(sandbox):
    mod, workspace, _ = sandbox
    # "." should resolve to the workspace root and pass
    assert mod._safe_path(".") == workspace


def test_nested_traversal_that_lands_inside_is_allowed(sandbox):
    mod, workspace, _ = sandbox
    # subdir/../file.txt resolves to workspace/file.txt — still inside
    resolved = mod._safe_path("subdir/../file.txt")
    assert resolved == workspace / "file.txt"


def test_parent_traversal_blocked(sandbox):
    mod, _, _ = sandbox
    with pytest.raises(ValueError, match="outside the workspace"):
        mod._safe_path("../../../etc/passwd")


def test_absolute_path_blocked(sandbox):
    mod, _, _ = sandbox
    # Use an absolute path that definitely isn't inside the tmp workspace.
    # On POSIX this is /etc/passwd; on Windows we use a drive-root path.
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
    # Ensure the function returns the fully resolved Path, not the raw input.
    resolved = mod._safe_path("a/b/../c/./d.txt")
    assert resolved == workspace / "a" / "c" / "d.txt"
    assert resolved.is_absolute()
