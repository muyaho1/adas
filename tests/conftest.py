from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

try:
    import _pytest.tmpdir as _pytest_tmpdir
except Exception:  # pragma: no cover
    _pytest_tmpdir = None


if _pytest_tmpdir is not None:
    _original_cleanup_dead_symlinks = _pytest_tmpdir.cleanup_dead_symlinks

    def _safe_cleanup_dead_symlinks(root):
        try:
            _original_cleanup_dead_symlinks(root)
        except PermissionError:
            return

    _pytest_tmpdir.cleanup_dead_symlinks = _safe_cleanup_dead_symlinks


@pytest.fixture
def tmp_path():
    """Workspace-local tmp path fixture to avoid sandbox-denied pytest basetemp scans."""
    root = Path.cwd() / ".pytest_runs" / "manual_tmp"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"tmp_{uuid.uuid4().hex}"
    case_dir.mkdir(parents=True, exist_ok=False)
    try:
        yield case_dir
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)
