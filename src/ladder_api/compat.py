"""Compatibility helpers for importing legacy codebase modules."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional



def _default_codebase_path() -> Path:
    """Resolve the default src/codebase path relative to this file."""
    # compat.py -> ladder_api -> src -> repo root
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "src" / "codebase").resolve()


def ensure_codebase_on_path(codebase_path: Optional[Path] = None) -> Path:
    """Ensure src/codebase is available on sys.path for legacy absolute imports.

    Args:
        codebase_path: Optional explicit path to the codebase directory.

    Returns:
        The resolved codebase path that was added or already present.
    """
    cb_path = (codebase_path or _default_codebase_path()).resolve()
    if not cb_path.exists():
        raise FileNotFoundError(f"Codebase path not found: {cb_path}")

    # Avoid duplicates while preserving existing ordering.
    cb_str = str(cb_path)
    if cb_str not in sys.path:
        sys.path.insert(0, cb_str)
    return cb_path
