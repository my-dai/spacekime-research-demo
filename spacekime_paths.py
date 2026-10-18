"""Portable paths for the synthetic SpaceKime research demo."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def portable_manifest_path(
    path: str | Path, *, root: str | Path = PROJECT_ROOT
) -> str:
    """Return a project-relative path without exposing a local machine path."""
    resolved = Path(path).expanduser().resolve()
    project_root = Path(root).expanduser().resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.name
