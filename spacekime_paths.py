"""Shared path resolution helpers for local Spacekime scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MAT_NAME = "subj1_run1_complex_all.mat"
DEFAULT_MAT = PROJECT_ROOT / "data" / DEFAULT_MAT_NAME
ROOT_MAT = PROJECT_ROOT / DEFAULT_MAT_NAME


def portable_manifest_path(
    path: str | Path, *, root: str | Path = PROJECT_ROOT
) -> str:
    """Return a portable path for generated manifests without leaking host paths."""
    resolved = Path(path).expanduser().resolve()
    project_root = Path(root).expanduser().resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.name


def candidate_mat_paths(cli_path: str | Path | None = None) -> List[Path]:
    """Return MAT-file candidates in the order users most likely intended."""
    candidates: list[Path] = []

    if cli_path:
        candidates.append(Path(cli_path).expanduser())

    env_path = os.environ.get("SPACEKIME_MAT_FILE")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates.extend([DEFAULT_MAT, ROOT_MAT])

    for folder in (PROJECT_ROOT / "data", PROJECT_ROOT):
        if folder.is_dir():
            candidates.extend(sorted(folder.glob("*.mat")))

    return _dedupe_paths(candidates)


def resolve_mat_path(cli_path: str | Path | None = None) -> Path:
    """Resolve a MAT path from CLI, env var, data/, or the project root."""
    candidates = candidate_mat_paths(cli_path)
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_file():
            return resolved

    searched = "\n".join(f"  - {path.resolve()}" for path in candidates)
    raise FileNotFoundError(
        "MAT file not found. Pass a path, set SPACEKIME_MAT_FILE, or place a "
        f".mat file under data/.\nSearched:\n{searched}"
    )


def _dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = str(path.expanduser().resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out
