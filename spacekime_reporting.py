"""Reproducible JSON manifests for Spacekime experiment scripts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from spacekime_loader import inspect_mat_variables
from spacekime_paths import PROJECT_ROOT


def build_run_manifest(
    tool: str,
    mat_path: Path,
    parameters: Mapping[str, Any],
    results: Mapping[str, Any],
    notes: Optional[list[str]] = None,
    include_local_path: bool = False,
) -> dict[str, Any]:
    """Capture dataset identity, parameters, runtime versions, and results."""
    path = Path(mat_path).resolve()
    stat = path.stat()
    dataset = {
        "basename": path.name,
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "fingerprint": _file_fingerprint(path),
        "mat_variables": inspect_mat_variables(path),
    }
    if include_local_path:
        dataset["local_path"] = str(path)

    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "representation": "spatial_phase_distribution_proxy",
        "representation_note": (
            "H(t,theta) is a magnitude-weighted spatial complex-signal phase "
            "histogram, not a repeated-measurement or KPT-reconstructed kimesurface."
        ),
        "dataset": dataset,
        "parameters": dict(parameters),
        "results": dict(results),
        "runtime": _runtime(),
        "notes": notes or [],
    }


def build_synthetic_run_manifest(
    tool: str,
    parameters: Mapping[str, Any],
    results: Mapping[str, Any],
    notes: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Capture a ground-truth synthetic reconstruction experiment."""
    return {
        "schema_version": 2,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tool": tool,
        "representation": "latent_repeated_measurement_kimesurface",
        "representation_note": (
            "The latent surface is known by construction; repeated observations "
            "sample kime-phase independently at each longitudinal time point."
        ),
        "dataset": {"kind": "synthetic", "ground_truth_available": True},
        "parameters": dict(parameters),
        "results": dict(results),
        "runtime": _runtime(),
        "notes": notes or [],
    }


def write_run_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Write strict, human-readable JSON without NaN/Infinity tokens."""
    path.write_text(
        json.dumps(_json_ready(manifest), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _file_fingerprint(path: Path, chunk_bytes: int = 1_048_576) -> dict[str, Any]:
    """Return a fast reproducibility fingerprint without hashing a huge file in full."""
    size = path.stat().st_size
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        head = handle.read(chunk_bytes)
        digest.update(head)
        if size > chunk_bytes:
            handle.seek(max(0, size - chunk_bytes))
            digest.update(handle.read(chunk_bytes))
    digest.update(str(size).encode("ascii"))
    return {
        "algorithm": "sha256(head_1MiB+tail_1MiB+size)",
        "value": digest.hexdigest(),
    }


def _package_version(name: str) -> Optional[str]:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _runtime() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "scipy", "matplotlib", "scikit-image")
        },
        "git": _git_state(),
    }


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
