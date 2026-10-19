#!/usr/bin/env python3
"""Inspect a .mat file for Spacekime compatibility without printing raw voxel data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spacekime_loader import (
    bigim_metadata,
    inspect_mat_keys,
    inspect_mat_variables,
    load_bigim,
)
from spacekime_paths import resolve_mat_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check that a MATLAB file contains a valid 4D complex bigim array."
    )
    parser.add_argument(
        "mat_file",
        nargs="?",
        default=None,
        help="Path to .mat file (default: $SPACEKIME_MAT_FILE, data/, or project root)",
    )
    parser.add_argument(
        "--keys-only",
        action="store_true",
        help="Only list variable names in the .mat file; do not load bigim.",
    )
    args = parser.parse_args()

    try:
        mat_path = resolve_mat_path(args.mat_file)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"MAT file: {mat_path}")
    print(f"File size: {mat_path.stat().st_size / (1024 * 1024):.2f} MiB")

    try:
        keys = inspect_mat_keys(mat_path)
        print(f"MAT keys: {keys}")
        for item in inspect_mat_variables(mat_path):
            print(
                "  "
                f"{item['name']}: shape={item['shape']}, "
                f"matlab_class={item['matlab_class']}"
            )
    except Exception as exc:
        print(f"ERROR reading MAT keys: {exc}", file=sys.stderr)
        return 1

    if args.keys_only:
        return 0

    if "bigim" not in keys:
        print("ERROR: 'bigim' variable missing.", file=sys.stderr)
        return 1

    try:
        bigim = load_bigim(mat_path)
        meta = bigim_metadata(bigim)
    except Exception as exc:
        print(f"ERROR loading bigim: {exc}", file=sys.stderr)
        return 1

    print("bigim OK")
    for key, value in meta.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
