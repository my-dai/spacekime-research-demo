#!/usr/bin/env python3
"""Map observed complex-phase structure across fMRI slices."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - minimal environments
    plt = None

from spacekime_core import (
    build_phase_distribution_surface,
    complex_phase_metrics,
    kime_surface_summary,
    phase_scope_mask,
)
from spacekime_loader import load_bigim
from spacekime_paths import PROJECT_ROOT, resolve_mat_path
from spacekime_reporting import build_run_manifest, write_run_manifest


OUTPUT_DIR = PROJECT_ROOT / "output"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report where observed spatial/temporal phase structure occurs by z slice."
    )
    parser.add_argument("mat_file", nargs="?", default=None)
    parser.add_argument("--max-t", type=int, default=50)
    parser.add_argument("--mask-t", type=int, default=0)
    parser.add_argument("--mask-percentile", type=float, default=50.0)
    parser.add_argument("--n-theta", type=int, default=64)
    parser.add_argument("--scope", choices=("slice", "roi"), default="slice")
    parser.add_argument("--roi-half", type=int, default=8)
    parser.add_argument(
        "--phase-mode", choices=("raw", "relative"), default="raw"
    )
    parser.add_argument("--skip-volume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    try:
        mat_path = resolve_mat_path(args.mat_file)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading: {mat_path}")
    bigim = load_bigim(mat_path)
    _, _, Z, T = bigim.shape
    t_list = list(range(min(T, max(2, args.max_t))))
    mask_t = max(0, min(T - 1, args.mask_t))

    rows: list[dict[str, float | int | str]] = []
    if not args.skip_volume:
        print("Analyzing whole-volume reference...")
        rows.append(
            analyze_scope(
                bigim,
                t_list,
                mask_t=mask_t,
                percentile=args.mask_percentile,
                n_theta=args.n_theta,
                scope="volume",
                z_idx=None,
                roi_half=args.roi_half,
                phase_mode=args.phase_mode,
            )
        )

    for z_idx in range(Z):
        print(f"Analyzing {args.scope} z={z_idx}")
        rows.append(
            analyze_scope(
                bigim,
                t_list,
                mask_t=mask_t,
                percentile=args.mask_percentile,
                n_theta=args.n_theta,
                scope=args.scope,
                z_idx=z_idx,
                roi_half=args.roi_half,
                phase_mode=args.phase_mode,
            )
        )

    csv_path = args.output_dir / "local_phase_report.csv"
    summary_path = args.output_dir / "local_phase_report.txt"
    fig_path = args.output_dir / "local_phase_report.png"
    manifest_path = args.output_dir / "local_phase_report_manifest.json"
    local_summary = summarize_local_rows(rows)
    write_rows(csv_path, rows)
    write_summary(summary_path, local_summary)
    if plt is not None:
        save_figure(fig_path, rows, scope=args.scope)

    manifest = build_run_manifest(
        "run_local_phase_report.py",
        mat_path,
        parameters={
            "max_t": len(t_list),
            "mask_t": mask_t,
            "mask_percentile": args.mask_percentile,
            "n_theta": args.n_theta,
            "scope": args.scope,
            "roi_half": args.roi_half,
            "include_volume": not args.skip_volume,
            "phase_mode": args.phase_mode,
        },
        results={"summary": local_summary, "rows": rows},
        notes=[
            "Adjacent-frame metrics mix acquisition noise, drift, motion, physiology, and true signal change.",
            "They are observed temporal-change diagnostics, not isolated scanner-noise estimates.",
        ],
    )
    write_run_manifest(manifest_path, manifest)

    print(f"Wrote: {csv_path.resolve()}")
    print(f"Wrote: {summary_path.resolve()}")
    if plt is not None:
        print(f"Wrote: {fig_path.resolve()}")
    print(f"Wrote: {manifest_path.resolve()}")
    return 0


def analyze_scope(
    bigim: np.ndarray,
    t_list: list[int],
    mask_t: int,
    percentile: float,
    n_theta: int,
    scope: str,
    z_idx: int | None,
    roi_half: int,
    phase_mode: str,
) -> dict[str, float | int | str]:
    mask = phase_scope_mask(
        bigim,
        t_idx=mask_t,
        percentile=percentile,
        scope=scope,
        z_idx=z_idx,
        roi_half=roi_half,
    )
    stats = build_phase_distribution_surface(
        bigim,
        mask,
        t_list=t_list,
        n_theta=n_theta,
        normalize_h=True,
        phase_mode=phase_mode,
        reference_t=mask_t,
    )
    summary = kime_surface_summary(stats)
    mask_magnitude = np.abs(bigim[:, :, :, mask_t])[mask]

    adjacent = [
        complex_phase_metrics(
            bigim[:, :, :, previous],
            bigim[:, :, :, current],
            mask=mask,
        )
        for previous, current in zip(t_list[:-1], t_list[1:])
    ]
    return {
        "scope": scope,
        "z": -1 if z_idx is None else z_idx,
        "mask_voxels": int(mask.sum()),
        "time_points": len(t_list),
        "mean_mask_magnitude": float(np.mean(mask_magnitude)),
        "median_mask_magnitude": float(np.median(mask_magnitude)),
        "mean_R": summary["mean_R"],
        "std_R": summary["std_R"],
        "mean_entropy_normalized": summary["mean_entropy_normalized"],
        "adjacent_phase_coherence": _mean(adjacent, "phase_coherence"),
        "adjacent_phase_rmse_rad": _mean(adjacent, "phase_rmse_rad"),
        "adjacent_complex_nrmse": _mean(adjacent, "complex_nrmse"),
    }


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    return math.nan if np.all(np.isnan(values)) else float(np.nanmean(values))


def summarize_local_rows(
    rows: list[dict[str, float | int | str]],
) -> dict[str, float | int]:
    local = [row for row in rows if int(row["z"]) >= 0]
    magnitude = np.asarray([float(row["mean_mask_magnitude"]) for row in local])
    phase_r = np.asarray([float(row["mean_R"]) for row in local])
    coherence = np.asarray([float(row["adjacent_phase_coherence"]) for row in local])
    max_r_row = max(local, key=lambda row: float(row["mean_R"]))
    return {
        "max_mean_R_z": int(max_r_row["z"]),
        "max_mean_R": float(max_r_row["mean_R"]),
        "corr_magnitude_vs_mean_R": float(np.corrcoef(magnitude, phase_r)[0, 1]),
        "corr_log_magnitude_vs_mean_R": float(
            np.corrcoef(np.log(np.maximum(magnitude, 1e-12)), phase_r)[0, 1]
        ),
        "corr_magnitude_vs_adjacent_coherence": float(
            np.corrcoef(magnitude, coherence)[0, 1]
        ),
    }


def write_summary(path: Path, summary: dict[str, float | int]) -> None:
    lines = [
        "# Local observed-phase report",
        f"max_mean_R_z: {summary['max_mean_R_z']}",
        f"max_mean_R: {float(summary['max_mean_R']):.6f}",
        f"corr_magnitude_vs_mean_R: {float(summary['corr_magnitude_vs_mean_R']):.6f}",
        "corr_log_magnitude_vs_mean_R: "
        f"{float(summary['corr_log_magnitude_vs_mean_R']):.6f}",
        "corr_magnitude_vs_adjacent_coherence: "
        f"{float(summary['corr_magnitude_vs_adjacent_coherence']):.6f}",
        "",
        "Negative magnitude/R correlation is a warning that high spatial phase",
        "concentration may be driven by low-signal or coverage-edge slices.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(
    path: Path,
    rows: list[dict[str, float | int | str]],
    scope: str,
) -> None:
    local = [row for row in rows if int(row["z"]) >= 0]
    volume = next((row for row in rows if row["scope"] == "volume"), None)
    z_axis = [int(row["z"]) for row in local]
    series = (
        ("mean_mask_magnitude", "mean masked magnitude"),
        ("mean_R", "mean spatial phase R"),
        ("mean_entropy_normalized", "normalized phase entropy"),
        ("adjacent_phase_coherence", "adjacent-frame phase coherence"),
    )

    fig, axes = plt.subplots(1, 4, figsize=(16.8, 3.8), constrained_layout=True)
    for ax, (key, title) in zip(axes, series):
        ax.plot(z_axis, [float(row[key]) for row in local], marker="o", markersize=3)
        if volume is not None:
            ax.axhline(
                float(volume[key]),
                color="#b54708",
                linestyle="--",
                label="whole volume",
            )
            ax.legend(fontsize=8)
        ax.set_xlabel("z slice")
        ax.set_title(title)
        ax.grid(True, alpha=0.35)
    fig.suptitle(f"Observed phase structure by {scope}")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
