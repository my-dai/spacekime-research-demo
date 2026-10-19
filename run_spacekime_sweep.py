#!/usr/bin/env python3
"""
Run a phase-noise sweep and compare kimesurface response curves.

This is a lightweight validation experiment: keep the same mask and time axis,
increase phase-noise sigma, and track how H(t, theta), circular concentration,
and phase entropy respond.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import math
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - exercised only in minimal environments
    plt = None

from spacekime_core import (
    build_phase_distribution_surface,
    kime_distance_L2,
    kime_js_distance,
    kime_surface_summary,
    phase_scope_mask,
)
from spacekime_loader import load_bigim
from spacekime_paths import PROJECT_ROOT, resolve_mat_path
from spacekime_reporting import build_run_manifest, write_run_manifest


OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_SIGMAS = "0,0.25,0.5,1,1.5,2,3"


def parse_sigmas(text: str) -> list[float]:
    sigmas = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not sigmas:
        raise ValueError("At least one sigma is required.")
    if any(sigma < 0 for sigma in sigmas):
        raise ValueError("Sigma values must be non-negative.")
    return sigmas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep phase-noise sigma and summarize the phase-distribution proxy."
    )
    parser.add_argument(
        "mat_file",
        nargs="?",
        default=None,
        help="Path to .mat (default: $SPACEKIME_MAT_FILE, data/, or project root)",
    )
    parser.add_argument("--sigmas", default=DEFAULT_SIGMAS, help="Comma-separated sigma values")
    parser.add_argument("--noise", choices=("gaussian", "vm"), default="gaussian")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t", type=int, default=0, help="Time index for mask creation")
    parser.add_argument("--z", type=int, default=10, help="Slice used by slice/ROI scope")
    parser.add_argument(
        "--scope", choices=("volume", "slice", "roi"), default="slice"
    )
    parser.add_argument("--roi-half", type=int, default=8)
    parser.add_argument(
        "--phase-mode", choices=("raw", "relative"), default="raw"
    )
    parser.add_argument("--max-t", type=int, default=50, help="Number of time points to compare")
    parser.add_argument("--n-theta", type=int, default=128, help="Number of phase bins")
    parser.add_argument("--mask-percentile", type=float, default=50.0)
    parser.add_argument(
        "--normalize-h",
        action="store_true",
        help="Normalize H rows before storing stats and raw L2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for CSV and figure outputs (default: ./output)",
    )
    args = parser.parse_args()

    try:
        sigmas = parse_sigmas(args.sigmas)
        mat_path = resolve_mat_path(args.mat_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {mat_path}")
    bigim = load_bigim(mat_path)
    _, _, _, T = bigim.shape
    t_list = list(range(0, min(max(args.max_t, 1), T)))

    z_idx = max(0, min(bigim.shape[2] - 1, args.z))
    t_idx = max(0, min(T - 1, args.t))
    print("Building original spatial phase-distribution proxy...")
    mask = phase_scope_mask(
        bigim,
        t_idx=t_idx,
        percentile=args.mask_percentile,
        scope=args.scope,
        z_idx=z_idx,
        roi_half=args.roi_half,
    )
    stats_orig = build_phase_distribution_surface(
        bigim,
        mask,
        t_list=t_list,
        n_theta=args.n_theta,
        normalize_h=args.normalize_h,
        phase_mode=args.phase_mode,
        reference_t=t_idx,
    )
    summary_orig = kime_surface_summary(stats_orig)

    rows: list[dict[str, float | str]] = []
    for sigma in sigmas:
        print(f"Sigma={sigma:g}")
        if sigma == 0:
            stats_noisy = stats_orig
        else:
            stats_noisy = build_phase_distribution_surface(
                bigim,
                mask,
                t_list=t_list,
                n_theta=args.n_theta,
                normalize_h=args.normalize_h,
                phase_noise_sigma=sigma,
                noise=args.noise,
                rng=np.random.default_rng(args.seed),
                phase_mode=args.phase_mode,
                reference_t=t_idx,
            )

        summary = kime_surface_summary(stats_noisy)
        rows.append(
            {
                "sigma": sigma,
                "noise": args.noise,
                "time_points": len(t_list),
                "l2": kime_distance_L2(stats_orig, stats_noisy),
                "l2_row_normalized": kime_distance_L2(
                    stats_orig, stats_noisy, normalize_rows=True
                ),
                "js_distance": kime_js_distance(stats_orig, stats_noisy),
                "mean_R": summary["mean_R"],
                "delta_mean_R": summary["mean_R"] - summary_orig["mean_R"],
                "mean_kappa": summary["mean_kappa"],
                "delta_mean_kappa": summary["mean_kappa"] - summary_orig["mean_kappa"],
                "mean_entropy": summary["mean_entropy"],
                "delta_mean_entropy": summary["mean_entropy"]
                - summary_orig["mean_entropy"],
                "mean_entropy_normalized": summary["mean_entropy_normalized"],
                "mean_phase_mae_rad": summary["mean_phase_mae_rad"],
                "mean_phase_rmse_rad": summary["mean_phase_rmse_rad"],
                "mean_phase_coherence": summary["mean_phase_coherence"],
                "expected_phase_coherence": (
                    math.exp(-0.5 * sigma**2) if args.noise == "gaussian" else math.nan
                ),
                "mean_complex_nrmse": summary["mean_complex_nrmse"],
            }
        )

    csv_path = args.output_dir / "sigma_sweep.csv"
    fig_path = args.output_dir / "sigma_sweep.png"
    manifest_path = args.output_dir / "sigma_sweep_manifest.json"
    write_rows(csv_path, rows)
    if plt is None:
        print("WARNING: matplotlib is not installed; skipped sweep figure.")
    else:
        save_sweep_figure(fig_path, rows)

    print(f"Wrote: {csv_path.resolve()}")
    if plt is not None:
        print(f"Wrote: {fig_path.resolve()}")
    manifest = build_run_manifest(
        "run_spacekime_sweep.py",
        mat_path,
        parameters={
            "sigmas": sigmas,
            "noise": args.noise,
            "seed": args.seed,
            "t": t_idx,
            "z": z_idx,
            "scope": args.scope,
            "roi_half": args.roi_half,
            "max_t": len(t_list),
            "n_theta": args.n_theta,
            "mask_percentile": args.mask_percentile,
            "normalize_h": args.normalize_h,
            "phase_mode": args.phase_mode,
            "reference_t": t_idx,
        },
        results={"mask_voxels": int(mask.sum()), "rows": rows},
        notes=[
            "Each Gaussian sigma reuses the same standardized random draw via a fixed seed.",
            "Histogram distances may plateau when the baseline spatial phase distribution is already near uniform.",
        ],
    )
    write_run_manifest(manifest_path, manifest)
    print(f"Wrote: {manifest_path.resolve()}")
    return 0


def write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_sweep_figure(path: Path, rows: list[dict[str, float | str]]) -> None:
    sigma = [float(row["sigma"]) for row in rows]
    l2_norm = [float(row["l2_row_normalized"]) for row in rows]
    js = [float(row["js_distance"]) for row in rows]
    phase_coherence = [float(row["mean_phase_coherence"]) for row in rows]
    expected_coherence = [float(row["expected_phase_coherence"]) for row in rows]
    complex_nrmse = [float(row["mean_complex_nrmse"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(sigma, l2_norm, marker="o", label="row-normalized L2")
    axes[0].plot(sigma, js, marker="o", label="JS distance")
    axes[0].set_xlabel("phase noise sigma (rad)")
    axes[0].set_title("Phase-distribution proxy distance")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(sigma, phase_coherence, marker="o", label="paired phase coherence")
    if np.any(np.isfinite(expected_coherence)):
        axes[1].plot(sigma, expected_coherence, "--", label="Gaussian expectation")
    axes[1].plot(sigma, complex_nrmse, marker="o", label="complex NRMSE")
    axes[1].set_xlabel("phase noise sigma (rad)")
    axes[1].set_title("Paired complex sensitivity")
    axes[1].grid(True)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
