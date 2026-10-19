#!/usr/bin/env python3
"""Multi-seed, leakage-controlled synthetic kimesurface benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from spacekime_paths import PROJECT_ROOT
from spacekime_reporting import build_synthetic_run_manifest, write_run_manifest
from spacekime_synthetic import (
    holdout_prediction_metrics,
    latent_kimesurface,
    make_theta_grid,
    nested_holdout_prediction_metrics,
    phase_coverage_metrics,
    reconstruct_kimesurface,
    select_kernel_kappa,
    simulate_repeated_measurements,
    sampling_weighted_surface_metrics,
    surface_recovery_metrics,
)


OUTPUT_DIR = PROJECT_ROOT / "output" / "synthetic_benchmark"
DEFAULT_REPEATS = "8,16,32,64,128"
DEFAULT_KAPPAS = "1,2,4,8,16,32,64,128,256"


def parse_number_list(text: str, cast) -> list:
    values = sorted({cast(item.strip()) for item in text.split(",") if item.strip()})
    if not values:
        raise ValueError("list cannot be empty")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fixed and observation-CV circular bandwidths across multiple "
            "synthetic seeds, with nested repeat-level holdout evaluation."
        )
    )
    parser.add_argument("--repeats", default=DEFAULT_REPEATS)
    parser.add_argument("--kappa-candidates", default=DEFAULT_KAPPAS)
    parser.add_argument("--fixed-kappa", type=float, default=4.0)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--n-time", type=int, default=60)
    parser.add_argument("--n-theta", type=int, default=128)
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--inner-validation-fraction", type=float, default=0.25)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument(
        "--phase-sampler", choices=("uniform", "vonmises"), default="uniform"
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    try:
        repeats = parse_number_list(args.repeats, int)
        candidates = parse_number_list(args.kappa_candidates, float)
        if repeats[0] < 6:
            raise ValueError("all repeat counts must be >= 6 for nested validation")
        if args.n_seeds < 2:
            raise ValueError("n-seeds must be >= 2 to estimate uncertainty")
        if args.n_time < 2 or args.n_theta < 8:
            raise ValueError("n-time must be >= 2 and n-theta must be >= 8")
        if args.fixed_kappa <= 0 or any(value <= 0 for value in candidates):
            raise ValueError("kernel kappas must be positive")
        if args.noise_sigma < 0:
            raise ValueError("noise-sigma must be non-negative")
        if not 0 < args.holdout_fraction < 1:
            raise ValueError("holdout-fraction must be between 0 and 1")
        if not 0 < args.inner_validation_fraction < 1:
            raise ValueError("inner-validation-fraction must be between 0 and 1")
        if args.inner_folds < 2:
            raise ValueError("inner-folds must be >= 2")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    theta_grid = make_theta_grid(args.n_theta)
    raw_rows: list[dict[str, float | int | str]] = []

    for seed_offset in range(args.n_seeds):
        seed = args.seed_start + seed_offset
        simulation = simulate_repeated_measurements(
            n_repeats=max(repeats),
            n_time=args.n_time,
            noise_sigma=args.noise_sigma,
            rng=np.random.default_rng(seed),
            phase_sampler=args.phase_sampler,
        )
        truth = latent_kimesurface(simulation["time"], theta_grid)

        for count in repeats:
            theta = simulation["theta"][:count]
            observations = simulation["observations"][:count]
            signal = simulation["signal"][:count]

            fixed_reconstruction = reconstruct_kimesurface(
                theta, observations, theta_grid, kappa=args.fixed_kappa
            )
            fixed_recovery = surface_recovery_metrics(truth, fixed_reconstruction)
            fixed_holdout = holdout_prediction_metrics(
                theta,
                observations,
                signal,
                holdout_fraction=args.holdout_fraction,
                kappa=args.fixed_kappa,
            )

            selection = select_kernel_kappa(
                theta,
                observations,
                candidates=candidates,
                validation_fraction=args.inner_validation_fraction,
                cv_folds=args.inner_folds,
                rng=np.random.default_rng(seed + 10_000 + count),
            )
            selected_kappa = float(selection["selected_kappa"])
            cv_reconstruction = reconstruct_kimesurface(
                theta, observations, theta_grid, kappa=selected_kappa
            )
            cv_recovery = surface_recovery_metrics(truth, cv_reconstruction)
            coverage_recovery = sampling_weighted_surface_metrics(
                truth,
                cv_reconstruction,
                theta,
                theta_grid,
                density_kappa=4.0,
            )

            nested = nested_holdout_prediction_metrics(
                theta,
                observations,
                signal,
                candidates=candidates,
                holdout_fraction=args.holdout_fraction,
                inner_validation_fraction=args.inner_validation_fraction,
                inner_folds=args.inner_folds,
                seed=seed + 20_000 + count,
            )
            coverage = phase_coverage_metrics(theta)

            raw_rows.append(
                {
                    "seed": seed,
                    "repeats": count,
                    "phase_sampler": args.phase_sampler,
                    "noise_sigma": args.noise_sigma,
                    "fixed_kappa": args.fixed_kappa,
                    "selected_kappa_full": selected_kappa,
                    "selected_kappa_nested": float(nested["selected_kappa"]),
                    "fixed_surface_nrmse": fixed_recovery["nrmse_std"],
                    "cv_surface_nrmse": cv_recovery["nrmse_std"],
                    "fixed_surface_correlation": fixed_recovery["correlation"],
                    "cv_surface_correlation": cv_recovery["correlation"],
                    "cv_sampling_weighted_nrmse": coverage_recovery["weighted_nrmse_std"],
                    "cv_low_coverage_nrmse": coverage_recovery["low_coverage_nrmse_std"],
                    "cv_high_coverage_nrmse": coverage_recovery["high_coverage_nrmse_std"],
                    "cv_low_to_high_rmse_ratio": coverage_recovery["low_to_high_rmse_ratio"],
                    "fixed_holdout_rmse": fixed_holdout["kernel_rmse"],
                    "nested_holdout_rmse": float(nested["kernel_rmse"]),
                    "nested_baseline_rmse": float(nested["time_mean_baseline_rmse"]),
                    "nested_relative_improvement": float(
                        nested["relative_rmse_improvement"]
                    ),
                    "inner_cv_rmse": float(nested["inner_cv_rmse"]),
                    "mean_max_phase_gap_rad": coverage["mean_max_phase_gap_rad"],
                    "mean_sampling_resultant": coverage["mean_sampling_resultant"],
                }
            )

        print(f"completed seed {seed} ({seed_offset + 1}/{args.n_seeds})")

    summary_rows = aggregate_rows(raw_rows, repeats)
    raw_path = args.output_dir / "synthetic_benchmark_runs.csv"
    summary_path = args.output_dir / "synthetic_benchmark_summary.csv"
    text_path = args.output_dir / "synthetic_benchmark_summary.txt"
    figure_path = args.output_dir / "synthetic_benchmark.png"
    manifest_path = args.output_dir / "synthetic_benchmark_manifest.json"

    write_csv(raw_path, raw_rows)
    write_csv(summary_path, summary_rows)
    write_text_summary(text_path, summary_rows)
    if plt is not None:
        save_figure(figure_path, summary_rows)

    manifest = build_synthetic_run_manifest(
        "run_synthetic_benchmark.py",
        parameters={
            "repeats": repeats,
            "kappa_candidates": candidates,
            "fixed_kappa": args.fixed_kappa,
            "n_seeds": args.n_seeds,
            "seed_start": args.seed_start,
            "n_time": args.n_time,
            "n_theta": args.n_theta,
            "noise_sigma": args.noise_sigma,
            "holdout_fraction": args.holdout_fraction,
            "inner_validation_fraction": args.inner_validation_fraction,
            "inner_folds": args.inner_folds,
            "phase_sampler": args.phase_sampler,
        },
        results={"summary_rows": summary_rows},
        notes=[
            "Bandwidth selection uses repeat-level K-fold CV on noisy observations only; latent truth is not used to choose kappa.",
            "Nested holdout first selects kappa inside outer-training repeats, then evaluates on untouched repeats.",
            "Surface recovery remains a synthetic-truth diagnostic and is not evidence that real fMRI follows this generator.",
        ],
    )
    write_run_manifest(manifest_path, manifest)

    for path in (raw_path, summary_path, text_path, manifest_path):
        print(f"Wrote: {path.resolve()}")
    if plt is not None:
        print(f"Wrote: {figure_path.resolve()}")
    return 0


def aggregate_rows(
    rows: list[dict[str, float | int | str]], repeats: Iterable[int]
) -> list[dict[str, float | int | str]]:
    summary: list[dict[str, float | int | str]] = []
    metric_names = (
        "fixed_surface_nrmse",
        "cv_surface_nrmse",
        "fixed_surface_correlation",
        "cv_surface_correlation",
        "cv_sampling_weighted_nrmse",
        "cv_low_coverage_nrmse",
        "cv_high_coverage_nrmse",
        "cv_low_to_high_rmse_ratio",
        "fixed_holdout_rmse",
        "nested_holdout_rmse",
        "nested_baseline_rmse",
        "nested_relative_improvement",
        "mean_max_phase_gap_rad",
        "mean_sampling_resultant",
    )
    for count in repeats:
        group = [row for row in rows if int(row["repeats"]) == int(count)]
        if not group:
            continue
        selected = [float(row["selected_kappa_nested"]) for row in group]
        counts = Counter(selected)
        row_out: dict[str, float | int | str] = {
            "repeats": int(count),
            "n_seeds": len(group),
            "selected_kappa_median": float(np.median(selected)),
            "selected_kappa_q25": float(np.quantile(selected, 0.25)),
            "selected_kappa_q75": float(np.quantile(selected, 0.75)),
            "selected_kappa_counts": json.dumps(
                {f"{key:g}": value for key, value in sorted(counts.items())},
                sort_keys=True,
            ),
        }
        for metric in metric_names:
            values = np.asarray([float(item[metric]) for item in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                mean = std = ci = float("nan")
            else:
                mean = float(np.mean(finite))
                std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
                ci = 1.96 * std / math.sqrt(finite.size)
            row_out[f"{metric}_mean"] = mean
            row_out[f"{metric}_std"] = std
            row_out[f"{metric}_ci95_low"] = mean - ci
            row_out[f"{metric}_ci95_high"] = mean + ci
        summary.append(row_out)
    return summary


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_text_summary(
    path: Path, rows: list[dict[str, float | int | str]]
) -> None:
    lines = [
        "# Leakage-controlled multi-seed synthetic benchmark",
        "",
        "repeats | fixed NRMSE | CV global NRMSE | CV weighted NRMSE | low/high error | holdout gain | median kappa",
    ]
    for row in rows:
        lines.append(
            f"{int(row['repeats']):7d} | "
            f"{float(row['fixed_surface_nrmse_mean']):.4f} | "
            f"{float(row['cv_surface_nrmse_mean']):.4f} | "
            f"{float(row['cv_sampling_weighted_nrmse_mean']):.4f} | "
            f"{float(row['cv_low_to_high_rmse_ratio_mean']):.4f} | "
            f"{float(row['nested_relative_improvement_mean']):.4f} | "
            f"{float(row['selected_kappa_median']):.2f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    x = np.asarray([int(row["repeats"]) for row in rows])
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0), constrained_layout=True)

    plot_ci(
        axes[0, 0], x, rows, "fixed_surface_nrmse", "fixed kappa NRMSE"
    )
    plot_ci(axes[0, 0], x, rows, "cv_surface_nrmse", "CV kappa NRMSE")
    axes[0, 0].set_title("Global surface recovery")
    axes[0, 0].set_ylabel("NRMSE / truth SD")
    axes[0, 0].legend()

    plot_ci(axes[0, 1], x, rows, "cv_surface_nrmse", "uniform-grid NRMSE")
    plot_ci(
        axes[0, 1],
        x,
        rows,
        "cv_sampling_weighted_nrmse",
        "sampling-weighted NRMSE",
    )
    axes[0, 1].set_title("Global vs in-distribution error")
    axes[0, 1].legend()

    plot_ci(
        axes[0, 2], x, rows, "cv_surface_correlation", "CV correlation"
    )
    axes[0, 2].set_title("Surface correlation")

    plot_ci(
        axes[1, 0],
        x,
        rows,
        "nested_relative_improvement",
        "phase-aware gain",
    )
    axes[1, 0].axhline(0.0, linewidth=1)
    axes[1, 0].set_title("Nested holdout vs time-only baseline")
    axes[1, 0].set_ylabel("relative RMSE improvement")

    plot_ci(
        axes[1, 1], x, rows, "cv_low_coverage_nrmse", "low-coverage NRMSE"
    )
    plot_ci(
        axes[1, 1], x, rows, "cv_high_coverage_nrmse", "high-coverage NRMSE"
    )
    axes[1, 1].set_title("Phase-coverage error gap")
    axes[1, 1].legend()

    median = np.asarray([float(row["selected_kappa_median"]) for row in rows])
    q25 = np.asarray([float(row["selected_kappa_q25"]) for row in rows])
    q75 = np.asarray([float(row["selected_kappa_q75"]) for row in rows])
    axes[1, 2].plot(x, median, marker="o")
    axes[1, 2].fill_between(x, q25, q75, alpha=0.2)
    axes[1, 2].set_title("Nested-CV bandwidth selection")
    axes[1, 2].set_ylabel("selected kappa")

    for ax in axes.ravel():
        ax.set_xlabel("repeated experiments")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.25)

    fig.suptitle("Synthetic kimesurface benchmark across random seeds")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_ci(ax, x, rows, metric: str, label: str) -> None:
    mean = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
    low = np.asarray([float(row[f"{metric}_ci95_low"]) for row in rows])
    high = np.asarray([float(row[f"{metric}_ci95_high"]) for row in rows])
    ax.plot(x, mean, marker="o", label=label)
    ax.fill_between(x, low, high, alpha=0.18)


if __name__ == "__main__":
    raise SystemExit(main())
