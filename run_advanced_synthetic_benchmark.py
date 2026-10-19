#!/usr/bin/env python3
"""Strict multi-seed comparison of advanced synthetic kimesurface estimators."""

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
from scipy.stats import t as student_t

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from spacekime_advanced import (
    SUPPORTED_METHODS,
    default_method_configs,
    nested_method_holdout_metrics,
    reconstruct_advanced_surface,
    select_reconstruction_config,
)
from spacekime_paths import PROJECT_ROOT
from spacekime_reporting import build_synthetic_run_manifest, write_run_manifest
from spacekime_synthetic import (
    latent_kimesurface,
    make_theta_grid,
    phase_coverage_metrics,
    sampling_weighted_surface_metrics,
    simulate_repeated_measurements,
    surface_recovery_metrics,
)


OUTPUT_DIR = PROJECT_ROOT / "output" / "advanced_synthetic_benchmark"
DEFAULT_METHODS = ",".join(SUPPORTED_METHODS)
DEFAULT_METRICS = (
    "surface_nrmse",
    "surface_correlation",
    "sampling_weighted_nrmse",
    "low_coverage_nrmse",
    "high_coverage_nrmse",
    "low_to_high_rmse_ratio",
    "outer_truth_rmse",
    "outer_observed_rmse",
    "relative_truth_rmse_improvement",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", default="16,32,64,128")
    parser.add_argument("--samplers", default="uniform,vonmises")
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--kappas", default="2,4,8,16,32,64,128")
    parser.add_argument("--joint-kappas", default="4,8,16,32,64")
    parser.add_argument("--time-bandwidths", default="0.025,0.05,0.1")
    parser.add_argument("--stitching-time-sigmas", default="0,0.5,1,2")
    parser.add_argument("--density-kappa", type=float, default=4.0)
    parser.add_argument("--max-density-weight-ratio", type=float, default=10.0)
    parser.add_argument("--n-seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=30_000)
    parser.add_argument("--n-time", type=int, default=40)
    parser.add_argument("--n-theta", type=int, default=64)
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    try:
        repeats = _parse_number_list(args.repeats, int)
        samplers = _parse_choices(args.samplers, ("uniform", "vonmises"))
        methods = _parse_choices(args.methods, SUPPORTED_METHODS)
        kappas = _parse_number_list(args.kappas, float)
        joint_kappas = _parse_number_list(args.joint_kappas, float)
        time_bandwidths = _parse_number_list(args.time_bandwidths, float)
        stitching_sigmas = _parse_number_list(args.stitching_time_sigmas, float)
        if repeats[0] < 6:
            raise ValueError("all repeat counts must be at least six")
        if args.n_seeds < 2:
            raise ValueError("n-seeds must be at least two")
        if args.n_time < 2 or args.n_theta < 8:
            raise ValueError("n-time must be >= 2 and n-theta must be >= 8")
        if args.noise_sigma < 0:
            raise ValueError("noise-sigma must be non-negative")
        if args.inner_folds < 2:
            raise ValueError("inner-folds must be at least two")
        if not 0 < args.holdout_fraction < 1:
            raise ValueError("holdout-fraction must be between zero and one")
        if args.density_kappa <= 0 or args.max_density_weight_ratio < 1:
            raise ValueError("density correction parameters are invalid")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    method_configs = {}
    for method in methods:
        method_configs[method] = default_method_configs(
            method,
            kappas=joint_kappas if method == "joint_kernel" else kappas,
            time_bandwidths=time_bandwidths,
            stitching_time_sigmas=stitching_sigmas,
            density_kappa=args.density_kappa,
            max_weight_ratio=args.max_density_weight_ratio,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    theta_grid = make_theta_grid(args.n_theta)
    raw_rows: list[dict[str, object]] = []
    sampler_codes = {name: idx + 1 for idx, name in enumerate(samplers)}

    # Each (seed, sampler, repeat count) gets an independent deterministic stream.
    # Methods remain paired within that dataset and share the exact outer split.
    total = len(samplers) * len(repeats) * args.n_seeds
    completed = 0
    for sampler in samplers:
        for count in repeats:
            for seed_offset in range(args.n_seeds):
                seed = args.seed_start + seed_offset
                data_sequence = np.random.SeedSequence(
                    [seed, int(count), sampler_codes[sampler], 91_337]
                )
                simulation = simulate_repeated_measurements(
                    n_repeats=count,
                    n_time=args.n_time,
                    noise_sigma=args.noise_sigma,
                    rng=np.random.default_rng(data_sequence),
                    phase_sampler=sampler,
                )
                truth = latent_kimesurface(simulation["time"], theta_grid)
                coverage = phase_coverage_metrics(simulation["theta"])
                split_seed = _derived_seed(seed, count, sampler_codes[sampler], 1)
                cv_seed = _derived_seed(seed, count, sampler_codes[sampler], 2)
                full_cv_seed = _derived_seed(seed, count, sampler_codes[sampler], 3)

                for method in methods:
                    configs = method_configs[method]
                    full_selection = select_reconstruction_config(
                        simulation["theta"],
                        simulation["observations"],
                        simulation["time"],
                        method,
                        configs,
                        cv_folds=args.inner_folds,
                        rng=np.random.default_rng(full_cv_seed),
                        theta_grid=theta_grid,
                    )
                    surface = reconstruct_advanced_surface(
                        simulation["theta"],
                        simulation["observations"],
                        simulation["time"],
                        theta_grid,
                        method,
                        full_selection["selected_config"],
                    )
                    recovery = surface_recovery_metrics(truth, surface)
                    weighted = sampling_weighted_surface_metrics(
                        truth,
                        surface,
                        simulation["theta"],
                        theta_grid,
                        density_kappa=args.density_kappa,
                    )
                    nested = nested_method_holdout_metrics(
                        simulation["theta"],
                        simulation["observations"],
                        simulation["signal"],
                        simulation["time"],
                        method,
                        configs,
                        holdout_fraction=args.holdout_fraction,
                        inner_folds=args.inner_folds,
                        split_seed=split_seed,
                        cv_seed=cv_seed,
                        theta_grid=theta_grid,
                    )
                    raw_rows.append(
                        {
                            "seed": seed,
                            "sampler": sampler,
                            "repeats": count,
                            "method": method,
                            "noise_sigma": args.noise_sigma,
                            "n_time": args.n_time,
                            "n_theta": args.n_theta,
                            "surface_selected_config": _json(full_selection["selected_config"]),
                            "surface_candidate_rmse": _json(full_selection["candidate_rmse"]),
                            "surface_cv_rmse": float(full_selection["cv_rmse"]),
                            "nested_selected_config": _json(nested["selected_config"]),
                            "nested_candidate_rmse": _json(nested["inner_candidate_rmse"]),
                            "nested_inner_cv_rmse": float(nested["inner_cv_rmse"]),
                            "surface_nrmse": float(recovery["nrmse_std"]),
                            "surface_correlation": float(recovery["correlation"]),
                            "surface_r2": float(recovery["r2"]),
                            "sampling_weighted_nrmse": float(weighted["weighted_nrmse_std"]),
                            "low_coverage_nrmse": float(weighted["low_coverage_nrmse_std"]),
                            "high_coverage_nrmse": float(weighted["high_coverage_nrmse_std"]),
                            "low_to_high_rmse_ratio": float(weighted["low_to_high_rmse_ratio"]),
                            "outer_truth_rmse": float(nested["outer_truth_rmse"]),
                            "outer_observed_rmse": float(nested["outer_observed_rmse"]),
                            "time_mean_baseline_truth_rmse": float(
                                nested["time_mean_baseline_truth_rmse"]
                            ),
                            "relative_truth_rmse_improvement": float(
                                nested["relative_truth_rmse_improvement"]
                            ),
                            "mean_max_phase_gap_rad": float(
                                coverage["mean_max_phase_gap_rad"]
                            ),
                            "mean_sampling_resultant": float(
                                coverage["mean_sampling_resultant"]
                            ),
                            "outer_split_seed": split_seed,
                            "inner_cv_seed": cv_seed,
                            "full_cv_seed": full_cv_seed,
                            "simulation_stream": "independent_seed_sampler_repeat_stream",
                        }
                    )
                completed += 1
                if completed % max(1, total // 20) == 0 or completed == total:
                    print(f"Completed {completed}/{total} paired datasets", flush=True)

    summary_rows = aggregate_rows(raw_rows, samplers, repeats, methods)
    contrast_rows = paired_contrasts(raw_rows, samplers, repeats, methods)
    raw_path = args.output_dir / "advanced_benchmark_runs.csv"
    summary_path = args.output_dir / "advanced_benchmark_summary.csv"
    contrast_path = args.output_dir / "advanced_benchmark_paired_contrasts.csv"
    text_path = args.output_dir / "advanced_benchmark_summary.txt"
    figure_path = args.output_dir / "advanced_benchmark.png"
    manifest_path = args.output_dir / "advanced_benchmark_manifest.json"
    _write_csv(raw_path, raw_rows)
    _write_csv(summary_path, summary_rows)
    _write_csv(contrast_path, contrast_rows)
    _write_text(text_path, summary_rows, contrast_rows, args.n_seeds)
    if plt is not None:
        _save_figure(figure_path, summary_rows, samplers, methods)

    manifest = build_synthetic_run_manifest(
        tool="run_advanced_synthetic_benchmark.py",
        parameters={
            "repeats": repeats,
            "samplers": samplers,
            "methods": methods,
            "method_configs": method_configs,
            "n_seeds": args.n_seeds,
            "seed_start": args.seed_start,
            "n_time": args.n_time,
            "n_theta": args.n_theta,
            "noise_sigma": args.noise_sigma,
            "holdout_fraction": args.holdout_fraction,
            "inner_folds": args.inner_folds,
            "seed_ci": "two_sided_student_t_95_percent",
            "simulation_stream": "independent_for_each_seed_sampler_repeat_count",
            "paired_methods": True,
            "shared_outer_split_across_methods": True,
        },
        results={"summary_rows": summary_rows, "paired_contrasts": contrast_rows},
        notes=[
            "Configuration selection uses noisy observations only; latent truth is never passed to a selector.",
            "Known latent truth is used only for final synthetic surface and untouched outer-holdout evaluation.",
            "Methods share each simulated dataset and outer split, enabling paired contrasts with seed-level Student-t intervals.",
            "Density correction uses stabilized leave-one-out inverse circular-KDE weights and may trade bias for variance.",
            "Stitching is periodic piecewise-linear interpolation, optionally smoothed only along time.",
            "These estimators require repeated phase-indexed measurements and are not applied to the real-fMRI observed phase proxy.",
        ],
    )
    write_run_manifest(manifest_path, manifest)
    for path in (raw_path, summary_path, contrast_path, text_path, manifest_path):
        print(f"Wrote: {path.resolve()}")
    if plt is not None:
        print(f"Wrote: {figure_path.resolve()}")
    return 0


def aggregate_rows(
    rows: list[dict[str, object]],
    samplers: Iterable[str],
    repeats: Iterable[int],
    methods: Iterable[str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for sampler in samplers:
        for count in repeats:
            for method in methods:
                group = [
                    row for row in rows
                    if row["sampler"] == sampler
                    and int(row["repeats"]) == int(count)
                    and row["method"] == method
                ]
                if not group:
                    continue
                config_counts = Counter(str(row["surface_selected_config"]) for row in group)
                item: dict[str, object] = {
                    "sampler": sampler,
                    "repeats": int(count),
                    "method": method,
                    "n_seeds": len(group),
                    "selected_config_counts": _json(dict(sorted(config_counts.items()))),
                    "ci_method": "student_t_seed_level_95_percent",
                }
                for metric in DEFAULT_METRICS:
                    stats = _mean_ci([float(row[metric]) for row in group])
                    for name, value in stats.items():
                        item[f"{metric}_{name}"] = value
                output.append(item)
    return output


def paired_contrasts(
    rows: list[dict[str, object]],
    samplers: Iterable[str],
    repeats: Iterable[int],
    methods: Iterable[str],
) -> list[dict[str, object]]:
    methods = list(methods)
    if "kernel" not in methods:
        return []
    lookup = {
        (str(row["sampler"]), int(row["repeats"]), int(row["seed"]), str(row["method"])): row
        for row in rows
    }
    output: list[dict[str, object]] = []
    for sampler in samplers:
        for count in repeats:
            seeds = sorted({
                int(row["seed"]) for row in rows
                if row["sampler"] == sampler and int(row["repeats"]) == int(count)
            })
            for method in methods:
                if method == "kernel":
                    continue
                for metric in ("surface_nrmse", "outer_truth_rmse"):
                    deltas = []
                    for seed in seeds:
                        baseline = lookup.get((sampler, int(count), seed, "kernel"))
                        candidate = lookup.get((sampler, int(count), seed, method))
                        if baseline is not None and candidate is not None:
                            deltas.append(float(candidate[metric]) - float(baseline[metric]))
                    stats = _mean_ci(deltas)
                    finite = np.asarray(deltas, dtype=np.float64)
                    finite = finite[np.isfinite(finite)]
                    output.append(
                        {
                            "sampler": sampler,
                            "repeats": int(count),
                            "method": method,
                            "reference_method": "kernel",
                            "metric": metric,
                            "n_paired_seeds": int(finite.size),
                            "mean_delta_candidate_minus_kernel": stats["mean"],
                            "std_delta": stats["std"],
                            "ci95_low": stats["ci95_low"],
                            "ci95_high": stats["ci95_high"],
                            "candidate_win_rate": (
                                float(np.mean(finite < 0)) if finite.size else float("nan")
                            ),
                            "direction": "negative_delta_favors_candidate",
                            "ci_method": "paired_student_t_seed_level_95_percent",
                        }
                    )
    return output


def _mean_ci(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "mean": float("nan"), "std": float("nan"), "sem": float("nan"),
            "ci95_low": float("nan"), "ci95_high": float("nan"), "n": 0,
        }
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    sem = std / math.sqrt(array.size)
    critical = float(student_t.ppf(0.975, df=array.size - 1)) if array.size > 1 else 0.0
    half = critical * sem
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "n": int(array.size),
    }


def _write_text(
    path: Path,
    summary: list[dict[str, object]],
    contrasts: list[dict[str, object]],
    n_seeds: int,
) -> None:
    lines = [
        "# Advanced synthetic latent-kimesurface benchmark",
        "",
        f"Independent datasets per seed/sampler/repeat count; {n_seeds} seeds; paired methods; Student-t 95% CIs.",
        "Hyperparameters were selected from noisy observations only. Latent truth was used only for final evaluation.",
        "",
        "sampler | repeats | method | global NRMSE [95% CI] | outer truth RMSE [95% CI] | low/high error",
    ]
    for row in summary:
        lines.append(
            f"{row['sampler']} | {int(row['repeats']):3d} | {row['method']} | "
            f"{float(row['surface_nrmse_mean']):.4f} "
            f"[{float(row['surface_nrmse_ci95_low']):.4f}, {float(row['surface_nrmse_ci95_high']):.4f}] | "
            f"{float(row['outer_truth_rmse_mean']):.4f} "
            f"[{float(row['outer_truth_rmse_ci95_low']):.4f}, {float(row['outer_truth_rmse_ci95_high']):.4f}] | "
            f"{float(row['low_to_high_rmse_ratio_mean']):.3f}"
        )
    lines.extend([
        "", "## Paired contrasts against ordinary circular kernel",
        "", "sampler | repeats | method | metric | mean delta [95% CI] | win rate",
    ])
    for row in contrasts:
        lines.append(
            f"{row['sampler']} | {int(row['repeats']):3d} | {row['method']} | {row['metric']} | "
            f"{float(row['mean_delta_candidate_minus_kernel']):+.4f} "
            f"[{float(row['ci95_low']):+.4f}, {float(row['ci95_high']):+.4f}] | "
            f"{float(row['candidate_win_rate']):.3f}"
        )
    lines.extend([
        "",
        "Negative paired deltas favor the candidate. These results apply only to the known synthetic repeated-measurement generator, not the observed real-fMRI phase proxy.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_figure(
    path: Path,
    summary: list[dict[str, object]],
    samplers: list[str],
    methods: list[str],
) -> None:
    fig, axes = plt.subplots(
        len(samplers), 2, figsize=(12.5, 4.2 * len(samplers)), squeeze=False,
        constrained_layout=True,
    )
    for row_idx, sampler in enumerate(samplers):
        sampler_rows = [row for row in summary if row["sampler"] == sampler]
        for method in methods:
            method_rows = sorted(
                [row for row in sampler_rows if row["method"] == method],
                key=lambda row: int(row["repeats"]),
            )
            if not method_rows:
                continue
            x = np.asarray([int(row["repeats"]) for row in method_rows])
            for col, metric in enumerate(("surface_nrmse", "outer_truth_rmse")):
                mean = np.asarray([float(row[f"{metric}_mean"]) for row in method_rows])
                low = np.asarray([float(row[f"{metric}_ci95_low"]) for row in method_rows])
                high = np.asarray([float(row[f"{metric}_ci95_high"]) for row in method_rows])
                axes[row_idx, col].plot(x, mean, marker="o", label=method)
                axes[row_idx, col].fill_between(x, low, high, alpha=0.14)
        axes[row_idx, 0].set_title(f"{sampler}: global latent-surface recovery")
        axes[row_idx, 0].set_ylabel("NRMSE / truth SD")
        axes[row_idx, 1].set_title(f"{sampler}: untouched outer-repeat prediction")
        axes[row_idx, 1].set_ylabel("RMSE against noiseless synthetic signal")
        for ax in axes[row_idx]:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("repeated experiments")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
    fig.suptitle("Advanced repeated-measurement reconstruction (seed-level 95% CI)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _parse_number_list(text: str, cast) -> list:
    values = sorted({cast(item.strip()) for item in text.split(",") if item.strip()})
    if not values:
        raise ValueError("numeric list cannot be empty")
    if any(not np.isfinite(float(value)) for value in values):
        raise ValueError("numeric lists must contain finite values")
    return values


def _parse_choices(text: str, choices: Iterable[str]) -> list[str]:
    allowed = set(choices)
    values = list(dict.fromkeys(item.strip().lower() for item in text.split(",") if item.strip()))
    if not values or any(value not in allowed for value in values):
        raise ValueError(f"choices must be drawn from {sorted(allowed)}")
    return values


def _derived_seed(seed: int, count: int, sampler_code: int, stream: int) -> int:
    state = np.random.SeedSequence([seed, count, sampler_code, stream]).generate_state(1)
    return int(state[0])


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
