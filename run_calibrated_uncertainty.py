#!/usr/bin/env python3
"""Multi-seed selection-aware bootstrap and split-conformal calibration study."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from spacekime_advanced import (
    SUPPORTED_METHODS,
    bootstrap_surface_uncertainty,
    bootstrap_surface_uncertainty_reselected,
    conformal_interval_metrics,
    default_method_configs,
    select_reconstruction_config,
    split_conformal_prediction_intervals,
    uncertainty_calibration_metrics,
)
from spacekime_paths import PROJECT_ROOT, portable_manifest_path
from spacekime_synthetic import latent_kimesurface, make_theta_grid, simulate_repeated_measurements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-seed", type=int, default=73_019)
    parser.add_argument("--conformal-seeds", type=int, default=16)
    parser.add_argument("--bootstrap-seeds", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=72)
    parser.add_argument("--n-time", type=int, default=28)
    parser.add_argument("--n-theta", type=int, default=48)
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--methods", default=",".join(SUPPORTED_METHODS))
    parser.add_argument("--bootstrap-methods", default="kernel,joint_kernel")
    parser.add_argument("--samplers", default="uniform,vonmises")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "calibrated_uncertainty",
    )
    args = parser.parse_args()
    methods = _choices(args.methods, SUPPORTED_METHODS)
    bootstrap_methods = _choices(args.bootstrap_methods, SUPPORTED_METHODS)
    samplers = _choices(args.samplers, ("uniform", "vonmises"))
    if args.n_bootstrap < 20 or args.repeats < 12:
        raise SystemExit("n-bootstrap >= 20 and repeats >= 12 are required")
    configs = {
        method: default_method_configs(
            method,
            kappas=(4.0, 8.0, 16.0, 32.0),
            time_bandwidths=(0.04, 0.08),
            stitching_time_sigmas=(0.0, 0.75),
        )
        for method in set(methods + bootstrap_methods)
    }
    theta_grid = make_theta_grid(args.n_theta)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    conformal_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []

    for sampler_idx, sampler in enumerate(samplers):
        for seed_offset in range(args.conformal_seeds):
            seed = args.base_seed + seed_offset
            simulation = _simulation(args, seed, sampler_idx, sampler)
            for method_idx, method in enumerate(methods):
                conformal = split_conformal_prediction_intervals(
                    simulation["theta"],
                    simulation["observations"],
                    simulation["time"],
                    method,
                    configs[method],
                    confidence=args.confidence,
                    cv_folds=3,
                    split_seed=seed + 1000,
                    cv_seed=seed + 2000 + method_idx,
                    theta_grid=theta_grid,
                )
                test_idx = np.asarray(conformal["test_indices"], dtype=int)
                observed_metrics = conformal_interval_metrics(
                    simulation["observations"][test_idx], conformal
                )
                latent_metrics = conformal_interval_metrics(
                    simulation["signal"][test_idx], conformal
                )
                conformal_rows.append(
                    {
                        "sampler": sampler,
                        "seed": seed,
                        "method": method,
                        "nominal_coverage": args.confidence,
                        "observed_pointwise_coverage": observed_metrics["pointwise_coverage"],
                        "observed_trajectory_coverage": observed_metrics["trajectory_coverage"],
                        "latent_pointwise_coverage_audit": latent_metrics["pointwise_coverage"],
                        "latent_trajectory_coverage_audit": latent_metrics["trajectory_coverage"],
                        "mean_interval_width": observed_metrics["mean_interval_width"],
                        "observed_prediction_rmse": observed_metrics["prediction_rmse"],
                        "radius": conformal["radius"],
                        "selected_config": json.dumps(conformal["config"], sort_keys=True),
                        "target": conformal["target"],
                        "truth_used_during_estimation": False,
                    }
                )
            print(f"Conformal completed {sampler} seed {seed}", flush=True)

        for seed_offset in range(args.bootstrap_seeds):
            seed = args.base_seed + 50_000 + seed_offset
            simulation = _simulation(args, seed, sampler_idx, sampler)
            truth_surface = latent_kimesurface(simulation["time"], theta_grid)
            for method_idx, method in enumerate(bootstrap_methods):
                selection = select_reconstruction_config(
                    simulation["theta"],
                    simulation["observations"],
                    simulation["time"],
                    method,
                    configs[method],
                    cv_folds=3,
                    rng=np.random.default_rng(seed + 3000 + method_idx),
                    theta_grid=theta_grid,
                )
                conditional = bootstrap_surface_uncertainty(
                    simulation["theta"],
                    simulation["observations"],
                    simulation["time"],
                    theta_grid,
                    method,
                    selection["selected_config"],
                    n_bootstrap=args.n_bootstrap,
                    confidence=args.confidence,
                    rng=np.random.default_rng(seed + 4000 + method_idx),
                )
                reselected = bootstrap_surface_uncertainty_reselected(
                    simulation["theta"],
                    simulation["observations"],
                    simulation["time"],
                    theta_grid,
                    method,
                    configs[method],
                    n_bootstrap=args.n_bootstrap,
                    confidence=args.confidence,
                    cv_folds=3,
                    rng=np.random.default_rng(seed + 5000 + method_idx),
                )
                conditional_metrics = uncertainty_calibration_metrics(
                    truth_surface, conditional, interval="percentile"
                )
                reselected_metrics = uncertainty_calibration_metrics(
                    truth_surface, reselected, interval="percentile"
                )
                bootstrap_rows.append(
                    {
                        "sampler": sampler,
                        "seed": seed,
                        "method": method,
                        "nominal_coverage": args.confidence,
                        "conditional_coverage": conditional_metrics["empirical_coverage"],
                        "reselected_coverage": reselected_metrics["empirical_coverage"],
                        "conditional_width": conditional_metrics["mean_interval_width"],
                        "reselected_width": reselected_metrics["mean_interval_width"],
                        "conditional_point_nrmse": conditional_metrics["point_nrmse_std"],
                        "reselected_point_nrmse": reselected_metrics["point_nrmse_std"],
                        "reselected_distinct_configs": len(reselected["selection_frequency"]),
                        "selection_frequency": json.dumps(
                            reselected["selection_frequency"], sort_keys=True
                        ),
                        "truth_used_during_estimation": False,
                    }
                )
            print(f"Bootstrap completed {sampler} seed {seed}", flush=True)

    conformal_summary = _group_summary(
        conformal_rows,
        (
            "observed_pointwise_coverage",
            "observed_trajectory_coverage",
            "latent_pointwise_coverage_audit",
            "mean_interval_width",
        ),
    )
    bootstrap_summary = _group_summary(
        bootstrap_rows,
        ("conditional_coverage", "reselected_coverage", "conditional_width", "reselected_width"),
    )
    _write_csv(args.output_dir / "split_conformal_runs.csv", conformal_rows)
    _write_csv(args.output_dir / "split_conformal_summary.csv", conformal_summary)
    _write_csv(args.output_dir / "selection_bootstrap_runs.csv", bootstrap_rows)
    _write_csv(args.output_dir / "selection_bootstrap_summary.csv", bootstrap_summary)
    manifest = {
        "parameters": {
            **vars(args),
            "output_dir": portable_manifest_path(args.output_dir),
        },
        "split_conformal_summary": conformal_summary,
        "selection_bootstrap_summary": bootstrap_summary,
        "interpretation": [
            "Split conformal targets future noisy repeat trajectories using repeat-level maximum residual scores.",
            "Latent-signal coverage is an after-construction synthetic audit, not the conformal target.",
            "Reselected bootstrap repeats observation-only configuration selection in every resample.",
            "Neither procedure is a Bayesian posterior interval or evidence that real-fMRI observed phase is a latent kimesurface.",
        ],
    }
    (args.output_dir / "calibrated_uncertainty_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "calibrated_uncertainty_summary.txt").write_text(
        _summary_text(conformal_summary, bootstrap_summary), encoding="utf-8"
    )
    if plt is not None:
        _save_figure(
            args.output_dir / "calibrated_uncertainty.png",
            conformal_summary,
            bootstrap_summary,
            args.confidence,
        )
    return 0


def _simulation(args, seed: int, sampler_idx: int, sampler: str):
    return simulate_repeated_measurements(
        n_repeats=args.repeats,
        n_time=args.n_time,
        noise_sigma=args.noise_sigma,
        rng=np.random.default_rng(np.random.SeedSequence([seed, sampler_idx + 1, 991])),
        phase_sampler=sampler,
    )


def _choices(raw: str, supported) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values or any(value not in supported for value in values):
        raise SystemExit(f"invalid choice in {raw}; supported: {', '.join(supported)}")
    return values


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _group_summary(rows: list[dict[str, object]], metrics) -> list[dict[str, object]]:
    groups = sorted({(str(row["sampler"]), str(row["method"])) for row in rows})
    output = []
    for sampler, method in groups:
        group = [row for row in rows if row["sampler"] == sampler and row["method"] == method]
        summary: dict[str, object] = {"sampler": sampler, "method": method, "datasets": len(group)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def _summary_text(conformal, bootstrap) -> str:
    lines = [
        "# Calibrated synthetic uncertainty",
        "",
        "## Split conformal (target: future noisy repeat trajectory)",
        "sampler | method | observed point/trajectory coverage | latent point audit | width",
    ]
    for row in conformal:
        lines.append(
            f"{row['sampler']} | {row['method']} | "
            f"{float(row['observed_pointwise_coverage_mean']):.3f}/"
            f"{float(row['observed_trajectory_coverage_mean']):.3f} | "
            f"{float(row['latent_pointwise_coverage_audit_mean']):.3f} | "
            f"{float(row['mean_interval_width_mean']):.3f}"
        )
    lines.extend(["", "## Pointwise surface repeat bootstrap", "sampler | method | conditional/reselected coverage | conditional/reselected width"])
    for row in bootstrap:
        lines.append(
            f"{row['sampler']} | {row['method']} | "
            f"{float(row['conditional_coverage_mean']):.3f}/"
            f"{float(row['reselected_coverage_mean']):.3f} | "
            f"{float(row['conditional_width_mean']):.3f}/"
            f"{float(row['reselected_width_mean']):.3f}"
        )
    lines.extend(
        [
            "",
            "Conformal intervals target noisy future observations, not a latent-surface confidence band.",
            "Bootstrap coverage uses synthetic truth only for the final audit.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_figure(path: Path, conformal, bootstrap, confidence: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    labels = [f"{row['sampler']}\n{row['method']}" for row in conformal]
    values = [float(row["observed_trajectory_coverage_mean"]) for row in conformal]
    axes[0].bar(np.arange(len(values)), values)
    axes[0].axhline(confidence, color="black", linestyle="--", label="nominal")
    axes[0].set_xticks(np.arange(len(values)), labels, rotation=45, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Split-conformal trajectory coverage")
    axes[0].legend()
    x = np.arange(len(bootstrap))
    width = 0.38
    axes[1].bar(x - width / 2, [float(row["conditional_coverage_mean"]) for row in bootstrap], width, label="conditional")
    axes[1].bar(x + width / 2, [float(row["reselected_coverage_mean"]) for row in bootstrap], width, label="reselected")
    axes[1].axhline(confidence, color="black", linestyle="--", label="nominal")
    axes[1].set_xticks(x, [f"{row['sampler']}\n{row['method']}" for row in bootstrap], rotation=45, ha="right")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Pointwise latent-surface bootstrap audit")
    axes[1].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
