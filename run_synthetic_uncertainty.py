#!/usr/bin/env python3
"""Repeat-bootstrap uncertainty study for synthetic latent-kimesurface estimators."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from spacekime_advanced import (
    SUPPORTED_METHODS,
    bootstrap_surface_uncertainty,
    default_method_configs,
    select_reconstruction_config,
    uncertainty_calibration_metrics,
)
from spacekime_paths import PROJECT_ROOT
from spacekime_reporting import build_synthetic_run_manifest, write_run_manifest
from spacekime_synthetic import latent_kimesurface, make_theta_grid, simulate_repeated_measurements


OUTPUT_DIR = PROJECT_ROOT / "output" / "synthetic_uncertainty"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samplers", default="uniform,vonmises")
    parser.add_argument("--methods", default=",".join(SUPPORTED_METHODS))
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--seed", type=int, default=71_919)
    parser.add_argument("--n-time", type=int, default=40)
    parser.add_argument("--n-theta", type=int, default=64)
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--kappas", default="2,4,8,16,32,64,128")
    parser.add_argument("--joint-kappas", default="4,8,16,32,64")
    parser.add_argument("--time-bandwidths", default="0.025,0.05,0.1")
    parser.add_argument("--stitching-time-sigmas", default="0,0.5,1,2")
    parser.add_argument("--density-kappa", type=float, default=4.0)
    parser.add_argument("--max-density-weight-ratio", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    try:
        samplers = _choices(args.samplers, ("uniform", "vonmises"))
        methods = _choices(args.methods, SUPPORTED_METHODS)
        kappas = _numbers(args.kappas)
        joint_kappas = _numbers(args.joint_kappas)
        time_bandwidths = _numbers(args.time_bandwidths)
        stitching_sigmas = _numbers(args.stitching_time_sigmas, allow_zero=True)
        if args.repeats < 6 or args.n_time < 2 or args.n_theta < 8:
            raise ValueError("repeats >= 6, n-time >= 2, and n-theta >= 8 are required")
        if args.n_bootstrap < 20:
            raise ValueError("n-bootstrap must be at least 20")
        if not 0 < args.confidence < 1:
            raise ValueError("confidence must be between zero and one")
        if args.inner_folds < 2:
            raise ValueError("inner-folds must be at least two")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    configs_by_method = {
        method: default_method_configs(
            method,
            kappas=joint_kappas if method == "joint_kernel" else kappas,
            time_bandwidths=time_bandwidths,
            stitching_time_sigmas=stitching_sigmas,
            density_kappa=args.density_kappa,
            max_weight_ratio=args.max_density_weight_ratio,
        )
        for method in methods
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    theta_grid = make_theta_grid(args.n_theta)
    rows: list[dict[str, object]] = []
    arrays: dict[str, np.ndarray] = {"theta_grid": theta_grid}

    for sampler_idx, sampler in enumerate(samplers):
        simulation = simulate_repeated_measurements(
            n_repeats=args.repeats,
            n_time=args.n_time,
            noise_sigma=args.noise_sigma,
            rng=np.random.default_rng(
                np.random.SeedSequence([args.seed, sampler_idx + 1, 8128])
            ),
            phase_sampler=sampler,
        )
        truth = latent_kimesurface(simulation["time"], theta_grid)
        arrays[f"{sampler}_time"] = simulation["time"]
        arrays[f"{sampler}_truth"] = truth
        for method_idx, method in enumerate(methods):
            selection = select_reconstruction_config(
                simulation["theta"],
                simulation["observations"],
                simulation["time"],
                method,
                configs_by_method[method],
                cv_folds=args.inner_folds,
                rng=np.random.default_rng(
                    np.random.SeedSequence([args.seed, sampler_idx + 1, 9000])
                ),
                theta_grid=theta_grid,
            )
            uncertainty = bootstrap_surface_uncertainty(
                simulation["theta"],
                simulation["observations"],
                simulation["time"],
                theta_grid,
                method,
                selection["selected_config"],
                n_bootstrap=args.n_bootstrap,
                confidence=args.confidence,
                rng=np.random.default_rng(
                    np.random.SeedSequence(
                        [args.seed, sampler_idx + 1, method_idx + 1, 10_000]
                    )
                ),
            )
            percentile_metrics = uncertainty_calibration_metrics(
                truth, uncertainty, interval="percentile", point_key="point_estimate"
            )
            basic_metrics = uncertainty_calibration_metrics(
                truth, uncertainty, interval="basic", point_key="bias_corrected_point"
            )
            prefix = f"{sampler}_{method}"
            for key in (
                "point_estimate", "bootstrap_mean", "bootstrap_bias",
                "bias_corrected_point", "bootstrap_std", "lower", "upper",
                "basic_lower", "basic_upper"
            ):
                arrays[f"{prefix}_{key}"] = np.asarray(uncertainty[key])
            rows.append(
                {
                    "sampler": sampler,
                    "method": method,
                    "repeats": args.repeats,
                    "n_time": args.n_time,
                    "n_theta": args.n_theta,
                    "noise_sigma": args.noise_sigma,
                    "selected_config": json.dumps(
                        selection["selected_config"], sort_keys=True, separators=(",", ":")
                    ),
                    "selection_cv_rmse": float(selection["cv_rmse"]),
                    "n_bootstrap": args.n_bootstrap,
                    "nominal_coverage": args.confidence,
                    "percentile_empirical_coverage": percentile_metrics["empirical_coverage"],
                    "percentile_coverage_error": percentile_metrics["empirical_coverage"] - args.confidence,
                    "percentile_mean_interval_width": percentile_metrics["mean_interval_width"],
                    "basic_empirical_coverage": basic_metrics["empirical_coverage"],
                    "basic_coverage_error": basic_metrics["empirical_coverage"] - args.confidence,
                    "basic_mean_interval_width": basic_metrics["mean_interval_width"],
                    "point_nrmse_std": percentile_metrics["point_nrmse_std"],
                    "bias_corrected_point_nrmse_std": basic_metrics["point_nrmse_std"],
                    "mean_abs_error_over_bootstrap_std": percentile_metrics[
                        "mean_abs_error_over_std"
                    ],
                    "uncertainty_kind": uncertainty["uncertainty_kind"],
                    "truth_used_during_estimation": False,
                }
            )
            print(f"Completed {sampler}/{method}", flush=True)

    csv_path = args.output_dir / "synthetic_uncertainty_summary.csv"
    npz_path = args.output_dir / "synthetic_uncertainty_surfaces.npz"
    text_path = args.output_dir / "synthetic_uncertainty_summary.txt"
    figure_path = args.output_dir / "synthetic_uncertainty.png"
    manifest_path = args.output_dir / "synthetic_uncertainty_manifest.json"
    _write_csv(csv_path, rows)
    np.savez_compressed(npz_path, **arrays)
    _write_text(text_path, rows, args.confidence)
    if plt is not None:
        _save_figure(figure_path, rows, samplers, methods, args.confidence)

    manifest = build_synthetic_run_manifest(
        tool="run_synthetic_uncertainty.py",
        parameters={
            "samplers": samplers,
            "methods": methods,
            "repeats": args.repeats,
            "seed": args.seed,
            "n_time": args.n_time,
            "n_theta": args.n_theta,
            "noise_sigma": args.noise_sigma,
            "n_bootstrap": args.n_bootstrap,
            "confidence": args.confidence,
            "inner_folds": args.inner_folds,
            "method_configs": configs_by_method,
        },
        results={"summary_rows": rows},
        notes=[
            "Repeat bootstrap intervals are estimated without latent truth and conditional on the observation-selected configuration.",
            "Both percentile and basic bias-corrected intervals are reported; neither uses latent truth during construction.",
            "Synthetic truth is consulted only afterward to measure empirical pointwise coverage.",
            "These intervals quantify repeat-sampling variability, not Bayesian posterior uncertainty, measurement-noise prediction intervals, or hyperparameter-selection uncertainty.",
            "Coverage below the nominal level is an empirical warning about smoothing bias or unmodeled uncertainty, not evidence that the observed real-fMRI proxy is a latent kimesurface.",
        ],
    )
    write_run_manifest(manifest_path, manifest)
    for path in (csv_path, npz_path, text_path, manifest_path):
        print(f"Wrote: {path.resolve()}")
    if plt is not None:
        print(f"Wrote: {figure_path.resolve()}")
    return 0


def _write_text(path: Path, rows: list[dict[str, object]], confidence: float) -> None:
    lines = [
        "# Synthetic repeat-bootstrap uncertainty calibration",
        "",
        "Intervals are conditional on the observation-selected configuration and do not use latent truth during estimation.",
        "",
        "sampler | method | point/basic-corrected NRMSE | percentile/basic/nominal coverage | percentile/basic mean width | |error| / bootstrap SD",
    ]
    for row in rows:
        lines.append(
            f"{row['sampler']} | {row['method']} | "
            f"{float(row['point_nrmse_std']):.4f}/"
            f"{float(row['bias_corrected_point_nrmse_std']):.4f} | "
            f"{float(row['percentile_empirical_coverage']):.3f}/"
            f"{float(row['basic_empirical_coverage']):.3f}/{confidence:.3f} | "
            f"{float(row['percentile_mean_interval_width']):.4f}/"
            f"{float(row['basic_mean_interval_width']):.4f} | "
            f"{float(row['mean_abs_error_over_bootstrap_std']):.3f}"
        )
    lines.extend([
        "",
        "Latent truth is used only to audit coverage after interval construction. Undercoverage exposes estimator bias or missing uncertainty components.",
        "This synthetic uncertainty analysis is scientifically separate from the observed real-fMRI spatial phase-distribution proxy.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_figure(
    path: Path,
    rows: list[dict[str, object]],
    samplers: list[str],
    methods: list[str],
    confidence: float,
) -> None:
    fig, axes = plt.subplots(
        len(samplers), 3, figsize=(13.5, 3.8 * len(samplers)), squeeze=False,
        constrained_layout=True,
    )
    x = np.arange(len(methods))
    for row_idx, sampler in enumerate(samplers):
        group = {str(row["method"]): row for row in rows if row["sampler"] == sampler}
        percentile_coverage = [
            float(group[method]["percentile_empirical_coverage"]) for method in methods
        ]
        basic_coverage = [
            float(group[method]["basic_empirical_coverage"]) for method in methods
        ]
        nrmse = [float(group[method]["point_nrmse_std"]) for method in methods]
        calibration = [
            float(group[method]["mean_abs_error_over_bootstrap_std"])
            for method in methods
        ]
        axes[row_idx, 0].bar(x - 0.18, percentile_coverage, width=0.36, label="percentile")
        axes[row_idx, 0].bar(x + 0.18, basic_coverage, width=0.36, label="basic")
        axes[row_idx, 0].axhline(confidence, color="black", linestyle="--", linewidth=1)
        axes[row_idx, 0].set_ylim(0, 1)
        axes[row_idx, 0].set_title(f"{sampler}: pointwise interval coverage")
        axes[row_idx, 0].set_ylabel("coverage")
        axes[row_idx, 0].legend(fontsize=8)
        axes[row_idx, 1].bar(x, nrmse)
        axes[row_idx, 1].set_title(f"{sampler}: point-estimate error")
        axes[row_idx, 1].set_ylabel("NRMSE / truth SD")
        axes[row_idx, 2].bar(x, calibration)
        axes[row_idx, 2].set_title(f"{sampler}: error relative to bootstrap SD")
        axes[row_idx, 2].set_ylabel("mean |error| / SD")
        for ax in axes[row_idx]:
            ax.set_xticks(x, methods, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Synthetic repeat-bootstrap uncertainty (truth used only for audit)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _numbers(text: str, allow_zero: bool = False) -> list[float]:
    values = sorted({float(item.strip()) for item in text.split(",") if item.strip()})
    if not values or any(not np.isfinite(value) for value in values):
        raise ValueError("numeric lists must contain finite values")
    if any(value < 0 if allow_zero else value <= 0 for value in values):
        raise ValueError("numeric list contains an out-of-range value")
    return values


def _choices(text: str, allowed_values) -> list[str]:
    allowed = set(allowed_values)
    values = list(dict.fromkeys(item.strip().lower() for item in text.split(",") if item.strip()))
    if not values or any(value not in allowed for value in values):
        raise ValueError(f"choices must be drawn from {sorted(allowed)}")
    return values


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
