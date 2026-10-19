#!/usr/bin/env python3
"""Multi-seed simultaneous latent-surface repeat-bootstrap sensitivity study."""

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
    default_method_configs,
    reconstruction_support_diagnostics,
    select_reconstruction_config,
    simultaneous_bootstrap_surface_band,
    simultaneous_surface_coverage_metrics,
    synthetic_population_smoothed_surface,
    undersmooth_reconstruction_config,
)
from spacekime_paths import PROJECT_ROOT, portable_manifest_path
from spacekime_synthetic import latent_kimesurface, make_theta_grid, simulate_repeated_measurements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-seed", type=int, default=75_019)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=72)
    parser.add_argument("--n-time", type=int, default=28)
    parser.add_argument("--n-theta", type=int, default=48)
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--methods", default="kernel,joint_kernel")
    parser.add_argument("--samplers", default="uniform,vonmises")
    parser.add_argument("--undersmooth-factor", type=float, default=2.0)
    parser.add_argument("--variants", default="selected_cv")
    parser.add_argument("--support-relative-mass", type=float, default=0.25)
    parser.add_argument("--support-min-repeat-ess", type=float, default=5.0)
    parser.add_argument(
        "--support-domain-mode",
        choices=("variant", "common_intersection"),
        default="variant",
        help=(
            "Use each variant's support mask or one intersection shared by all "
            "requested variants."
        ),
    )
    parser.add_argument("--mc-confidence", type=float, default=0.95)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "simultaneous_uncertainty_phase5",
    )
    args = parser.parse_args()
    methods = _choices(args.methods, ("kernel", "joint_kernel"))
    samplers = _choices(args.samplers, ("uniform", "vonmises"))
    requested_variants = _choices(
        args.variants, ("selected_cv", "prespecified_undersmoothed")
    )
    if args.seeds < 2 or args.n_bootstrap < 20 or args.repeats < 12:
        raise SystemExit("seeds >= 2, n-bootstrap >= 20, and repeats >= 12 are required")
    configs = {
        method: default_method_configs(
            method,
            kappas=(4.0, 8.0, 16.0, 32.0),
            time_bandwidths=(0.04, 0.08),
        )
        for method in methods
    }
    theta_grid = make_theta_grid(args.n_theta)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for sampler_idx, sampler in enumerate(samplers):
        for seed_offset in range(args.seeds):
            seed = args.base_seed + seed_offset
            simulation = simulate_repeated_measurements(
                n_repeats=args.repeats,
                n_time=args.n_time,
                noise_sigma=args.noise_sigma,
                rng=np.random.default_rng(
                    np.random.SeedSequence([seed, sampler_idx + 1, 1771])
                ),
                phase_sampler=sampler,
            )
            truth = latent_kimesurface(simulation["time"], theta_grid)
            for method_idx, method in enumerate(methods):
                selection = select_reconstruction_config(
                    simulation["theta"],
                    simulation["observations"],
                    simulation["time"],
                    method,
                    configs[method],
                    cv_folds=3,
                    rng=np.random.default_rng(seed + 1000 + method_idx),
                    theta_grid=theta_grid,
                )
                selected = dict(selection["selected_config"])
                all_variants = {
                    "selected_cv": selected,
                    "prespecified_undersmoothed": undersmooth_reconstruction_config(
                        method, selected, factor=args.undersmooth_factor
                    ),
                }
                variants = {
                    name: all_variants[name] for name in requested_variants
                }
                variant_supports = {
                    name: reconstruction_support_diagnostics(
                        simulation["theta"],
                        simulation["time"],
                        theta_grid,
                        method,
                        variant_config,
                        relative_mass_threshold=args.support_relative_mass,
                        min_repeat_ess=args.support_min_repeat_ess,
                    )
                    for name, variant_config in variants.items()
                }
                common_support_mask = np.logical_and.reduce(
                    [
                        np.asarray(item["supported_mask"], dtype=bool)
                        for item in variant_supports.values()
                    ]
                )
                if (
                    args.support_domain_mode == "common_intersection"
                    and not np.any(common_support_mask)
                ):
                    raise RuntimeError(
                        "requested variants have no common supported cells"
                    )
                paired_bootstrap_seed = seed + 5000 + method_idx
                for variant, config in variants.items():
                    support = variant_supports[variant]
                    supported_mask = (
                        common_support_mask
                        if args.support_domain_mode == "common_intersection"
                        else np.asarray(support["supported_mask"], dtype=bool)
                    )
                    band = simultaneous_bootstrap_surface_band(
                        simulation["theta"],
                        simulation["observations"],
                        simulation["time"],
                        theta_grid,
                        method,
                        config,
                        n_bootstrap=args.n_bootstrap,
                        confidence=args.confidence,
                        rng=np.random.default_rng(paired_bootstrap_seed),
                        mc_confidence=args.mc_confidence,
                        return_bootstrap_samples=True,
                    )
                    bootstrap_samples = np.asarray(band.pop("bootstrap_samples"))
                    supported_band = simultaneous_bootstrap_surface_band(
                        simulation["theta"],
                        simulation["observations"],
                        simulation["time"],
                        theta_grid,
                        method,
                        config,
                        n_bootstrap=args.n_bootstrap,
                        confidence=args.confidence,
                        domain_mask=supported_mask,
                        mc_confidence=args.mc_confidence,
                        precomputed_bootstrap_samples=bootstrap_samples,
                    )
                    raw_audit = simultaneous_surface_coverage_metrics(truth, band)
                    smoothed_target = synthetic_population_smoothed_surface(
                        simulation["time"],
                        theta_grid,
                        method,
                        config,
                        sampler,
                    )
                    smoothed_audit = simultaneous_surface_coverage_metrics(
                        smoothed_target, band
                    )
                    supported_raw_audit = simultaneous_surface_coverage_metrics(
                        truth, supported_band
                    )
                    supported_smoothed_audit = simultaneous_surface_coverage_metrics(
                        smoothed_target, supported_band
                    )
                    mc = dict(band["critical_value_mc_diagnostics"])
                    supported_mc = dict(
                        supported_band["critical_value_mc_diagnostics"]
                    )
                    supported_ess = np.asarray(
                        support["repeat_effective_sample_size"]
                    )[supported_mask]
                    rows.append(
                        {
                            "sampler": sampler,
                            "seed": seed,
                            "method": method,
                            "variant": variant,
                            "nominal_confidence": args.confidence,
                            "pointwise_coverage": raw_audit["pointwise_coverage"],
                            "whole_surface_covered": raw_audit["whole_surface_covered"],
                            "mean_band_width": raw_audit["mean_band_width"],
                            "max_studentized_truth_error": raw_audit[
                                "max_studentized_truth_error"
                            ],
                            "critical_value": band["critical_value"],
                            "critical_value_mc_lower": mc[
                                "critical_value_mc_lower"
                            ],
                            "critical_value_mc_upper": mc[
                                "critical_value_mc_upper"
                            ],
                            "critical_value_mc_width": mc[
                                "critical_value_mc_width"
                            ],
                            "max_error_to_critical_ratio": raw_audit[
                                "max_error_to_critical_ratio"
                            ],
                            "band_center_surface_nrmse": raw_audit[
                                "band_center_surface_nrmse"
                            ],
                            "smoothed_estimand_pointwise_coverage": smoothed_audit[
                                "pointwise_coverage"
                            ],
                            "smoothed_estimand_whole_surface_covered": smoothed_audit[
                                "whole_surface_covered"
                            ],
                            "smoothed_estimand_max_error_to_critical_ratio": smoothed_audit[
                                "max_error_to_critical_ratio"
                            ],
                            "smoothed_estimand_band_center_nrmse": smoothed_audit[
                                "band_center_surface_nrmse"
                            ],
                            "supported_fraction": float(np.mean(supported_mask)),
                            "variant_supported_fraction": support[
                                "supported_fraction"
                            ],
                            "support_domain_mode": args.support_domain_mode,
                            "supported_min_repeat_ess": float(
                                np.min(supported_ess)
                            ),
                            "supported_raw_pointwise_coverage": supported_raw_audit[
                                "pointwise_coverage"
                            ],
                            "supported_raw_whole_surface_covered": supported_raw_audit[
                                "whole_surface_covered"
                            ],
                            "supported_raw_max_error_to_critical_ratio": supported_raw_audit[
                                "max_error_to_critical_ratio"
                            ],
                            "supported_smoothed_pointwise_coverage": supported_smoothed_audit[
                                "pointwise_coverage"
                            ],
                            "supported_smoothed_whole_surface_covered": supported_smoothed_audit[
                                "whole_surface_covered"
                            ],
                            "supported_smoothed_max_error_to_critical_ratio": supported_smoothed_audit[
                                "max_error_to_critical_ratio"
                            ],
                            "supported_critical_value": supported_band[
                                "critical_value"
                            ],
                            "supported_critical_value_mc_width": supported_mc[
                                "critical_value_mc_width"
                            ],
                            "selected_cv_rmse": selection["cv_rmse"],
                            "selected_config": json.dumps(selected, sort_keys=True),
                            "band_config": json.dumps(config, sort_keys=True),
                            "truth_used_during_estimation": False,
                            "is_bayesian": False,
                            "bootstrap_surface_draws_reused": True,
                        }
                    )
            print(f"Completed {sampler} seed {seed}", flush=True)

    summary = _summarize(rows)
    _write_csv(args.output_dir / "simultaneous_uncertainty_runs.csv", rows)
    _write_csv(args.output_dir / "simultaneous_uncertainty_summary.csv", summary)
    (args.output_dir / "simultaneous_uncertainty_summary.txt").write_text(
        _summary_text(summary), encoding="utf-8"
    )
    manifest = {
        "experiment": "simultaneous_studentized_repeat_bootstrap_surface_band",
        "parameters": {
            **vars(args),
            "output_dir": portable_manifest_path(args.output_dir),
        },
        "summary": summary,
        "scientific_limits": [
            "Bands quantify repeat-sampling variability conditional on a fixed reconstruction configuration.",
            "The selected configuration uses noisy observations only; undersmoothing is a prespecified sensitivity transform.",
            "Synthetic truth is used only after band construction for coverage auditing.",
            "The population-smoothed target is also synthetic-only and is computed after band construction.",
            "The supported-domain mask uses only phase-design kernel mass and repeat-level ESS with thresholds fixed by CLI parameters.",
            "common_intersection mode fixes one supported domain across requested variants; variant mode preserves historical behavior.",
            "Full- and supported-domain bands reuse the exact same bootstrap surface draws.",
            "DKW critical-value intervals quantify bootstrap Monte Carlo resolution, not statistical calibration.",
            "These are empirical bootstrap bands, not Bayesian posterior intervals or future-observation prediction bands.",
            "Real-fMRI H(t,theta) remains an observed spatial phase-distribution proxy, not the synthetic latent surface audited here.",
        ],
    }
    (args.output_dir / "simultaneous_uncertainty_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if plt is not None:
        _save_figure(args.output_dir / "simultaneous_uncertainty.png", summary, args.confidence)
    return 0


def _choices(raw: str, supported) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    if not values or any(value not in supported for value in values):
        raise SystemExit(f"invalid choices; supported: {', '.join(supported)}")
    return values


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = sorted(
        {
            (str(row["sampler"]), str(row["method"]), str(row["variant"]))
            for row in rows
        }
    )
    metrics = (
        "pointwise_coverage",
        "mean_band_width",
        "max_error_to_critical_ratio",
        "band_center_surface_nrmse",
        "critical_value",
        "critical_value_mc_width",
        "smoothed_estimand_pointwise_coverage",
        "smoothed_estimand_max_error_to_critical_ratio",
        "smoothed_estimand_band_center_nrmse",
        "supported_fraction",
        "variant_supported_fraction",
        "supported_raw_pointwise_coverage",
        "supported_raw_max_error_to_critical_ratio",
        "supported_smoothed_pointwise_coverage",
        "supported_smoothed_max_error_to_critical_ratio",
        "supported_critical_value",
        "supported_critical_value_mc_width",
    )
    output: list[dict[str, object]] = []
    for sampler, method, variant in groups:
        group = [
            row
            for row in rows
            if row["sampler"] == sampler
            and row["method"] == method
            and row["variant"] == variant
        ]
        item: dict[str, object] = {
            "sampler": sampler,
            "method": method,
            "variant": variant,
            "datasets": len(group),
            "whole_surface_coverage_fraction": float(
                np.mean([bool(row["whole_surface_covered"]) for row in group])
            ),
            "smoothed_estimand_whole_surface_coverage_fraction": float(
                np.mean(
                    [
                        bool(row["smoothed_estimand_whole_surface_covered"])
                        for row in group
                    ]
                )
            ),
            "supported_raw_whole_surface_coverage_fraction": float(
                np.mean(
                    [
                        bool(row["supported_raw_whole_surface_covered"])
                        for row in group
                    ]
                )
            ),
            "supported_smoothed_whole_surface_coverage_fraction": float(
                np.mean(
                    [
                        bool(row["supported_smoothed_whole_surface_covered"])
                        for row in group
                    ]
                )
            ),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def _summary_text(summary: list[dict[str, object]]) -> str:
    lines = [
        "# Simultaneous latent-surface repeat-bootstrap audit",
        "",
        "sampler | method | variant | full raw point/whole | full smoothed point/whole | supported fraction | supported raw point/whole | supported smoothed point/whole | critical MC width",
    ]
    for row in summary:
        lines.append(
            f"{row['sampler']} | {row['method']} | {row['variant']} | "
            f"{float(row['pointwise_coverage_mean']):.3f}/"
            f"{float(row['whole_surface_coverage_fraction']):.3f} | "
            f"{float(row['smoothed_estimand_pointwise_coverage_mean']):.3f}/"
            f"{float(row['smoothed_estimand_whole_surface_coverage_fraction']):.3f} | "
            f"{float(row['supported_fraction_mean']):.3f} | "
            f"{float(row['supported_raw_pointwise_coverage_mean']):.3f}/"
            f"{float(row['supported_raw_whole_surface_coverage_fraction']):.3f} | "
            f"{float(row['supported_smoothed_pointwise_coverage_mean']):.3f}/"
            f"{float(row['supported_smoothed_whole_surface_coverage_fraction']):.3f} | "
            f"{float(row['critical_value_mc_width_mean']):.3f}"
        )
    lines.extend(
        [
            "",
            "Whole-surface coverage requires every finite grid cell to be inside the band.",
            "Truth enters only the final synthetic audit; any undercoverage is reported rather than hidden.",
            "Smoothed-estimand coverage isolates sampling calibration from raw-surface smoothing bias.",
            "Supported-domain results use a prespecified observation-only design mask and a separately calibrated maximum statistic.",
            "Critical-value MC width is a conservative DKW quantile-resolution diagnostic.",
            "The real-fMRI observed phase proxy is outside this latent-surface coverage claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_figure(path: Path, summary, confidence: float) -> None:
    labels = [
        f"{row['sampler']}\n{row['method']}\n{row['variant'].replace('_', ' ')}"
        for row in summary
    ]
    x = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    width = 0.38
    axes[0].bar(
        x - width / 2,
        [float(row["pointwise_coverage_mean"]) for row in summary],
        width,
        label="raw latent surface",
    )
    axes[0].bar(
        x + width / 2,
        [float(row["smoothed_estimand_pointwise_coverage_mean"]) for row in summary],
        width,
        label="population-smoothed estimand",
    )
    axes[0].axhline(confidence, color="black", linestyle="--", label="nominal")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x, labels, rotation=45, ha="right")
    axes[0].set_title("Pointwise coverage audit")
    axes[0].legend()
    axes[1].bar(
        x - width / 2,
        [float(row["whole_surface_coverage_fraction"]) for row in summary],
        width,
        label="raw latent surface",
    )
    axes[1].bar(
        x + width / 2,
        [
            float(row["smoothed_estimand_whole_surface_coverage_fraction"])
            for row in summary
        ],
        width,
        label="population-smoothed estimand",
    )
    axes[1].axhline(confidence, color="black", linestyle="--", label="nominal")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xticks(x, labels, rotation=45, ha="right")
    axes[1].set_title("Whole-surface coverage audit")
    axes[1].legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
