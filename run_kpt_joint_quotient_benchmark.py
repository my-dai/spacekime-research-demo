#!/usr/bin/env python3
"""Benchmark permutation-calibrated multichannel observable-law quotients."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from spacekime_paths import PROJECT_ROOT, portable_manifest_path
from spacekime_quotient import (
    QUOTIENT_REPRESENTATION,
    observable_wasserstein_distance,
    permutation_mmd_test,
    weighted_gaussian_mmd_squared,
)


RESEARCH_SOURCES = {
    "mmd_two_sample": "https://www.jmlr.org/papers/v13/gretton12a.html",
    "finite_sample_mmd_permutation": "https://www.jmlr.org/papers/v24/21-1289.html",
    "multi_view_identifiability": "https://arxiv.org/abs/1905.06642",
}
DEPENDENCE_SCENARIOS = (
    "strong_nonlinear_dependence",
    "weak_lissajous_dependence",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--base-seed", type=int, default=811_501)
    parser.add_argument("--sample-sizes", default="48,96")
    parser.add_argument("--noise-sigmas", default="0,0.15")
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--calibration-seeds", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "kpt_joint_quotient_phase5",
    )
    args = parser.parse_args()
    sample_sizes = _parse_ints(args.sample_sizes, "sample-sizes")
    noise_sigmas = _parse_floats(args.noise_sigmas, "noise-sigmas")
    if args.seeds < 2:
        raise SystemExit("seeds must be at least 2")
    if args.permutations < 19:
        raise SystemExit("permutations must be at least 19")
    if args.calibration_seeds < 100:
        raise SystemExit("calibration-seeds must be at least 100")
    if not 0.0 < args.alpha < 1.0:
        raise SystemExit("alpha must lie in (0, 1)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    population = [
        _population_diagnostics(scenario) for scenario in DEPENDENCE_SCENARIOS
    ]
    rows: list[dict[str, object]] = []
    for seed_offset in range(args.seeds):
        seed = args.base_seed + seed_offset
        for scenario_index, scenario in enumerate(DEPENDENCE_SCENARIOS):
            for sample_size in sample_sizes:
                for noise_sigma in noise_sigmas:
                    rows.append(
                        _run_condition(
                            scenario,
                            scenario_index,
                            seed,
                            sample_size,
                            noise_sigma,
                            args.permutations,
                            args.alpha,
                        )
                    )
        print(f"Completed joint-quotient seed {seed}", flush=True)

    summary = _summarize(rows)
    calibration_rows = _run_null_calibration(
        args.calibration_seeds,
        max(sample_sizes),
        args.permutations,
        args.alpha,
    )
    calibration_summary = _summarize_calibration(calibration_rows)
    _write_csv(args.output_dir / "joint_quotient_runs.csv", rows)
    _write_csv(args.output_dir / "joint_quotient_summary.csv", summary)
    _write_csv(
        args.output_dir / "joint_quotient_null_calibration.csv",
        calibration_rows,
    )
    summary_text = _summary_text(
        population, summary, calibration_summary, args.alpha
    )
    (args.output_dir / "joint_quotient_summary.txt").write_text(
        summary_text, encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "kpt_multichannel_joint_quotient_phase5",
        "representation": QUOTIENT_REPRESENTATION,
        "parameters": {
            "seeds": args.seeds,
            "base_seed": args.base_seed,
            "sample_sizes": sample_sizes,
            "noise_sigmas_relative": noise_sigmas,
            "permutations": args.permutations,
            "calibration_seeds": args.calibration_seeds,
            "alpha": args.alpha,
            "dependence_scenarios": list(DEPENDENCE_SCENARIOS),
            "output_dir": portable_manifest_path(args.output_dir),
        },
        "population_diagnostics": population,
        "finite_sample_summary": summary,
        "null_calibration_summary": calibration_summary,
        "research_sources": RESEARCH_SOURCES,
        "scientific_limits": [
            "Joint-law separation does not by itself identify the latent phase coordinate.",
            "Multiple views require explicit distinct-view, noise, calibration, or injectivity assumptions before latent recovery can be claimed.",
            "The permutation test assumes independent experimental units within each sample; it is not valid for untreated consecutive fMRI frames.",
            "The pooled median bandwidth is fixed before label permutations, but this benchmark does not aggregate over multiple kernels.",
            "The weak Lissajous alternative is retained as a low-power stress test rather than presented as a successful detector.",
            "Twenty-four seeds give an exploratory rejection-rate audit, not a precise proof of nominal calibration.",
            "The separate 500-run null audit checks implementation calibration for one representative joint law and sample size, not every possible distribution.",
            "This synthetic benchmark is not the official SOCR KPT implementation.",
        ],
        "is_official_kpt": False,
    }
    (args.output_dir / "joint_quotient_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if plt is not None:
        _save_figure(
            args.output_dir / "joint_quotient_benchmark.png",
            population,
            summary,
            args.alpha,
        )
    print(summary_text, end="")
    return 0


def _population_diagnostics(scenario: str) -> dict[str, object]:
    # With prime grid size 257, multiplication by two permutes these circle
    # points exactly, so first- and second-harmonic channel marginals match.
    theta = np.linspace(0.0, 2.0 * np.pi, 257, endpoint=False)
    mass = np.full(theta.size, 1.0 / theta.size)
    original = _joint_surface(theta, scenario=scenario, alternative=False)
    alternative = _joint_surface(theta, scenario=scenario, alternative=True)
    first_w2 = observable_wasserstein_distance(
        original[:, 0], mass, alternative[:, 0], mass, order=2
    )
    second_w2 = observable_wasserstein_distance(
        original[:, 1], mass, alternative[:, 1], mass, order=2
    )
    equivalent_mmd = weighted_gaussian_mmd_squared(
        original, mass, original.copy(), mass.copy(), bandwidth=1.0
    )
    alternative_mmd = weighted_gaussian_mmd_squared(
        original, mass, alternative, mass, bandwidth=1.0
    )
    return {
        "scenario": scenario,
        "grid_size": theta.size,
        "channel_1_marginal_wasserstein_2": first_w2,
        "channel_2_marginal_wasserstein_2": second_w2,
        "equivalent_joint_mmd_squared": equivalent_mmd,
        "dependence_alternative_joint_mmd_squared": alternative_mmd,
        "marginals_equal_on_discrete_grid": bool(
            max(first_w2, second_w2) < 1e-12
        ),
        "joint_law_separates_dependence_alternative": bool(
            alternative_mmd > 1e-3
        ),
    }


def _run_condition(
    scenario: str,
    scenario_index: int,
    seed: int,
    sample_size: int,
    noise_sigma: float,
    permutations: int,
    alpha: float,
) -> dict[str, object]:
    generator = np.random.default_rng(
        np.random.SeedSequence(
            [
                seed,
                scenario_index,
                sample_size,
                int(round(noise_sigma * 10_000)),
                31,
            ]
        )
    )
    theta_original = generator.uniform(-np.pi, np.pi, size=sample_size)
    theta_null = generator.uniform(-np.pi, np.pi, size=sample_size)
    theta_alternative = generator.uniform(-np.pi, np.pi, size=sample_size)
    original = _joint_surface(
        theta_original, scenario=scenario, alternative=False
    )
    same_law = _joint_surface(
        theta_null, scenario=scenario, alternative=False
    )
    alternative = _joint_surface(
        theta_alternative, scenario=scenario, alternative=True
    )
    if noise_sigma > 0.0:
        original += generator.normal(0.0, noise_sigma, size=original.shape)
        same_law += generator.normal(0.0, noise_sigma, size=same_law.shape)
        alternative += generator.normal(0.0, noise_sigma, size=alternative.shape)

    joint_null = permutation_mmd_test(
        original,
        same_law,
        n_permutations=permutations,
        alpha=alpha,
        rng=np.random.default_rng(
            np.random.SeedSequence([seed, scenario_index, sample_size, 41])
        ),
    )
    joint_alternative = permutation_mmd_test(
        original,
        alternative,
        n_permutations=permutations,
        alpha=alpha,
        rng=np.random.default_rng(
            np.random.SeedSequence([seed, scenario_index, sample_size, 43])
        ),
    )
    marginal_1 = permutation_mmd_test(
        original[:, 0],
        alternative[:, 0],
        n_permutations=permutations,
        alpha=alpha,
        rng=np.random.default_rng(
            np.random.SeedSequence([seed, scenario_index, sample_size, 47])
        ),
    )
    marginal_2 = permutation_mmd_test(
        original[:, 1],
        alternative[:, 1],
        n_permutations=permutations,
        alpha=alpha,
        rng=np.random.default_rng(
            np.random.SeedSequence([seed, scenario_index, sample_size, 53])
        ),
    )
    return {
        "scenario": scenario,
        "seed": seed,
        "sample_size": sample_size,
        "noise_sigma": noise_sigma,
        "alpha": alpha,
        "permutations": permutations,
        "joint_null_mmd_squared": joint_null["mmd_squared_biased"],
        "joint_null_p_value": joint_null["p_value"],
        "joint_null_reject": joint_null["reject"],
        "joint_alternative_mmd_squared": joint_alternative[
            "mmd_squared_biased"
        ],
        "joint_alternative_p_value": joint_alternative["p_value"],
        "joint_alternative_reject": joint_alternative["reject"],
        "marginal_1_alternative_p_value": marginal_1["p_value"],
        "marginal_1_alternative_reject": marginal_1["reject"],
        "marginal_2_alternative_p_value": marginal_2["p_value"],
        "marginal_2_alternative_reject": marginal_2["reject"],
        "joint_bandwidth": joint_alternative["bandwidth"],
        "bandwidth_reselected_within_permutations": False,
        "is_official_kpt": False,
    }


def _joint_surface(
    theta: np.ndarray, *, scenario: str, alternative: bool
) -> np.ndarray:
    if scenario == "strong_nonlinear_dependence":
        second = np.cos(2.0 * theta) if alternative else np.cos(theta)
    elif scenario == "weak_lissajous_dependence":
        second = np.sin(2.0 * theta) if alternative else np.sin(theta)
    else:  # pragma: no cover
        raise ValueError(f"unknown dependence scenario: {scenario}")
    return np.column_stack([np.cos(theta), second])


def _run_null_calibration(
    seeds: int,
    sample_size: int,
    permutations: int,
    alpha: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed_offset in range(seeds):
        seed = 911_000 + seed_offset
        generator = np.random.default_rng(seed)
        theta_a = generator.uniform(-np.pi, np.pi, size=sample_size)
        theta_b = generator.uniform(-np.pi, np.pi, size=sample_size)
        sample_a = _joint_surface(
            theta_a, scenario="weak_lissajous_dependence", alternative=False
        )
        sample_b = _joint_surface(
            theta_b, scenario="weak_lissajous_dependence", alternative=False
        )
        result = permutation_mmd_test(
            sample_a,
            sample_b,
            n_permutations=permutations,
            alpha=alpha,
            rng=np.random.default_rng(seed + 100_000),
        )
        rows.append(
            {
                "seed": seed,
                "sample_size": sample_size,
                "permutations": permutations,
                "p_value": result["p_value"],
                "reject": result["reject"],
                "mmd_squared_biased": result["mmd_squared_biased"],
            }
        )
    return rows


def _parse_ints(value: str, name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be comma-separated integers") from exc
    if not parsed or any(item < 8 for item in parsed):
        raise SystemExit(f"{name} values must be at least 8")
    return parsed


def _parse_floats(value: str, name: str) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be comma-separated numbers") from exc
    if not parsed or any(not np.isfinite(item) or item < 0.0 for item in parsed):
        raise SystemExit(f"{name} values must be finite and non-negative")
    return parsed


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = sorted(
        {
            (
                str(row["scenario"]),
                int(row["sample_size"]),
                float(row["noise_sigma"]),
            )
            for row in rows
        }
    )
    summary: list[dict[str, object]] = []
    for scenario, sample_size, noise_sigma in groups:
        group = [
            row
            for row in rows
            if row["scenario"] == scenario
            and int(row["sample_size"]) == sample_size
            and float(row["noise_sigma"]) == noise_sigma
        ]
        null_fraction, null_low, null_high = _binary_rate_summary(
            group, "joint_null_reject"
        )
        power, power_low, power_high = _binary_rate_summary(
            group, "joint_alternative_reject"
        )
        summary.append(
            {
                "scenario": scenario,
                "sample_size": sample_size,
                "noise_sigma": noise_sigma,
                "datasets": len(group),
                "joint_null_rejection_fraction": null_fraction,
                "joint_null_rejection_wilson_95_low": null_low,
                "joint_null_rejection_wilson_95_high": null_high,
                "joint_dependence_power": power,
                "joint_dependence_power_wilson_95_low": power_low,
                "joint_dependence_power_wilson_95_high": power_high,
                "marginal_1_rejection_fraction": _fraction(
                    group, "marginal_1_alternative_reject"
                ),
                "marginal_2_rejection_fraction": _fraction(
                    group, "marginal_2_alternative_reject"
                ),
                "joint_null_p_value_mean": float(
                    np.mean([float(row["joint_null_p_value"]) for row in group])
                ),
                "joint_alternative_p_value_mean": float(
                    np.mean(
                        [float(row["joint_alternative_p_value"]) for row in group]
                    )
                ),
                "joint_mmd_squared_mean": float(
                    np.mean(
                        [
                            float(row["joint_alternative_mmd_squared"])
                            for row in group
                        ]
                    )
                ),
            }
        )
    return summary


def _fraction(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([bool(row[key]) for row in rows]))


def _binary_rate_summary(
    rows: list[dict[str, object]], key: str
) -> tuple[float, float, float]:
    successes = sum(bool(row[key]) for row in rows)
    total = len(rows)
    fraction = successes / total
    z_value = 1.959963984540054
    denominator = 1.0 + z_value**2 / total
    center = (fraction + z_value**2 / (2.0 * total)) / denominator
    radius = (
        z_value
        * np.sqrt(
            fraction * (1.0 - fraction) / total
            + z_value**2 / (4.0 * total**2)
        )
        / denominator
    )
    return float(fraction), float(center - radius), float(center + radius)


def _summarize_calibration(rows: list[dict[str, object]]) -> dict[str, object]:
    p_values = np.asarray([float(row["p_value"]) for row in rows])
    rejection, rejection_low, rejection_high = _binary_rate_summary(rows, "reject")
    return {
        "null_runs": len(rows),
        "sample_size": int(rows[0]["sample_size"]),
        "permutations": int(rows[0]["permutations"]),
        "rejection_fraction": rejection,
        "rejection_fraction_wilson_95_low": rejection_low,
        "rejection_fraction_wilson_95_high": rejection_high,
        "p_value_mean": float(np.mean(p_values)),
        "p_value_quantiles": {
            str(level): float(value)
            for level, value in zip(
                (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0),
                np.quantile(p_values, (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)),
            )
        },
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_text(
    population: list[dict[str, object]],
    summary: list[dict[str, object]],
    calibration: dict[str, object],
    alpha: float,
) -> str:
    lines = [
        "# KPT multichannel joint-quotient benchmark",
        "",
        f"nominal alpha: {alpha:.3f}",
        "",
        "scenario | marginal W2 max | population joint MMD^2",
    ]
    for row in population:
        lines.append(
            f"{row['scenario']} | "
            f"{max(float(row['channel_1_marginal_wasserstein_2']), float(row['channel_2_marginal_wasserstein_2'])):.3e} | "
            f"{float(row['dependence_alternative_joint_mmd_squared']):.6f}"
        )
    lines.extend(
        [
            "",
            "scenario | samples | noise | null rejection | joint power | marginal 1 rejection | marginal 2 rejection",
        ]
    )
    for row in summary:
        lines.append(
            f"{row['scenario']} | {row['sample_size']} | "
            f"{float(row['noise_sigma']):.3g} | "
            f"{float(row['joint_null_rejection_fraction']):.3f} | "
            f"{float(row['joint_dependence_power']):.3f} | "
            f"{float(row['marginal_1_rejection_fraction']):.3f} | "
            f"{float(row['marginal_2_rejection_fraction']):.3f}"
        )
    lines.extend(
        [
            "",
            "independent null calibration: "
            f"{calibration['null_runs']} runs, rejection fraction "
            f"{float(calibration['rejection_fraction']):.3f} "
            f"(Wilson 95% {float(calibration['rejection_fraction_wilson_95_low']):.3f}-"
            f"{float(calibration['rejection_fraction_wilson_95_high']):.3f}), "
            "mean p-value "
            f"{float(calibration['p_value_mean']):.3f}",
            "",
            "The two channel marginals are equal; only their joint dependence changes.",
            "Permutation calibration assumes independent units and does not validate untreated time-series frames.",
            "Joint-law identification is weaker than latent phase-coordinate identification.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_figure(
    path: Path,
    population: list[dict[str, object]],
    summary: list[dict[str, object]],
    alpha: float,
) -> None:
    theta = np.linspace(-np.pi, np.pi, 600, endpoint=False)
    original = _joint_surface(
        theta, scenario="strong_nonlinear_dependence", alternative=False
    )
    alternative = _joint_surface(
        theta, scenario="strong_nonlinear_dependence", alternative=True
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].plot(original[:, 0], original[:, 1], label="original joint law")
    axes[0].plot(
        alternative[:, 0], alternative[:, 1], label="dependence alternative"
    )
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("channel 1")
    axes[0].set_ylabel("channel 2")
    axes[0].set_title("Same marginals, different joint law")
    axes[0].legend(fontsize=8)

    metrics = [
        float(row["dependence_alternative_joint_mmd_squared"])
        for row in population
    ]
    axes[1].bar(
        ["strong nonlinear", "weak Lissajous"],
        metrics,
        color=["#b23a48", "#2f6690"],
    )
    axes[1].set_ylabel("population discrepancy")
    axes[1].set_title("Joint quotient reveals dependence")
    axes[1].tick_params(axis="x", rotation=18)

    combinations = sorted(
        {(str(row["scenario"]), float(row["noise_sigma"])) for row in summary}
    )
    colors = ("#2f6690", "#d17c0f", "#4f772d", "#7b2cbf")
    for (scenario, noise_sigma), color in zip(combinations, colors):
        group = sorted(
            [
                row
                for row in summary
                if row["scenario"] == scenario
                and float(row["noise_sigma"]) == noise_sigma
            ],
            key=lambda row: int(row["sample_size"]),
        )
        rates = np.asarray(
            [float(row["joint_dependence_power"]) for row in group]
        )
        lower = np.asarray(
            [float(row["joint_dependence_power_wilson_95_low"]) for row in group]
        )
        upper = np.asarray(
            [float(row["joint_dependence_power_wilson_95_high"]) for row in group]
        )
        axes[2].errorbar(
            [int(row["sample_size"]) for row in group],
            rates,
            yerr=np.vstack(
                [
                    np.maximum(rates - lower, 0.0),
                    np.maximum(upper - rates, 0.0),
                ]
            ),
            marker="o",
            color=color,
            capsize=3,
            label=f"{scenario.replace('_dependence', '')}, noise={noise_sigma:g}",
        )
    axes[2].axhline(alpha, color="black", linewidth=0.9, label="nominal alpha")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("independent samples per law")
    axes[2].set_ylabel("rejection fraction")
    axes[2].set_title("Permutation-calibrated joint test")
    axes[2].legend(fontsize=7)
    fig.suptitle("KPT Phase 5: multichannel observable quotient")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
