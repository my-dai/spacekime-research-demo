#!/usr/bin/env python3
"""Run a compact, reproducible showcase of the current KPT research line."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None

from spacekime_identifiability import (
    circular_mixture_observable,
    fourier_density,
    operator_diagnostics,
    phase_reparameterization_counterexample,
    quadrature_probe_operator,
    random_positive_fourier_coefficients,
    recover_phase_law_known_surface,
)
from spacekime_paths import PROJECT_ROOT, portable_manifest_path
from spacekime_quotient import (
    canonical_observable_representative,
    compare_observable_laws,
    observable_wasserstein_distance,
    weighted_gaussian_mmd_squared,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=812_101)
    parser.add_argument("--grid-size", type=int, default=257)
    parser.add_argument("--no-plot", action="store_true", help="Write only numeric evidence.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "research_showcase",
    )
    args = parser.parse_args()
    if args.grid_size < 65 or args.grid_size % 2 == 0:
        raise SystemExit("grid-size must be an odd integer of at least 65")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(-np.pi, np.pi, args.grid_size, endpoint=False)
    density = fourier_density(
        theta,
        random_positive_fourier_coefficients(
            4, rng=np.random.default_rng(args.seed), positivity_margin=0.75
        ),
    )
    mass = density / np.sum(density)
    surface = (
        0.25
        + np.cos(theta + 0.2)
        + 0.45 * np.cos(2.0 * theta - 0.5)
        + 0.25 * np.cos(3.0 * theta + 0.4)
        + 0.15 * np.cos(4.0 * theta - 0.1)
    )

    single = operator_diagnostics(quadrature_probe_operator(4, [0.0]))
    paired = operator_diagnostics(
        quadrature_probe_operator(4, [0.0, np.pi / 2.0])
    )

    reparameterized = phase_reparameterization_counterexample(
        theta, surface, density, warp_strength=0.65
    )
    transformed_theta = np.mod(
        np.asarray(reparameterized["transformed_theta"]), 2.0 * np.pi
    )
    transformed_surface = np.asarray(
        reparameterized["transformed_surface_at_transformed_theta"]
    )
    order = np.argsort(transformed_theta)
    warped_on_grid = np.interp(
        theta,
        transformed_theta[order],
        transformed_surface[order],
        period=2.0 * np.pi,
    )
    surface_change = _relative_rmse(warped_on_grid, surface)

    observable = circular_mixture_observable(surface, density)
    known_surface_fit = recover_phase_law_known_surface(
        observable, surface, maximum_harmonic=4
    )
    phase_recovery_error = _relative_rmse(
        np.asarray(known_surface_fit["phase_probability_mass"]), mass
    )

    equivalent = compare_observable_laws(
        surface,
        mass,
        transformed_surface,
        np.asarray(reparameterized["transformed_probability_mass"]),
    )
    alternative_surface = np.mean(surface) + 1.35 * (
        surface - np.mean(surface)
    )
    alternative = compare_observable_laws(
        surface, mass, alternative_surface, mass
    )
    canonical = canonical_observable_representative(surface, mass)

    joint_theta = np.linspace(0.0, 2.0 * np.pi, args.grid_size, endpoint=False)
    joint_original = np.column_stack(
        [np.cos(joint_theta), np.cos(joint_theta)]
    )
    joint_alternative = np.column_stack(
        [np.cos(joint_theta), np.cos(2.0 * joint_theta)]
    )
    uniform_mass = np.full(joint_theta.size, 1.0 / joint_theta.size)
    marginal_wasserstein = max(
        observable_wasserstein_distance(
            joint_original[:, channel],
            uniform_mass,
            joint_alternative[:, channel],
            uniform_mass,
            order=2,
        )
        for channel in range(2)
    )
    joint_mmd = weighted_gaussian_mmd_squared(
        joint_original,
        uniform_mass,
        joint_alternative,
        uniform_mass,
        bandwidth=1.0,
    )

    results = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "spacekime_research_showcase",
        "parameters": {
            "seed": args.seed,
            "grid_size": args.grid_size,
            "output_dir": portable_manifest_path(args.output_dir),
        },
        "measurement_design": {
            "single_quadrature_rank": single["rank"],
            "single_quadrature_nullity": single["nullity"],
            "paired_quadrature_rank": paired["rank"],
            "paired_quadrature_nullity": paired["nullity"],
            "paired_quadrature_condition_number": paired["condition_number"],
        },
        "nonidentifiability": {
            "coordinate_surface_relative_rmse": surface_change,
            "observable_characteristic_max_abs_difference": reparameterized[
                "observable_characteristic_max_abs_difference"
            ],
        },
        "known_surface_rescue": {
            "forward_model": "observed_offset_indexed_circular_convolution",
            "phase_probability_mass_relative_rmse": phase_recovery_error,
            "all_requested_harmonics_observable": known_surface_fit[
                "all_requested_harmonics_observable"
            ],
        },
        "observable_quotient": {
            "equivalent_wasserstein_2": equivalent["wasserstein_2"],
            "observable_alternative_wasserstein_2": alternative["wasserstein_2"],
        },
        "multichannel_joint_quotient": {
            "maximum_marginal_wasserstein_2": marginal_wasserstein,
            "joint_mmd_squared": joint_mmd,
        },
        "scientific_scope": [
            "Known-surface recovery assumes an observed circular-convolution response; it does not prove recovery from unlabelled scalar pushforward samples.",
            "Synthetic inverse-problem demonstration; not the official KPT algorithm.",
            "The quotient represents an observable law, not a recovered latent phase coordinate.",
            "The real fMRI file is not used because it lacks a declared controlled-repeat axis.",
        ],
    }
    manifest_path = args.output_dir / "research_showcase.json"
    summary_path = args.output_dir / "research_showcase.txt"
    figure_path = args.output_dir / "research_showcase.png"
    manifest_path.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    summary = _summary_text(results)
    summary_path.write_text(summary, encoding="utf-8")
    if plt is not None and not args.no_plot:
        _save_figure(
            figure_path,
            theta,
            surface,
            warped_on_grid,
            np.asarray(canonical["uniform_coordinate"]),
            np.asarray(canonical["quantile_values"]),
            single,
            paired,
            phase_recovery_error,
            equivalent,
            alternative,
            joint_original,
            joint_alternative,
            joint_mmd,
        )
    print(summary, end="")
    return 0


def _relative_rmse(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2))
    scale = max(float(np.sqrt(np.mean(np.asarray(right) ** 2))), 1e-12)
    return float(difference / scale)


def _summary_text(results: dict[str, object]) -> str:
    design = results["measurement_design"]
    nonidentifiability = results["nonidentifiability"]
    rescue = results["known_surface_rescue"]
    quotient = results["observable_quotient"]
    joint = results["multichannel_joint_quotient"]
    return "\n".join(
        [
            "# Spacekime research showcase",
            "",
            "Measurement design:",
            f"- single quadrature rank/nullity: {design['single_quadrature_rank']}/{design['single_quadrature_nullity']}",
            f"- paired quadrature rank/nullity: {design['paired_quadrature_rank']}/{design['paired_quadrature_nullity']}",
            "",
            "Identifiability boundary:",
            f"- coordinate surface relative RMSE after phase warp: {float(nonidentifiability['coordinate_surface_relative_rmse']):.4f}",
            f"- observable discrepancy: {float(nonidentifiability['observable_characteristic_max_abs_difference']):.3e}",
            "",
            "Rescue and quotient:",
            f"- known-surface convolution phase-law relative RMSE: {float(rescue['phase_probability_mass_relative_rmse']):.3e}",
            f"- equivalent-pair Wasserstein-2: {float(quotient['equivalent_wasserstein_2']):.3e}",
            f"- observable-alternative Wasserstein-2: {float(quotient['observable_alternative_wasserstein_2']):.4f}",
            "",
            "Multichannel joint law:",
            f"- maximum marginal Wasserstein-2: {float(joint['maximum_marginal_wasserstein_2']):.3e}",
            f"- joint MMD squared: {float(joint['joint_mmd_squared']):.6f}",
            "",
            "This compact run uses synthetic data and does not claim latent-phase recovery from the current fMRI file.",
        ]
    ) + "\n"


def _save_figure(
    path: Path,
    theta: np.ndarray,
    surface: np.ndarray,
    warped_surface: np.ndarray,
    uniform_coordinate: np.ndarray,
    canonical_quantiles: np.ndarray,
    single: dict[str, object],
    paired: dict[str, object],
    recovery_error: float,
    equivalent: dict[str, object],
    alternative: dict[str, object],
    joint_original: np.ndarray,
    joint_alternative: np.ndarray,
    joint_mmd: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    axes[0, 0].bar(
        ["single", "paired"],
        [single["rank"], paired["rank"]],
        color=["#b23a48", "#2f6690"],
    )
    axes[0, 0].axhline(8, color="black", linewidth=0.9, linestyle="--")
    axes[0, 0].set_ylabel("operator rank")
    axes[0, 0].set_title("Measurement completeness")

    axes[0, 1].plot(theta, surface, label="original coordinate")
    axes[0, 1].plot(theta, warped_surface, label="warped coordinate")
    axes[0, 1].plot(
        -np.pi + 2.0 * np.pi * uniform_coordinate,
        canonical_quantiles,
        label="observable quantile",
        linewidth=2.0,
    )
    axes[0, 1].set_xlabel("coordinate")
    axes[0, 1].set_ylabel("observable")
    axes[0, 1].set_title("Different coordinates, same observable law")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(
        ["convolution\nrecovery", "equivalent\nquotient", "observable\nalternative"],
        [
            recovery_error,
            float(equivalent["wasserstein_2"]),
            float(alternative["wasserstein_2"]),
        ],
        color=["#4f772d", "#2f6690", "#b23a48"],
    )
    axes[1, 0].set_yscale("symlog", linthresh=1e-14)
    axes[1, 0].set_ylabel("error or distance")
    axes[1, 0].set_title("Convolution recovery and observable-law comparisons")

    axes[1, 1].plot(
        joint_original[:, 0], joint_original[:, 1], label="original joint law"
    )
    axes[1, 1].plot(
        joint_alternative[:, 0],
        joint_alternative[:, 1],
        label="same marginals, different dependence",
    )
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].set_xlabel("channel 1")
    axes[1, 1].set_ylabel("channel 2")
    axes[1, 1].set_title(f"Joint MMD squared = {joint_mmd:.4f}")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Spacekime / KPT research showcase")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
