#!/usr/bin/env python3
"""Benchmark an observable-law quotient target for non-identifiable KPT models."""

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

from spacekime_identifiability import (
    fourier_density,
    phase_reparameterization_counterexample,
    random_positive_fourier_coefficients,
)
from spacekime_paths import PROJECT_ROOT, portable_manifest_path
from spacekime_quotient import (
    QUOTIENT_REPRESENTATION,
    canonical_observable_representative,
    compare_observable_laws,
    empirical_observable_law,
    observable_wasserstein_distance,
)


RESEARCH_SOURCES = {
    "optimal_transport": "https://arxiv.org/abs/1803.00567",
    "kpt_v5": "https://www.socr.umich.edu/TCIU/HTMLs/TCIU_SK_Appendix03_KPT_DS_fMRI_V5.html",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=809_401)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--density-order", type=int, default=5)
    parser.add_argument("--surface-order", type=int, default=5)
    parser.add_argument("--warp-strength", type=float, default=0.65)
    parser.add_argument("--alternative-strength", type=float, default=0.35)
    parser.add_argument("--sample-sizes", default="64,256,1024")
    parser.add_argument("--noise-sigmas", default="0,0.15")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "kpt_quotient_phase4",
    )
    args = parser.parse_args()
    sample_sizes = _parse_ints(args.sample_sizes, "sample-sizes")
    noise_sigmas = _parse_floats(args.noise_sigmas, "noise-sigmas")
    if args.seeds < 2:
        raise SystemExit("seeds must be at least 2")
    if args.grid_size < 64 or args.grid_size % 2:
        raise SystemExit("grid-size must be an even integer at least 64")
    if args.density_order < 1 or args.surface_order < 1:
        raise SystemExit("density-order and surface-order must be positive")
    if not 0.0 < args.warp_strength < 1.0:
        raise SystemExit("warp-strength must lie in (0, 1)")
    if not np.isfinite(args.alternative_strength) or args.alternative_strength <= 0:
        raise SystemExit("alternative-strength must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(-np.pi, np.pi, args.grid_size, endpoint=False)
    population_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    example: dict[str, np.ndarray] = {}

    for seed_offset in range(args.seeds):
        seed = args.base_seed + seed_offset
        coefficients = random_positive_fourier_coefficients(
            args.density_order,
            rng=np.random.default_rng(np.random.SeedSequence([seed, 11])),
            positivity_margin=0.75,
        )
        density = fourier_density(theta, coefficients)
        mass = density / np.sum(density)
        surface = _random_surface(
            theta, args.surface_order, np.random.default_rng(seed)
        )
        counterexample = phase_reparameterization_counterexample(
            theta,
            surface,
            density,
            warp_strength=args.warp_strength,
        )
        transformed_theta = np.asarray(counterexample["transformed_theta"])
        equivalent_surface = np.asarray(
            counterexample["transformed_surface_at_transformed_theta"]
        )
        equivalent_mass = np.asarray(
            counterexample["transformed_probability_mass"]
        )
        warped_surface_on_original_grid = np.interp(
            theta,
            transformed_theta,
            equivalent_surface,
            period=2.0 * np.pi,
        )
        scale = max(float(np.std(surface)), 1e-12)
        alternative_surface = np.mean(surface) + (
            1.0 + args.alternative_strength
        ) * (surface - np.mean(surface))

        equivalent_comparison = compare_observable_laws(
            surface, mass, equivalent_surface, equivalent_mass
        )
        alternative_comparison = compare_observable_laws(
            surface, mass, alternative_surface, mass
        )
        canonical_original = canonical_observable_representative(surface, mass)
        canonical_equivalent = canonical_observable_representative(
            equivalent_surface, equivalent_mass
        )
        canonical_difference = float(
            np.max(
                np.abs(
                    np.asarray(canonical_original["quantile_values"])
                    - np.asarray(canonical_equivalent["quantile_values"])
                )
            )
        )
        population_rows.append(
            {
                "seed": seed,
                "warp_strength": args.warp_strength,
                "alternative_strength": args.alternative_strength,
                "coordinate_surface_relative_rmse": _rmse(
                    warped_surface_on_original_grid, surface
                )
                / scale,
                "equivalent_wasserstein_1": equivalent_comparison[
                    "wasserstein_1"
                ],
                "equivalent_wasserstein_2": equivalent_comparison[
                    "wasserstein_2"
                ],
                "equivalent_canonical_max_abs_difference": canonical_difference,
                "alternative_wasserstein_1": alternative_comparison[
                    "wasserstein_1"
                ],
                "alternative_wasserstein_2": alternative_comparison[
                    "wasserstein_2"
                ],
                "alternative_canonical_quantile_rmse": alternative_comparison[
                    "canonical_quantile_rmse"
                ],
                "requires_phase_alignment": False,
                "is_official_kpt": False,
            }
        )

        for sample_size in sample_sizes:
            for noise_sigma in noise_sigmas:
                generator = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, sample_size, int(round(noise_sigma * 10_000)), 17]
                    )
                )
                original_sample = generator.choice(
                    surface, size=sample_size, replace=True, p=mass
                )
                equivalent_sample = generator.choice(
                    equivalent_surface,
                    size=sample_size,
                    replace=True,
                    p=equivalent_mass,
                )
                alternative_sample = generator.choice(
                    alternative_surface, size=sample_size, replace=True, p=mass
                )
                if noise_sigma > 0.0:
                    noise_scale = noise_sigma * scale
                    original_sample += generator.normal(
                        0.0, noise_scale, size=sample_size
                    )
                    equivalent_sample += generator.normal(
                        0.0, noise_scale, size=sample_size
                    )
                    alternative_sample += generator.normal(
                        0.0, noise_scale, size=sample_size
                    )
                original_law = empirical_observable_law(original_sample)
                equivalent_law = empirical_observable_law(equivalent_sample)
                alternative_law = empirical_observable_law(alternative_sample)
                equivalent_distance = observable_wasserstein_distance(
                    original_law["values"],
                    original_law["probability_mass"],
                    equivalent_law["values"],
                    equivalent_law["probability_mass"],
                    order=2,
                )
                alternative_distance = observable_wasserstein_distance(
                    original_law["values"],
                    original_law["probability_mass"],
                    alternative_law["values"],
                    alternative_law["probability_mass"],
                    order=2,
                )
                sample_rows.append(
                    {
                        "seed": seed,
                        "sample_size": sample_size,
                        "noise_sigma_relative": noise_sigma,
                        "equivalent_empirical_wasserstein_2": equivalent_distance,
                        "alternative_empirical_wasserstein_2": alternative_distance,
                        "alternative_detected": alternative_distance
                        > equivalent_distance,
                        "distance_margin": alternative_distance
                        - equivalent_distance,
                        "is_official_kpt": False,
                    }
                )

        if seed_offset == 0:
            example = {
                "theta": theta,
                "surface": surface,
                "warped_surface": warped_surface_on_original_grid,
                "uniform_coordinate": np.asarray(
                    canonical_original["uniform_coordinate"]
                ),
                "canonical_surface": np.asarray(
                    canonical_original["quantile_values"]
                ),
                "alternative_surface": alternative_surface,
            }
        print(f"Completed quotient seed {seed}", flush=True)

    population_summary = _summarize_population(population_rows)
    sample_summary = _summarize_samples(sample_rows)
    _write_csv(args.output_dir / "quotient_population_runs.csv", population_rows)
    _write_csv(args.output_dir / "quotient_finite_sample_runs.csv", sample_rows)
    _write_csv(args.output_dir / "quotient_finite_sample_summary.csv", sample_summary)
    (args.output_dir / "kpt_quotient_summary.txt").write_text(
        _summary_text(population_summary, sample_summary), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "kpt_observable_law_quotient_phase4",
        "representation": QUOTIENT_REPRESENTATION,
        "parameters": {
            "seeds": args.seeds,
            "base_seed": args.base_seed,
            "grid_size": args.grid_size,
            "density_order": args.density_order,
            "surface_order": args.surface_order,
            "warp_strength": args.warp_strength,
            "alternative_strength": args.alternative_strength,
            "sample_sizes": sample_sizes,
            "noise_sigmas_relative": noise_sigmas,
            "output_dir": portable_manifest_path(args.output_dir),
        },
        "population_summary": population_summary,
        "finite_sample_summary": sample_summary,
        "research_sources": RESEARCH_SOURCES,
        "scientific_limits": [
            "The quotient target is the observable pushforward law, not a recovered latent phase coordinate or Kime surface.",
            "With additive noise, empirical observations identify the convolved observable law unless the noise distribution is known or separately identified.",
            "The finite-sample study uses independent scalar draws from a synthetic marginal law and is not the official KPT likelihood.",
            "A zero quotient distance for reparameterized pairs does not make the original surface and phase law separately identifiable.",
        ],
        "is_official_kpt": False,
    }
    (args.output_dir / "kpt_quotient_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if plt is not None:
        _save_figure(
            args.output_dir / "kpt_quotient_benchmark.png",
            population_rows,
            sample_summary,
            example,
        )
    print(_summary_text(population_summary, sample_summary), end="")
    return 0


def _random_surface(
    theta: np.ndarray, order: int, generator: np.random.Generator
) -> np.ndarray:
    amplitudes = generator.uniform(0.25, 1.0, size=order)
    amplitudes *= 0.72 ** np.arange(order)
    phases = generator.uniform(-np.pi, np.pi, size=order)
    surface = np.full(theta.shape, generator.normal(0.0, 0.2))
    for harmonic, (amplitude, phase) in enumerate(
        zip(amplitudes, phases), start=1
    ):
        surface += amplitude * np.cos(harmonic * theta + phase)
    return surface


def _parse_ints(value: str, name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be comma-separated integers") from exc
    if not parsed or any(item < 2 for item in parsed):
        raise SystemExit(f"{name} values must be at least 2")
    return parsed


def _parse_floats(value: str, name: str) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be comma-separated numbers") from exc
    if not parsed or any(not np.isfinite(item) or item < 0.0 for item in parsed):
        raise SystemExit(f"{name} values must be finite and non-negative")
    return parsed


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2)))


def _sample_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def _summarize_population(rows: list[dict[str, object]]) -> dict[str, object]:
    coordinate = np.asarray(
        [float(row["coordinate_surface_relative_rmse"]) for row in rows]
    )
    equivalent = np.asarray(
        [float(row["equivalent_wasserstein_2"]) for row in rows]
    )
    alternative = np.asarray(
        [float(row["alternative_wasserstein_2"]) for row in rows]
    )
    canonical = np.asarray(
        [float(row["equivalent_canonical_max_abs_difference"]) for row in rows]
    )
    return {
        "datasets": len(rows),
        "coordinate_surface_relative_rmse_mean": float(np.mean(coordinate)),
        "coordinate_surface_relative_rmse_sd": _sample_sd(coordinate),
        "equivalent_wasserstein_2_max": float(np.max(equivalent)),
        "equivalent_canonical_max_abs_difference": float(np.max(canonical)),
        "alternative_wasserstein_2_mean": float(np.mean(alternative)),
        "alternative_wasserstein_2_min": float(np.min(alternative)),
        "population_separation_fraction": float(np.mean(alternative > equivalent)),
    }


def _summarize_samples(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    groups = sorted(
        {
            (int(row["sample_size"]), float(row["noise_sigma_relative"]))
            for row in rows
        }
    )
    summary: list[dict[str, object]] = []
    for sample_size, noise_sigma in groups:
        group = [
            row
            for row in rows
            if int(row["sample_size"]) == sample_size
            and float(row["noise_sigma_relative"]) == noise_sigma
        ]
        equivalent = np.asarray(
            [float(row["equivalent_empirical_wasserstein_2"]) for row in group]
        )
        alternative = np.asarray(
            [float(row["alternative_empirical_wasserstein_2"]) for row in group]
        )
        summary.append(
            {
                "sample_size": sample_size,
                "noise_sigma_relative": noise_sigma,
                "datasets": len(group),
                "equivalent_wasserstein_2_mean": float(np.mean(equivalent)),
                "equivalent_wasserstein_2_sd": _sample_sd(equivalent),
                "alternative_wasserstein_2_mean": float(np.mean(alternative)),
                "alternative_wasserstein_2_sd": _sample_sd(alternative),
                "alternative_detected_fraction": float(
                    np.mean([bool(row["alternative_detected"]) for row in group])
                ),
                "distance_margin_mean": float(
                    np.mean([float(row["distance_margin"]) for row in group])
                ),
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_text(
    population: dict[str, object], sample_summary: list[dict[str, object]]
) -> str:
    lines = [
        "# KPT observable-law quotient benchmark",
        "",
        f"population datasets: {population['datasets']}",
        "mean coordinate surface change: "
        f"{float(population['coordinate_surface_relative_rmse_mean']):.4f}",
        "maximum equivalent-pair W2: "
        f"{float(population['equivalent_wasserstein_2_max']):.3e}",
        "maximum equivalent canonical difference: "
        f"{float(population['equivalent_canonical_max_abs_difference']):.3e}",
        "mean observable-alternative W2: "
        f"{float(population['alternative_wasserstein_2_mean']):.4f}",
        "",
        "sample size | relative noise | same-law W2 | alternative W2 | detection fraction",
    ]
    for row in sample_summary:
        lines.append(
            f"{row['sample_size']} | {float(row['noise_sigma_relative']):.3g} | "
            f"{float(row['equivalent_wasserstein_2_mean']):.4f} | "
            f"{float(row['alternative_wasserstein_2_mean']):.4f} | "
            f"{float(row['alternative_detected_fraction']):.3f}"
        )
    lines.extend(
        [
            "",
            "The canonical quantile curve represents the identifiable observable law, not latent phase.",
            "With unknown additive noise, only the convolved observable law is directly identified.",
            "This is a synthetic marginal-law benchmark, not the official SOCR KPT implementation.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_figure(
    path: Path,
    population_rows: list[dict[str, object]],
    sample_summary: list[dict[str, object]],
    example: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    axes[0].plot(example["theta"], example["surface"], label="original")
    axes[0].plot(
        example["theta"], example["warped_surface"], label="warped coordinate"
    )
    axes[0].plot(
        -np.pi + 2.0 * np.pi * example["uniform_coordinate"],
        example["canonical_surface"],
        label="canonical quantile",
        linewidth=2.0,
    )
    axes[0].set_xlabel("coordinate")
    axes[0].set_ylabel("observable value")
    axes[0].set_title("Different coordinates, same quotient")
    axes[0].legend(fontsize=8)

    coordinate = [
        float(row["coordinate_surface_relative_rmse"]) for row in population_rows
    ]
    equivalent = [
        max(float(row["equivalent_wasserstein_2"]), 1e-18)
        for row in population_rows
    ]
    alternative = [
        float(row["alternative_wasserstein_2"]) for row in population_rows
    ]
    axes[1].scatter(coordinate, equivalent, label="equivalent pair", color="#2f6690")
    axes[1].scatter(coordinate, alternative, label="observable alternative", color="#b23a48")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("coordinate surface relative RMSE")
    axes[1].set_ylabel("population Wasserstein-2")
    axes[1].set_title("Quotient ignores gauge, detects change")
    axes[1].legend(fontsize=8)

    noises = sorted({float(row["noise_sigma_relative"]) for row in sample_summary})
    colors = ("#2f6690", "#d17c0f", "#4f772d", "#7b2cbf")
    for noise_sigma, color in zip(noises, colors):
        group = sorted(
            [
                row
                for row in sample_summary
                if float(row["noise_sigma_relative"]) == noise_sigma
            ],
            key=lambda row: int(row["sample_size"]),
        )
        axes[2].plot(
            [int(row["sample_size"]) for row in group],
            [float(row["alternative_detected_fraction"]) for row in group],
            marker="o",
            label=f"noise={noise_sigma:g}",
            color=color,
        )
    axes[2].set_xscale("log")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("independent samples")
    axes[2].set_ylabel("alternative detected fraction")
    axes[2].set_title("Finite-sample discrimination")
    axes[2].legend(fontsize=8)
    fig.suptitle("KPT Phase 4: inference on the identifiable quotient")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
