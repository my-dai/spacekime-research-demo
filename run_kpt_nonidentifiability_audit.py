#!/usr/bin/env python3
"""Audit non-uniqueness in the marginal and spectral KPT inverse models."""

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
    IDENTIFIABILITY_REPRESENTATION,
    fourier_density,
    known_surface_spectral_diagnostics,
    phase_reparameterization_counterexample,
    random_positive_fourier_coefficients,
    spectral_gauge_counterexample,
)
from spacekime_paths import PROJECT_ROOT, portable_manifest_path


SOCR_SOURCES = {
    "kpt_v5": {
        "url": "https://www.socr.umich.edu/TCIU/HTMLs/TCIU_SK_Appendix03_KPT_DS_fMRI_V5.html",
        "sha256": "ADA5993543B7126FECF1FCA0C04EB1F8BFDFD3EAE34177F6B4170B6F27BB8F3B",
    },
    "kpt_v1_simulation": {
        "url": "https://www.socr.umich.edu/TCIU/HTMLs/TCIU_SK_Appendix03_KPT_sim.html",
        "sha256": "A7D612D3B0EAB84FE0B82F07F90DC7AFD253745FE7023D18CE2294927793C50E",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=809_126)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--density-order", type=int, default=5)
    parser.add_argument("--surface-order", type=int, default=5)
    parser.add_argument("--warp-strengths", default="0.15,0.35,0.65")
    parser.add_argument("--gauge-strengths", default="0.05,0.15,0.35")
    parser.add_argument("--gauge-harmonic", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "kpt_nonidentifiability_phase2",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        raise SystemExit("seeds must be at least 1")
    if args.grid_size < 64 or args.grid_size % 2:
        raise SystemExit("grid-size must be an even integer at least 64")
    if args.density_order < 1 or args.surface_order < 1:
        raise SystemExit("density-order and surface-order must be positive")
    if args.gauge_harmonic < 1 or args.gauge_harmonic > args.surface_order:
        raise SystemExit("gauge-harmonic must be represented in the surface")
    warp_strengths = _parse_strengths(
        args.warp_strengths, "warp-strengths", upper=1.0
    )
    gauge_strengths = _parse_strengths(
        args.gauge_strengths, "gauge-strengths", upper=None
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(-np.pi, np.pi, args.grid_size, endpoint=False)
    rows: list[dict[str, object]] = []
    example: dict[str, np.ndarray] = {}
    known_surface_rows: list[dict[str, object]] = []

    for seed_offset in range(args.seeds):
        seed = args.base_seed + seed_offset
        generator = np.random.default_rng(seed)
        density_coefficients = random_positive_fourier_coefficients(
            args.density_order,
            rng=np.random.default_rng(np.random.SeedSequence([seed, 11])),
            positivity_margin=0.75,
        )
        density = fourier_density(theta, density_coefficients)
        surface = _random_surface(theta, args.surface_order, generator)
        surface_scale = max(float(np.std(surface)), 1e-12)
        density_scale = max(float(np.mean(density)), 1e-12)
        known_diagnostics = known_surface_spectral_diagnostics(
            surface, maximum_required_harmonic=args.density_order
        )
        known_surface_rows.append(
            {"seed": seed, **known_diagnostics}
        )

        for strength in warp_strengths:
            counterexample = phase_reparameterization_counterexample(
                theta, surface, density, warp_strength=strength
            )
            transformed_surface = _periodic_interpolate(
                np.asarray(counterexample["transformed_theta"]),
                np.asarray(counterexample["transformed_surface_at_transformed_theta"]),
                theta,
            )
            transformed_density = _periodic_interpolate(
                np.asarray(counterexample["transformed_theta"]),
                np.asarray(
                    counterexample[
                        "transformed_phase_density_at_transformed_theta"
                    ]
                ),
                theta,
            )
            transformed_density *= np.sum(density) / np.sum(transformed_density)
            surface_error = _rmse(transformed_surface, surface) / surface_scale
            density_error = _rmse(transformed_density, density) / density_scale
            rows.append(
                {
                    "family": "smooth_phase_reparameterization",
                    "seed": seed,
                    "strength": strength,
                    "surface_relative_rmse": surface_error,
                    "phase_density_relative_rmse": density_error,
                    "combined_object_change": max(surface_error, density_error),
                    "observable_max_abs_difference": counterexample[
                        "observable_characteristic_max_abs_difference"
                    ],
                    "nonrotation_witness": counterexample[
                        "nonrotation_displacement_rms"
                    ],
                    "transformed_phase_density_min": float(
                        np.min(transformed_density)
                    ),
                    "complete_observable_law_preserved": counterexample[
                        "complete_discrete_observable_law_preserved"
                    ],
                    "is_official_kpt": False,
                }
            )
            if seed_offset == 0 and strength == max(warp_strengths):
                example.update(
                    {
                        "theta": theta,
                        "surface": surface,
                        "density": density,
                        "warped_surface": transformed_surface,
                    }
                )

        for strength in gauge_strengths:
            counterexample = spectral_gauge_counterexample(
                surface,
                density,
                gauge_harmonic=args.gauge_harmonic,
                gauge_strength=strength,
            )
            transformed_surface = np.asarray(counterexample["transformed_surface"])
            transformed_density = np.asarray(
                counterexample["transformed_phase_density"]
            )
            surface_error = _rmse(transformed_surface, surface) / surface_scale
            density_error = _rmse(transformed_density, density) / density_scale
            rows.append(
                {
                    "family": "spectral_factorization_gauge",
                    "seed": seed,
                    "strength": strength,
                    "surface_relative_rmse": surface_error,
                    "phase_density_relative_rmse": density_error,
                    "combined_object_change": max(surface_error, density_error),
                    "observable_max_abs_difference": counterexample[
                        "observable_max_abs_difference"
                    ],
                    "nonrotation_witness": counterexample[
                        "surface_spectral_magnitude_relative_change"
                    ],
                    "transformed_phase_density_min": counterexample[
                        "transformed_phase_density_min"
                    ],
                    "complete_observable_law_preserved": True,
                    "is_official_kpt": False,
                }
            )
            if seed_offset == 0 and strength == max(gauge_strengths):
                example["gauge_density"] = transformed_density
        print(f"Completed seed {seed}", flush=True)

    summary = _summarize(rows)
    _write_csv(args.output_dir / "kpt_nonidentifiability_runs.csv", rows)
    _write_csv(args.output_dir / "kpt_nonidentifiability_summary.csv", summary)
    _write_csv(
        args.output_dir / "known_surface_spectral_diagnostics.csv",
        known_surface_rows,
    )
    (args.output_dir / "kpt_nonidentifiability_summary.txt").write_text(
        _summary_text(summary, known_surface_rows), encoding="utf-8"
    )
    (args.output_dir / "kpt_nonidentifiability_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "experiment": "kpt_nonidentifiability_counterexample_audit",
                "representation": IDENTIFIABILITY_REPRESENTATION,
                "parameters": {
                    **vars(args),
                    "warp_strengths": warp_strengths,
                    "gauge_strengths": gauge_strengths,
                    "output_dir": portable_manifest_path(args.output_dir),
                },
                "official_source_snapshots": SOCR_SOURCES,
                "summary": summary,
                "known_surface_diagnostics": known_surface_rows,
                "scientific_limits": [
                    "This is an independent mathematical audit, not the official SOCR KPT implementation.",
                    "The smooth reparameterization counterexample applies to the marginal Y=S(t,Theta)+epsilon model without phase anchors or a specified cross-time phase law.",
                    "The spectral gauge counterexample establishes non-uniqueness of M_hat=F_hat*Phi_hat alone, not every possible constrained likelihood model.",
                    "Regularization can select a factorization but does not by itself establish identifiability of the selected factors.",
                    "Known calibration surfaces, phase anchors, injective shared-phase multichannel observations, or validated phase dynamics may reduce the ambiguity and require separate audits.",
                    "MRI acquisition phase is not assumed to equal latent kime phase.",
                ],
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if plt is not None:
        _save_figure(
            args.output_dir / "kpt_nonidentifiability.png", rows, example
        )
    return 0


def _random_surface(
    theta: np.ndarray, order: int, generator: np.random.Generator
) -> np.ndarray:
    amplitudes = generator.uniform(0.35, 1.0, size=order)
    amplitudes *= 0.7 ** np.arange(order)
    phases = generator.uniform(-np.pi, np.pi, size=order)
    surface = np.full(theta.shape, generator.normal(0.0, 0.15))
    for harmonic, (amplitude, phase) in enumerate(
        zip(amplitudes, phases), start=1
    ):
        surface += amplitude * np.cos(harmonic * theta + phase)
    return surface


def _periodic_interpolate(
    source_theta: np.ndarray, values: np.ndarray, query_theta: np.ndarray
) -> np.ndarray:
    extended_theta = np.concatenate(
        [source_theta[-1:] - 2.0 * np.pi, source_theta, source_theta[:1] + 2.0 * np.pi]
    )
    extended_values = np.concatenate([values[-1:], values, values[:1]])
    return np.interp(query_theta, extended_theta, extended_values)


def _parse_strengths(value: str, name: str, *, upper: float | None) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be comma-separated numbers") from exc
    if (
        not parsed
        or any(not np.isfinite(item) or item <= 0 for item in parsed)
        or (upper is not None and any(item >= upper for item in parsed))
    ):
        bound = f" and less than {upper}" if upper is not None else ""
        raise SystemExit(f"{name} values must be positive, finite{bound}")
    return parsed


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2)))


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = sorted(
        {(str(row["family"]), float(row["strength"])) for row in rows}
    )
    output: list[dict[str, object]] = []
    for family, strength in groups:
        group = [
            row
            for row in rows
            if str(row["family"]) == family
            and float(row["strength"]) == strength
        ]
        output.append(
            {
                "family": family,
                "strength": strength,
                "datasets": len(group),
                "surface_relative_rmse_mean": _mean(group, "surface_relative_rmse"),
                "phase_density_relative_rmse_mean": _mean(
                    group, "phase_density_relative_rmse"
                ),
                "combined_object_change_mean": _mean(
                    group, "combined_object_change"
                ),
                "observable_max_abs_difference_max": max(
                    float(row["observable_max_abs_difference"]) for row in group
                ),
                "nonrotation_witness_mean": _mean(group, "nonrotation_witness"),
                "minimum_transformed_phase_density": min(
                    float(row["transformed_phase_density_min"]) for row in group
                ),
                "observable_law_preserved_fraction": float(
                    np.mean(
                        [bool(row["complete_observable_law_preserved"]) for row in group]
                    )
                ),
            }
        )
    return output


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_text(
    summary: list[dict[str, object]], known_surface_rows: list[dict[str, object]]
) -> str:
    lines = [
        "# KPT non-identifiability counterexample audit",
        "",
        "family | strength | surface change | phase-law change | max observable discrepancy | non-rotation witness",
    ]
    for row in summary:
        lines.append(
            f"{row['family']} | {float(row['strength']):.3f} | "
            f"{float(row['surface_relative_rmse_mean']):.4f} | "
            f"{float(row['phase_density_relative_rmse_mean']):.4f} | "
            f"{float(row['observable_max_abs_difference_max']):.3e} | "
            f"{float(row['nonrotation_witness_mean']):.4f}"
        )
    unobservable = [int(row["unobservable_modes"]) for row in known_surface_rows]
    required_unobservable = [
        int(row["unobservable_required_modes"]) for row in known_surface_rows
    ]
    lines.extend(
        [
            "",
            f"Known-surface diagnostic: unobservable full-grid Fourier modes ranged from {min(unobservable)} to {max(unobservable)}; unobservable modes required by the prespecified density order ranged from {min(required_unobservable)} to {max(required_unobservable)}.",
            "The marginal reparameterization examples preserve the complete discrete observable law exactly.",
            "The spectral examples preserve the stated M_hat=F_hat*Phi_hat observable to floating-point precision while changing both factors in a way that is not a rotation.",
            "These counterexamples identify missing assumptions; they are not an implementation of official KPT.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_figure(
    path: Path, rows: list[dict[str, object]], example: dict[str, np.ndarray]
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    theta = example["theta"]
    axes[0].plot(theta, example["surface"], label="original", color="#1f4e79")
    axes[0].plot(
        theta,
        example["warped_surface"],
        label="nonlinear phase reparameterization",
        color="#b23a48",
    )
    axes[0].set_title("Different surfaces")
    axes[0].set_xlabel("phase coordinate")
    axes[0].set_ylabel("surface value")
    axes[0].legend(fontsize=8)

    axes[1].plot(theta, example["density"], label="original", color="#1f4e79")
    axes[1].plot(
        theta,
        example["gauge_density"],
        label="spectral gauge alternative",
        color="#d17c0f",
    )
    axes[1].set_title("Different phase laws")
    axes[1].set_xlabel("phase coordinate")
    axes[1].set_ylabel("density")
    axes[1].legend(fontsize=8)

    colors = {
        "smooth_phase_reparameterization": "#b23a48",
        "spectral_factorization_gauge": "#d17c0f",
    }
    for family, color in colors.items():
        group = [row for row in rows if row["family"] == family]
        axes[2].scatter(
            [float(row["combined_object_change"]) for row in group],
            [
                max(float(row["observable_max_abs_difference"]), 1e-18)
                for row in group
            ],
            label=family,
            color=color,
            alpha=0.75,
        )
    axes[2].set_yscale("log")
    axes[2].set_xlabel("surface or phase-law relative change")
    axes[2].set_ylabel("observable discrepancy")
    axes[2].set_title("Changed factors, same observations")
    axes[2].legend(fontsize=8)
    fig.suptitle("KPT inverse-model non-identifiability counterexamples")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
