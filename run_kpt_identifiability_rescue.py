#!/usr/bin/env python3
"""Benchmark known-surface and phase-anchor KPT identifiability conditions."""

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
    circular_mixture_observable,
    fourier_density,
    phase_anchor_gauge_diagnostics,
    random_positive_fourier_coefficients,
    recover_phase_law_known_surface,
)
from spacekime_paths import PROJECT_ROOT, portable_manifest_path


CALIBRATION_PROFILES = ("balanced", "decaying", "missing_highest")
ANCHOR_PLACEMENTS = ("random", "uniform")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=809_226)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--density-order", type=int, default=4)
    parser.add_argument("--noise-sigmas", default="0,0.0001,0.001,0.01")
    parser.add_argument("--warp-orders", default="1,2,4,6")
    parser.add_argument("--anchor-counts", default="0,1,2,3,4,5,8,9,12,13")
    parser.add_argument("--ridge", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "kpt_identifiability_rescue_phase3",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        raise SystemExit("seeds must be at least 1")
    if args.grid_size < 64 or args.grid_size % 2:
        raise SystemExit("grid-size must be an even integer at least 64")
    if args.density_order < 1 or args.density_order >= args.grid_size // 2:
        raise SystemExit("density-order is outside the supported grid range")
    if args.ridge < 0 or not np.isfinite(args.ridge):
        raise SystemExit("ridge must be finite and non-negative")
    noise_sigmas = _parse_floats(args.noise_sigmas, "noise-sigmas", minimum=0.0)
    warp_orders = _parse_ints(args.warp_orders, "warp-orders", minimum=1)
    anchor_counts = _parse_ints(
        args.anchor_counts, "anchor-counts", minimum=0, allow_zero=True
    )
    if max(anchor_counts) > args.grid_size:
        raise SystemExit("anchor-counts cannot exceed grid-size")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(-np.pi, np.pi, args.grid_size, endpoint=False)
    known_rows = _run_known_surface_benchmark(
        theta, args, noise_sigmas
    )
    anchor_rows = _run_anchor_benchmark(args, warp_orders, anchor_counts)
    known_summary = _summarize_known(known_rows)
    anchor_summary = _summarize_anchors(anchor_rows)

    _write_csv(args.output_dir / "known_surface_recovery_runs.csv", known_rows)
    _write_csv(
        args.output_dir / "known_surface_recovery_summary.csv", known_summary
    )
    _write_csv(args.output_dir / "anchor_gauge_runs.csv", anchor_rows)
    _write_csv(args.output_dir / "anchor_gauge_summary.csv", anchor_summary)
    (args.output_dir / "kpt_identifiability_rescue_summary.txt").write_text(
        _summary_text(known_summary, anchor_summary), encoding="utf-8"
    )
    (args.output_dir / "kpt_identifiability_rescue_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "experiment": "kpt_identifiability_rescue_conditions",
                "representation": IDENTIFIABILITY_REPRESENTATION,
                "parameters": {
                    **vars(args),
                    "noise_sigmas": noise_sigmas,
                    "warp_orders": warp_orders,
                    "anchor_counts": anchor_counts,
                    "output_dir": portable_manifest_path(args.output_dir),
                },
                "known_surface_summary": known_summary,
                "anchor_gauge_summary": anchor_summary,
                "scientific_limits": [
                    "This is a finite circular-convolution and harmonic-warp analog, not the official KPT implementation.",
                    "Known-surface recovery assumes the calibration surface is exact and tests only a prespecified finite density order.",
                    "A zero or near-zero calibration coefficient makes the corresponding phase-law harmonic unidentified or unstable.",
                    "Anchor rank results apply to a finite harmonic warp family; finitely many anchors do not fix an unrestricted smooth diffeomorphism class.",
                    "Multichannel and cross-time dynamic rescue conditions remain to be tested separately.",
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
            args.output_dir / "kpt_identifiability_rescue.png",
            known_summary,
            anchor_summary,
        )
    return 0


def _run_known_surface_benchmark(
    theta: np.ndarray, args: argparse.Namespace, noise_sigmas: list[float]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed_offset in range(args.seeds):
        seed = args.base_seed + seed_offset
        coefficients = random_positive_fourier_coefficients(
            args.density_order,
            rng=np.random.default_rng(np.random.SeedSequence([seed, 31])),
            positivity_margin=0.75,
        )
        density = fourier_density(theta, coefficients)
        truth_mass = density / np.sum(density)
        mass_scale = 1.0 / theta.size
        phases = np.random.default_rng(
            np.random.SeedSequence([seed, 37])
        ).uniform(-np.pi, np.pi, size=args.density_order)
        for profile_idx, profile in enumerate(CALIBRATION_PROFILES):
            surface = _calibration_surface(
                theta, args.density_order, phases, profile
            )
            clean_observable = circular_mixture_observable(surface, density)
            observable_scale = max(float(np.std(clean_observable)), 1e-12)
            for noise_idx, noise_sigma in enumerate(noise_sigmas):
                generator = np.random.default_rng(
                    np.random.SeedSequence(
                        [seed, profile_idx + 1, noise_idx + 1, 43]
                    )
                )
                noisy_observable = clean_observable + generator.normal(
                    0.0,
                    noise_sigma * observable_scale,
                    size=clean_observable.shape,
                )
                fit = recover_phase_law_known_surface(
                    noisy_observable,
                    surface,
                    maximum_harmonic=args.density_order,
                    ridge=args.ridge,
                )
                estimate = np.asarray(fit["phase_probability_mass"])
                rows.append(
                    {
                        "profile": profile,
                        "seed": seed,
                        "noise_sigma_relative": noise_sigma,
                        "density_order": args.density_order,
                        "phase_mass_relative_rmse": _rmse(
                            estimate, truth_mass
                        )
                        / mass_scale,
                        "observable_fit_relative_rmse": _rmse(
                            np.asarray(fit["fitted_observable"]), noisy_observable
                        )
                        / observable_scale,
                        "all_requested_harmonics_observable": fit[
                            "all_requested_harmonics_observable"
                        ],
                        "unobservable_harmonic_count": len(
                            fit["unobservable_harmonics"]
                        ),
                        "minimum_requested_transfer_magnitude": fit[
                            "minimum_requested_transfer_magnitude"
                        ],
                        "truth_used_during_recovery": False,
                        "is_official_kpt": False,
                    }
                )
        print(f"Completed known-surface seed {seed}", flush=True)
    return rows


def _run_anchor_benchmark(
    args: argparse.Namespace, warp_orders: list[int], anchor_counts: list[int]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed_offset in range(args.seeds):
        seed = args.base_seed + seed_offset
        for order in warp_orders:
            for count in anchor_counts:
                for placement_idx, placement in enumerate(ANCHOR_PLACEMENTS):
                    if count == 0:
                        anchors = np.empty(0)
                    elif placement == "uniform":
                        anchors = np.linspace(-np.pi, np.pi, count, endpoint=False)
                    else:
                        anchors = np.sort(
                            np.random.default_rng(
                                np.random.SeedSequence(
                                    [seed, order, count, placement_idx + 1, 47]
                                )
                            ).uniform(-np.pi, np.pi, size=count)
                        )
                    diagnostic = phase_anchor_gauge_diagnostics(
                        anchors, warp_order=order
                    )
                    singular = np.asarray(diagnostic["singular_values"])
                    positive = singular[singular > float(diagnostic["rank_tolerance"])]
                    rows.append(
                        {
                            "placement": placement,
                            "seed": seed,
                            "warp_order": order,
                            "anchor_count": count,
                            "warp_parameters": diagnostic["n_warp_parameters"],
                            "anchor_operator_rank": diagnostic[
                                "anchor_operator_rank"
                            ],
                            "residual_gauge_nullity": diagnostic[
                                "residual_gauge_nullity"
                            ],
                            "harmonic_warp_fully_fixed": diagnostic[
                                "harmonic_warp_fully_fixed"
                            ],
                            "minimum_nonzero_singular_value": (
                                float(np.min(positive)) if positive.size else 0.0
                            ),
                            "witness_displacement_rms": diagnostic[
                                "witness_displacement_rms"
                            ],
                            "minimum_warp_derivative": diagnostic[
                                "minimum_warp_derivative"
                            ],
                            "maximum_anchor_displacement": diagnostic[
                                "maximum_anchor_displacement"
                            ],
                            "is_official_kpt": False,
                        }
                    )
        print(f"Completed anchor seed {seed}", flush=True)
    return rows


def _calibration_surface(
    theta: np.ndarray, order: int, phases: np.ndarray, profile: str
) -> np.ndarray:
    if profile == "balanced":
        amplitudes = np.ones(order)
    elif profile == "decaying":
        amplitudes = 0.25 ** np.arange(order)
    elif profile == "missing_highest":
        amplitudes = np.ones(order)
        amplitudes[-1] = 0.0
    else:  # pragma: no cover
        raise ValueError(f"unknown calibration profile: {profile}")
    surface = np.ones(theta.shape)
    for harmonic, (amplitude, phase) in enumerate(
        zip(amplitudes, phases), start=1
    ):
        surface += amplitude * np.cos(harmonic * theta + phase)
    return surface


def _parse_ints(
    value: str, name: str, *, minimum: int, allow_zero: bool = False
) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be comma-separated integers") from exc
    if not parsed or any(item < minimum for item in parsed):
        qualifier = "non-negative" if allow_zero else f"at least {minimum}"
        raise SystemExit(f"{name} values must be {qualifier}")
    return parsed


def _parse_floats(value: str, name: str, *, minimum: float) -> list[float]:
    try:
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be comma-separated numbers") from exc
    if not parsed or any(not np.isfinite(item) or item < minimum for item in parsed):
        raise SystemExit(f"{name} values must be finite and at least {minimum}")
    return parsed


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2)))


def _summarize_known(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = sorted(
        {
            (str(row["profile"]), float(row["noise_sigma_relative"]))
            for row in rows
        }
    )
    output: list[dict[str, object]] = []
    for profile, noise in groups:
        group = [
            row
            for row in rows
            if row["profile"] == profile
            and float(row["noise_sigma_relative"]) == noise
        ]
        errors = np.asarray(
            [float(row["phase_mass_relative_rmse"]) for row in group]
        )
        output.append(
            {
                "profile": profile,
                "noise_sigma_relative": noise,
                "datasets": len(group),
                "phase_mass_relative_rmse_mean": float(np.mean(errors)),
                "phase_mass_relative_rmse_sd": _sample_sd(errors),
                "observable_fit_relative_rmse_mean": float(
                    np.mean(
                        [float(row["observable_fit_relative_rmse"]) for row in group]
                    )
                ),
                "all_requested_harmonics_observable_fraction": float(
                    np.mean(
                        [
                            bool(row["all_requested_harmonics_observable"])
                            for row in group
                        ]
                    )
                ),
                "exact_recovery_fraction": float(np.mean(errors < 1e-8)),
            }
        )
    return output


def _summarize_anchors(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = sorted(
        {
            (
                str(row["placement"]),
                int(row["warp_order"]),
                int(row["anchor_count"]),
            )
            for row in rows
        }
    )
    output: list[dict[str, object]] = []
    for placement, order, count in groups:
        group = [
            row
            for row in rows
            if row["placement"] == placement
            and int(row["warp_order"]) == order
            and int(row["anchor_count"]) == count
        ]
        output.append(
            {
                "placement": placement,
                "warp_order": order,
                "anchor_count": count,
                "datasets": len(group),
                "anchor_operator_rank_mean": float(
                    np.mean([int(row["anchor_operator_rank"]) for row in group])
                ),
                "residual_gauge_nullity_mean": float(
                    np.mean([int(row["residual_gauge_nullity"]) for row in group])
                ),
                "fully_fixed_fraction": float(
                    np.mean([bool(row["harmonic_warp_fully_fixed"]) for row in group])
                ),
                "minimum_nonzero_singular_value_mean": float(
                    np.mean(
                        [
                            float(row["minimum_nonzero_singular_value"])
                            for row in group
                        ]
                    )
                ),
                "maximum_anchor_displacement_max": max(
                    float(row["maximum_anchor_displacement"]) for row in group
                ),
            }
        )
    return output


def _sample_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_text(
    known_summary: list[dict[str, object]],
    anchor_summary: list[dict[str, object]],
) -> str:
    lines = [
        "# KPT identifiability rescue benchmark",
        "",
        "Known calibration surface:",
        "profile | relative noise | phase-law relative RMSE | all target modes observable | exact recovery",
    ]
    for row in known_summary:
        lines.append(
            f"{row['profile']} | {float(row['noise_sigma_relative']):.4g} | "
            f"{float(row['phase_mass_relative_rmse_mean']):.4f} | "
            f"{float(row['all_requested_harmonics_observable_fraction']):.3f} | "
            f"{float(row['exact_recovery_fraction']):.3f}"
        )
    lines.extend(
        [
            "",
            "Anchor gauge fixing thresholds (first anchor count with fully-fixed fraction 1):",
            "placement | warp order | anchors required in tested grid",
        ]
    )
    for placement in ANCHOR_PLACEMENTS:
        orders = sorted(
            {int(row["warp_order"]) for row in anchor_summary if row["placement"] == placement}
        )
        for order in orders:
            candidates = [
                row
                for row in anchor_summary
                if row["placement"] == placement
                and int(row["warp_order"]) == order
                and float(row["fully_fixed_fraction"]) == 1.0
            ]
            threshold = min(int(row["anchor_count"]) for row in candidates) if candidates else "not reached"
            lines.append(f"{placement} | {order} | {threshold}")
    lines.extend(
        [
            "",
            "Known-factor recovery succeeds only on target modes where the calibration surface has nonzero spectral transfer.",
            "Anchor counts are dimension bounds, not guarantees: placement controls rank and conditioning.",
            "These finite-model rescue results do not establish identifiability of unrestricted KPT.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_figure(
    path: Path,
    known_summary: list[dict[str, object]],
    anchor_summary: list[dict[str, object]],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for profile, color in zip(
        CALIBRATION_PROFILES, ("#2f6690", "#d17c0f", "#b23a48")
    ):
        group = sorted(
            [row for row in known_summary if row["profile"] == profile],
            key=lambda row: float(row["noise_sigma_relative"]),
        )
        axes[0].plot(
            [max(float(row["noise_sigma_relative"]), 1e-6) for row in group],
            [
                max(float(row["phase_mass_relative_rmse_mean"]), 1e-12)
                for row in group
            ],
            marker="o",
            label=profile,
            color=color,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("relative observable noise")
    axes[0].set_ylabel("phase-law relative RMSE")
    axes[0].set_title("Known calibration surface")
    axes[0].legend(fontsize=8)

    for axis, placement in zip(axes[1:], ANCHOR_PLACEMENTS):
        for order in sorted(
            {int(row["warp_order"]) for row in anchor_summary}
        ):
            group = sorted(
                [
                    row
                    for row in anchor_summary
                    if row["placement"] == placement
                    and int(row["warp_order"]) == order
                ],
                key=lambda row: int(row["anchor_count"]),
            )
            axis.plot(
                [int(row["anchor_count"]) for row in group],
                [float(row["residual_gauge_nullity_mean"]) for row in group],
                marker="o",
                label=f"warp order {order}",
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("phase anchors")
        axis.set_ylabel("residual gauge nullity")
        axis.set_title(f"{placement} anchor placement")
        axis.legend(fontsize=8)
    fig.suptitle("Finite-model conditions that rescue KPT identifiability")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
