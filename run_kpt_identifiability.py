#!/usr/bin/env python3
"""Scan rank and noise stability in a finite-basis KPT analog."""

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
    operator_diagnostics,
    quadrature_probe_operator,
    random_positive_fourier_coefficients,
    recover_fourier_coefficients,
    simulate_probe_measurements,
)
from spacekime_paths import PROJECT_ROOT, portable_manifest_path


AVAILABLE_DESIGNS = (
    "single_quadrature",
    "paired_quadrature",
    "near_collinear",
    "three_quadrature",
    "paired_missing_highest",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default="2,4,6")
    parser.add_argument("--designs", default=",".join(AVAILABLE_DESIGNS))
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=809_026)
    parser.add_argument("--repeats", default="1,8,32")
    parser.add_argument("--noise-sigmas", default="0,0.02,0.08")
    parser.add_argument("--density-grid-size", type=int, default=512)
    parser.add_argument("--constraint-grid-size", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1e-10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "kpt_identifiability",
    )
    args = parser.parse_args()

    orders = _parse_ints(args.orders, "orders", minimum=2)
    repeats = _parse_ints(args.repeats, "repeats", minimum=1)
    noise_sigmas = _parse_floats(args.noise_sigmas, "noise-sigmas", minimum=0.0)
    designs = [item.strip() for item in args.designs.split(",") if item.strip()]
    unknown_designs = sorted(set(designs).difference(AVAILABLE_DESIGNS))
    if not designs or unknown_designs:
        raise SystemExit(
            "designs must contain only: " + ", ".join(AVAILABLE_DESIGNS)
        )
    if args.seeds < 1:
        raise SystemExit("seeds must be at least 1")
    if args.density_grid_size < 64:
        raise SystemExit("density-grid-size must be at least 64")
    if args.constraint_grid_size < 4 * max(orders) + 1:
        raise SystemExit("constraint-grid-size must be at least 4 * max(order) + 1")
    if args.ridge < 0 or not np.isfinite(args.ridge):
        raise SystemExit("ridge must be finite and non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    theta = np.linspace(
        -np.pi, np.pi, args.density_grid_size, endpoint=False, dtype=np.float64
    )
    rows: list[dict[str, object]] = []
    design_metadata: dict[str, dict[str, object]] = {}

    for order in orders:
        operators = {
            design: _build_design(order, design) for design in designs
        }
        diagnostics = {
            design: operator_diagnostics(operator)
            for design, operator in operators.items()
        }
        for design in designs:
            diagnostic = diagnostics[design]
            design_metadata[f"order_{order}:{design}"] = {
                key: value
                for key, value in diagnostic.items()
                if key != "singular_values"
            }
            design_metadata[f"order_{order}:{design}"]["singular_values"] = (
                np.asarray(diagnostic["singular_values"]).tolist()
            )

        for seed_offset in range(args.seeds):
            seed = args.base_seed + seed_offset
            coefficients = random_positive_fourier_coefficients(
                order,
                rng=np.random.default_rng(
                    np.random.SeedSequence([seed, order, 41])
                ),
            )
            truth_density = fourier_density(theta, coefficients)
            density_scale = 1.0 / (2.0 * np.pi)
            for design_idx, design in enumerate(designs):
                operator = operators[design]
                diagnostic = diagnostics[design]
                for repeat_count in repeats:
                    for noise_idx, noise_sigma in enumerate(noise_sigmas):
                        simulation = simulate_probe_measurements(
                            operator,
                            coefficients,
                            n_repeats=repeat_count,
                            noise_sigma=noise_sigma,
                            rng=np.random.default_rng(
                                np.random.SeedSequence(
                                    [
                                        seed,
                                        order,
                                        design_idx + 1,
                                        repeat_count,
                                        noise_idx + 1,
                                    ]
                                )
                            ),
                        )
                        fit = recover_fourier_coefficients(
                            operator,
                            simulation["mean"],
                            ridge=args.ridge,
                            constraint_grid_size=args.constraint_grid_size,
                        )
                        estimate = np.asarray(fit["coefficients"])
                        estimated_density = fourier_density(theta, estimate)
                        rows.append(
                            {
                                "order": order,
                                "design": design,
                                "seed": seed,
                                "n_repeats": repeat_count,
                                "noise_sigma": noise_sigma,
                                "mean_noise_standard_error": noise_sigma
                                / np.sqrt(repeat_count),
                                "n_measurements": diagnostic["n_measurements"],
                                "n_parameters": diagnostic["n_parameters"],
                                "operator_rank": diagnostic["rank"],
                                "operator_nullity": diagnostic["nullity"],
                                "operator_full_column_rank": diagnostic[
                                    "full_column_rank"
                                ],
                                "operator_condition_number": diagnostic[
                                    "condition_number"
                                ],
                                "operator_smallest_singular_value": diagnostic[
                                    "smallest_singular_value"
                                ],
                                "coefficient_rmse": _rmse(
                                    estimate, coefficients
                                ),
                                "density_relative_rmse": _rmse(
                                    estimated_density, truth_density
                                )
                                / density_scale,
                                "estimated_density_min": float(
                                    np.min(estimated_density)
                                ),
                                "measurement_rmse": fit["measurement_rmse"],
                                "optimizer_success": fit["optimizer_success"],
                                "truth_used_during_recovery": False,
                                "is_official_kpt": False,
                            }
                        )
            print(f"Completed order {order} seed {seed}", flush=True)

    summary = _summarize(rows)
    _write_csv(args.output_dir / "kpt_identifiability_runs.csv", rows)
    _write_csv(args.output_dir / "kpt_identifiability_summary.csv", summary)
    (args.output_dir / "kpt_identifiability_summary.txt").write_text(
        _summary_text(summary), encoding="utf-8"
    )
    (args.output_dir / "kpt_identifiability_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "experiment": "finite_basis_phase_density_identifiability",
                "representation": IDENTIFIABILITY_REPRESENTATION,
                "parameters": {
                    **vars(args),
                    "output_dir": portable_manifest_path(args.output_dir),
                },
                "operator_diagnostics": design_metadata,
                "summary": summary,
                "scientific_limits": [
                    "This is a synthetic finite-Fourier inverse-problem analog, not the official SOCR KPT algorithm.",
                    "Rotated harmonic expectation probes are explicitly specified; they are not claimed to be the exact KPT measurement bases.",
                    "Full matrix rank is sufficient only for this finite linear coefficient model and does not prove nonparametric KPT uniqueness.",
                    "Synthetic truth is used only for final recovery error metrics.",
                    "MRI acquisition phase and latent kime phase are not assumed to be equivalent.",
                ],
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if plt is not None:
        _save_figure(args.output_dir / "kpt_identifiability.png", summary, max(orders))
    return 0


def _build_design(order: int, design: str) -> np.ndarray:
    if design == "single_quadrature":
        return quadrature_probe_operator(order, [0.0])
    if design == "paired_quadrature":
        return quadrature_probe_operator(order, [0.0, np.pi / 2.0])
    if design == "near_collinear":
        return quadrature_probe_operator(order, [0.0, 0.05])
    if design == "three_quadrature":
        return quadrature_probe_operator(
            order, [0.0, np.pi / 3.0, 2.0 * np.pi / 3.0]
        )
    if design == "paired_missing_highest":
        return quadrature_probe_operator(
            order, [0.0, np.pi / 2.0], harmonics=range(1, order)
        )
    raise ValueError(f"unknown design: {design}")


def _parse_ints(value: str, name: str, *, minimum: int) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{name} must be comma-separated integers") from exc
    if not parsed or any(item < minimum for item in parsed):
        raise SystemExit(f"{name} values must be at least {minimum}")
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


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                int(row["order"]),
                str(row["design"]),
                int(row["n_repeats"]),
                float(row["noise_sigma"]),
            )
            for row in rows
        }
    )
    output: list[dict[str, object]] = []
    for order, design, repeat_count, noise_sigma in keys:
        group = [
            row
            for row in rows
            if int(row["order"]) == order
            and str(row["design"]) == design
            and int(row["n_repeats"]) == repeat_count
            and float(row["noise_sigma"]) == noise_sigma
        ]
        coefficient_error = np.asarray(
            [float(row["coefficient_rmse"]) for row in group]
        )
        density_error = np.asarray(
            [float(row["density_relative_rmse"]) for row in group]
        )
        first = group[0]
        output.append(
            {
                "order": order,
                "design": design,
                "n_repeats": repeat_count,
                "noise_sigma": noise_sigma,
                "mean_noise_standard_error": noise_sigma / np.sqrt(repeat_count),
                "datasets": len(group),
                "operator_rank": first["operator_rank"],
                "operator_nullity": first["operator_nullity"],
                "operator_full_column_rank": first["operator_full_column_rank"],
                "operator_condition_number": first["operator_condition_number"],
                "coefficient_rmse_mean": float(np.mean(coefficient_error)),
                "coefficient_rmse_sd": _sample_sd(coefficient_error),
                "density_relative_rmse_mean": float(np.mean(density_error)),
                "density_relative_rmse_sd": _sample_sd(density_error),
                "optimizer_success_fraction": float(
                    np.mean([bool(row["optimizer_success"]) for row in group])
                ),
                "exact_recovery_fraction": float(
                    np.mean(coefficient_error < 1e-6)
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


def _summary_text(summary: list[dict[str, object]]) -> str:
    zero_noise = [
        row
        for row in summary
        if float(row["noise_sigma"]) == 0.0 and int(row["n_repeats"]) == 1
    ]
    lines = [
        "# Finite-basis phase-density identifiability laboratory",
        "",
        "order | design | rank/nullity | condition | exact noiseless recovery | density relative RMSE",
    ]
    for row in zero_noise:
        condition = row["operator_condition_number"]
        condition_text = "rank-deficient" if condition is None else f"{float(condition):.3g}"
        lines.append(
            f"{row['order']} | {row['design']} | {row['operator_rank']}/{row['operator_nullity']} | "
            f"{condition_text} | {float(row['exact_recovery_fraction']):.3f} | "
            f"{float(row['density_relative_rmse_mean']):.4f}"
        )
    lines.extend(
        [
            "",
            "Rank-deficient designs can fail even without noise. Full rank does not imply stability: near-collinear probes amplify noise according to their condition number.",
            "This is an explicit synthetic inverse-problem analog and is not an implementation or validation of official KPT.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_figure(
    path: Path, summary: list[dict[str, object]], maximum_order: int
) -> None:
    selected = [row for row in summary if int(row["order"]) == maximum_order]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    labels = list(dict.fromkeys(str(row["design"]) for row in selected))
    diagnostics = [
        next(row for row in selected if str(row["design"]) == label)
        for label in labels
    ]
    condition_values = [
        float(row["operator_condition_number"])
        if row["operator_condition_number"] is not None
        else np.nan
        for row in diagnostics
    ]
    finite = [value for value in condition_values if np.isfinite(value)]
    ceiling = max(finite, default=1.0) * 1.35
    bars = axes[0].bar(
        np.arange(len(labels)),
        [value if np.isfinite(value) else ceiling for value in condition_values],
        color=["#2f6690" if np.isfinite(value) else "#b23a48" for value in condition_values],
    )
    for bar, value, diagnostic in zip(bars, condition_values, diagnostics):
        if not np.isfinite(value):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                np.sqrt(ceiling),
                f"rank-def.\nnullity={diagnostic['operator_nullity']}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white",
                fontweight="bold",
            )
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.8, ceiling * 1.08)
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    axes[0].set_ylabel("condition number")
    axes[0].set_title(f"Operator diagnostics (order {maximum_order})")

    for label in labels:
        group = [
            row
            for row in selected
            if str(row["design"]) == label and float(row["noise_sigma"]) > 0
        ]
        grouped: dict[float, list[float]] = {}
        for row in group:
            grouped.setdefault(float(row["mean_noise_standard_error"]), []).append(
                float(row["density_relative_rmse_mean"])
            )
        x_values = sorted(grouped)
        y_values = [float(np.mean(grouped[value])) for value in x_values]
        axes[1].plot(x_values, y_values, marker="o", label=label)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("noise sigma / sqrt(repeats)")
    axes[1].set_ylabel("density relative RMSE")
    axes[1].set_title(f"Noise stability (order {maximum_order})")
    axes[1].legend(fontsize=8)
    fig.suptitle("Finite-basis circular density identifiability (synthetic analog)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
