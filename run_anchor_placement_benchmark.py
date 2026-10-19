#!/usr/bin/env python3
"""Multi-seed observation-only phase-anchor placement benchmark."""

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

from spacekime_advanced import reconstruct_advanced_surface
from spacekime_blind import truth_grid_to_estimated_coordinate
from spacekime_paths import PROJECT_ROOT, portable_manifest_path
from spacekime_phase_dynamics import (
    designed_phase_anchor_masks,
    estimate_smooth_phase_trajectories,
    phase_trajectory_alignment_metrics,
    simulate_smooth_phase_repeats,
)
from spacekime_synthetic import latent_kimesurface, make_theta_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=76_019)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--n-time", type=int, default=32)
    parser.add_argument("--n-theta", type=int, default=48)
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--drift-scale", type=float, default=0.45)
    parser.add_argument("--samplers", default="uniform,vonmises")
    parser.add_argument(
        "--strategies",
        default=(
            "random,uniform,endpoints,ambiguity,ambiguity_spread,"
            "surface_variance,surface_variance_spread"
        ),
    )
    parser.add_argument("--anchors-per-repeat", default="1,2,4")
    parser.add_argument("--phase-grid", type=int, default=48)
    parser.add_argument("--phase-iterations", type=int, default=3)
    parser.add_argument("--kappa", type=float, default=8.0)
    parser.add_argument("--transition-strength", type=float, default=2.0)
    parser.add_argument("--max-step-rad", type=float, default=0.65)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "anchor_placement_benchmark_phase5",
    )
    args = parser.parse_args()
    samplers = _choices(args.samplers, ("uniform", "vonmises"))
    strategies = _choices(
        args.strategies,
        (
            "random",
            "uniform",
            "endpoints",
            "ambiguity",
            "ambiguity_spread",
            "surface_variance",
            "surface_variance_spread",
        ),
    )
    anchor_counts = sorted({int(value) for value in args.anchors_per_repeat.split(",")})
    if not anchor_counts or anchor_counts[0] < 1 or anchor_counts[-1] > args.n_time:
        raise SystemExit("anchors-per-repeat must be integers between 1 and n-time")
    if "random" not in strategies:
        raise SystemExit("strategies must include random for paired contrasts")
    theta_grid = make_theta_grid(args.n_theta)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for sampler_idx, sampler in enumerate(samplers):
        for seed_offset in range(args.seeds):
            seed = args.base_seed + seed_offset
            simulation = simulate_smooth_phase_repeats(
                n_repeats=args.repeats,
                n_time=args.n_time,
                noise_sigma=args.noise_sigma,
                drift_scale=args.drift_scale,
                rng=np.random.default_rng(
                    np.random.SeedSequence([seed, sampler_idx + 1, 4111])
                ),
                phase_sampler=sampler,
            )
            truth_surface = latent_kimesurface(simulation["time"], theta_grid)
            scale = max(float(np.std(truth_surface)), 1e-12)
            no_anchor_fit = estimate_smooth_phase_trajectories(
                simulation["observations"],
                simulation["time"],
                n_phase_grid=args.phase_grid,
                kappa=args.kappa,
                transition_strength=args.transition_strength,
                max_step_rad=args.max_step_rad,
                max_iterations=args.phase_iterations,
            )
            no_anchor_alignment = phase_trajectory_alignment_metrics(
                np.asarray(no_anchor_fit["phase"]), simulation["phase"]
            )
            no_anchor_surface = _aligned_surface(
                np.asarray(no_anchor_fit["phase"]),
                simulation["observations"],
                simulation["time"],
                theta_grid,
                no_anchor_alignment,
                args.kappa,
            )
            no_anchor_oracle_o2_nrmse = _rmse(no_anchor_surface, truth_surface) / scale
            dataset_rows: list[dict[str, object]] = []
            for strategy_idx, strategy in enumerate(strategies):
                masks = designed_phase_anchor_masks(
                    args.repeats,
                    args.n_time,
                    max(anchor_counts),
                    strategy,
                    rng=np.random.default_rng(
                        np.random.SeedSequence(
                            [seed, sampler_idx + 1, strategy_idx + 1, 6221]
                        )
                    ),
                    ambiguity=(
                        np.asarray(no_anchor_fit["phase_ambiguity"])
                        if strategy.startswith("ambiguity")
                        else None
                    ),
                    downstream_variance=(
                        np.asarray(no_anchor_fit["downstream_surface_variance"])
                        if strategy.startswith("surface_variance")
                        else None
                    ),
                )
                for anchor_count in anchor_counts:
                    mask = masks[anchor_count]
                    fit = estimate_smooth_phase_trajectories(
                        simulation["observations"],
                        simulation["time"],
                        anchor_phase=np.where(mask, simulation["phase"], np.nan),
                        anchor_mask=mask,
                        n_phase_grid=args.phase_grid,
                        kappa=args.kappa,
                        transition_strength=args.transition_strength,
                        max_step_rad=args.max_step_rad,
                        max_iterations=args.phase_iterations,
                    )
                    alignment = phase_trajectory_alignment_metrics(
                        np.asarray(fit["phase"]), simulation["phase"]
                    )
                    absolute_surface = reconstruct_advanced_surface(
                        np.asarray(fit["phase"]),
                        simulation["observations"],
                        simulation["time"],
                        theta_grid,
                        "kernel",
                        {"kappa": float(args.kappa)},
                    )
                    oracle_o2_surface = _aligned_surface(
                        np.asarray(fit["phase"]),
                        simulation["observations"],
                        simulation["time"],
                        theta_grid,
                        alignment,
                        args.kappa,
                    )
                    dataset_rows.append(
                        {
                            "sampler": sampler,
                            "seed": seed,
                            "strategy": strategy,
                            "anchors_per_repeat": anchor_count,
                            "anchor_fraction": anchor_count / args.n_time,
                            "phase_circular_rmse": alignment["circular_rmse"],
                            "phase_step_rmse": alignment["step_rmse"],
                            "surface_nrmse": _rmse(absolute_surface, truth_surface) / scale,
                            "absolute_anchor_surface_nrmse": _rmse(
                                absolute_surface, truth_surface
                            )
                            / scale,
                            "oracle_o2_aligned_surface_nrmse": _rmse(
                                oracle_o2_surface, truth_surface
                            )
                            / scale,
                            "no_anchor_oracle_o2_surface_nrmse": no_anchor_oracle_o2_nrmse,
                            "selected_ambiguity_mean": float(
                                np.mean(np.asarray(no_anchor_fit["phase_ambiguity"])[mask])
                            ),
                            "selected_downstream_variance_mean": float(
                                np.mean(
                                    np.asarray(
                                        no_anchor_fit["downstream_surface_variance"]
                                    )[mask]
                                )
                            ),
                            "anchor_locations_selected_with_truth": False,
                            "phase_truth_used_beyond_declared_anchors": False,
                            "truth_used_for_primary_surface_coordinate": False,
                            "truth_used_for_oracle_o2_metric": True,
                            "is_official_kpt": False,
                        }
                    )
            random_lookup = {
                int(row["anchors_per_repeat"]): float(row["surface_nrmse"])
                for row in dataset_rows
                if row["strategy"] == "random"
            }
            for row in dataset_rows:
                random_value = random_lookup[int(row["anchors_per_repeat"])]
                row["paired_surface_delta_vs_random"] = (
                    float(row["surface_nrmse"]) - random_value
                )
                row["beats_random"] = float(row["surface_nrmse"]) < random_value
            rows.extend(dataset_rows)
            print(f"Completed {sampler} seed {seed}", flush=True)

    summary = _summarize(rows)
    _write_csv(args.output_dir / "anchor_placement_runs.csv", rows)
    _write_csv(args.output_dir / "anchor_placement_summary.csv", summary)
    (args.output_dir / "anchor_placement_summary.txt").write_text(
        _summary_text(summary), encoding="utf-8"
    )
    manifest = {
        "experiment": "observation_only_phase_anchor_location_design",
        "parameters": {
            **vars(args),
            "output_dir": portable_manifest_path(args.output_dir),
        },
        "summary": summary,
        "scientific_limits": [
            "Anchor locations are selected without true phase; only selected cells reveal synthetic phase labels.",
            "Ambiguity is an energy-based pseudo-posterior circular variance, not calibrated Bayesian uncertainty.",
            "Global O(2) alignment uses truth only for post-estimation synthetic evaluation.",
            "Primary surface NRMSE stays in the absolute coordinate fixed by declared anchors; oracle O(2)-aligned NRMSE is a separate sensitivity metric.",
            "Surface-variance acquisition is observation-only LOO predictive variance under an uncalibrated path pseudo-posterior.",
            "The smooth-path estimator is not official KPT and is not applied to real fMRI.",
            "Real-fMRI H(t,theta) remains an observed spatial phase-distribution proxy.",
        ],
    }
    (args.output_dir / "anchor_placement_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if plt is not None:
        _save_figure(args.output_dir / "anchor_placement_benchmark.png", summary, samplers)
    return 0


def _aligned_surface(phase, observations, time, theta_grid, alignment, kappa):
    query = truth_grid_to_estimated_coordinate(
        theta_grid,
        sign=float(alignment["sign"]),
        offset=float(alignment["offset"]),
    )
    return reconstruct_advanced_surface(
        phase, observations, time, query, "kernel", {"kappa": float(kappa)}
    )


def _rmse(left, right) -> float:
    return float(np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2)))


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
            (str(row["sampler"]), str(row["strategy"]), int(row["anchors_per_repeat"]))
            for row in rows
        }
    )
    metrics = (
        "phase_circular_rmse",
        "phase_step_rmse",
        "surface_nrmse",
        "oracle_o2_aligned_surface_nrmse",
        "paired_surface_delta_vs_random",
        "selected_ambiguity_mean",
        "selected_downstream_variance_mean",
    )
    output: list[dict[str, object]] = []
    for sampler, strategy, anchors in groups:
        group = [
            row
            for row in rows
            if row["sampler"] == sampler
            and row["strategy"] == strategy
            and row["anchors_per_repeat"] == anchors
        ]
        item: dict[str, object] = {
            "sampler": sampler,
            "strategy": strategy,
            "anchors_per_repeat": anchors,
            "datasets": len(group),
            "beats_random_fraction": float(np.mean([bool(row["beats_random"]) for row in group])),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(item)
    return output


def _summary_text(summary: list[dict[str, object]]) -> str:
    lines = [
        "# Observation-only phase-anchor placement benchmark",
        "",
        "sampler | strategy | anchors | phase RMSE | absolute/oracle-O2 surface NRMSE | delta vs random | P(beat random)",
    ]
    for row in summary:
        lines.append(
            f"{row['sampler']} | {row['strategy']} | {row['anchors_per_repeat']} | "
            f"{float(row['phase_circular_rmse_mean']):.3f} | "
            f"{float(row['surface_nrmse_mean']):.3f}/"
            f"{float(row['oracle_o2_aligned_surface_nrmse_mean']):.3f} | "
            f"{float(row['paired_surface_delta_vs_random_mean']):+.3f} | "
            f"{float(row['beats_random_fraction']):.3f}"
        )
    lines.extend(
        [
            "",
            "Negative paired deltas favor the named strategy over random anchors.",
            "Anchor locations never use truth; selected cells reveal phase labels only after selection.",
            "Ambiguity is a pseudo-posterior design score, not calibrated uncertainty.",
            "Primary surface NRMSE uses the coordinate fixed by declared anchors; oracle O(2) alignment is reported separately.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_figure(path: Path, summary, samplers) -> None:
    fig, axes = plt.subplots(1, len(samplers), figsize=(7 * len(samplers), 5), squeeze=False)
    for column, sampler in enumerate(samplers):
        group = [row for row in summary if row["sampler"] == sampler]
        for strategy in sorted({str(row["strategy"]) for row in group}):
            values = [row for row in group if row["strategy"] == strategy]
            axes[0, column].plot(
                [int(row["anchors_per_repeat"]) for row in values],
                [float(row["surface_nrmse_mean"]) for row in values],
                marker="o",
                label=strategy,
            )
        axes[0, column].set_title(f"{sampler}: anchor placement")
        axes[0, column].set_xlabel("anchors per repeat")
        axes[0, column].set_ylabel("latent-surface NRMSE")
        axes[0, column].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
