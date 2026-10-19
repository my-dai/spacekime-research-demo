#!/usr/bin/env python3
"""Multi-seed smooth latent-phase and sparse-anchor identifiability benchmark."""

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
from spacekime_blind import (
    align_circular_phase,
    estimate_blind_repeat_phase,
    truth_grid_to_estimated_coordinate,
)
from spacekime_paths import PROJECT_ROOT, portable_manifest_path
from spacekime_phase_dynamics import (
    estimate_smooth_phase_trajectories,
    nested_phase_anchor_masks,
    phase_trajectory_alignment_metrics,
    simulate_smooth_phase_repeats,
)
from spacekime_synthetic import latent_kimesurface, make_theta_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=74_019)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--n-time", type=int, default=32)
    parser.add_argument("--n-theta", type=int, default=48)
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--drift-scale", type=float, default=0.45)
    parser.add_argument("--samplers", default="uniform,vonmises")
    parser.add_argument("--anchors-per-repeat", default="0,1,2,4")
    parser.add_argument("--phase-grid", type=int, default=48)
    parser.add_argument("--phase-iterations", type=int, default=3)
    parser.add_argument("--kappa", type=float, default=8.0)
    parser.add_argument("--transition-strength", type=float, default=2.0)
    parser.add_argument("--max-step-rad", type=float, default=0.65)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "phase_dynamics_benchmark",
    )
    args = parser.parse_args()
    samplers = _choices(args.samplers, ("uniform", "vonmises"))
    anchor_counts = sorted({int(value) for value in args.anchors_per_repeat.split(",")})
    if not anchor_counts or anchor_counts[0] < 0 or anchor_counts[-1] > args.n_time:
        raise SystemExit("anchors-per-repeat must be integers between 0 and n-time")
    if args.seeds < 2 or args.repeats < 8 or args.n_time < 8:
        raise SystemExit("seeds >= 2 and repeats/n-time >= 8 are required")
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
                    np.random.SeedSequence([seed, sampler_idx + 1, 7331])
                ),
                phase_sampler=sampler,
            )
            truth_surface = latent_kimesurface(simulation["time"], theta_grid)
            scale = max(float(np.std(truth_surface)), 1e-12)
            time_mean_surface = np.broadcast_to(
                np.mean(simulation["observations"], axis=0)[:, None],
                truth_surface.shape,
            )
            oracle_surface = reconstruct_advanced_surface(
                simulation["phase"],
                simulation["observations"],
                simulation["time"],
                theta_grid,
                "kernel",
                {"kappa": args.kappa},
            )

            constant_fit = estimate_blind_repeat_phase(
                simulation["observations"],
                n_phase_grid=args.phase_grid,
                kappa=args.kappa,
                max_iterations=4,
            )
            constant_phase = np.broadcast_to(
                np.asarray(constant_fit["phase"])[:, None], simulation["phase"].shape
            )
            constant_alignment = phase_trajectory_alignment_metrics(
                constant_phase, simulation["phase"]
            )
            constant_surface = _aligned_surface(
                constant_phase,
                simulation["observations"],
                simulation["time"],
                theta_grid,
                constant_alignment,
                args.kappa,
            )
            constant_nrmse = _rmse(constant_surface, truth_surface) / scale

            masks = nested_phase_anchor_masks(
                args.repeats,
                args.n_time,
                max(anchor_counts),
                rng=np.random.default_rng(
                    np.random.SeedSequence([seed, sampler_idx + 1, 9441])
                ),
            )
            for anchor_count in anchor_counts:
                mask = masks[anchor_count]
                fit = estimate_smooth_phase_trajectories(
                    simulation["observations"],
                    simulation["time"],
                    anchor_phase=simulation["phase"] if anchor_count else None,
                    anchor_mask=mask if anchor_count else None,
                    n_phase_grid=args.phase_grid,
                    kappa=args.kappa,
                    transition_strength=args.transition_strength,
                    max_step_rad=args.max_step_rad,
                    max_iterations=args.phase_iterations,
                )
                alignment = phase_trajectory_alignment_metrics(
                    np.asarray(fit["phase"]), simulation["phase"]
                )
                surface = _aligned_surface(
                    np.asarray(fit["phase"]),
                    simulation["observations"],
                    simulation["time"],
                    theta_grid,
                    alignment,
                    args.kappa,
                )
                surface_nrmse = _rmse(surface, truth_surface) / scale
                rows.append(
                    {
                        "sampler": sampler,
                        "seed": seed,
                        "anchors_per_repeat": anchor_count,
                        "anchor_fraction": anchor_count / args.n_time,
                        "repeats": args.repeats,
                        "n_time": args.n_time,
                        "drift_scale": args.drift_scale,
                        "phase_circular_rmse": alignment["circular_rmse"],
                        "phase_circular_mae": alignment["circular_mae"],
                        "phase_step_rmse": alignment["step_rmse"],
                        "phase_resultant_coherence": alignment["resultant_coherence"],
                        "dynamic_surface_nrmse": surface_nrmse,
                        "constant_blind_phase_rmse": constant_alignment["circular_rmse"],
                        "constant_blind_surface_nrmse": constant_nrmse,
                        "oracle_surface_nrmse": _rmse(oracle_surface, truth_surface) / scale,
                        "time_mean_surface_nrmse": _rmse(time_mean_surface, truth_surface) / scale,
                        "dynamic_beats_constant": surface_nrmse < constant_nrmse,
                        "declared_anchor_cells": int(np.sum(mask)),
                        "phase_truth_used_beyond_declared_anchors": False,
                        "is_official_kpt": False,
                    }
                )
            print(f"Completed {sampler} seed {seed}", flush=True)

    summary = _summarize(rows)
    _write_csv(args.output_dir / "phase_dynamics_runs.csv", rows)
    _write_csv(args.output_dir / "phase_dynamics_summary.csv", summary)
    (args.output_dir / "phase_dynamics_summary.txt").write_text(
        _summary_text(summary), encoding="utf-8"
    )
    manifest = {
        "experiment": "smooth_latent_phase_with_declared_sparse_anchors",
        "parameters": {
            **vars(args),
            "output_dir": portable_manifest_path(args.output_dir),
        },
        "summary": summary,
        "scientific_limits": [
            "The simulator assumes one low-dimensional smooth phase path per repeat.",
            "Anchor values are declared observed phase labels; truth is not used beyond those cells.",
            "Global rotation/reflection alignment uses truth only after estimation for synthetic evaluation.",
            "This is an identifiability stress test and not official KPT.",
            "Real-fMRI H(t,theta) remains an observed spatial phase-distribution proxy.",
        ],
    }
    (args.output_dir / "phase_dynamics_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if plt is not None:
        _save_figure(args.output_dir / "phase_dynamics_benchmark.png", summary, samplers)
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
        {(str(row["sampler"]), int(row["anchors_per_repeat"])) for row in rows}
    )
    metrics = (
        "phase_circular_rmse",
        "phase_step_rmse",
        "dynamic_surface_nrmse",
        "constant_blind_surface_nrmse",
        "oracle_surface_nrmse",
        "time_mean_surface_nrmse",
    )
    output: list[dict[str, object]] = []
    for sampler, anchors in groups:
        group = [
            row
            for row in rows
            if row["sampler"] == sampler and row["anchors_per_repeat"] == anchors
        ]
        summary: dict[str, object] = {
            "sampler": sampler,
            "anchors_per_repeat": anchors,
            "anchor_fraction": float(group[0]["anchor_fraction"]),
            "datasets": len(group),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary["dynamic_beats_constant_fraction"] = float(
            np.mean([bool(row["dynamic_beats_constant"]) for row in group])
        )
        output.append(summary)
    return output


def _summary_text(summary: list[dict[str, object]]) -> str:
    lines = [
        "# Smooth latent phase / sparse-anchor benchmark",
        "",
        "sampler | anchors/repeat | phase RMSE | dynamic/constant/oracle/time-mean surface NRMSE | P(dynamic<constant)",
    ]
    for row in summary:
        lines.append(
            f"{row['sampler']} | {row['anchors_per_repeat']} | "
            f"{float(row['phase_circular_rmse_mean']):.3f} | "
            f"{float(row['dynamic_surface_nrmse_mean']):.3f}/"
            f"{float(row['constant_blind_surface_nrmse_mean']):.3f}/"
            f"{float(row['oracle_surface_nrmse_mean']):.3f}/"
            f"{float(row['time_mean_surface_nrmse_mean']):.3f} | "
            f"{float(row['dynamic_beats_constant_fraction']):.3f}"
        )
    lines.extend(
        [
            "",
            "Anchor values are declared observed labels; truth is not used beyond those cells.",
            "Global O(2) alignment uses truth only for final synthetic evaluation.",
            "This smooth-path baseline is not official KPT and is not applied to real fMRI.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_figure(path: Path, summary, samplers) -> None:
    fig, axes = plt.subplots(len(samplers), 2, figsize=(11, 4 * len(samplers)), squeeze=False)
    for row_idx, sampler in enumerate(samplers):
        group = [row for row in summary if row["sampler"] == sampler]
        x = [int(row["anchors_per_repeat"]) for row in group]
        axes[row_idx, 0].errorbar(
            x,
            [float(row["phase_circular_rmse_mean"]) for row in group],
            yerr=[float(row["phase_circular_rmse_sd"]) for row in group],
            marker="o",
        )
        axes[row_idx, 0].set_title(f"{sampler}: phase path recovery")
        axes[row_idx, 0].set_xlabel("anchors per repeat")
        axes[row_idx, 0].set_ylabel("circular phase RMSE")
        axes[row_idx, 1].errorbar(
            x,
            [float(row["dynamic_surface_nrmse_mean"]) for row in group],
            yerr=[float(row["dynamic_surface_nrmse_sd"]) for row in group],
            marker="o",
            label="dynamic phase",
        )
        axes[row_idx, 1].plot(
            x,
            [float(row["constant_blind_surface_nrmse_mean"]) for row in group],
            "--",
            label="constant blind",
        )
        axes[row_idx, 1].plot(
            x,
            [float(row["time_mean_surface_nrmse_mean"]) for row in group],
            ":",
            label="time mean",
        )
        axes[row_idx, 1].set_title(f"{sampler}: latent-surface audit")
        axes[row_idx, 1].set_xlabel("anchors per repeat")
        axes[row_idx, 1].set_ylabel("surface NRMSE")
        axes[row_idx, 1].legend()
    fig.suptitle("Smooth latent phase with declared sparse anchors (synthetic only)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
