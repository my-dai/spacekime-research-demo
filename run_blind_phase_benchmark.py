#!/usr/bin/env python3
"""Multi-seed benchmark for an observation-only synthetic latent-phase baseline."""

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
    reconstruct_advanced_surface,
    select_reconstruction_config,
)
from spacekime_blind import (
    align_circular_phase,
    estimate_blind_repeat_phase,
    simulate_constant_phase_repeats,
    truth_grid_to_estimated_coordinate,
)
from spacekime_paths import PROJECT_ROOT, portable_manifest_path
from spacekime_synthetic import latent_kimesurface, make_theta_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=72_019)
    parser.add_argument("--repeats", type=int, default=48)
    parser.add_argument("--n-time", type=int, default=40)
    parser.add_argument("--n-theta", type=int, default=64)
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--samplers", default="uniform,vonmises")
    parser.add_argument("--phase-grid", type=int, default=64)
    parser.add_argument("--phase-iterations", type=int, default=5)
    parser.add_argument("--phase-kappa", type=float, default=8.0)
    parser.add_argument("--surface-kappas", default="2,4,8,16,32,64")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "output" / "blind_phase_benchmark"
    )
    args = parser.parse_args()
    samplers = [value.strip() for value in args.samplers.split(",") if value.strip()]
    if any(value not in ("uniform", "vonmises") for value in samplers):
        raise SystemExit("samplers must contain only uniform and/or vonmises")
    kappas = [float(value) for value in args.surface_kappas.split(",")]
    configs = default_method_configs("kernel", kappas=kappas)
    theta_grid = make_theta_grid(args.n_theta)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for sampler_idx, sampler in enumerate(samplers):
        for seed_offset in range(args.seeds):
            seed = args.base_seed + seed_offset
            simulation = simulate_constant_phase_repeats(
                n_repeats=args.repeats,
                n_time=args.n_time,
                noise_sigma=args.noise_sigma,
                rng=np.random.default_rng(
                    np.random.SeedSequence([seed, sampler_idx + 1, 1123])
                ),
                phase_sampler=sampler,
            )
            fit = estimate_blind_repeat_phase(
                simulation["observations"],
                n_phase_grid=args.phase_grid,
                kappa=args.phase_kappa,
                max_iterations=args.phase_iterations,
            )
            initial_alignment = align_circular_phase(
                np.asarray(fit["initial_phase"]), simulation["phase"]
            )
            alignment = align_circular_phase(np.asarray(fit["phase"]), simulation["phase"])
            true_phase_matrix = np.broadcast_to(
                simulation["phase"][:, None], simulation["observations"].shape
            )
            estimated_phase_matrix = np.broadcast_to(
                np.asarray(fit["phase"])[:, None], simulation["observations"].shape
            )
            oracle_selection = select_reconstruction_config(
                true_phase_matrix,
                simulation["observations"],
                simulation["time"],
                "kernel",
                configs,
                cv_folds=4,
                rng=np.random.default_rng(seed + 10_000),
                theta_grid=theta_grid,
            )
            blind_selection = select_reconstruction_config(
                estimated_phase_matrix,
                simulation["observations"],
                simulation["time"],
                "kernel",
                configs,
                cv_folds=4,
                rng=np.random.default_rng(seed + 20_000),
                theta_grid=theta_grid,
            )
            truth_surface = latent_kimesurface(simulation["time"], theta_grid)
            oracle_surface = reconstruct_advanced_surface(
                true_phase_matrix,
                simulation["observations"],
                simulation["time"],
                theta_grid,
                "kernel",
                oracle_selection["selected_config"],
            )
            estimated_query_grid = truth_grid_to_estimated_coordinate(
                theta_grid,
                sign=float(alignment["sign"]),
                offset=float(alignment["offset"]),
            )
            blind_surface = reconstruct_advanced_surface(
                estimated_phase_matrix,
                simulation["observations"],
                simulation["time"],
                estimated_query_grid,
                "kernel",
                blind_selection["selected_config"],
            )
            time_mean = np.broadcast_to(
                np.mean(simulation["observations"], axis=0)[:, None], truth_surface.shape
            )
            scale = max(float(np.std(truth_surface)), 1e-12)
            rows.append(
                {
                    "sampler": sampler,
                    "seed": seed,
                    "repeats": args.repeats,
                    "n_time": args.n_time,
                    "noise_sigma": args.noise_sigma,
                    "initial_phase_circular_rmse": initial_alignment["circular_rmse"],
                    "blind_phase_circular_rmse": alignment["circular_rmse"],
                    "blind_phase_circular_mae": alignment["circular_mae"],
                    "blind_phase_resultant_coherence": alignment["resultant_coherence"],
                    "alignment_reflected": bool(float(alignment["sign"]) < 0),
                    "phase_iterations": fit["iterations"],
                    "oracle_surface_nrmse": _rmse(oracle_surface, truth_surface) / scale,
                    "blind_surface_nrmse": _rmse(blind_surface, truth_surface) / scale,
                    "time_mean_surface_nrmse": _rmse(time_mean, truth_surface) / scale,
                    "oracle_selected_config": json.dumps(
                        oracle_selection["selected_config"], sort_keys=True
                    ),
                    "blind_selected_config": json.dumps(
                        blind_selection["selected_config"], sort_keys=True
                    ),
                    "truth_used_during_phase_estimation": False,
                    "is_official_kpt": False,
                }
            )
            print(f"Completed {sampler} seed {seed}", flush=True)

    summary = _summarize(rows, samplers)
    run_path = args.output_dir / "blind_phase_benchmark_runs.csv"
    summary_path = args.output_dir / "blind_phase_benchmark_summary.csv"
    manifest_path = args.output_dir / "blind_phase_benchmark_manifest.json"
    text_path = args.output_dir / "blind_phase_benchmark_summary.txt"
    _write_csv(run_path, rows)
    _write_csv(summary_path, summary)
    manifest_path.write_text(
        json.dumps(
            {
                "experiment": "blind_constant_repeat_phase_baseline",
                "parameters": {
                    **vars(args),
                    "output_dir": portable_manifest_path(args.output_dir),
                },
                "summary": summary,
                "scientific_limits": [
                    "Estimator assumes one constant latent phase per repeat.",
                    "Truth is used only for rotation/reflection alignment and final evaluation.",
                    "This is an observation-only baseline closer to the KPT problem setting, not an implementation of official KPT.",
                    "A free phase at every repeat-time cell is not identifiable from one scalar observation without additional structure.",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    text_path.write_text(_summary_text(summary), encoding="utf-8")
    if plt is not None:
        _save_figure(args.output_dir / "blind_phase_benchmark.png", rows, samplers)
    return 0


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(left) - np.asarray(right)) ** 2)))


def _summarize(rows: list[dict[str, object]], samplers: list[str]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    metrics = (
        "initial_phase_circular_rmse",
        "blind_phase_circular_rmse",
        "blind_phase_resultant_coherence",
        "oracle_surface_nrmse",
        "blind_surface_nrmse",
        "time_mean_surface_nrmse",
    )
    for sampler in samplers:
        group = [row for row in rows if row["sampler"] == sampler]
        summary: dict[str, object] = {"sampler": sampler, "datasets": len(group)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_sd"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        summary["blind_beats_time_mean_fraction"] = float(
            np.mean(
                [
                    float(row["blind_surface_nrmse"]) < float(row["time_mean_surface_nrmse"])
                    for row in group
                ]
            )
        )
        output.append(summary)
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summary_text(summary: list[dict[str, object]]) -> str:
    lines = [
        "# Blind synthetic latent-phase benchmark",
        "",
        "sampler | phase RMSE initial -> refined | surface NRMSE oracle / blind / time mean | P(blind < mean)",
    ]
    for row in summary:
        lines.append(
            f"{row['sampler']} | {float(row['initial_phase_circular_rmse_mean']):.3f} -> "
            f"{float(row['blind_phase_circular_rmse_mean']):.3f} | "
            f"{float(row['oracle_surface_nrmse_mean']):.3f} / "
            f"{float(row['blind_surface_nrmse_mean']):.3f} / "
            f"{float(row['time_mean_surface_nrmse_mean']):.3f} | "
            f"{float(row['blind_beats_time_mean_fraction']):.3f}"
        )
    lines.extend(
        [
            "",
            "The estimator uses observations only and assumes one constant latent phase per repeat.",
            "Truth appears only in alignment/evaluation. This baseline is not official KPT.",
        ]
    )
    return "\n".join(lines) + "\n"


def _save_figure(path: Path, rows: list[dict[str, object]], samplers: list[str]) -> None:
    fig, axes = plt.subplots(1, len(samplers), figsize=(6 * len(samplers), 4.5), squeeze=False)
    for index, sampler in enumerate(samplers):
        group = [row for row in rows if row["sampler"] == sampler]
        ax = axes[0, index]
        ax.scatter(
            [float(row["time_mean_surface_nrmse"]) for row in group],
            [float(row["blind_surface_nrmse"]) for row in group],
            label="seed",
        )
        limits = ax.get_xlim()
        low = min(limits[0], ax.get_ylim()[0])
        high = max(limits[1], ax.get_ylim()[1])
        ax.plot([low, high], [low, high], "--", color="black", label="equal")
        ax.set_xlabel("time-mean surface NRMSE")
        ax.set_ylabel("blind-phase surface NRMSE")
        ax.set_title(sampler)
        ax.legend()
    fig.suptitle("Observation-only blind phase baseline (synthetic constant-phase assumption)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
