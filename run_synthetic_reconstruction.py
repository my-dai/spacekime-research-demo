#!/usr/bin/env python3
"""Validate kimesurface reconstruction against a known synthetic ground truth."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - minimal environments
    plt = None

from spacekime_paths import PROJECT_ROOT
from spacekime_reporting import build_synthetic_run_manifest, write_run_manifest
from spacekime_synthetic import (
    holdout_prediction_metrics,
    latent_kimesurface,
    make_theta_grid,
    phase_coverage_metrics,
    reconstruct_kimesurface,
    simulate_repeated_measurements,
    surface_recovery_metrics,
)


OUTPUT_DIR = PROJECT_ROOT / "output" / "synthetic"
DEFAULT_REPEATS = "8,16,32,64,128"


def parse_repeats(text: str) -> list[int]:
    values = sorted({int(item.strip()) for item in text.split(",") if item.strip()})
    if not values or values[0] < 4:
        raise ValueError("Provide repeat counts >= 4")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct a known latent kimesurface from repeated phase samples."
    )
    parser.add_argument("--repeats", default=DEFAULT_REPEATS)
    parser.add_argument("--n-time", type=int, default=60)
    parser.add_argument("--n-theta", type=int, default=128)
    parser.add_argument("--noise-sigma", type=float, default=0.08)
    parser.add_argument("--kernel-kappa", type=float, default=4.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument(
        "--phase-sampler", choices=("uniform", "vonmises"), default="uniform"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    try:
        repeats = parse_repeats(args.repeats)
        if args.n_time < 2 or args.n_theta < 8:
            raise ValueError("n-time must be >= 2 and n-theta must be >= 8")
        if args.kernel_kappa <= 0 or args.noise_sigma < 0:
            raise ValueError("kernel-kappa must be positive and noise-sigma non-negative")
        if not 0 < args.holdout_fraction < 1:
            raise ValueError("holdout-fraction must be between 0 and 1")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    theta_grid = make_theta_grid(args.n_theta)
    simulation = simulate_repeated_measurements(
        n_repeats=max(repeats),
        n_time=args.n_time,
        noise_sigma=args.noise_sigma,
        rng=np.random.default_rng(args.seed),
        phase_sampler=args.phase_sampler,
    )
    truth = latent_kimesurface(simulation["time"], theta_grid)

    rows: list[dict[str, float | int | str]] = []
    reconstructions: dict[int, np.ndarray] = {}
    for count in repeats:
        theta = simulation["theta"][:count]
        observations = simulation["observations"][:count]
        signal = simulation["signal"][:count]
        reconstructed = reconstruct_kimesurface(
            theta,
            observations,
            theta_grid,
            kappa=args.kernel_kappa,
        )
        reconstructions[count] = reconstructed
        recovery = surface_recovery_metrics(truth, reconstructed)
        holdout = holdout_prediction_metrics(
            theta,
            observations,
            signal,
            holdout_fraction=args.holdout_fraction,
            kappa=args.kernel_kappa,
        )
        coverage = phase_coverage_metrics(theta)
        rows.append(
            {
                "repeats": count,
                "phase_sampler": args.phase_sampler,
                "noise_sigma": args.noise_sigma,
                "kernel_kappa": args.kernel_kappa,
                **recovery,
                **holdout,
                "mean_max_phase_gap_rad": coverage["mean_max_phase_gap_rad"],
                "mean_sampling_resultant": coverage["mean_sampling_resultant"],
            }
        )
        print(
            f"repeats={count:3d} nrmse={recovery['nrmse_std']:.4f} "
            f"corr={recovery['correlation']:.4f} "
            f"holdout_gain={holdout['relative_rmse_improvement']:.3f}"
        )

    csv_path = args.output_dir / "synthetic_reconstruction.csv"
    summary_path = args.output_dir / "synthetic_reconstruction.txt"
    figure_path = args.output_dir / "synthetic_reconstruction.png"
    manifest_path = args.output_dir / "synthetic_reconstruction_manifest.json"
    write_rows(csv_path, rows)
    write_summary(summary_path, rows)
    if plt is not None:
        save_figure(
            figure_path,
            simulation,
            theta_grid,
            truth,
            reconstructions[max(repeats)],
            rows,
        )

    manifest = build_synthetic_run_manifest(
        "run_synthetic_reconstruction.py",
        parameters={
            "repeats": repeats,
            "n_time": args.n_time,
            "n_theta": args.n_theta,
            "noise_sigma": args.noise_sigma,
            "kernel_kappa": args.kernel_kappa,
            "holdout_fraction": args.holdout_fraction,
            "phase_sampler": args.phase_sampler,
            "seed": args.seed,
        },
        results={"rows": rows},
        notes=[
            "Surface recovery uses all samples at each repeat count.",
            "Holdout prediction is trained only on the non-held-out repeats.",
            "The baseline predicts each held-out observation with the training time mean and ignores phase.",
        ],
    )
    write_run_manifest(manifest_path, manifest)

    print(f"Wrote: {csv_path.resolve()}")
    print(f"Wrote: {summary_path.resolve()}")
    if plt is not None:
        print(f"Wrote: {figure_path.resolve()}")
    print(f"Wrote: {manifest_path.resolve()}")
    return 0


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    first = rows[0]
    last = rows[-1]
    lines = [
        "# Synthetic repeated-measurement kimesurface reconstruction",
        f"repeat_range: {first['repeats']} -> {last['repeats']}",
        f"nrmse_std: {float(first['nrmse_std']):.6f} -> {float(last['nrmse_std']):.6f}",
        f"correlation: {float(first['correlation']):.6f} -> {float(last['correlation']):.6f}",
        "largest_repeat_holdout_kernel_rmse: "
        f"{float(last['kernel_rmse']):.6f}",
        "largest_repeat_time_mean_baseline_rmse: "
        f"{float(last['time_mean_baseline_rmse']):.6f}",
        "largest_repeat_relative_holdout_improvement: "
        f"{float(last['relative_rmse_improvement']):.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(
    path: Path,
    simulation: dict[str, np.ndarray],
    theta_grid: np.ndarray,
    truth: np.ndarray,
    reconstructed: np.ndarray,
    rows: list[dict[str, float | int | str]],
) -> None:
    time = simulation["time"]
    error = np.abs(reconstructed - truth)
    extent = (float(time[0]), float(time[-1]), -np.pi, np.pi)
    vmin = float(min(truth.min(), reconstructed.min()))
    vmax = float(max(truth.max(), reconstructed.max()))

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), constrained_layout=True)
    for ax, surface, title in (
        (axes[0, 0], truth, "Known latent kimesurface"),
        (axes[0, 2], reconstructed, "Circular-kernel reconstruction"),
    ):
        image = ax.imshow(
            surface.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(title)
        ax.set_xlabel("longitudinal time")
        ax.set_ylabel("kime-phase theta")
        fig.colorbar(image, ax=ax, shrink=0.78)

    shown = min(12, simulation["theta"].shape[0])
    scatter = axes[0, 1].scatter(
        np.tile(time, shown),
        simulation["theta"][:shown].ravel(),
        c=simulation["observations"][:shown].ravel(),
        s=7,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0, 1].set_title(f"Irregular samples ({shown} repeats shown)")
    axes[0, 1].set_xlabel("longitudinal time")
    axes[0, 1].set_ylabel("sampled kime-phase theta")
    fig.colorbar(scatter, ax=axes[0, 1], shrink=0.78)

    error_image = axes[1, 0].imshow(
        error.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
    )
    axes[1, 0].set_title("Absolute reconstruction error")
    axes[1, 0].set_xlabel("longitudinal time")
    axes[1, 0].set_ylabel("kime-phase theta")
    fig.colorbar(error_image, ax=axes[1, 0], shrink=0.78)

    repeat_axis = [int(row["repeats"]) for row in rows]
    axes[1, 1].plot(
        repeat_axis,
        [float(row["nrmse_std"]) for row in rows],
        marker="o",
        label="surface NRMSE",
    )
    axes[1, 1].plot(
        repeat_axis,
        [1.0 - float(row["correlation"]) for row in rows],
        marker="o",
        label="1 - correlation",
    )
    axes[1, 1].set_title("Ground-truth recovery")
    axes[1, 1].set_xlabel("repeated experiments")
    axes[1, 1].grid(True, alpha=0.35)
    axes[1, 1].legend()

    axes[1, 2].plot(
        repeat_axis,
        [float(row["kernel_rmse"]) for row in rows],
        marker="o",
        label="phase-aware kernel",
    )
    axes[1, 2].plot(
        repeat_axis,
        [float(row["time_mean_baseline_rmse"]) for row in rows],
        marker="o",
        label="time-mean baseline",
    )
    axes[1, 2].set_title("Held-out true-signal prediction")
    axes[1, 2].set_xlabel("repeated experiments")
    axes[1, 2].grid(True, alpha=0.35)
    axes[1, 2].legend()

    fig.suptitle("Falsifiable repeated-measurement kimesurface experiment")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
