#!/usr/bin/env python3
"""Grid report for local Spacekime fMRI phase-noise sensitivity."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

from spacekime_core import slice_metrics_report
from spacekime_loader import load_bigim
from spacekime_paths import PROJECT_ROOT, resolve_mat_path
from spacekime_reporting import build_run_manifest, write_run_manifest


OUTPUT_DIR = PROJECT_ROOT / "output"


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer.")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write real-channel and paired complex metrics over a t/z grid."
    )
    parser.add_argument(
        "mat_file",
        nargs="?",
        default=None,
        help="Path to .mat (default: $SPACEKIME_MAT_FILE, data/, or project root)",
    )
    parser.add_argument("--t-list", default="0,10,20,30,40", help="Comma-separated t indices")
    parser.add_argument("--z-list", default="5,10,20,30", help="Comma-separated z indices")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--noise", choices=("gaussian", "vm"), default="gaussian")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--roi-half", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for CSV and summary outputs (default: ./output)",
    )
    args = parser.parse_args()

    try:
        t_list = parse_int_list(args.t_list)
        z_list = parse_int_list(args.z_list)
        mat_path = resolve_mat_path(args.mat_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {mat_path}")
    bigim = load_bigim(mat_path)
    _, _, Z, T = bigim.shape

    rows: list[dict[str, float | int | str]] = []
    for t in t_list:
        if t < 0 or t >= T:
            print(f"Skipping t={t}: out of range 0..{T - 1}")
            continue
        for z in z_list:
            if z < 0 or z >= Z:
                print(f"Skipping z={z}: out of range 0..{Z - 1}")
                continue
            metrics = slice_metrics_report(
                bigim,
                t_idx=t,
                z_idx=z,
                sigma=args.sigma,
                noise=args.noise,
                roi_half=args.roi_half,
                rng=np.random.default_rng(args.seed + t * 1000 + z),
            )
            rows.append(
                {
                    "t": t,
                    "z": z,
                    "sigma": args.sigma,
                    "noise": args.noise,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"],
                    "roi_snr_orig": metrics["roi_snr_orig"],
                    "roi_snr_noisy": metrics["roi_snr_noisy"],
                    "roi_snr_delta": metrics["roi_snr_delta"],
                    "complex_nrmse": metrics["complex_nrmse"],
                    "magnitude_nrmse": metrics["magnitude_nrmse"],
                    "phase_mae_rad": metrics["phase_mae_rad"],
                    "phase_rmse_rad": metrics["phase_rmse_rad"],
                    "phase_bias_rad": metrics["phase_bias_rad"],
                    "phase_coherence": metrics["phase_coherence"],
                }
            )

    if not rows:
        print("ERROR: no valid t/z combinations were evaluated.", file=sys.stderr)
        return 1

    csv_path = args.output_dir / "quick_report.csv"
    summary_path = args.output_dir / "quick_report.txt"
    manifest_path = args.output_dir / "quick_report_manifest.json"
    write_rows(csv_path, rows)
    write_summary(summary_path, mat_path, rows, args.sigma, args.noise)

    print(f"Wrote: {csv_path.resolve()}")
    print(f"Wrote: {summary_path.resolve()}")
    manifest = build_run_manifest(
        "run_quick_report.py",
        mat_path,
        parameters={
            "t_list": t_list,
            "z_list": z_list,
            "sigma": args.sigma,
            "noise": args.noise,
            "seed": args.seed,
            "roi_half": args.roi_half,
        },
        results={"evaluated_slices": len(rows), "rows": rows},
        notes=["PSNR/SSIM and ROI SNR are real-channel metrics."],
    )
    write_run_manifest(manifest_path, manifest)
    print(f"Wrote: {manifest_path.resolve()}")
    return 0


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(
    path: Path,
    mat_path: Path,
    rows: list[dict[str, float | int | str]],
    sigma: float,
    noise: str,
) -> None:
    psnr_vals = np.asarray([float(row["psnr"]) for row in rows], dtype=np.float64)
    ssim_vals = np.asarray([float(row["ssim"]) for row in rows], dtype=np.float64)
    snr_orig = np.asarray([float(row["roi_snr_orig"]) for row in rows], dtype=np.float64)
    snr_noisy = np.asarray([float(row["roi_snr_noisy"]) for row in rows], dtype=np.float64)
    snr_delta = np.asarray([float(row["roi_snr_delta"]) for row in rows], dtype=np.float64)
    complex_nrmse = np.asarray([float(row["complex_nrmse"]) for row in rows])
    magnitude_nrmse = np.asarray([float(row["magnitude_nrmse"]) for row in rows])
    phase_mae = np.asarray([float(row["phase_mae_rad"]) for row in rows])
    phase_rmse = np.asarray([float(row["phase_rmse_rad"]) for row in rows])
    phase_coherence = np.asarray([float(row["phase_coherence"]) for row in rows])

    lines = [
        "# Spacekime quick grid report",
        f"mat_file: {mat_path}",
        f"sigma: {sigma}",
        f"noise: {noise}",
        f"evaluated_slices: {len(rows)}",
        "",
        f"mean_psnr: {_nanmean(psnr_vals):.4f}",
        f"mean_ssim: {_nanmean(ssim_vals):.6f}",
        f"mean_roi_snr_orig: {_nanmean(snr_orig):.6f}",
        f"mean_roi_snr_noisy: {_nanmean(snr_noisy):.6f}",
        f"mean_roi_snr_delta: {_nanmean(snr_delta):.6f}",
        "",
        f"mean_complex_nrmse: {_nanmean(complex_nrmse):.6f}",
        f"mean_magnitude_nrmse: {_nanmean(magnitude_nrmse):.6f}",
        f"mean_phase_mae_rad: {_nanmean(phase_mae):.6f}",
        f"mean_phase_rmse_rad: {_nanmean(phase_rmse):.6f}",
        f"mean_phase_coherence: {_nanmean(phase_coherence):.6f}",
        "",
        "See quick_report.csv for per-slice rows.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _nanmean(values: np.ndarray) -> float:
    if np.all(np.isnan(values)):
        return math.nan
    return float(np.nanmean(values))


if __name__ == "__main__":
    raise SystemExit(main())
