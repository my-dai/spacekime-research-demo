#!/usr/bin/env python3
"""
Lightweight local Spacekime / KIME demo (no neural-network training).

Loads a real bigim .mat file, runs phase-noise metrics and kimesurface analysis,
and writes figures + a text summary under output/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from spacekime_core import (
    build_phase_distribution_surface,
    kime_js_distance,
    kime_distance_L2,
    kime_surface_summary,
    phase_difference_noise_diagnostics,
    phase_scope_mask,
    save_kimesurface_compare_figure,
    save_kimesurface_figure,
    slice_metrics_report,
)
from spacekime_loader import bigim_metadata, load_bigim
from spacekime_paths import PROJECT_ROOT, resolve_mat_path
from spacekime_reporting import build_run_manifest, write_run_manifest

OUTPUT_DIR = PROJECT_ROOT / "output"


def write_summary(path: Path, lines: list[str]) -> None:
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    print(text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Spacekime KIME demo on real bigim data.")
    parser.add_argument(
        "mat_file",
        nargs="?",
        default=None,
        help="Path to .mat (default: $SPACEKIME_MAT_FILE, data/, or project root)",
    )
    parser.add_argument("--t", type=int, default=0, help="Demo time index")
    parser.add_argument("--z", type=int, default=10, help="Demo slice index")
    parser.add_argument("--sigma", type=float, default=2.0, help="Phase-noise sigma (rad)")
    parser.add_argument("--noise", choices=("gaussian", "vm"), default="gaussian")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scope",
        choices=("volume", "slice", "roi"),
        default="slice",
        help="Spatial scope for the phase-distribution proxy (default: slice)",
    )
    parser.add_argument("--mask-percentile", type=float, default=50.0)
    parser.add_argument("--roi-half", type=int, default=8)
    parser.add_argument("--n-theta", type=int, default=128)
    parser.add_argument(
        "--phase-mode",
        choices=("raw", "relative"),
        default="raw",
        help="Use raw phase or phase relative to --t (default: raw)",
    )
    parser.add_argument(
        "--max-t",
        type=int,
        default=None,
        help="Optional cap on time points for kimesurface (faster on large T)",
    )
    parser.add_argument(
        "--normalize-h",
        action="store_true",
        help="Normalize each row of H(t,theta) before L2 distance",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for figures and summary (default: ./output)",
    )
    args = parser.parse_args()

    try:
        mat_path = resolve_mat_path(args.mat_file)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {mat_path}")
    bigim = load_bigim(mat_path)
    meta = bigim_metadata(bigim)
    X, Y, Z, T = bigim.shape

    t_idx = int(np_clip(args.t, 0, T - 1))
    z_idx = int(np_clip(args.z, 0, Z - 1))

    lines: list[str] = [
        "# Spacekime local demo summary",
        f"mat_file: {mat_path}",
        f"shape (X,Y,Z,T): {meta['shape']}",
        f"dtype: {meta['dtype']}",
        f"demo t={t_idx}, z={z_idx}, sigma={args.sigma}, noise={args.noise}",
        "",
    ]

    try:
        phase_diag = phase_difference_noise_diagnostics(
            bigim, z_idx=z_idx, t_start=0, t_end=min(T, 20)
        )
        lines.append(
            "phase_difference_sigma_proxy_linear_per_frame: "
            f"{phase_diag['linear_per_frame_sigma']:.4f} rad"
        )
        lines.append(
            "phase_difference_sigma_proxy_circular_per_frame: "
            f"{phase_diag['circular_per_frame_sigma']:.4f} rad"
        )
        lines.append(
            "phase_difference_proxy_note: includes true temporal phase change; "
            "it is not an identified scanner-noise estimate"
        )
    except Exception as exc:
        phase_diag = {}
        lines.append(f"phase_difference_sigma_proxy: failed ({exc})")

    metrics = slice_metrics_report(
        bigim,
        t_idx,
        z_idx,
        sigma=args.sigma,
        noise=args.noise,
        roi_half=args.roi_half,
        rng=np.random.default_rng(args.seed),
    )
    lines.append(f"real-channel slice PSNR: {metrics['psnr']:.2f} dB")
    lines.append(f"real-channel slice SSIM: {metrics['ssim']:.4f}")
    lines.append(f"real-channel ROI SNR orig: {metrics['roi_snr_orig']:.4f}")
    lines.append(f"real-channel ROI SNR noisy: {metrics['roi_snr_noisy']:.4f}")
    lines.append(f"paired complex NRMSE: {metrics['complex_nrmse']:.6f}")
    lines.append(f"paired magnitude NRMSE: {metrics['magnitude_nrmse']:.6f}")
    lines.append(f"paired phase MAE: {metrics['phase_mae_rad']:.6f} rad")
    lines.append(f"paired phase RMSE: {metrics['phase_rmse_rad']:.6f} rad")
    lines.append(f"paired phase coherence: {metrics['phase_coherence']:.6f}")
    lines.append("")

    mask = phase_scope_mask(
        bigim,
        t_idx=t_idx,
        percentile=args.mask_percentile,
        scope=args.scope,
        z_idx=z_idx,
        roi_half=args.roi_half,
    )
    t_list = None
    if args.max_t is not None and args.max_t < T:
        t_list = list(range(0, args.max_t))

    print("Building spatial phase-distribution proxy for original data...")
    stats_orig = build_phase_distribution_surface(
        bigim,
        mask,
        t_list=t_list,
        n_theta=args.n_theta,
        normalize_h=args.normalize_h,
        phase_mode=args.phase_mode,
        reference_t=t_idx,
    )

    print(f"Building perturbed proxy in streaming mode (sigma={args.sigma})...")
    stats_noisy = build_phase_distribution_surface(
        bigim,
        mask,
        t_list=t_list,
        n_theta=args.n_theta,
        normalize_h=args.normalize_h,
        phase_noise_sigma=args.sigma,
        noise=args.noise,
        rng=np.random.default_rng(args.seed),
        phase_mode=args.phase_mode,
        reference_t=t_idx,
    )

    d_h = kime_distance_L2(stats_orig, stats_noisy)
    d_js = kime_js_distance(stats_orig, stats_noisy)
    summary_orig = kime_surface_summary(stats_orig)
    summary_noisy = kime_surface_summary(stats_noisy)
    lines.append("representation: spatial phase-distribution proxy (not KPT reconstruction)")
    lines.append(f"phase_mode: {args.phase_mode}; reference_t: {t_idx}")
    lines.append(f"scope: {args.scope}; mask_voxels: {int(mask.sum())}")
    lines.append(f"phase-proxy L2 distance ||H_orig - H_noisy||_2: {d_h:.4f}")
    lines.append(f"phase-proxy JS distance: {d_js:.6f}")
    lines.append(f"mean R(t) orig: {summary_orig['mean_R']:.6f}")
    lines.append(f"mean R(t) noisy: {summary_noisy['mean_R']:.6f}")
    lines.append(f"mean kappa(t) orig: {summary_orig['mean_kappa']:.6f}")
    lines.append(f"mean kappa(t) noisy: {summary_noisy['mean_kappa']:.6f}")
    lines.append(f"mean phase entropy orig: {summary_orig['mean_entropy']:.6f}")
    lines.append(f"mean phase entropy noisy: {summary_noisy['mean_entropy']:.6f}")
    lines.append(
        "mean normalized phase entropy orig: "
        f"{summary_orig['mean_entropy_normalized']:.6f}"
    )
    lines.append(
        "mean normalized phase entropy noisy: "
        f"{summary_noisy['mean_entropy_normalized']:.6f}"
    )
    lines.append(
        f"mean paired phase coherence: {summary_noisy['mean_phase_coherence']:.6f}"
    )
    lines.append(f"mean paired complex NRMSE: {summary_noisy['mean_complex_nrmse']:.6f}")
    lines.append("")
    lines.append(f"figures: {args.output_dir.resolve()}")

    fig_orig = args.output_dir / "kimesurface_original.png"
    fig_compare = args.output_dir / "kimesurface_compare.png"
    summary_path = args.output_dir / "demo_summary.txt"
    manifest_path = args.output_dir / "demo_manifest.json"

    save_kimesurface_figure(stats_orig, str(fig_orig), title_prefix="Original")
    save_kimesurface_compare_figure(
        stats_orig,
        stats_noisy,
        str(fig_compare),
        label_a="orig",
        label_b=f"noisy sigma={args.sigma}",
    )

    write_summary(summary_path, lines)
    manifest = build_run_manifest(
        "run_real_bigim_demo.py",
        mat_path,
        parameters={
            "t": t_idx,
            "z": z_idx,
            "sigma": args.sigma,
            "noise": args.noise,
            "seed": args.seed,
            "scope": args.scope,
            "mask_percentile": args.mask_percentile,
            "roi_half": args.roi_half,
            "max_t": args.max_t,
            "n_theta": args.n_theta,
            "normalize_h": args.normalize_h,
            "phase_mode": args.phase_mode,
            "reference_t": t_idx,
        },
        results={
            "slice": metrics,
            "phase_difference_proxy": phase_diag,
            "surface_original": summary_orig,
            "surface_noisy": summary_noisy,
            "surface_l2": d_h,
            "surface_js_distance": d_js,
            "mask_voxels": int(mask.sum()),
        },
        notes=[
            "PSNR/SSIM and ROI SNR are computed on the real channel.",
            "Magnitude NRMSE should remain near zero for phase-only perturbations.",
        ],
    )
    write_run_manifest(manifest_path, manifest)
    print(f"Wrote: {manifest_path.resolve()}")
    print("Done.")
    return 0


def np_clip(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


if __name__ == "__main__":
    raise SystemExit(main())
