"""Spacekime / KIME core: phase noise, metrics, and kimesurface statistics."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - exercised only in minimal environments
    plt = None

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:  # pragma: no cover - exercised only in minimal environments
    psnr = None
    ssim = None


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def make_phase_noisy_complex_slice(
    vol3d: np.ndarray,
    z_idx: int,
    sigma: Optional[float] = 1.0,
    noise: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add phase noise to one z-slice and preserve the complex representation."""
    if vol3d.ndim != 3:
        raise ValueError(f"vol3d must be 3D (X,Y,Z), got shape {vol3d.shape}")
    z_idx = int(np.clip(z_idx, 0, vol3d.shape[2] - 1))
    orig = vol3d[:, :, z_idx]
    mag = np.abs(orig)
    phase = np.angle(orig)

    if sigma is None or sigma <= 0:
        noisy = orig
    else:
        eps = _sample_phase_noise(phase.shape, sigma=sigma, noise=noise, rng=rng)
        phase_noisy = phase + eps
        noisy = mag * np.exp(1j * phase_noisy)

    return orig.astype(np.complex64, copy=False), noisy.astype(np.complex64, copy=False)


def make_phase_noisy_slice(
    vol3d: np.ndarray,
    z_idx: int,
    sigma: Optional[float] = 1.0,
    noise: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add phase noise to one z-slice; return its original/noisy real channels."""
    orig, noisy = make_phase_noisy_complex_slice(
        vol3d,
        z_idx=z_idx,
        sigma=sigma,
        noise=noise,
        rng=rng,
    )
    return np.real(orig).astype(np.float32), np.real(noisy).astype(np.float32)


def apply_phase_noise_volume(
    bigim: np.ndarray,
    sigma: float,
    noise: str = "gaussian",
    t_list: Optional[List[int]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Return a copy of bigim with phase noise at selected or all time points."""
    if bigim.ndim != 4:
        raise ValueError(f"bigim must be 4D (X,Y,Z,T), got shape {bigim.shape}")

    out = bigim.copy()
    _, _, _, T = bigim.shape

    if sigma <= 0:
        return out

    if t_list is None:
        t_idx = range(T)
    else:
        t_idx = sorted({int(t) for t in t_list if 0 <= int(t) < T})
        if not t_idx:
            raise ValueError("t_list is empty or all indices are out of range.")

    for t in t_idx:
        vol = bigim[:, :, :, t]
        mag = np.abs(vol)
        phase = np.angle(vol)
        eps = _sample_phase_noise(phase.shape, sigma=sigma, noise=noise, rng=rng)
        out[:, :, :, t] = mag * np.exp(1j * (phase + eps))

    return out


def _sample_phase_noise(
    shape: Tuple[int, ...],
    sigma: float,
    noise: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample phase perturbations in radians."""
    random_source = np.random if rng is None else rng
    if noise.lower() in ("gaussian", "gauss", "normal"):
        return random_source.normal(loc=0.0, scale=sigma, size=shape)
    if noise.lower() in ("vm", "vonmises", "von_mises"):
        kappa = 1.0 / max(sigma**2, 1e-6)
        return random_source.vonmises(mu=0.0, kappa=kappa, size=shape)
    raise ValueError(f"Unknown noise type: {noise}")


def complex_phase_metrics(
    ref: np.ndarray,
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Paired complex-signal and circular phase errors.

    Unlike a histogram distance, these metrics retain voxel correspondence.
    Phase errors are wrapped to ``[-pi, pi)`` and weighted by reference
    magnitude.  ``phase_coherence`` is one for identical phase and approaches
    zero as paired phase offsets become uniform.
    """
    ref_arr = np.asarray(ref)
    img_arr = np.asarray(img)
    if ref_arr.shape != img_arr.shape:
        raise ValueError(f"ref/img shapes differ: {ref_arr.shape} vs {img_arr.shape}")

    valid = (
        np.isfinite(ref_arr.real)
        & np.isfinite(ref_arr.imag)
        & np.isfinite(img_arr.real)
        & np.isfinite(img_arr.imag)
    )
    if mask is not None:
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != ref_arr.shape:
            raise ValueError(f"mask shape {mask_arr.shape} does not match {ref_arr.shape}")
        valid &= mask_arr
    if not np.any(valid):
        return {
            "complex_nrmse": float("nan"),
            "magnitude_nrmse": float("nan"),
            "phase_mae_rad": float("nan"),
            "phase_rmse_rad": float("nan"),
            "phase_bias_rad": float("nan"),
            "phase_coherence": float("nan"),
        }

    ref_sel = ref_arr[valid].astype(np.complex128, copy=False)
    img_sel = img_arr[valid].astype(np.complex128, copy=False)
    mag_ref = np.abs(ref_sel)
    weight_sum = float(mag_ref.sum())
    weights = mag_ref if weight_sum > 0 else np.ones_like(mag_ref)
    weight_sum = float(weights.sum())

    delta = np.angle(img_sel * np.conj(ref_sel))
    phase_vec = np.sum(weights * np.exp(1j * delta)) / max(weight_sum, 1e-12)
    ref_energy = float(np.sum(np.abs(ref_sel) ** 2))
    denom = max(ref_energy, 1e-12)

    return {
        "complex_nrmse": float(np.sqrt(np.sum(np.abs(img_sel - ref_sel) ** 2) / denom)),
        "magnitude_nrmse": float(
            np.sqrt(np.sum((np.abs(img_sel) - mag_ref) ** 2) / denom)
        ),
        "phase_mae_rad": float(np.sum(weights * np.abs(delta)) / weight_sum),
        "phase_rmse_rad": float(np.sqrt(np.sum(weights * delta**2) / weight_sum)),
        "phase_bias_rad": float(np.angle(phase_vec)),
        "phase_coherence": float(np.abs(phase_vec)),
    }


def metrics_psnr_ssim(ref: np.ndarray, img: np.ndarray) -> Tuple[float, float]:
    if psnr is None or ssim is None:
        raise ImportError("scikit-image is required for PSNR/SSIM metrics.")

    ref = np.asarray(ref, dtype=np.float32)
    img = np.asarray(img, dtype=np.float32)
    data_range = ref.max() - ref.min()
    if data_range <= 0:
        return float("nan"), float("nan")
    return float(psnr(ref, img, data_range=data_range)), float(
        ssim(ref, img, data_range=data_range)
    )


def roi_center_box(
    img2d: np.ndarray,
    half: Optional[int] = 8,
    size: Optional[int] = None,
) -> np.ndarray:
    """Extract a central square ROI from a 2D image."""
    img = np.asarray(img2d)
    if img.ndim != 2:
        raise ValueError(f"img2d must be 2D, got shape {img.shape}")

    h, w = img.shape
    if size is not None:
        half = max(1, int(size) // 2)
    elif half is None:
        half = max(1, min(h, w) // 8)
    else:
        half = max(1, int(half))

    cx, cy = h // 2, w // 2
    x0 = max(cx - half, 0)
    x1 = min(cx + half, h)
    y0 = max(cy - half, 0)
    y1 = min(cy + half, w)
    return img[x0:x1, y0:y1]


def snr_mean_over_std(roi2d: np.ndarray) -> float:
    """SNR proxy: absolute ROI mean divided by ROI standard deviation."""
    roi = np.asarray(roi2d, dtype=np.float32)
    sd = float(np.std(roi))
    if sd <= 0:
        return float("nan")
    return float(abs(np.mean(roi)) / sd)


def estimate_phase_noise_sigma(
    bigim: np.ndarray,
    z_idx: int = 10,
    t_start: int = 0,
    t_end: Optional[int] = None,
    background_margin: int = 8,
    per_frame: bool = True,
    method: str = "linear",
) -> float:
    """Estimate a phase-noise proxy from border temporal phase differences.

    This is not an identifiable scanner-noise estimate unless the underlying
    phase is stationary and frame errors are independent.  Use
    :func:`phase_difference_noise_diagnostics` to inspect both estimators.
    """
    diagnostics = phase_difference_noise_diagnostics(
        bigim,
        z_idx=z_idx,
        t_start=t_start,
        t_end=t_end,
        background_margin=background_margin,
    )
    scale = "per_frame" if per_frame else "diff"
    method_key = method.strip().lower()
    if method_key not in ("linear", "circular"):
        raise ValueError("method must be 'linear' or 'circular'")
    return diagnostics[f"{method_key}_{scale}_sigma"]


def phase_difference_noise_diagnostics(
    bigim: np.ndarray,
    z_idx: int = 10,
    t_start: int = 0,
    t_end: Optional[int] = None,
    background_margin: int = 8,
) -> Dict[str, float]:
    """Return linear and circular phase-difference noise proxies."""
    if bigim.ndim != 4:
        raise ValueError("bigim must have shape (X, Y, Z, T)")

    X, Y, Z, T = bigim.shape
    z_idx = int(np.clip(z_idx, 0, Z - 1))
    t_start = int(max(0, t_start))
    t_end = T if t_end is None or t_end > T else int(t_end)

    if t_end - t_start < 2:
        raise ValueError("Need at least 2 time points to estimate sigma.")

    vol = bigim[:, :, z_idx, t_start:t_end]
    phase = np.angle(vol)
    dtheta = phase[:, :, 1:] - phase[:, :, :-1]
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

    margin = int(max(1, background_margin))
    if 2 * margin >= X or 2 * margin >= Y:
        mask = np.ones((X, Y), dtype=bool)
    else:
        mask = np.zeros((X, Y), dtype=bool)
        mask[:margin, :] = True
        mask[-margin:, :] = True
        mask[:, :margin] = True
        mask[:, -margin:] = True

    vals = dtheta[mask, :].ravel()
    if vals.size == 0:
        vals = dtheta.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {
            "linear_diff_sigma": float("nan"),
            "linear_per_frame_sigma": float("nan"),
            "circular_resultant": float("nan"),
            "circular_diff_sigma": float("nan"),
            "circular_per_frame_sigma": float("nan"),
            "sample_count": 0.0,
        }

    linear_diff_sigma = float(np.std(vals))
    resultant = float(np.abs(np.mean(np.exp(1j * vals))))
    resultant = float(np.clip(resultant, 1e-12, 1.0))
    circular_diff_sigma = float(np.sqrt(max(-2.0 * math.log(resultant), 0.0)))
    return {
        "linear_diff_sigma": linear_diff_sigma,
        "linear_per_frame_sigma": linear_diff_sigma / math.sqrt(2.0),
        "circular_resultant": resultant,
        "circular_diff_sigma": circular_diff_sigma,
        "circular_per_frame_sigma": circular_diff_sigma / math.sqrt(2.0),
        "sample_count": float(vals.size),
    }


def brain_mask_from_magnitude(
    bigim: np.ndarray,
    t_idx: int = 0,
    percentile: float = 50.0,
) -> np.ndarray:
    """Backward-compatible whole-volume magnitude percentile mask."""
    return phase_scope_mask(
        bigim,
        t_idx=t_idx,
        percentile=percentile,
        scope="volume",
    )


def phase_scope_mask(
    bigim: np.ndarray,
    t_idx: int = 0,
    percentile: float = 50.0,
    scope: str = "volume",
    z_idx: Optional[int] = None,
    roi_half: int = 8,
) -> np.ndarray:
    """Build a magnitude mask for a volume, selected slice, or central slice ROI."""
    if bigim.ndim != 4:
        raise ValueError(f"bigim must be 4D (X,Y,Z,T), got shape {bigim.shape}")
    X, Y, Z, T = bigim.shape
    t_idx = int(np.clip(t_idx, 0, bigim.shape[3] - 1))
    percentile = float(np.clip(percentile, 0.0, 100.0))
    mag_vol = np.abs(bigim[:, :, :, t_idx])

    scope_key = scope.strip().lower()
    region = np.zeros((X, Y, Z), dtype=bool)
    if scope_key == "volume":
        region[:] = True
    elif scope_key in ("slice", "roi"):
        if z_idx is None:
            raise ValueError(f"z_idx is required for scope={scope_key!r}")
        z = int(np.clip(z_idx, 0, Z - 1))
        if scope_key == "slice":
            region[:, :, z] = True
        else:
            half = max(1, int(roi_half))
            cx, cy = X // 2, Y // 2
            region[
                max(0, cx - half) : min(X, cx + half),
                max(0, cy - half) : min(Y, cy + half),
                z,
            ] = True
    else:
        raise ValueError("scope must be 'volume', 'slice', or 'roi'")

    valid = region & np.isfinite(mag_vol)
    values = mag_vol[valid]
    positive = values[values > 0]
    if positive.size:
        values = positive
    if values.size == 0:
        raise ValueError("No finite magnitude samples are available in the selected scope.")
    threshold = float(np.percentile(values, percentile))
    mask = valid & (mag_vol > threshold)
    if not np.any(mask):
        mask = valid & (mag_vol >= threshold)
    return mask


def _kappa_from_R(R: np.ndarray) -> np.ndarray:
    R = np.clip(R, 1e-6, 0.999999)
    kappa = np.zeros_like(R)

    small = R < 0.53
    mid = (R >= 0.53) & (R < 0.85)
    large = R >= 0.85

    kappa[small] = 2 * R[small] + R[small] ** 3 + 5 * R[small] ** 5 / 6
    kappa[mid] = -0.4 + 1.39 * R[mid] + 0.43 / (1 - R[mid])
    kappa[large] = 1 / (R[large] ** 3 - 4 * R[large] ** 2 + 3 * R[large])

    return np.clip(kappa, 1e-6, None)


def build_phase_distribution_surface(
    bigim: np.ndarray,
    mask: np.ndarray,
    t_list: Optional[List[int]] = None,
    n_theta: int = 128,
    normalize_h: bool = False,
    phase_noise_sigma: float = 0.0,
    noise: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
    phase_mode: str = "raw",
    reference_t: int = 0,
) -> Dict[str, object]:
    """Build a spatial phase-distribution surface and circular statistics.

    ``H(t, theta)`` is a magnitude-weighted histogram of observed complex
    signal phase across the selected spatial mask.  It is a useful Spacekime-
    inspired proxy, but it is not a kimesurface reconstructed from repeated
    measurements or Kime-Phase Tomography.

    ``phase_mode='relative'`` subtracts each voxel's phase at ``reference_t``
    before spatial aggregation. This removes static spatial phase offsets, but
    does not replace motion, field-drift, physiological, or phase-unwrapping
    corrections. Optional perturbations are generated inside the time loop,
    avoiding a second 4D array for sensitivity experiments.
    """
    bigim = np.asarray(bigim)
    if bigim.ndim != 4:
        raise ValueError(f"bigim must be 4D (X,Y,Z,T), got shape {bigim.shape}")
    n_theta = int(n_theta)
    if n_theta < 4:
        raise ValueError("n_theta must be at least 4")

    X, Y, Z, T = bigim.shape
    phase_mode_key = phase_mode.strip().lower()
    if phase_mode_key not in ("raw", "relative"):
        raise ValueError("phase_mode must be 'raw' or 'relative'")
    reference_t = int(np.clip(reference_t, 0, T - 1))
    reference_phase = (
        np.angle(bigim[:, :, :, reference_t])
        if phase_mode_key == "relative"
        else None
    )
    mask_arr = np.asarray(mask).astype(bool)

    if mask_arr.ndim == 3:
        if mask_arr.shape != (X, Y, Z):
            raise ValueError(
                f"3D mask shape {mask_arr.shape} does not match (X,Y,Z)=({X},{Y},{Z})"
            )
        mask_3d = mask_arr
        mask_is_4d = False
    elif mask_arr.ndim == 4:
        if mask_arr.shape != (X, Y, Z, T):
            raise ValueError(
                f"4D mask shape {mask_arr.shape} does not match (X,Y,Z,T)=({X},{Y},{Z},{T})"
            )
        mask_is_4d = True
        mask_3d = None
    else:
        raise ValueError("mask must be 3D (X,Y,Z) or 4D (X,Y,Z,T).")

    if t_list is None:
        t_idx = np.arange(T, dtype=int)
    else:
        t_idx = np.asarray(sorted({int(t) for t in t_list if 0 <= int(t) < T}), dtype=int)
        if t_idx.size == 0:
            raise ValueError("t_list is empty or all indices are out of range.")

    Nt = t_idx.shape[0]
    theta_edges = np.linspace(-np.pi, np.pi, n_theta + 1, endpoint=True)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    H = np.zeros((Nt, n_theta), dtype=np.float32)
    mu_t = np.full(Nt, np.nan, dtype=np.float32)
    R_t = np.full(Nt, np.nan, dtype=np.float32)
    kappa_t = np.full(Nt, np.nan, dtype=np.float32)
    row_valid = np.zeros(Nt, dtype=bool)
    sample_count_t = np.zeros(Nt, dtype=np.int64)
    total_weight_t = np.zeros(Nt, dtype=np.float64)
    phase_mae_t = np.full(Nt, np.nan, dtype=np.float32)
    phase_rmse_t = np.full(Nt, np.nan, dtype=np.float32)
    phase_coherence_t = np.full(Nt, np.nan, dtype=np.float32)
    complex_nrmse_t = np.full(Nt, np.nan, dtype=np.float32)

    for i, t in enumerate(t_idx):
        vol_t = bigim[:, :, :, t]
        mag_t = np.abs(vol_t)
        phase_t = np.angle(vol_t)
        m = mask_arr[:, :, :, t] if mask_is_4d else mask_3d

        mag_sel = mag_t[m]
        phase_sel = phase_t[m]
        if reference_phase is not None:
            ref_sel_phase = reference_phase[m]
            finite = np.isfinite(mag_sel) & np.isfinite(phase_sel) & np.isfinite(ref_sel_phase)
            mag_sel = mag_sel[finite]
            phase_sel = np.angle(np.exp(1j * (phase_sel[finite] - ref_sel_phase[finite])))
        else:
            finite = np.isfinite(mag_sel) & np.isfinite(phase_sel)
            mag_sel = mag_sel[finite]
            phase_sel = phase_sel[finite]

        positive = mag_sel > 0
        mag_sel = mag_sel[positive]
        phase_sel = phase_sel[positive]
        if mag_sel.size == 0:
            continue
        row_valid[i] = True
        sample_count_t[i] = int(mag_sel.size)
        total_weight_t[i] = float(np.sum(mag_sel, dtype=np.float64))

        if phase_noise_sigma > 0:
            eps = _sample_phase_noise(
                phase_sel.shape,
                sigma=phase_noise_sigma,
                noise=noise,
                rng=rng,
            )
            phase_eval = phase_sel + eps
            ref_sel = mag_sel * np.exp(1j * phase_sel)
            noisy_sel = mag_sel * np.exp(1j * phase_eval)
            paired = complex_phase_metrics(ref_sel, noisy_sel)
            phase_mae_t[i] = paired["phase_mae_rad"]
            phase_rmse_t[i] = paired["phase_rmse_rad"]
            phase_coherence_t[i] = paired["phase_coherence"]
            complex_nrmse_t[i] = paired["complex_nrmse"]
        else:
            phase_eval = phase_sel
            phase_mae_t[i] = 0.0
            phase_rmse_t[i] = 0.0
            phase_coherence_t[i] = 1.0
            complex_nrmse_t[i] = 0.0

        H[i, :], _ = np.histogram(phase_eval, bins=theta_edges, weights=mag_sel)

        w = mag_sel.astype(np.float64)
        z_circ = np.exp(1j * phase_eval)
        R_vec = (w * z_circ).sum() / (w.sum() + 1e-8)
        mu_t[i] = float(np.angle(R_vec))
        R_t[i] = float(np.abs(R_vec))

    if normalize_h:
        row_sums = H.sum(axis=1, keepdims=True)
        H[row_valid] = H[row_valid] / np.maximum(row_sums[row_valid], 1e-8)

    H_prob = _normalize_rows(H)
    entropy_t = np.full(Nt, np.nan, dtype=np.float64)
    if np.any(row_valid):
        entropy_t[row_valid] = -np.sum(
            H_prob[row_valid] * np.log(np.maximum(H_prob[row_valid], 1e-12)), axis=1
        )
    entropy_normalized_t = entropy_t / math.log(n_theta)

    nonzero = row_valid & (R_t > 0)
    kappa_t[nonzero] = _kappa_from_R(R_t[nonzero])

    return {
        "representation": "spatial_phase_distribution_proxy",
        "phase_mode": phase_mode_key,
        "reference_t": reference_t,
        "theta": theta_centers,
        "t_axis": t_idx,
        "H": H,
        "row_valid": row_valid,
        "sample_count_t": sample_count_t,
        "total_weight_t": total_weight_t.astype(np.float32),
        "mu_t": mu_t,
        "R_t": R_t,
        "kappa_t": kappa_t,
        "entropy_t": entropy_t.astype(np.float32),
        "entropy_normalized_t": entropy_normalized_t.astype(np.float32),
        "phase_noise_sigma": float(max(0.0, phase_noise_sigma)),
        "phase_mae_t": phase_mae_t,
        "phase_rmse_t": phase_rmse_t,
        "phase_coherence_t": phase_coherence_t,
        "complex_nrmse_t": complex_nrmse_t,
    }


def build_circular_stats_over_time(
    bigim: np.ndarray,
    mask: np.ndarray,
    t_list: Optional[List[int]] = None,
    n_theta: int = 128,
    normalize_h: bool = False,
    phase_noise_sigma: float = 0.0,
    noise: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
    phase_mode: str = "raw",
    reference_t: int = 0,
) -> Dict[str, object]:
    """Backward-compatible alias for :func:`build_phase_distribution_surface`."""
    return build_phase_distribution_surface(
        bigim,
        mask,
        t_list=t_list,
        n_theta=n_theta,
        normalize_h=normalize_h,
        phase_noise_sigma=phase_noise_sigma,
        noise=noise,
        rng=rng,
        phase_mode=phase_mode,
        reference_t=reference_t,
    )


def _normalize_rows(H: np.ndarray) -> np.ndarray:
    """Normalize non-empty rows while leaving empty rows at zero."""
    H = np.asarray(H, dtype=np.float64)
    row_sums = H.sum(axis=1, keepdims=True)
    out = np.zeros_like(H, dtype=np.float64)
    np.divide(H, row_sums, out=out, where=row_sums > 0)
    return out


def kime_distance_L2(
    stats_a: Dict,
    stats_b: Dict,
    normalize_rows: bool = False,
) -> float:
    H_a = np.asarray(stats_a["H"], dtype=np.float32)
    H_b = np.asarray(stats_b["H"], dtype=np.float32)
    if H_a.shape != H_b.shape:
        raise ValueError(f"H shapes differ: {H_a.shape} vs {H_b.shape}")
    if normalize_rows:
        H_a = _normalize_rows(H_a)
        H_b = _normalize_rows(H_b)
    return float(np.linalg.norm(H_a - H_b))


def kime_js_distance(stats_a: Dict, stats_b: Dict) -> float:
    """Jensen-Shannon distance between row-normalized kimesurfaces."""
    H_a = np.asarray(stats_a["H"], dtype=np.float64)
    H_b = np.asarray(stats_b["H"], dtype=np.float64)
    if H_a.shape != H_b.shape:
        raise ValueError(f"H shapes differ: {H_a.shape} vs {H_b.shape}")

    P = _normalize_rows(H_a)
    Q = _normalize_rows(H_b)
    valid_a = np.asarray(stats_a.get("row_valid", H_a.sum(axis=1) > 0), dtype=bool)
    valid_b = np.asarray(stats_b.get("row_valid", H_b.sum(axis=1) > 0), dtype=bool)
    valid = valid_a & valid_b
    if valid.shape != (H_a.shape[0],):
        raise ValueError("row_valid must have one entry per time row")
    if not np.any(valid):
        return float("nan")
    P = P[valid]
    Q = Q[valid]
    M = 0.5 * (P + Q)
    kl_pm = np.sum(P * np.log(np.maximum(P, 1e-12) / np.maximum(M, 1e-12)), axis=1)
    kl_qm = np.sum(Q * np.log(np.maximum(Q, 1e-12) / np.maximum(M, 1e-12)), axis=1)
    js = 0.5 * (kl_pm + kl_qm)
    return float(np.sqrt(np.mean(np.maximum(js, 0.0))))


def kime_surface_summary(stats: Dict) -> Dict[str, float]:
    """Compact scalar summary for comparing kimesurface experiments."""
    H = np.asarray(stats["H"], dtype=np.float64)
    R = np.asarray(stats["R_t"], dtype=np.float64)
    kappa = np.asarray(stats["kappa_t"], dtype=np.float64)
    entropy = np.asarray(stats.get("entropy_t", []), dtype=np.float64)
    entropy_normalized = np.asarray(
        stats.get("entropy_normalized_t", []), dtype=np.float64
    )
    if entropy.size == 0:
        H_prob = _normalize_rows(H)
        entropy = -np.sum(H_prob * np.log(np.maximum(H_prob, 1e-12)), axis=1)
    if entropy_normalized.size == 0:
        entropy_normalized = entropy / math.log(max(H.shape[1], 2))

    row_valid = np.asarray(stats.get("row_valid", H.sum(axis=1) > 0), dtype=bool)
    summary = {
        "time_points": float(H.shape[0]),
        "valid_time_points": float(np.sum(row_valid)),
        "theta_bins": float(H.shape[1]),
        "total_intensity": float(H.sum()),
        "mean_R": float(np.nanmean(R)),
        "std_R": float(np.nanstd(R)),
        "mean_kappa": float(np.nanmean(kappa)),
        "std_kappa": float(np.nanstd(kappa)),
        "mean_entropy": float(np.nanmean(entropy)),
        "std_entropy": float(np.nanstd(entropy)),
        "mean_entropy_normalized": float(np.nanmean(entropy_normalized)),
        "std_entropy_normalized": float(np.nanstd(entropy_normalized)),
    }
    for source, target in (
        ("phase_mae_t", "mean_phase_mae_rad"),
        ("phase_rmse_t", "mean_phase_rmse_rad"),
        ("phase_coherence_t", "mean_phase_coherence"),
        ("complex_nrmse_t", "mean_complex_nrmse"),
    ):
        values = np.asarray(stats.get(source, []), dtype=np.float64)
        summary[target] = float(np.nanmean(values)) if values.size else float("nan")
    return summary


def save_kimesurface_figure(
    stats: Dict,
    out_path: str,
    title_prefix: str = "Kimesurface",
) -> None:
    """Save the phase-distribution proxy and its circular summaries."""
    if plt is None:
        raise ImportError("matplotlib is required to save kimesurface figures.")

    theta = np.asarray(stats["theta"], dtype=np.float32)
    H = np.asarray(stats["H"], dtype=np.float32)
    t_axis = np.asarray(stats.get("t_axis", np.arange(H.shape[0])), dtype=int)
    R = np.asarray(stats["R_t"], dtype=np.float32)
    kappa = np.asarray(stats["kappa_t"], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # H is (time, theta); transpose so x=time, y=theta
    im = axes[0].imshow(
        H.T,
        aspect="auto",
        origin="lower",
        extent=(t_axis[0], t_axis[-1], theta[0], theta[-1]),
        cmap="viridis",
    )
    axes[0].set_xlabel("time index t")
    axes[0].set_ylabel("theta (rad)")
    axes[0].set_title(f"{title_prefix}: phase-distribution H(t, theta)")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    axes[1].plot(t_axis, R, label="R(t)")
    axes[1].plot(t_axis, kappa, label="kappa(t)")
    axes[1].set_xlabel("time index t")
    axes[1].set_title("Spatial phase circular stats")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_kimesurface_compare_figure(
    stats_a: Dict,
    stats_b: Dict,
    out_path: str,
    label_a: str = "orig",
    label_b: str = "noisy",
) -> None:
    if plt is None:
        raise ImportError("matplotlib is required to save kimesurface figures.")

    theta = np.asarray(stats_a["theta"], dtype=np.float32)
    H_a = np.asarray(stats_a["H"], dtype=np.float32)
    H_b = np.asarray(stats_b["H"], dtype=np.float32)
    t_axis = np.asarray(stats_a.get("t_axis", np.arange(H_a.shape[0])), dtype=int)
    R_a = np.asarray(stats_a["R_t"], dtype=np.float32)
    R_b = np.asarray(stats_b["R_t"], dtype=np.float32)
    k_a = np.asarray(stats_a["kappa_t"], dtype=np.float32)
    k_b = np.asarray(stats_b["kappa_t"], dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    extent = (t_axis[0], t_axis[-1], theta[0], theta[-1])

    for ax, H, label in zip(axes[:2], (H_a, H_b), (label_a, label_b)):
        im = ax.imshow(H.T, aspect="auto", origin="lower", extent=extent, cmap="viridis")
        ax.set_title(f"Phase proxy H(t,theta) {label}")
        ax.set_xlabel("t")
        ax.set_ylabel("theta")
        plt.colorbar(im, ax=ax, shrink=0.8)

    axes[2].plot(t_axis, R_a, label=f"R {label_a}")
    axes[2].plot(t_axis, R_b, label=f"R {label_b}")
    axes[2].plot(t_axis, k_a, "--", label=f"kappa {label_a}")
    axes[2].plot(t_axis, k_b, "--", label=f"kappa {label_b}")
    axes[2].set_xlabel("t")
    axes[2].set_title("R(t), kappa(t) comparison")
    axes[2].legend()
    axes[2].grid(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def slice_metrics_report(
    bigim: np.ndarray,
    t_idx: int,
    z_idx: int,
    sigma: float,
    noise: str = "gaussian",
    roi_half: Optional[int] = 8,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Image-channel and paired complex metrics for one noisy (t, z) slice."""
    vol = bigim[:, :, :, t_idx]
    orig_complex, noisy_complex = make_phase_noisy_complex_slice(
        vol,
        z_idx,
        sigma=sigma,
        noise=noise,
        rng=rng,
    )
    orig = np.real(orig_complex).astype(np.float32)
    noisy = np.real(noisy_complex).astype(np.float32)
    p, s = metrics_psnr_ssim(orig, noisy)
    roi_orig = roi_center_box(orig, half=roi_half)
    roi_noisy = roi_center_box(noisy, half=roi_half)
    snr_orig = snr_mean_over_std(roi_orig)
    snr_noisy = snr_mean_over_std(roi_noisy)
    paired = complex_phase_metrics(orig_complex, noisy_complex)
    return {
        "psnr": p,
        "ssim": s,
        "roi_snr_orig": snr_orig,
        "roi_snr_noisy": snr_noisy,
        "roi_snr_delta": snr_noisy - snr_orig,
        **paired,
    }
