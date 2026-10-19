"""Ground-truth repeated-measurement kimesurface simulation and reconstruction.

The synthetic line is intentionally separate from the observed fMRI phase proxy.
Here the latent surface and sampled phase are known by construction, which makes
reconstruction error, bandwidth selection, and held-out prediction falsifiable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter


def make_theta_grid(n_theta: int = 128) -> np.ndarray:
    """Return an endpoint-free grid on the phase circle ``[-pi, pi)``."""
    n_theta = int(n_theta)
    if n_theta < 8:
        raise ValueError("n_theta must be at least 8")
    return np.linspace(-np.pi, np.pi, n_theta, endpoint=False, dtype=np.float64)


def latent_kimesurface(time: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Evaluate a smooth positive latent surface with two moving phase ridges."""
    time_arr = np.asarray(time, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64)
    if time_arr.ndim == 1 and theta_arr.ndim == 1:
        t_eval = time_arr[:, None]
        theta_eval = theta_arr[None, :]
    else:
        t_eval, theta_eval = np.broadcast_arrays(time_arr, theta_arr)

    mu_primary = 0.9 * np.sin(2 * np.pi * t_eval) + 0.3 * np.cos(4 * np.pi * t_eval)
    mu_secondary = mu_primary + np.pi * (
        0.65 + 0.10 * np.sin(2 * np.pi * t_eval)
    )
    amplitude_primary = 1.05 + 0.30 * np.sin(2 * np.pi * t_eval + 0.4)
    amplitude_secondary = 0.60 + 0.20 * np.cos(2 * np.pi * t_eval - 0.3)
    baseline = 0.38 + 0.08 * np.cos(2 * np.pi * t_eval)

    primary = np.exp(5.0 * (np.cos(theta_eval - mu_primary) - 1.0))
    secondary = np.exp(2.8 * (np.cos(theta_eval - mu_secondary) - 1.0))
    texture = 0.08 * np.sin(6 * np.pi * t_eval + 2.0 * theta_eval)
    surface = (
        baseline
        + amplitude_primary * primary
        + amplitude_secondary * secondary
        + texture
    )
    return np.maximum(surface, 1e-6)


def simulate_repeated_measurements(
    n_repeats: int = 64,
    n_time: int = 60,
    noise_sigma: float = 0.08,
    rng: Optional[np.random.Generator] = None,
    phase_sampler: str = "uniform",
) -> Dict[str, np.ndarray]:
    """Sample one latent kime-phase and intensity per repeat and time point."""
    n_repeats = int(n_repeats)
    n_time = int(n_time)
    if n_repeats < 2 or n_time < 2:
        raise ValueError("n_repeats and n_time must both be at least 2")
    if noise_sigma < 0:
        raise ValueError("noise_sigma must be non-negative")

    generator = np.random.default_rng() if rng is None else rng
    time = np.linspace(0.0, 1.0, n_time, dtype=np.float64)
    sampler = phase_sampler.strip().lower()
    if sampler == "uniform":
        theta = generator.uniform(-np.pi, np.pi, size=(n_repeats, n_time))
    elif sampler == "vonmises":
        center = 0.7 * np.sin(2 * np.pi * time)[None, :]
        theta = generator.vonmises(center, 1.5, size=(n_repeats, n_time))
    else:
        raise ValueError("phase_sampler must be 'uniform' or 'vonmises'")

    signal = latent_kimesurface(time[None, :], theta)
    observations = signal + generator.normal(0.0, noise_sigma, size=signal.shape)
    return {
        "time": time,
        "theta": theta,
        "signal": signal,
        "observations": observations,
    }


def circular_distance(theta_a: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    """Return absolute shortest-arc distance on the phase circle."""
    delta = np.asarray(theta_a, dtype=np.float64) - np.asarray(theta_b, dtype=np.float64)
    return np.abs(np.angle(np.exp(1j * delta)))


def circular_kernel_predict(
    theta_train: np.ndarray,
    values_train: np.ndarray,
    theta_query: np.ndarray,
    kappa: float = 4.0,
) -> np.ndarray:
    """Nadaraya-Watson prediction using a von-Mises-shaped circular kernel."""
    train_theta = np.asarray(theta_train, dtype=np.float64).ravel()
    train_values = np.asarray(values_train, dtype=np.float64).ravel()
    query = np.asarray(theta_query, dtype=np.float64)
    if train_theta.size != train_values.size or train_theta.size == 0:
        raise ValueError("theta_train and values_train must be non-empty and equal length")
    if kappa <= 0 or not np.isfinite(kappa):
        raise ValueError("kappa must be finite and positive")

    valid_train = np.isfinite(train_theta) & np.isfinite(train_values)
    if not np.any(valid_train):
        return np.full(query.shape, np.nan, dtype=np.float64)
    train_theta = train_theta[valid_train]
    train_values = train_values[valid_train]

    flat_query = query.ravel()
    output = np.full(flat_query.shape, np.nan, dtype=np.float64)
    valid_query = np.isfinite(flat_query)
    if np.any(valid_query):
        delta = flat_query[valid_query, None] - train_theta[None, :]
        log_weights = kappa * (np.cos(delta) - 1.0)
        # Subtracting the row maximum preserves the weighted mean while avoiding
        # artificial shrinkage or NaNs when every raw kernel weight is tiny in a
        # poorly sampled phase sector at high kappa.
        log_weights -= np.max(log_weights, axis=1, keepdims=True)
        weights = np.exp(log_weights)
        output[valid_query] = (weights @ train_values) / weights.sum(axis=1)
    return output.reshape(query.shape)


def reconstruct_kimesurface(
    theta_samples: np.ndarray,
    observations: np.ndarray,
    theta_grid: np.ndarray,
    kappa: float = 4.0,
) -> np.ndarray:
    """Reconstruct ``H(t, theta)`` independently at each longitudinal time."""
    theta_arr, obs_arr = _validate_repeated_arrays(theta_samples, observations)
    grid = np.asarray(theta_grid, dtype=np.float64)
    if grid.ndim != 1:
        raise ValueError("theta_grid must be one-dimensional")

    _, n_time = theta_arr.shape
    reconstructed = np.empty((n_time, grid.size), dtype=np.float64)
    for t_idx in range(n_time):
        reconstructed[t_idx] = circular_kernel_predict(
            theta_arr[:, t_idx],
            obs_arr[:, t_idx],
            grid,
            kappa=kappa,
        )
    return reconstructed


def smooth_kimesurface_time_phase(
    surface: np.ndarray,
    time_sigma: float = 0.0,
    theta_sigma: float = 0.0,
) -> np.ndarray:
    """Apply optional 2D smoothing with a circular boundary on the phase axis.

    This is an experimental regularizer, not an automatically validated default.
    ``time_sigma`` and ``theta_sigma`` are in grid-index units. The time boundary
    uses nearest extension, while the phase boundary wraps around ``S^1``.
    """
    arr = np.asarray(surface, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("surface must have shape (time, theta)")
    if time_sigma < 0 or theta_sigma < 0:
        raise ValueError("smoothing sigmas must be non-negative")
    if time_sigma == 0 and theta_sigma == 0:
        return arr.copy()
    return gaussian_filter(
        arr,
        sigma=(float(time_sigma), float(theta_sigma)),
        mode=("nearest", "wrap"),
        output=np.float64,
    )


def select_kernel_kappa(
    theta_samples: np.ndarray,
    observations: np.ndarray,
    candidates: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
    validation_fraction: float = 0.25,
    cv_folds: Optional[int] = 3,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Choose circular bandwidth from noisy observations using repeat-level CV.

    No latent truth is used. By default, repeat-level K-fold cross-validation
    predicts held-out phases and chooses the candidate with the smallest
    observed-value RMSE. Set ``cv_folds=None`` to use one validation split.
    """
    theta_arr, obs_arr = _validate_repeated_arrays(theta_samples, observations)
    n_repeats, n_time = theta_arr.shape
    if n_repeats < 4:
        raise ValueError("at least four repeats are required for bandwidth selection")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    candidate_values = sorted({float(value) for value in candidates})
    if not candidate_values or any(
        (not np.isfinite(value)) or value <= 0 for value in candidate_values
    ):
        raise ValueError("candidates must contain finite positive values")

    generator = np.random.default_rng(0) if rng is None else rng
    permutation = generator.permutation(n_repeats)
    if cv_folds is not None:
        fold_count = int(np.clip(int(cv_folds), 2, n_repeats))
        validation_folds = [fold for fold in np.array_split(permutation, fold_count) if fold.size]
        scheme = "repeat_kfold"
    else:
        validation_count = int(
            np.clip(round(n_repeats * validation_fraction), 1, n_repeats - 2)
        )
        validation_folds = [permutation[:validation_count]]
        scheme = "single_repeat_holdout"

    candidate_rmse: dict[str, float] = {}
    all_indices = np.arange(n_repeats)
    for kappa in candidate_values:
        squared_errors: list[np.ndarray] = []
        for validation_idx in validation_folds:
            train_idx = np.setdiff1d(all_indices, validation_idx, assume_unique=False)
            if train_idx.size < 2:
                continue
            for t_idx in range(n_time):
                prediction = circular_kernel_predict(
                    theta_arr[train_idx, t_idx],
                    obs_arr[train_idx, t_idx],
                    theta_arr[validation_idx, t_idx],
                    kappa=kappa,
                )
                target = obs_arr[validation_idx, t_idx]
                valid = np.isfinite(prediction) & np.isfinite(target)
                if np.any(valid):
                    squared_errors.append((prediction[valid] - target[valid]) ** 2)
        rmse = (
            float(np.sqrt(np.mean(np.concatenate(squared_errors))))
            if squared_errors
            else float("inf")
        )
        candidate_rmse[f"{kappa:g}"] = rmse

    selected = min(candidate_values, key=lambda value: candidate_rmse[f"{value:g}"])
    return {
        "selected_kappa": float(selected),
        "cv_rmse": float(candidate_rmse[f"{selected:g}"]),
        "candidate_rmse": candidate_rmse,
        "cv_scheme": scheme,
        "cv_folds": len(validation_folds),
        "repeats": int(n_repeats),
    }


def nested_holdout_prediction_metrics(
    theta_samples: np.ndarray,
    observations: np.ndarray,
    true_signal: np.ndarray,
    candidates: Sequence[float] = (1.0, 2.0, 4.0, 8.0, 16.0),
    holdout_fraction: float = 0.25,
    inner_validation_fraction: float = 0.25,
    inner_folds: Optional[int] = 3,
    seed: int = 0,
) -> Dict[str, object]:
    """Nested repeat-level validation for phase-aware prediction.

    The outer split estimates generalization against the known synthetic signal.
    The circular bandwidth is selected only within the outer-training repeats and
    only from noisy observations, preventing truth leakage.
    """
    theta_arr, obs_arr = _validate_repeated_arrays(theta_samples, observations)
    truth_arr = np.asarray(true_signal, dtype=np.float64)
    if truth_arr.shape != theta_arr.shape:
        raise ValueError("true_signal must share shape with theta_samples")
    n_repeats, n_time = theta_arr.shape
    if n_repeats < 6:
        raise ValueError("at least six repeats are required for nested holdout")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")

    generator = np.random.default_rng(seed)
    permutation = generator.permutation(n_repeats)
    test_count = int(np.clip(round(n_repeats * holdout_fraction), 1, n_repeats - 4))
    test_idx = permutation[:test_count]
    train_idx = permutation[test_count:]

    selection = select_kernel_kappa(
        theta_arr[train_idx],
        obs_arr[train_idx],
        candidates=candidates,
        validation_fraction=inner_validation_fraction,
        cv_folds=inner_folds,
        rng=np.random.default_rng(seed + 1),
    )
    selected_kappa = float(selection["selected_kappa"])

    kernel_predictions = np.empty((test_count, n_time), dtype=np.float64)
    baseline_predictions = np.empty((test_count, n_time), dtype=np.float64)
    for t_idx in range(n_time):
        kernel_predictions[:, t_idx] = circular_kernel_predict(
            theta_arr[train_idx, t_idx],
            obs_arr[train_idx, t_idx],
            theta_arr[test_idx, t_idx],
            kappa=selected_kappa,
        )
        baseline_predictions[:, t_idx] = np.nanmean(obs_arr[train_idx, t_idx])

    target = truth_arr[test_idx]
    valid_kernel = np.isfinite(kernel_predictions) & np.isfinite(target)
    valid_baseline = np.isfinite(baseline_predictions) & np.isfinite(target)
    kernel_rmse = float(
        np.sqrt(np.mean((kernel_predictions[valid_kernel] - target[valid_kernel]) ** 2))
    )
    baseline_rmse = float(
        np.sqrt(np.mean((baseline_predictions[valid_baseline] - target[valid_baseline]) ** 2))
    )
    return {
        "train_repeats": int(train_idx.size),
        "test_repeats": int(test_idx.size),
        "selected_kappa": selected_kappa,
        "inner_cv_rmse": float(selection["cv_rmse"]),
        "candidate_rmse": selection["candidate_rmse"],
        "inner_cv_scheme": selection["cv_scheme"],
        "inner_cv_folds": selection["cv_folds"],
        "kernel_rmse": kernel_rmse,
        "time_mean_baseline_rmse": baseline_rmse,
        "relative_rmse_improvement": (
            baseline_rmse - kernel_rmse
        ) / max(baseline_rmse, 1e-12),
    }


def surface_recovery_metrics(
    truth: np.ndarray,
    reconstructed: np.ndarray,
) -> Dict[str, float]:
    """Compare a reconstructed surface with known latent ground truth."""
    truth_arr = np.asarray(truth, dtype=np.float64)
    recon_arr = np.asarray(reconstructed, dtype=np.float64)
    if truth_arr.shape != recon_arr.shape:
        raise ValueError(f"surface shapes differ: {truth_arr.shape} vs {recon_arr.shape}")
    valid = np.isfinite(truth_arr) & np.isfinite(recon_arr)
    if not np.any(valid):
        return {key: float("nan") for key in ("rmse", "nrmse_std", "correlation", "r2")}
    truth_valid = truth_arr[valid]
    recon_valid = recon_arr[valid]
    residual = recon_valid - truth_valid
    rmse = float(np.sqrt(np.mean(residual**2)))
    truth_std = float(np.std(truth_valid))
    centered_energy = float(np.sum((truth_valid - np.mean(truth_valid)) ** 2))
    residual_energy = float(np.sum(residual**2))
    correlation = (
        float(np.corrcoef(truth_valid, recon_valid)[0, 1])
        if truth_valid.size > 1 and np.std(recon_valid) > 0 and truth_std > 0
        else float("nan")
    )
    return {
        "rmse": rmse,
        "nrmse_std": rmse / max(truth_std, 1e-12),
        "correlation": correlation,
        "r2": 1.0 - residual_energy / max(centered_energy, 1e-12),
    }


def holdout_prediction_metrics(
    theta_samples: np.ndarray,
    observations: np.ndarray,
    true_signal: np.ndarray,
    holdout_fraction: float = 0.25,
    kappa: float = 4.0,
) -> Dict[str, float]:
    """Compare fixed-bandwidth phase-aware prediction with a time-mean baseline.

    This backward-compatible helper uses the last repeats as holdout. For a less
    optimistic bandwidth-tuned estimate, use :func:`nested_holdout_prediction_metrics`.
    """
    theta_arr, obs_arr = _validate_repeated_arrays(theta_samples, observations)
    truth_arr = np.asarray(true_signal, dtype=np.float64)
    if theta_arr.shape != truth_arr.shape:
        raise ValueError("theta, observations, and true_signal must share shape")
    n_repeats, n_time = theta_arr.shape
    test_count = int(np.clip(round(n_repeats * holdout_fraction), 1, n_repeats - 1))
    train_count = n_repeats - test_count

    kernel_predictions = np.empty((test_count, n_time), dtype=np.float64)
    baseline_predictions = np.empty((test_count, n_time), dtype=np.float64)
    for t_idx in range(n_time):
        kernel_predictions[:, t_idx] = circular_kernel_predict(
            theta_arr[:train_count, t_idx],
            obs_arr[:train_count, t_idx],
            theta_arr[train_count:, t_idx],
            kappa=kappa,
        )
        baseline_predictions[:, t_idx] = np.nanmean(obs_arr[:train_count, t_idx])

    target = truth_arr[train_count:]
    kernel_rmse = float(np.sqrt(np.nanmean((kernel_predictions - target) ** 2)))
    baseline_rmse = float(np.sqrt(np.nanmean((baseline_predictions - target) ** 2)))
    return {
        "train_repeats": float(train_count),
        "test_repeats": float(test_count),
        "kernel_rmse": kernel_rmse,
        "time_mean_baseline_rmse": baseline_rmse,
        "relative_rmse_improvement": (
            baseline_rmse - kernel_rmse
        ) / max(baseline_rmse, 1e-12),
    }


def phase_sampling_density(
    theta_samples: np.ndarray,
    theta_grid: np.ndarray,
    kappa: float = 4.0,
) -> np.ndarray:
    """Estimate per-time sampling density on a circular phase grid.

    Rows are normalized to sum to one. This density describes where repeated
    observations occur; it is not the latent process intensity surface.
    """
    theta_arr = np.asarray(theta_samples, dtype=np.float64)
    grid = np.asarray(theta_grid, dtype=np.float64)
    if theta_arr.ndim != 2 or grid.ndim != 1:
        raise ValueError("theta_samples must be (repeats,time) and theta_grid must be 1D")
    if kappa <= 0 or not np.isfinite(kappa):
        raise ValueError("kappa must be finite and positive")
    _, n_time = theta_arr.shape
    density = np.zeros((n_time, grid.size), dtype=np.float64)
    for t_idx in range(n_time):
        samples = theta_arr[:, t_idx]
        samples = samples[np.isfinite(samples)]
        if samples.size == 0:
            continue
        weights = np.exp(kappa * (np.cos(grid[:, None] - samples[None, :]) - 1.0))
        density[t_idx] = np.mean(weights, axis=1)
    row_sums = density.sum(axis=1, keepdims=True)
    np.divide(density, row_sums, out=density, where=row_sums > 0)
    return density


def sampling_weighted_surface_metrics(
    truth: np.ndarray,
    reconstructed: np.ndarray,
    theta_samples: np.ndarray,
    theta_grid: np.ndarray,
    density_kappa: float = 4.0,
    coverage_quantile: float = 0.25,
) -> Dict[str, float]:
    """Separate in-distribution recovery from low-coverage phase recovery.

    ``weighted_nrmse_std`` emphasizes phase regions sampled by the repeats.
    ``low_coverage_nrmse_std`` and ``high_coverage_nrmse_std`` use the lowest
    and highest empirical-density quantiles at each time. A large low/high ratio
    signals that ordinary same-distribution holdout can hide phase-sector gaps.
    """
    truth_arr = np.asarray(truth, dtype=np.float64)
    recon_arr = np.asarray(reconstructed, dtype=np.float64)
    if truth_arr.shape != recon_arr.shape:
        raise ValueError("truth and reconstructed must share shape")
    if not 0 < coverage_quantile < 0.5:
        raise ValueError("coverage_quantile must be between 0 and 0.5")
    density = phase_sampling_density(theta_samples, theta_grid, kappa=density_kappa)
    if density.shape != truth_arr.shape:
        raise ValueError("surface shape must match (time, len(theta_grid))")

    valid = np.isfinite(truth_arr) & np.isfinite(recon_arr) & np.isfinite(density)
    weights = np.where(valid, density, 0.0)
    weight_sum = float(weights.sum())
    if weight_sum <= 0:
        return {
            "weighted_rmse": float("nan"),
            "weighted_nrmse_std": float("nan"),
            "low_coverage_nrmse_std": float("nan"),
            "high_coverage_nrmse_std": float("nan"),
            "low_to_high_rmse_ratio": float("nan"),
        }

    residual_sq = (recon_arr - truth_arr) ** 2
    weighted_rmse = float(np.sqrt(np.sum(weights * residual_sq) / weight_sum))
    weighted_mean = float(np.sum(weights * truth_arr) / weight_sum)
    weighted_std = float(
        np.sqrt(np.sum(weights * (truth_arr - weighted_mean) ** 2) / weight_sum)
    )

    low_mask = np.zeros_like(valid, dtype=bool)
    high_mask = np.zeros_like(valid, dtype=bool)
    for t_idx in range(density.shape[0]):
        row = density[t_idx]
        row_valid = valid[t_idx]
        if not np.any(row_valid):
            continue
        low_threshold = float(np.quantile(row[row_valid], coverage_quantile))
        high_threshold = float(np.quantile(row[row_valid], 1.0 - coverage_quantile))
        low_mask[t_idx] = row_valid & (row <= low_threshold)
        high_mask[t_idx] = row_valid & (row >= high_threshold)

    truth_std = float(np.std(truth_arr[valid]))
    low_rmse = float(np.sqrt(np.mean(residual_sq[low_mask]))) if np.any(low_mask) else float("nan")
    high_rmse = float(np.sqrt(np.mean(residual_sq[high_mask]))) if np.any(high_mask) else float("nan")
    return {
        "weighted_rmse": weighted_rmse,
        "weighted_nrmse_std": weighted_rmse / max(weighted_std, 1e-12),
        "low_coverage_nrmse_std": low_rmse / max(truth_std, 1e-12),
        "high_coverage_nrmse_std": high_rmse / max(truth_std, 1e-12),
        "low_to_high_rmse_ratio": low_rmse / max(high_rmse, 1e-12),
    }


def phase_coverage_metrics(theta_samples: np.ndarray) -> Dict[str, float]:
    """Summarize circular phase coverage across repeated observations."""
    theta_arr = np.asarray(theta_samples, dtype=np.float64)
    if theta_arr.ndim != 2 or theta_arr.shape[0] < 2:
        raise ValueError("theta_samples must have shape (repeats,time), repeats >= 2")
    max_gaps = []
    resultants = []
    valid_time_points = 0
    for t_idx in range(theta_arr.shape[1]):
        values = theta_arr[:, t_idx]
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        values = np.sort((values + 2 * np.pi) % (2 * np.pi))
        gaps = np.diff(np.concatenate([values, values[:1] + 2 * np.pi]))
        max_gaps.append(float(np.max(gaps)))
        resultants.append(float(np.abs(np.mean(np.exp(1j * values)))))
        valid_time_points += 1
    if not max_gaps:
        return {
            "mean_max_phase_gap_rad": float("nan"),
            "mean_sampling_resultant": float("nan"),
            "uniformity_score": float("nan"),
            "repeats": float(theta_arr.shape[0]),
            "time_points": float(theta_arr.shape[1]),
            "valid_time_points": 0.0,
            "circle_radians": 2.0 * math.pi,
        }
    return {
        "mean_max_phase_gap_rad": float(np.mean(max_gaps)),
        "mean_sampling_resultant": float(np.mean(resultants)),
        "uniformity_score": float(1.0 - np.mean(resultants)),
        "repeats": float(theta_arr.shape[0]),
        "time_points": float(theta_arr.shape[1]),
        "valid_time_points": float(valid_time_points),
        "circle_radians": 2.0 * math.pi,
    }


def _validate_repeated_arrays(
    theta_samples: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    theta_arr = np.asarray(theta_samples, dtype=np.float64)
    obs_arr = np.asarray(observations, dtype=np.float64)
    if theta_arr.shape != obs_arr.shape or theta_arr.ndim != 2:
        raise ValueError("theta_samples and observations must share shape (repeats,time)")
    if theta_arr.shape[0] < 2 or theta_arr.shape[1] < 1:
        raise ValueError("at least two repeats and one time point are required")
    return theta_arr, obs_arr
