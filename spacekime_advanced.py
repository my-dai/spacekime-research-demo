"""Advanced reconstruction methods for the synthetic latent-kimesurface line.

This module must not be used to relabel the real-fMRI spatial phase-distribution
proxy as a reconstructed kimesurface.  Every estimator here assumes repeated
measurements with an observed phase coordinate at each longitudinal time.
Known synthetic truth is accepted only by evaluation helpers, never by model
selection or uncertainty estimation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Dict, Optional

import numpy as np

from spacekime_synthetic import (
    circular_kernel_predict,
    latent_kimesurface,
    reconstruct_kimesurface,
    smooth_kimesurface_time_phase,
)


SYNTHETIC_REPRESENTATION = "known_latent_repeated_measurement_kimesurface"
SUPPORTED_METHODS = ("kernel", "joint_kernel", "density_kernel", "stitching")


def synthetic_population_smoothed_surface(
    time: np.ndarray,
    theta_grid: np.ndarray,
    method: str,
    config: Mapping[str, float],
    phase_sampler: str,
    integration_points: int = 4096,
) -> np.ndarray:
    """Return the infinite-repeat smoothing target for the bundled simulator.

    This is a synthetic evaluation oracle, not an estimator and not an input to
    configuration selection or interval construction.  It numerically
    integrates the known latent surface against the simulator's phase-design
    density and the fixed reconstruction kernel.  Comparing a sampling band to
    this target separates finite-repeat variability from the structural bias of
    asking a smoothed estimator to cover the unsmoothed latent surface.

    Only ``kernel`` and ``joint_kernel`` are supported because their population
    targets are explicit ratios of design-weighted kernel integrals.
    """
    time_arr = np.asarray(time, dtype=np.float64)
    grid = np.asarray(theta_grid, dtype=np.float64)
    if time_arr.ndim != 1 or time_arr.size < 2 or np.any(~np.isfinite(time_arr)):
        raise ValueError("time must be a finite one-dimensional array")
    if np.any(np.diff(time_arr) <= 0):
        raise ValueError("time must be strictly increasing")
    if grid.ndim != 1 or grid.size < 2 or np.any(~np.isfinite(grid)):
        raise ValueError("theta_grid must be a finite one-dimensional array")
    method_name = _validate_method(method)
    if method_name not in ("kernel", "joint_kernel"):
        raise ValueError("population smoothing target supports kernel/joint_kernel only")
    n_integration = int(integration_points)
    if n_integration < 512:
        raise ValueError("integration_points must be at least 512")

    phase = np.linspace(-np.pi, np.pi, n_integration, endpoint=False)
    truth = latent_kimesurface(time_arr, phase)
    sampler = str(phase_sampler).strip().lower()
    if sampler == "uniform":
        density = np.ones_like(truth)
    elif sampler == "vonmises":
        center = 0.7 * np.sin(2.0 * np.pi * time_arr)[:, None]
        density = np.exp(1.5 * np.cos(phase[None, :] - center)) / np.i0(1.5)
    else:
        raise ValueError("phase_sampler must be 'uniform' or 'vonmises'")

    params = dict(config)
    default_kappa = 4.0 if method_name == "kernel" else 16.0
    kappa = float(params.get("kappa", default_kappa))
    if kappa <= 0 or not np.isfinite(kappa):
        raise ValueError("kappa must be finite and positive")
    phase_weights = np.exp(
        kappa * (np.cos(grid[:, None] - phase[None, :]) - 1.0)
    )
    output = np.empty((time_arr.size, grid.size), dtype=np.float64)
    if method_name == "kernel":
        for time_idx in range(time_arr.size):
            denominator = phase_weights @ density[time_idx]
            output[time_idx] = (phase_weights @ (density[time_idx] * truth[time_idx])) / denominator
        return output

    time_bandwidth = float(params.get("time_bandwidth", 0.05))
    truncate = float(params.get("truncate", 4.0))
    if time_bandwidth <= 0 or truncate <= 0:
        raise ValueError("time_bandwidth and truncate must be positive")
    for time_idx, query_time in enumerate(time_arr):
        distance = np.abs(time_arr - query_time)
        time_weights = np.exp(-0.5 * (distance / time_bandwidth) ** 2)
        time_weights[distance > truncate * time_bandwidth] = 0.0
        weighted_density = np.sum(time_weights[:, None] * density, axis=0)
        weighted_signal = np.sum(time_weights[:, None] * density * truth, axis=0)
        denominator = phase_weights @ weighted_density
        output[time_idx] = (phase_weights @ weighted_signal) / denominator
    return output


def reconstruction_support_diagnostics(
    theta_samples: np.ndarray,
    time: np.ndarray,
    theta_grid: np.ndarray,
    method: str,
    config: Mapping[str, float],
    *,
    relative_mass_threshold: float = 0.25,
    min_repeat_ess: float = 5.0,
) -> Dict[str, object]:
    """Define an observation-only supported time-phase domain.

    Support is deliberately a property of the observed phase design, not of
    the synthetic latent truth or of interval coverage.  A grid cell is kept
    when both of the following prespecified criteria hold:

    * its mean kernel mass is at least ``relative_mass_threshold`` times the
      largest mass at the same longitudinal time; and
    * its Kish effective sample size, computed across repeat-level kernel
      contributions, is at least ``min_repeat_ess``.

    For ``joint_kernel`` the within-repeat time-kernel contributions are first
    aggregated, so correlated time cells are not counted as independent
    repeats.  The mask is suitable for a separate supported-domain audit; it
    must not be chosen after inspecting synthetic coverage.
    """
    theta = np.asarray(theta_samples, dtype=np.float64)
    if theta.ndim != 2 or theta.shape[0] < 2 or theta.shape[1] < 1:
        raise ValueError("theta_samples must have shape (repeats,time)")
    time_arr = _validate_time(time, theta.shape[1])
    grid = np.asarray(theta_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or np.any(~np.isfinite(grid)):
        raise ValueError("theta_grid must be a finite one-dimensional array")
    method_name = _validate_method(method)
    if method_name not in ("kernel", "joint_kernel"):
        raise ValueError("support diagnostics support kernel/joint_kernel only")
    mass_threshold = float(relative_mass_threshold)
    ess_threshold = float(min_repeat_ess)
    if not 0 < mass_threshold <= 1 or not np.isfinite(mass_threshold):
        raise ValueError("relative_mass_threshold must be finite and in (0,1]")
    if ess_threshold < 2 or not np.isfinite(ess_threshold):
        raise ValueError("min_repeat_ess must be finite and at least two")

    params = dict(config)
    default_kappa = 4.0 if method_name == "kernel" else 16.0
    kappa = float(params.get("kappa", default_kappa))
    if kappa <= 0 or not np.isfinite(kappa):
        raise ValueError("kappa must be finite and positive")
    time_bandwidth = float(params.get("time_bandwidth", 0.05))
    truncate = float(params.get("truncate", 4.0))
    if method_name == "joint_kernel" and (
        time_bandwidth <= 0
        or truncate <= 0
        or not np.isfinite(time_bandwidth)
        or not np.isfinite(truncate)
    ):
        raise ValueError("time_bandwidth and truncate must be finite and positive")

    design_mass = np.full((time_arr.size, grid.size), np.nan, dtype=np.float64)
    repeat_ess = np.full_like(design_mass, np.nan)
    for time_idx, query_time in enumerate(time_arr):
        if method_name == "kernel":
            phase = theta[:, time_idx : time_idx + 1]
            valid = np.isfinite(phase)
            repeat_weight = np.exp(
                kappa * (np.cos(phase - grid[None, :]) - 1.0)
            )
            repeat_weight[~valid[:, 0], :] = 0.0
            valid_repeat = valid[:, 0]
        else:
            distance = np.abs(time_arr - query_time)
            window = distance <= truncate * time_bandwidth
            if not np.any(window):
                window[int(np.argmin(distance))] = True
            time_weight = np.exp(-0.5 * (distance[window] / time_bandwidth) ** 2)
            phase = theta[:, window]
            valid_cell = np.isfinite(phase)
            phase_weight = np.exp(
                kappa * (np.cos(phase[:, :, None] - grid[None, None, :]) - 1.0)
            )
            weighted_valid = valid_cell * time_weight[None, :]
            denominator = np.sum(weighted_valid, axis=1)
            repeat_weight = np.divide(
                np.sum(
                    phase_weight * weighted_valid[:, :, None],
                    axis=1,
                ),
                denominator[:, None],
                out=np.zeros((theta.shape[0], grid.size), dtype=np.float64),
                where=denominator[:, None] > 0,
            )
            valid_repeat = denominator > 0

        if np.any(valid_repeat):
            weights = repeat_weight[valid_repeat]
            design_mass[time_idx] = np.mean(weights, axis=0)
            weight_sum = np.sum(weights, axis=0)
            weight_square_sum = np.sum(weights**2, axis=0)
            repeat_ess[time_idx] = np.divide(
                weight_sum**2,
                weight_square_sum,
                out=np.zeros(grid.size, dtype=np.float64),
                where=weight_square_sum > 0,
            )

    row_max = np.nanmax(design_mass, axis=1, keepdims=True)
    relative_mass = np.divide(
        design_mass,
        row_max,
        out=np.zeros_like(design_mass),
        where=np.isfinite(row_max) & (row_max > 0),
    )
    supported = (
        np.isfinite(relative_mass)
        & np.isfinite(repeat_ess)
        & (relative_mass >= mass_threshold)
        & (repeat_ess >= ess_threshold)
    )
    return {
        "method": method_name,
        "config": dict(config),
        "design_mass": design_mass,
        "relative_design_mass": relative_mass,
        "repeat_effective_sample_size": repeat_ess,
        "supported_mask": supported,
        "supported_fraction": float(np.mean(supported)),
        "relative_mass_threshold": mass_threshold,
        "min_repeat_ess": ess_threshold,
        "effective_unit": "repeat",
        "truth_used_during_definition": False,
    }


def bootstrap_critical_value_diagnostics(
    max_statistics: np.ndarray,
    quantile: float,
    mc_confidence: float = 0.95,
) -> Dict[str, float | int]:
    """Quantify empirical critical-value Monte Carlo resolution.

    The interval uses the Dvoretzky--Kiefer--Wolfowitz bound for the empirical
    CDF.  It is intentionally conservative and diagnoses finite-bootstrap
    quantile resolution only; it does not establish statistical coverage of
    the resulting surface band.
    """
    values = np.sort(np.asarray(max_statistics, dtype=np.float64).ravel())
    values = values[np.isfinite(values)]
    probability = float(quantile)
    coverage = float(mc_confidence)
    if values.size < 20:
        raise ValueError("at least twenty finite maximum statistics are required")
    if not 0 < probability < 1 or not 0 < coverage < 1:
        raise ValueError("quantile and mc_confidence must be between zero and one")
    epsilon = math.sqrt(math.log(2.0 / (1.0 - coverage)) / (2.0 * values.size))
    lower_probability = max(0.0, probability - epsilon)
    upper_probability = min(1.0, probability + epsilon)
    estimate = float(np.quantile(values, probability, method="higher"))
    lower = float(np.quantile(values, lower_probability, method="lower"))
    upper = float(np.quantile(values, upper_probability, method="higher"))
    return {
        "bootstrap_resamples": int(values.size),
        "target_quantile": probability,
        "mc_confidence": coverage,
        "dkw_probability_epsilon": float(epsilon),
        "critical_value": estimate,
        "critical_value_mc_lower": lower,
        "critical_value_mc_upper": upper,
        "critical_value_mc_width": upper - lower,
    }


def inverse_density_sample_weights(
    theta_train: np.ndarray,
    density_kappa: float = 4.0,
    max_weight_ratio: float = 10.0,
) -> np.ndarray:
    """Estimate stabilized inverse-design weights with leave-one-out circular KDE.

    The returned weights have mean one over finite samples.  Isolated phase
    samples receive more weight, but clipping limits the variance inflation that
    unregularized inverse-density weighting can cause in poorly covered sectors.
    This is a sampling-design correction, not an uncertainty probability.
    """
    theta = np.asarray(theta_train, dtype=np.float64).ravel()
    if density_kappa <= 0 or not np.isfinite(density_kappa):
        raise ValueError("density_kappa must be finite and positive")
    if max_weight_ratio < 1 or not np.isfinite(max_weight_ratio):
        raise ValueError("max_weight_ratio must be finite and at least one")

    output = np.full(theta.shape, np.nan, dtype=np.float64)
    valid_idx = np.flatnonzero(np.isfinite(theta))
    if valid_idx.size < 3:
        output[valid_idx] = 1.0
        return output

    values = theta[valid_idx]
    delta = values[:, None] - values[None, :]
    kernels = np.exp(density_kappa * (np.cos(delta) - 1.0))
    np.fill_diagonal(kernels, 0.0)
    density = kernels.sum(axis=1) / max(values.size - 1, 1)
    positive = density[density > 0]
    if positive.size == 0:
        output[valid_idx] = 1.0
        return output

    # A small empirical floor plus explicit clipping stabilizes sparse designs.
    floor = max(float(np.quantile(positive, 0.05)) * 0.5, 1e-8)
    inverse = 1.0 / np.maximum(density, floor)
    median = max(float(np.median(inverse)), 1e-12)
    inverse = np.clip(inverse, median / max_weight_ratio, median * max_weight_ratio)
    inverse /= max(float(np.mean(inverse)), 1e-12)
    output[valid_idx] = inverse
    return output


def density_corrected_circular_predict(
    theta_train: np.ndarray,
    values_train: np.ndarray,
    theta_query: np.ndarray,
    kappa: float = 4.0,
    density_kappa: float = 4.0,
    max_weight_ratio: float = 10.0,
) -> np.ndarray:
    """Circular kernel prediction with stabilized inverse sampling-density weights."""
    theta = np.asarray(theta_train, dtype=np.float64).ravel()
    values = np.asarray(values_train, dtype=np.float64).ravel()
    query = np.asarray(theta_query, dtype=np.float64)
    if theta.size != values.size or theta.size == 0:
        raise ValueError("theta_train and values_train must be non-empty and equal length")
    if kappa <= 0 or not np.isfinite(kappa):
        raise ValueError("kappa must be finite and positive")

    design_weights = inverse_density_sample_weights(
        theta, density_kappa=density_kappa, max_weight_ratio=max_weight_ratio
    )
    valid_train = np.isfinite(theta) & np.isfinite(values) & np.isfinite(design_weights)
    flat_query = query.ravel()
    output = np.full(flat_query.shape, np.nan, dtype=np.float64)
    valid_query = np.isfinite(flat_query)
    if not np.any(valid_train) or not np.any(valid_query):
        return output.reshape(query.shape)

    delta = flat_query[valid_query, None] - theta[valid_train][None, :]
    log_weights = kappa * (np.cos(delta) - 1.0)
    log_weights += np.log(design_weights[valid_train][None, :])
    log_weights -= np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights)
    denominator = weights.sum(axis=1)
    numerator = weights @ values[valid_train]
    output[valid_query] = np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    return output.reshape(query.shape)


def periodic_stitching_predict(
    theta_train: np.ndarray,
    values_train: np.ndarray,
    theta_query: np.ndarray,
) -> np.ndarray:
    """Piecewise-linear periodic interpolation after sorting samples by phase.

    Exact duplicate phase samples are averaged before interpolation.  The first
    and last sorted samples are connected across the ``-pi/pi`` seam, making
    this a genuine circular stitching baseline rather than ordinary linear
    interpolation with edge extrapolation.
    """
    theta = np.asarray(theta_train, dtype=np.float64).ravel()
    values = np.asarray(values_train, dtype=np.float64).ravel()
    query = np.asarray(theta_query, dtype=np.float64)
    if theta.size != values.size or theta.size == 0:
        raise ValueError("theta_train and values_train must be non-empty and equal length")
    valid = np.isfinite(theta) & np.isfinite(values)
    if not np.any(valid):
        return np.full(query.shape, np.nan, dtype=np.float64)

    wrapped = np.mod(theta[valid] + np.pi, 2.0 * np.pi)
    order = np.argsort(wrapped)
    wrapped = wrapped[order]
    values = values[valid][order]
    unique, inverse = np.unique(wrapped, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    averaged = sums / np.maximum(counts, 1)
    if unique.size == 1:
        return np.full(query.shape, averaged[0], dtype=np.float64)

    x = np.concatenate([unique[-1:] - 2.0 * np.pi, unique, unique[:1] + 2.0 * np.pi])
    y = np.concatenate([averaged[-1:], averaged, averaged[:1]])
    flat_query = query.ravel()
    output = np.full(flat_query.shape, np.nan, dtype=np.float64)
    valid_query = np.isfinite(flat_query)
    output[valid_query] = np.interp(
        np.mod(flat_query[valid_query] + np.pi, 2.0 * np.pi), x, y
    )
    return output.reshape(query.shape)


def joint_time_phase_predict(
    theta_train: np.ndarray,
    values_train: np.ndarray,
    time: np.ndarray,
    theta_query: np.ndarray,
    kappa: float = 16.0,
    time_bandwidth: float = 0.05,
    truncate: float = 4.0,
) -> np.ndarray:
    """Predict repeated trajectories with a Gaussian-time × circular-phase kernel.

    ``time_bandwidth`` uses the physical units of ``time`` (the bundled simulator
    uses a normalized ``[0, 1]`` interval).  Phase remains periodic while time is
    not wrapped.  Query arrays must have shape ``(queries, time)``.
    """
    theta, values = _validate_repeated_arrays(theta_train, values_train)
    time_arr = _validate_time(time, theta.shape[1])
    query = np.asarray(theta_query, dtype=np.float64)
    if query.ndim == 1:
        query = query[None, :]
    if query.ndim != 2 or query.shape[1] != time_arr.size:
        raise ValueError("theta_query must have shape (queries,time)")
    if kappa <= 0 or not np.isfinite(kappa):
        raise ValueError("kappa must be finite and positive")
    if time_bandwidth <= 0 or not np.isfinite(time_bandwidth):
        raise ValueError("time_bandwidth must be finite and positive")
    if truncate <= 0 or not np.isfinite(truncate):
        raise ValueError("truncate must be finite and positive")

    output = np.full(query.shape, np.nan, dtype=np.float64)
    for t_idx, query_time in enumerate(time_arr):
        output[:, t_idx] = _joint_predict_at_time(
            theta,
            values,
            time_arr,
            query[:, t_idx],
            query_time,
            kappa,
            time_bandwidth,
            truncate,
        )
    return output


def reconstruct_advanced_surface(
    theta_samples: np.ndarray,
    observations: np.ndarray,
    time: np.ndarray,
    theta_grid: np.ndarray,
    method: str,
    config: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Reconstruct a synthetic repeated-measurement surface with one named method."""
    theta, values = _validate_repeated_arrays(theta_samples, observations)
    time_arr = _validate_time(time, theta.shape[1])
    grid = np.asarray(theta_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("theta_grid must be one-dimensional with at least two points")
    method_name = _validate_method(method)
    params = dict(config or {})

    if method_name == "kernel":
        return reconstruct_kimesurface(
            theta, values, grid, kappa=float(params.get("kappa", 4.0))
        )
    if method_name == "density_kernel":
        surface = np.empty((time_arr.size, grid.size), dtype=np.float64)
        for t_idx in range(time_arr.size):
            surface[t_idx] = density_corrected_circular_predict(
                theta[:, t_idx],
                values[:, t_idx],
                grid,
                kappa=float(params.get("kappa", 4.0)),
                density_kappa=float(params.get("density_kappa", 4.0)),
                max_weight_ratio=float(params.get("max_weight_ratio", 10.0)),
            )
        return surface
    if method_name == "stitching":
        surface = np.empty((time_arr.size, grid.size), dtype=np.float64)
        for t_idx in range(time_arr.size):
            surface[t_idx] = periodic_stitching_predict(
                theta[:, t_idx], values[:, t_idx], grid
            )
        return smooth_kimesurface_time_phase(
            surface, time_sigma=float(params.get("time_sigma", 0.0)), theta_sigma=0.0
        )

    kappa = float(params.get("kappa", 16.0))
    time_bandwidth = float(params.get("time_bandwidth", 0.05))
    surface = np.empty((time_arr.size, grid.size), dtype=np.float64)
    for t_idx, query_time in enumerate(time_arr):
        surface[t_idx] = _joint_predict_at_time(
            theta,
            values,
            time_arr,
            grid,
            query_time,
            kappa,
            time_bandwidth,
            float(params.get("truncate", 4.0)),
        )
    return surface


def predict_repeated_measurements(
    theta_train: np.ndarray,
    observations_train: np.ndarray,
    time: np.ndarray,
    theta_query: np.ndarray,
    method: str,
    config: Optional[Mapping[str, float]] = None,
    theta_grid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Predict phase-indexed repeated trajectories for observation-only CV."""
    theta, values = _validate_repeated_arrays(theta_train, observations_train)
    time_arr = _validate_time(time, theta.shape[1])
    query = np.asarray(theta_query, dtype=np.float64)
    if query.ndim == 1:
        query = query[None, :]
    if query.ndim != 2 or query.shape[1] != time_arr.size:
        raise ValueError("theta_query must have shape (queries,time)")
    method_name = _validate_method(method)
    params = dict(config or {})

    if method_name == "joint_kernel":
        return joint_time_phase_predict(
            theta,
            values,
            time_arr,
            query,
            kappa=float(params.get("kappa", 16.0)),
            time_bandwidth=float(params.get("time_bandwidth", 0.05)),
            truncate=float(params.get("truncate", 4.0)),
        )
    if method_name == "stitching" and float(params.get("time_sigma", 0.0)) > 0:
        grid = (
            np.linspace(-np.pi, np.pi, 128, endpoint=False)
            if theta_grid is None
            else np.asarray(theta_grid, dtype=np.float64)
        )
        surface = reconstruct_advanced_surface(
            theta, values, time_arr, grid, method_name, params
        )
        output = np.empty(query.shape, dtype=np.float64)
        for t_idx in range(time_arr.size):
            output[:, t_idx] = periodic_stitching_predict(
                grid, surface[t_idx], query[:, t_idx]
            )
        return output

    output = np.empty(query.shape, dtype=np.float64)
    for t_idx in range(time_arr.size):
        if method_name == "kernel":
            output[:, t_idx] = circular_kernel_predict(
                theta[:, t_idx],
                values[:, t_idx],
                query[:, t_idx],
                kappa=float(params.get("kappa", 4.0)),
            )
        elif method_name == "density_kernel":
            output[:, t_idx] = density_corrected_circular_predict(
                theta[:, t_idx],
                values[:, t_idx],
                query[:, t_idx],
                kappa=float(params.get("kappa", 4.0)),
                density_kappa=float(params.get("density_kappa", 4.0)),
                max_weight_ratio=float(params.get("max_weight_ratio", 10.0)),
            )
        else:
            output[:, t_idx] = periodic_stitching_predict(
                theta[:, t_idx], values[:, t_idx], query[:, t_idx]
            )
    return output


def default_method_configs(
    method: str,
    kappas: Sequence[float] = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0),
    time_bandwidths: Sequence[float] = (0.025, 0.05, 0.10),
    stitching_time_sigmas: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    density_kappa: float = 4.0,
    max_weight_ratio: float = 10.0,
) -> list[dict[str, float]]:
    """Build deterministic observation-CV candidate grids for each method."""
    method_name = _validate_method(method)
    kappa_values = _positive_unique(kappas, "kappas")
    if method_name == "kernel":
        return [{"kappa": value} for value in kappa_values]
    if method_name == "density_kernel":
        return [
            {
                "kappa": value,
                "density_kappa": float(density_kappa),
                "max_weight_ratio": float(max_weight_ratio),
            }
            for value in kappa_values
        ]
    if method_name == "joint_kernel":
        bandwidths = _positive_unique(time_bandwidths, "time_bandwidths")
        return [
            {"kappa": kappa, "time_bandwidth": bandwidth}
            for kappa in kappa_values
            for bandwidth in bandwidths
        ]
    sigmas = sorted({float(value) for value in stitching_time_sigmas})
    if not sigmas or any(value < 0 or not np.isfinite(value) for value in sigmas):
        raise ValueError("stitching_time_sigmas must be finite and non-negative")
    return [{"time_sigma": value} for value in sigmas]


def select_reconstruction_config(
    theta_samples: np.ndarray,
    observations: np.ndarray,
    time: np.ndarray,
    method: str,
    configs: Sequence[Mapping[str, float]],
    cv_folds: int = 4,
    rng: Optional[np.random.Generator] = None,
    theta_grid: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Select a reconstruction configuration using noisy observations only."""
    theta, values = _validate_repeated_arrays(theta_samples, observations)
    time_arr = _validate_time(time, theta.shape[1])
    if theta.shape[0] < 4:
        raise ValueError("at least four repeats are required for model selection")
    config_values = [dict(config) for config in configs]
    if not config_values:
        raise ValueError("configs cannot be empty")
    fold_count = int(np.clip(int(cv_folds), 2, theta.shape[0]))
    generator = np.random.default_rng(0) if rng is None else rng
    folds = [fold for fold in np.array_split(generator.permutation(theta.shape[0]), fold_count) if fold.size]
    all_idx = np.arange(theta.shape[0])

    candidate_rmse: dict[str, float] = {}
    for config in config_values:
        squared_errors: list[np.ndarray] = []
        for validation_idx in folds:
            train_idx = np.setdiff1d(all_idx, validation_idx, assume_unique=False)
            prediction = predict_repeated_measurements(
                theta[train_idx],
                values[train_idx],
                time_arr,
                theta[validation_idx],
                method,
                config,
                theta_grid=theta_grid,
            )
            target = values[validation_idx]
            valid = np.isfinite(prediction) & np.isfinite(target)
            if np.any(valid):
                squared_errors.append((prediction[valid] - target[valid]) ** 2)
        key = _config_key(config)
        candidate_rmse[key] = (
            float(np.sqrt(np.mean(np.concatenate(squared_errors))))
            if squared_errors
            else float("inf")
        )

    selected = min(config_values, key=lambda config: candidate_rmse[_config_key(config)])
    return {
        "method": _validate_method(method),
        "selected_config": selected,
        "cv_rmse": candidate_rmse[_config_key(selected)],
        "candidate_rmse": candidate_rmse,
        "cv_scheme": "repeat_kfold_observations_only",
        "cv_folds": len(folds),
        "repeats": int(theta.shape[0]),
    }


def nested_method_holdout_metrics(
    theta_samples: np.ndarray,
    observations: np.ndarray,
    true_signal: np.ndarray,
    time: np.ndarray,
    method: str,
    configs: Sequence[Mapping[str, float]],
    holdout_fraction: float = 0.25,
    inner_folds: int = 4,
    split_seed: int = 0,
    cv_seed: int = 1,
    theta_grid: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Leakage-controlled outer holdout shared by all advanced methods.

    Latent truth is used only after configuration selection to score untouched
    outer repeats.  ``outer_observed_rmse`` is also reported so the effect of
    measurement noise remains explicit.
    """
    theta, values = _validate_repeated_arrays(theta_samples, observations)
    truth = np.asarray(true_signal, dtype=np.float64)
    time_arr = _validate_time(time, theta.shape[1])
    if truth.shape != theta.shape:
        raise ValueError("true_signal must share shape with theta_samples")
    if theta.shape[0] < 6:
        raise ValueError("at least six repeats are required for nested holdout")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")

    permutation = np.random.default_rng(split_seed).permutation(theta.shape[0])
    test_count = int(np.clip(round(theta.shape[0] * holdout_fraction), 1, theta.shape[0] - 4))
    test_idx = permutation[:test_count]
    train_idx = permutation[test_count:]
    selection = select_reconstruction_config(
        theta[train_idx],
        values[train_idx],
        time_arr,
        method,
        configs,
        cv_folds=inner_folds,
        rng=np.random.default_rng(cv_seed),
        theta_grid=theta_grid,
    )
    prediction = predict_repeated_measurements(
        theta[train_idx],
        values[train_idx],
        time_arr,
        theta[test_idx],
        method,
        selection["selected_config"],
        theta_grid=theta_grid,
    )
    baseline = np.broadcast_to(
        np.nanmean(values[train_idx], axis=0), prediction.shape
    )
    observed_rmse = _paired_rmse(prediction, values[test_idx])
    truth_rmse = _paired_rmse(prediction, truth[test_idx])
    baseline_truth_rmse = _paired_rmse(baseline, truth[test_idx])
    return {
        "method": _validate_method(method),
        "train_repeats": int(train_idx.size),
        "test_repeats": int(test_idx.size),
        "selected_config": selection["selected_config"],
        "inner_cv_rmse": float(selection["cv_rmse"]),
        "inner_candidate_rmse": selection["candidate_rmse"],
        "inner_cv_scheme": selection["cv_scheme"],
        "inner_cv_folds": selection["cv_folds"],
        "outer_observed_rmse": observed_rmse,
        "outer_truth_rmse": truth_rmse,
        "time_mean_baseline_truth_rmse": baseline_truth_rmse,
        "relative_truth_rmse_improvement": (
            baseline_truth_rmse - truth_rmse
        ) / max(baseline_truth_rmse, 1e-12),
        "split_seed": int(split_seed),
        "cv_seed": int(cv_seed),
    }


def bootstrap_surface_uncertainty(
    theta_samples: np.ndarray,
    observations: np.ndarray,
    time: np.ndarray,
    theta_grid: np.ndarray,
    method: str,
    config: Mapping[str, float],
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Estimate pointwise repeat-sampling uncertainty without consulting truth.

    This percentile bootstrap conditions on the already selected configuration;
    it measures repeat-sampling variability, not measurement-noise prediction
    intervals, Bayesian posterior uncertainty, or hyperparameter-selection
    uncertainty.
    """
    theta, values = _validate_repeated_arrays(theta_samples, observations)
    time_arr = _validate_time(time, theta.shape[1])
    grid = np.asarray(theta_grid, dtype=np.float64)
    if int(n_bootstrap) < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    generator = np.random.default_rng(0) if rng is None else rng
    point = reconstruct_advanced_surface(theta, values, time_arr, grid, method, config)
    samples = np.empty((int(n_bootstrap),) + point.shape, dtype=np.float64)
    for bootstrap_idx in range(int(n_bootstrap)):
        repeat_idx = generator.integers(0, theta.shape[0], size=theta.shape[0])
        samples[bootstrap_idx] = reconstruct_advanced_surface(
            theta[repeat_idx], values[repeat_idx], time_arr, grid, method, config
        )
    alpha = (1.0 - confidence) / 2.0
    bootstrap_mean = np.nanmean(samples, axis=0)
    lower_quantile = np.nanquantile(samples, alpha, axis=0)
    upper_quantile = np.nanquantile(samples, 1.0 - alpha, axis=0)
    return {
        "representation": SYNTHETIC_REPRESENTATION,
        "uncertainty_kind": "repeat_bootstrap_conditional_on_selected_configuration",
        "method": _validate_method(method),
        "config": dict(config),
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
        "point_estimate": point,
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_bias": bootstrap_mean - point,
        "bias_corrected_point": 2.0 * point - bootstrap_mean,
        "bootstrap_std": np.nanstd(samples, axis=0, ddof=1),
        "lower": lower_quantile,
        "upper": upper_quantile,
        "basic_lower": 2.0 * point - upper_quantile,
        "basic_upper": 2.0 * point - lower_quantile,
    }


def bootstrap_surface_uncertainty_reselected(
    theta_samples: np.ndarray,
    observations: np.ndarray,
    time: np.ndarray,
    theta_grid: np.ndarray,
    method: str,
    configs: Sequence[Mapping[str, float]],
    n_bootstrap: int = 100,
    confidence: float = 0.95,
    cv_folds: int = 4,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Repeat bootstrap that re-runs observation-only configuration selection.

    Unlike :func:`bootstrap_surface_uncertainty`, every bootstrap resample runs
    cross-validation again.  The resulting distribution therefore includes a
    measured component of hyperparameter-selection instability.  It still does
    not constitute a Bayesian posterior or a measurement-noise prediction band.
    """
    theta, values = _validate_repeated_arrays(theta_samples, observations)
    time_arr = _validate_time(time, theta.shape[1])
    grid = np.asarray(theta_grid, dtype=np.float64)
    config_values = [dict(config) for config in configs]
    if not config_values:
        raise ValueError("configs cannot be empty")
    if int(n_bootstrap) < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    generator = np.random.default_rng(0) if rng is None else rng
    full_selection = select_reconstruction_config(
        theta,
        values,
        time_arr,
        method,
        config_values,
        cv_folds=cv_folds,
        rng=np.random.default_rng(generator.integers(0, 2**32 - 1)),
        theta_grid=grid,
    )
    point = reconstruct_advanced_surface(
        theta, values, time_arr, grid, method, full_selection["selected_config"]
    )
    samples = np.empty((int(n_bootstrap),) + point.shape, dtype=np.float64)
    selected_keys: list[str] = []
    for bootstrap_idx in range(int(n_bootstrap)):
        repeat_idx = generator.integers(0, theta.shape[0], size=theta.shape[0])
        selection = select_reconstruction_config(
            theta[repeat_idx],
            values[repeat_idx],
            time_arr,
            method,
            config_values,
            cv_folds=cv_folds,
            rng=np.random.default_rng(generator.integers(0, 2**32 - 1)),
            theta_grid=grid,
        )
        selected_keys.append(_config_key(selection["selected_config"]))
        samples[bootstrap_idx] = reconstruct_advanced_surface(
            theta[repeat_idx],
            values[repeat_idx],
            time_arr,
            grid,
            method,
            selection["selected_config"],
        )
    alpha = (1.0 - confidence) / 2.0
    bootstrap_mean = np.nanmean(samples, axis=0)
    lower_quantile = np.nanquantile(samples, alpha, axis=0)
    upper_quantile = np.nanquantile(samples, 1.0 - alpha, axis=0)
    frequency = {
        key: selected_keys.count(key) for key in sorted(set(selected_keys))
    }
    return {
        "representation": SYNTHETIC_REPRESENTATION,
        "uncertainty_kind": "repeat_bootstrap_with_observation_only_config_reselection",
        "method": _validate_method(method),
        "config": dict(full_selection["selected_config"]),
        "selection_frequency": frequency,
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
        "point_estimate": point,
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_bias": bootstrap_mean - point,
        "bias_corrected_point": 2.0 * point - bootstrap_mean,
        "bootstrap_std": np.nanstd(samples, axis=0, ddof=1),
        "lower": lower_quantile,
        "upper": upper_quantile,
        "basic_lower": 2.0 * point - upper_quantile,
        "basic_upper": 2.0 * point - lower_quantile,
        "truth_used_during_estimation": False,
    }


def undersmooth_reconstruction_config(
    method: str,
    config: Mapping[str, float],
    factor: float = 2.0,
) -> Dict[str, float]:
    """Return a prespecified lower-bias configuration for sensitivity analysis.

    ``factor`` is chosen before seeing synthetic truth.  Kernel concentration is
    increased, joint time bandwidth is reduced, and stitching time smoothing is
    reduced.  The result is a variance-increasing sensitivity configuration,
    not an automatically optimal or theoretically bias-free choice.
    """
    method_name = _validate_method(method)
    multiplier = float(factor)
    if multiplier <= 1.0 or not np.isfinite(multiplier):
        raise ValueError("factor must be finite and greater than one")
    output = {str(key): float(value) for key, value in dict(config).items()}
    if method_name in ("kernel", "density_kernel"):
        output["kappa"] = float(output.get("kappa", 4.0)) * multiplier
    elif method_name == "joint_kernel":
        output["kappa"] = float(output.get("kappa", 16.0)) * multiplier
        output["time_bandwidth"] = (
            float(output.get("time_bandwidth", 0.05)) / np.sqrt(multiplier)
        )
    else:
        output["time_sigma"] = float(output.get("time_sigma", 0.0)) / multiplier
    return output


def simultaneous_bootstrap_surface_band(
    theta_samples: np.ndarray,
    observations: np.ndarray,
    time: np.ndarray,
    theta_grid: np.ndarray,
    method: str,
    config: Mapping[str, float],
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    rng: Optional[np.random.Generator] = None,
    scale_floor_fraction: float = 0.05,
    domain_mask: Optional[np.ndarray] = None,
    mc_confidence: float = 0.95,
    precomputed_bootstrap_samples: Optional[np.ndarray] = None,
    return_bootstrap_samples: bool = False,
) -> Dict[str, object]:
    """Build an empirical simultaneous repeat-bootstrap band for a surface.

    The critical value is the bootstrap quantile of the maximum absolute
    studentized difference between each resampled estimate and the original
    estimate over the full time-phase grid.  The primary band is centered on
    the original estimate, matching the standard symmetric bootstrap inversion.
    A first-order bias-corrected center and band are returned only as a separate
    sensitivity result.  The procedure conditions on a fixed, observation-only
    or prespecified reconstruction configuration and targets repeat-sampling
    uncertainty.  It is neither a Bayesian credible band nor a prediction
    interval for future noisy measurements.  Callers auditing several fixed
    domains for the same estimator may request and reuse the bootstrap surface
    draws so domain comparisons do not repeat the expensive reconstructions.
    """
    theta, values = _validate_repeated_arrays(theta_samples, observations)
    time_arr = _validate_time(time, theta.shape[1])
    grid = np.asarray(theta_grid, dtype=np.float64)
    if grid.ndim != 1 or grid.size < 2 or np.any(~np.isfinite(grid)):
        raise ValueError("theta_grid must be a finite one-dimensional array")
    if int(n_bootstrap) < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    floor_fraction = float(scale_floor_fraction)
    if floor_fraction <= 0 or not np.isfinite(floor_fraction):
        raise ValueError("scale_floor_fraction must be finite and positive")
    if domain_mask is None:
        domain = np.ones((time_arr.size, grid.size), dtype=bool)
        domain_kind = "full_time_phase_grid"
    else:
        domain = np.asarray(domain_mask, dtype=bool)
        if domain.shape != (time_arr.size, grid.size) or not np.any(domain):
            raise ValueError("domain_mask must match the surface and contain a cell")
        domain_kind = "prespecified_observation_only_supported_grid"

    generator = np.random.default_rng(0) if rng is None else rng
    point = reconstruct_advanced_surface(theta, values, time_arr, grid, method, config)
    if precomputed_bootstrap_samples is None:
        samples = np.empty((int(n_bootstrap),) + point.shape, dtype=np.float64)
        for bootstrap_idx in range(int(n_bootstrap)):
            repeat_idx = generator.integers(0, theta.shape[0], size=theta.shape[0])
            samples[bootstrap_idx] = reconstruct_advanced_surface(
                theta[repeat_idx], values[repeat_idx], time_arr, grid, method, config
            )
    else:
        samples = np.asarray(precomputed_bootstrap_samples, dtype=np.float64)
        expected_shape = (int(n_bootstrap),) + point.shape
        if samples.shape != expected_shape:
            raise ValueError(
                "precomputed_bootstrap_samples must have shape "
                f"{expected_shape}"
            )

    bootstrap_mean = np.nanmean(samples, axis=0)
    bootstrap_std = np.nanstd(samples, axis=0, ddof=1)
    positive_scale = bootstrap_std[np.isfinite(bootstrap_std) & (bootstrap_std > 0)]
    reference_scale = float(np.median(positive_scale)) if positive_scale.size else 1.0
    scale_floor = max(reference_scale * floor_fraction, np.finfo(float).eps)
    studentization_scale = np.maximum(bootstrap_std, scale_floor)
    standardized = np.abs(samples - point[None, ...]) / studentization_scale
    max_statistics = np.nanmax(standardized[:, domain], axis=1)
    finite_statistics = max_statistics[np.isfinite(max_statistics)]
    if finite_statistics.size < max(2, int(n_bootstrap) // 2):
        raise ValueError("too few finite bootstrap maximum statistics")
    critical_value = float(
        np.quantile(finite_statistics, confidence, method="higher")
    )
    bias_corrected = 2.0 * point - bootstrap_mean
    radius = critical_value * studentization_scale
    centered_statistics = np.nanmax(
        (
            np.abs(samples - bootstrap_mean[None, ...])
            / studentization_scale
        )[:, domain],
        axis=1,
    )
    centered_statistics = centered_statistics[np.isfinite(centered_statistics)]
    bias_corrected_critical = float(
        np.quantile(centered_statistics, confidence, method="higher")
    )
    bias_corrected_radius = bias_corrected_critical * studentization_scale
    result: Dict[str, object] = {
        "representation": SYNTHETIC_REPRESENTATION,
        "uncertainty_kind": "simultaneous_studentized_repeat_bootstrap_fixed_config",
        "target": (
            "full_latent_surface_repeat_sampling_variability"
            if domain_mask is None
            else "supported_domain_latent_surface_repeat_sampling_variability"
        ),
        "method": _validate_method(method),
        "config": dict(config),
        "confidence": float(confidence),
        "n_bootstrap": int(n_bootstrap),
        "point_estimate": point,
        "band_center": point,
        "center_method": "original_estimate_symmetric_bootstrap_inversion",
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_bias": bootstrap_mean - point,
        "bias_corrected_point": bias_corrected,
        "bootstrap_std": bootstrap_std,
        "studentization_scale": studentization_scale,
        "scale_floor": float(scale_floor),
        "max_statistics": finite_statistics,
        "critical_value": critical_value,
        "critical_value_mc_diagnostics": bootstrap_critical_value_diagnostics(
            finite_statistics, confidence, mc_confidence=mc_confidence
        ),
        "domain_mask": domain,
        "domain_kind": domain_kind,
        "domain_fraction": float(np.mean(domain)),
        "lower": point - radius,
        "upper": point + radius,
        "bias_corrected_critical_value": bias_corrected_critical,
        "bias_corrected_lower": bias_corrected - bias_corrected_radius,
        "bias_corrected_upper": bias_corrected + bias_corrected_radius,
        "truth_used_during_estimation": False,
        "is_bayesian": False,
    }
    if return_bootstrap_samples:
        result["bootstrap_samples"] = samples
    return result


def simultaneous_surface_coverage_metrics(
    truth: np.ndarray,
    band: Mapping[str, object],
    evaluation_mask: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Audit a completed simultaneous band against known synthetic truth."""
    truth_arr = np.asarray(truth, dtype=np.float64)
    center = np.asarray(
        band.get("band_center", band["bias_corrected_point"]), dtype=np.float64
    )
    lower = np.asarray(band["lower"], dtype=np.float64)
    upper = np.asarray(band["upper"], dtype=np.float64)
    scale = np.asarray(band["studentization_scale"], dtype=np.float64)
    if not (truth_arr.shape == center.shape == lower.shape == upper.shape == scale.shape):
        raise ValueError("truth and simultaneous-band surfaces must share shape")
    if evaluation_mask is None:
        domain = np.asarray(
            band.get("domain_mask", np.ones(truth_arr.shape, dtype=bool)),
            dtype=bool,
        )
    else:
        domain = np.asarray(evaluation_mask, dtype=bool)
    if domain.shape != truth_arr.shape or not np.any(domain):
        raise ValueError("evaluation mask must match truth and contain a cell")
    valid = (
        np.isfinite(truth_arr)
        & np.isfinite(center)
        & np.isfinite(lower)
        & np.isfinite(upper)
        & np.isfinite(scale)
        & domain
    )
    if not np.any(valid):
        return {
            "pointwise_coverage": float("nan"),
            "whole_surface_covered": False,
            "mean_band_width": float("nan"),
            "max_studentized_truth_error": float("nan"),
            "max_error_to_critical_ratio": float("nan"),
            "band_center_surface_nrmse": float("nan"),
            "evaluated_cells": 0,
            "evaluated_fraction": 0.0,
        }
    covered = (truth_arr >= lower) & (truth_arr <= upper) & valid
    max_error = float(np.max(np.abs(center[valid] - truth_arr[valid]) / scale[valid]))
    critical = float(band["critical_value"])
    return {
        "pointwise_coverage": float(np.mean(covered[valid])),
        "whole_surface_covered": bool(np.all(covered[valid])),
        "mean_band_width": float(np.mean((upper - lower)[valid])),
        "max_studentized_truth_error": max_error,
        "max_error_to_critical_ratio": max_error / max(critical, 1e-12),
        "band_center_surface_nrmse": float(
            np.sqrt(np.mean((center[valid] - truth_arr[valid]) ** 2))
            / max(float(np.std(truth_arr[valid])), 1e-12)
        ),
        "evaluated_cells": int(np.sum(valid)),
        "evaluated_fraction": float(np.mean(valid)),
    }


def split_conformal_prediction_intervals(
    theta_samples: np.ndarray,
    observations: np.ndarray,
    time: np.ndarray,
    method: str,
    configs: Sequence[Mapping[str, float]],
    confidence: float = 0.95,
    calibration_fraction: float = 0.25,
    test_fraction: float = 0.25,
    cv_folds: int = 4,
    split_seed: int = 0,
    cv_seed: int = 1,
    theta_grid: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Construct split-conformal prediction bands for unseen noisy repeats.

    A maximum absolute residual per calibration repeat is the conformity score,
    so repeats--rather than correlated time cells--are the exchangeable units.
    The interval targets a future *observed trajectory*.  It is not a confidence
    band for the full latent surface; synthetic latent signal may be audited
    afterward with :func:`conformal_interval_metrics`.
    """
    theta, values = _validate_repeated_arrays(theta_samples, observations)
    time_arr = _validate_time(time, theta.shape[1])
    config_values = [dict(config) for config in configs]
    if theta.shape[0] < 12:
        raise ValueError("at least twelve repeats are required for split conformal")
    if not config_values:
        raise ValueError("configs cannot be empty")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    if not 0 < calibration_fraction < 0.5 or not 0 < test_fraction < 0.5:
        raise ValueError("calibration_fraction and test_fraction must be in (0,0.5)")
    permutation = np.random.default_rng(split_seed).permutation(theta.shape[0])
    calibration_count = int(max(2, round(theta.shape[0] * calibration_fraction)))
    test_count = int(max(1, round(theta.shape[0] * test_fraction)))
    if calibration_count + test_count > theta.shape[0] - 4:
        raise ValueError("split leaves fewer than four training repeats")
    calibration_idx = permutation[:calibration_count]
    test_idx = permutation[calibration_count : calibration_count + test_count]
    train_idx = permutation[calibration_count + test_count :]
    selection = select_reconstruction_config(
        theta[train_idx],
        values[train_idx],
        time_arr,
        method,
        config_values,
        cv_folds=cv_folds,
        rng=np.random.default_rng(cv_seed),
        theta_grid=theta_grid,
    )
    calibration_prediction = predict_repeated_measurements(
        theta[train_idx],
        values[train_idx],
        time_arr,
        theta[calibration_idx],
        method,
        selection["selected_config"],
        theta_grid=theta_grid,
    )
    residual = np.abs(values[calibration_idx] - calibration_prediction)
    repeat_scores = np.nanmax(residual, axis=1)
    repeat_scores = repeat_scores[np.isfinite(repeat_scores)]
    if repeat_scores.size < 2:
        raise ValueError("too few finite calibration-repeat scores")
    quantile_level = min(
        1.0, np.ceil((repeat_scores.size + 1) * confidence) / repeat_scores.size
    )
    radius = float(np.quantile(repeat_scores, quantile_level, method="higher"))
    prediction = predict_repeated_measurements(
        theta[train_idx],
        values[train_idx],
        time_arr,
        theta[test_idx],
        method,
        selection["selected_config"],
        theta_grid=theta_grid,
    )
    return {
        "representation": SYNTHETIC_REPRESENTATION,
        "uncertainty_kind": "split_conformal_future_observed_repeat_trajectory",
        "target": "future_noisy_observed_trajectory",
        "exchangeable_unit": "repeat",
        "score": "maximum_absolute_residual_over_time",
        "method": _validate_method(method),
        "config": dict(selection["selected_config"]),
        "confidence": float(confidence),
        "quantile_level": float(quantile_level),
        "radius": radius,
        "prediction": prediction,
        "lower": prediction - radius,
        "upper": prediction + radius,
        "train_indices": train_idx,
        "calibration_indices": calibration_idx,
        "test_indices": test_idx,
        "calibration_repeat_scores": repeat_scores,
        "truth_used_during_estimation": False,
    }


def conformal_interval_metrics(
    target: np.ndarray, conformal: Mapping[str, object]
) -> Dict[str, float]:
    """Score pointwise and whole-trajectory coverage for a supplied target."""
    target_arr = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(conformal["prediction"], dtype=np.float64)
    lower = np.asarray(conformal["lower"], dtype=np.float64)
    upper = np.asarray(conformal["upper"], dtype=np.float64)
    if not (target_arr.shape == prediction.shape == lower.shape == upper.shape):
        raise ValueError("target and conformal interval arrays must share shape")
    valid = np.isfinite(target_arr) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(valid):
        return {
            "pointwise_coverage": float("nan"),
            "trajectory_coverage": float("nan"),
            "mean_interval_width": float("nan"),
            "prediction_rmse": float("nan"),
        }
    pointwise = (target_arr >= lower) & (target_arr <= upper) & valid
    trajectory_valid = np.any(valid, axis=1)
    trajectory_covered = np.all(pointwise | ~valid, axis=1)
    return {
        "pointwise_coverage": float(np.mean(pointwise[valid])),
        "trajectory_coverage": float(np.mean(trajectory_covered[trajectory_valid])),
        "mean_interval_width": float(np.mean((upper - lower)[valid])),
        "prediction_rmse": _paired_rmse(prediction, target_arr),
    }


def uncertainty_calibration_metrics(
    truth: np.ndarray,
    uncertainty: Mapping[str, object],
    interval: str = "percentile",
    point_key: str = "point_estimate",
) -> Dict[str, float]:
    """Evaluate synthetic-truth coverage after uncertainty estimation is complete."""
    truth_arr = np.asarray(truth, dtype=np.float64)
    if interval == "percentile":
        lower_key, upper_key = "lower", "upper"
    elif interval == "basic":
        lower_key, upper_key = "basic_lower", "basic_upper"
    else:
        raise ValueError("interval must be 'percentile' or 'basic'")
    point = np.asarray(uncertainty[point_key], dtype=np.float64)
    lower = np.asarray(uncertainty[lower_key], dtype=np.float64)
    upper = np.asarray(uncertainty[upper_key], dtype=np.float64)
    std = np.asarray(uncertainty["bootstrap_std"], dtype=np.float64)
    if not (truth_arr.shape == point.shape == lower.shape == upper.shape == std.shape):
        raise ValueError("truth and uncertainty surfaces must share shape")
    valid = (
        np.isfinite(truth_arr)
        & np.isfinite(point)
        & np.isfinite(lower)
        & np.isfinite(upper)
        & np.isfinite(std)
    )
    if not np.any(valid):
        return {key: float("nan") for key in (
            "empirical_coverage", "mean_interval_width", "median_interval_width",
            "point_nrmse_std", "mean_abs_error_over_std"
        )}
    width = upper[valid] - lower[valid]
    error = point[valid] - truth_arr[valid]
    return {
        "empirical_coverage": float(
            np.mean((truth_arr[valid] >= lower[valid]) & (truth_arr[valid] <= upper[valid]))
        ),
        "mean_interval_width": float(np.mean(width)),
        "median_interval_width": float(np.median(width)),
        "point_nrmse_std": float(
            np.sqrt(np.mean(error**2)) / max(float(np.std(truth_arr[valid])), 1e-12)
        ),
        "mean_abs_error_over_std": float(
            np.mean(np.abs(error) / np.maximum(std[valid], 1e-12))
        ),
    }


def _joint_predict_at_time(
    theta: np.ndarray,
    values: np.ndarray,
    time: np.ndarray,
    theta_query: np.ndarray,
    query_time: float,
    kappa: float,
    time_bandwidth: float,
    truncate: float,
) -> np.ndarray:
    query = np.asarray(theta_query, dtype=np.float64).ravel()
    time_distance = np.abs(time - query_time)
    window = time_distance <= truncate * time_bandwidth
    if not np.any(window):
        window[int(np.argmin(time_distance))] = True
    time_weights = np.exp(-0.5 * (time_distance[window] / time_bandwidth) ** 2)
    train_theta = theta[:, window].ravel()
    train_values = values[:, window].ravel()
    design_time_weights = np.tile(time_weights, theta.shape[0])
    valid_train = (
        np.isfinite(train_theta)
        & np.isfinite(train_values)
        & np.isfinite(design_time_weights)
    )
    output = np.full(query.shape, np.nan, dtype=np.float64)
    valid_query = np.isfinite(query)
    if not np.any(valid_train) or not np.any(valid_query):
        return output
    delta = query[valid_query, None] - train_theta[valid_train][None, :]
    log_weights = kappa * (np.cos(delta) - 1.0)
    log_weights += np.log(design_time_weights[valid_train][None, :])
    log_weights -= np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights)
    denominator = weights.sum(axis=1)
    numerator = weights @ train_values[valid_train]
    output[valid_query] = np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan, dtype=np.float64),
        where=denominator > 0,
    )
    return output


def _validate_repeated_arrays(
    theta_samples: np.ndarray, observations: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.asarray(theta_samples, dtype=np.float64)
    values = np.asarray(observations, dtype=np.float64)
    if theta.shape != values.shape or theta.ndim != 2:
        raise ValueError("theta_samples and observations must share shape (repeats,time)")
    if theta.shape[0] < 2 or theta.shape[1] < 1:
        raise ValueError("at least two repeats and one time point are required")
    return theta, values


def _validate_time(time: np.ndarray, n_time: int) -> np.ndarray:
    time_arr = np.asarray(time, dtype=np.float64)
    if time_arr.ndim != 1 or time_arr.size != n_time or not np.all(np.isfinite(time_arr)):
        raise ValueError("time must be a finite vector matching the time dimension")
    if time_arr.size > 1 and np.any(np.diff(time_arr) <= 0):
        raise ValueError("time must be strictly increasing")
    return time_arr


def _validate_method(method: str) -> str:
    name = str(method).strip().lower()
    if name not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {SUPPORTED_METHODS}")
    return name


def _positive_unique(values: Sequence[float], name: str) -> list[float]:
    output = sorted({float(value) for value in values})
    if not output or any(value <= 0 or not np.isfinite(value) for value in output):
        raise ValueError(f"{name} must contain finite positive values")
    return output


def _config_key(config: Mapping[str, float]) -> str:
    return json.dumps(dict(config), sort_keys=True, separators=(",", ":"))


def _paired_rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction_arr = np.asarray(prediction, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    if prediction_arr.shape != target_arr.shape:
        raise ValueError("prediction and target must share shape")
    valid = np.isfinite(prediction_arr) & np.isfinite(target_arr)
    return (
        float(np.sqrt(np.mean((prediction_arr[valid] - target_arr[valid]) ** 2)))
        if np.any(valid)
        else float("nan")
    )
