"""Observation-only latent phase baselines for repeated synthetic trajectories.

The routines in this module are deliberately named *blind phase baselines*, not
KPT.  They assume that each repeated trajectory has one constant latent circular
coordinate.  Without such structure, one scalar observation at every ``(r,t)``
cannot identify a free latent phase at every cell.

Synthetic truth is accepted only by alignment/evaluation helpers.  The phase
estimator itself receives repeated observations and never receives true phase or
the latent surface.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.sparse.csgraph import shortest_path

from spacekime_synthetic import latent_kimesurface


BLIND_REPRESENTATION = "synthetic_constant_latent_phase_repeated_measurements"


def simulate_constant_phase_repeats(
    n_repeats: int = 64,
    n_time: int = 48,
    noise_sigma: float = 0.08,
    rng: Optional[np.random.Generator] = None,
    phase_sampler: str = "uniform",
) -> Dict[str, np.ndarray]:
    """Simulate trajectories with one withheld latent phase per repeat."""
    n_repeats, n_time = int(n_repeats), int(n_time)
    if n_repeats < 6 or n_time < 4:
        raise ValueError("n_repeats >= 6 and n_time >= 4 are required")
    if noise_sigma < 0 or not np.isfinite(noise_sigma):
        raise ValueError("noise_sigma must be finite and non-negative")
    generator = np.random.default_rng() if rng is None else rng
    time = np.linspace(0.0, 1.0, n_time, dtype=np.float64)
    sampler = str(phase_sampler).strip().lower()
    if sampler == "uniform":
        phase = generator.uniform(-np.pi, np.pi, size=n_repeats)
    elif sampler == "vonmises":
        phase = generator.vonmises(0.35, 1.5, size=n_repeats)
    else:
        raise ValueError("phase_sampler must be 'uniform' or 'vonmises'")
    phase_matrix = np.broadcast_to(phase[:, None], (n_repeats, n_time))
    signal = latent_kimesurface(time[None, :], phase_matrix)
    observations = signal + generator.normal(0.0, noise_sigma, size=signal.shape)
    return {
        "time": time,
        "phase": phase,
        "signal": signal,
        "observations": observations,
    }


def spectral_phase_initialization(observations: np.ndarray) -> np.ndarray:
    """Select an observation-only circular embedding by LOO trajectory error."""
    values = _validate_observations(observations)
    candidates = _phase_initialization_candidates(values)
    return min(
        candidates.items(), key=lambda item: _loo_phase_prediction_rmse(item[1], values, 8.0)
    )[1]


def _phase_initialization_candidates(values: np.ndarray) -> dict[str, np.ndarray]:
    """Build complementary circular manifold embeddings without latent truth."""
    filled = values.copy()
    column_mean = np.nanmean(filled, axis=0)
    missing = ~np.isfinite(filled)
    if np.any(missing):
        filled[missing] = np.take(column_mean, np.nonzero(missing)[1])
    scale = np.nanstd(filled, axis=0)
    standardized = (filled - np.mean(filled, axis=0)) / np.maximum(scale, 1e-8)
    squared_distance = np.mean(
        (standardized[:, None, :] - standardized[None, :, :]) ** 2, axis=2
    )
    positive = squared_distance[squared_distance > 0]
    bandwidth2 = float(np.median(positive)) if positive.size else 1.0
    affinity = np.exp(-squared_distance / max(2.0 * bandwidth2, 1e-12))
    np.fill_diagonal(affinity, 0.0)
    sampling_density = np.sum(affinity, axis=1)
    corrected = affinity / np.maximum(
        sampling_density[:, None] * sampling_density[None, :], 1e-12
    )
    candidates = {
        "diffusion_alpha0": _affinity_circle(affinity),
        "diffusion_alpha1": _affinity_circle(corrected),
    }

    # PCA is a useful low-curvature fallback when the sampled circle contains a
    # large gap and a graph embedding effectively sees an open arc.
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    scores = standardized @ right[:2].T
    scores *= singular[:2] / np.maximum(singular[:2].max(), 1e-12)
    candidates["pca"] = np.angle(scores[:, 0] + 1j * scores[:, 1])

    # Isomap approximates geodesic rather than chordal trajectory distance.
    distance = np.sqrt(np.maximum(squared_distance, 0.0))
    neighbors = int(np.clip(round(np.sqrt(values.shape[0])) + 2, 4, values.shape[0] - 1))
    graph = np.full(distance.shape, np.inf)
    np.fill_diagonal(graph, 0.0)
    neighbor_idx = np.argsort(distance, axis=1)[:, 1 : neighbors + 1]
    for row_idx in range(values.shape[0]):
        graph[row_idx, neighbor_idx[row_idx]] = distance[row_idx, neighbor_idx[row_idx]]
    graph = np.minimum(graph, graph.T)
    geodesic = shortest_path(graph, directed=False)
    if np.all(np.isfinite(geodesic)):
        centered = np.eye(values.shape[0]) - np.ones((values.shape[0], values.shape[0])) / values.shape[0]
        gram = -0.5 * centered @ (geodesic**2) @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        coords = eigenvectors[:, order[:2]] * np.sqrt(np.maximum(eigenvalues[order[:2]], 0.0))
        candidates["isomap"] = np.angle(coords[:, 0] + 1j * coords[:, 1])
    return candidates


def _affinity_circle(affinity: np.ndarray) -> np.ndarray:
    degree = np.sum(affinity, axis=1)
    normalized = affinity / np.sqrt(
        np.maximum(degree[:, None] * degree[None, :], 1e-12)
    )
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    order = np.argsort(eigenvalues)[::-1]
    first = eigenvectors[:, order[1]]
    second = eigenvectors[:, order[2]]
    return np.angle(first + 1j * second)


def estimate_blind_repeat_phase(
    observations: np.ndarray,
    *,
    n_phase_grid: int = 96,
    kappa: float = 8.0,
    max_iterations: int = 8,
    tolerance: float = 1e-3,
    initialization: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Estimate one circular phase per repeat by spectral init + LOO refinement.

    Each coordinate update predicts a candidate trajectory from all *other*
    repeats with a circular kernel, preventing a repeat from trivially fitting
    itself.  A small rank-preserving regularizer anchors the refinement to the
    observation-only spectral embedding and reduces phase collapse.
    """
    values = _validate_observations(observations)
    if int(n_phase_grid) < 16:
        raise ValueError("n_phase_grid must be at least 16")
    if kappa <= 0 or max_iterations < 1 or tolerance <= 0:
        raise ValueError("kappa/max_iterations/tolerance must be positive")
    if initialization is None:
        initialization_candidates = _phase_initialization_candidates(values)
        initialization_method, initial = min(
            initialization_candidates.items(),
            key=lambda item: _loo_phase_prediction_rmse(item[1], values, float(kappa)),
        )
        initial = initial.copy()
    else:
        initialization_method = "user_supplied"
        initial = np.asarray(initialization, dtype=np.float64).copy()
    if initial.shape != (values.shape[0],) or np.any(~np.isfinite(initial)):
        raise ValueError("initialization must be a finite vector matching repeats")
    initial = _wrap(initial)
    phase = initial.copy()
    grid = np.linspace(-np.pi, np.pi, int(n_phase_grid), endpoint=False)
    history: list[float] = []
    ridge_strength = 0.015 * float(np.nanvar(values))

    for _ in range(int(max_iterations)):
        previous = phase.copy()
        updated = phase.copy()
        for repeat_idx in range(values.shape[0]):
            train_mask = np.arange(values.shape[0]) != repeat_idx
            delta = grid[:, None] - phase[train_mask][None, :]
            log_weights = float(kappa) * (np.cos(delta) - 1.0)
            log_weights -= np.max(log_weights, axis=1, keepdims=True)
            weights = np.exp(log_weights)
            predictions = (weights @ values[train_mask]) / np.maximum(
                np.sum(weights, axis=1, keepdims=True), 1e-12
            )
            squared_error = np.nanmean((predictions - values[repeat_idx]) ** 2, axis=1)
            anchor_penalty = ridge_strength * (1.0 - np.cos(grid - initial[repeat_idx]))
            updated[repeat_idx] = grid[int(np.nanargmin(squared_error + anchor_penalty))]
        phase = _wrap(updated)
        change = float(np.sqrt(np.mean(_signed_circular_difference(phase, previous) ** 2)))
        history.append(change)
        if change <= tolerance:
            break
    return {
        "representation": BLIND_REPRESENTATION,
        "estimator": "spectral_circle_plus_leave_one_repeat_out_kernel_refinement",
        "assumption": "one_constant_latent_phase_per_repeat",
        "phase": phase,
        "initial_phase": initial,
        "initialization_method": initialization_method,
        "iterations": len(history),
        "change_history": history,
        "n_phase_grid": int(n_phase_grid),
        "kappa": float(kappa),
        "truth_used_during_estimation": False,
        "is_official_kpt": False,
    }


def align_circular_phase(
    estimated: np.ndarray, truth: np.ndarray, *, allow_reflection: bool = True
) -> Dict[str, object]:
    """Align phase for synthetic evaluation under rotation/reflection ambiguity."""
    estimate = np.asarray(estimated, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if estimate.shape != target.shape or estimate.ndim != 1:
        raise ValueError("estimated and truth must be matching vectors")
    if np.any(~np.isfinite(estimate)) or np.any(~np.isfinite(target)):
        raise ValueError("estimated and truth must be finite")
    candidates: list[dict[str, object]] = []
    signs = (1.0, -1.0) if allow_reflection else (1.0,)
    for sign in signs:
        offset = float(np.angle(np.mean(np.exp(1j * (target - sign * estimate)))))
        aligned = _wrap(sign * estimate + offset)
        error = _signed_circular_difference(aligned, target)
        candidates.append(
            {
                "sign": sign,
                "offset": offset,
                "aligned": aligned,
                "circular_rmse": float(np.sqrt(np.mean(error**2))),
                "circular_mae": float(np.mean(np.abs(error))),
                "resultant_coherence": float(np.abs(np.mean(np.exp(1j * error)))),
            }
        )
    return min(candidates, key=lambda item: float(item["circular_rmse"]))


def truth_grid_to_estimated_coordinate(
    theta_grid: np.ndarray, *, sign: float, offset: float
) -> np.ndarray:
    """Map a truth-coordinate grid into an aligned estimator coordinate."""
    grid = np.asarray(theta_grid, dtype=np.float64)
    if sign not in (-1.0, 1.0):
        raise ValueError("sign must be -1 or 1")
    return _wrap(sign * (grid - float(offset)))


def _validate_observations(observations: np.ndarray) -> np.ndarray:
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 6 or values.shape[1] < 4:
        raise ValueError("observations must have shape (repeats,time), at least (6,4)")
    if np.any(np.all(~np.isfinite(values), axis=0)):
        raise ValueError("each time point needs at least one finite observation")
    return values


def _wrap(theta: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(theta, dtype=np.float64)))


def _signed_circular_difference(theta_a: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (theta_a - theta_b)))


def _loo_phase_prediction_rmse(
    phase: np.ndarray, values: np.ndarray, kappa: float
) -> float:
    delta = phase[:, None] - phase[None, :]
    log_weights = float(kappa) * (np.cos(delta) - 1.0)
    np.fill_diagonal(log_weights, -np.inf)
    row_max = np.max(log_weights, axis=1, keepdims=True)
    weights = np.exp(log_weights - row_max)
    prediction = (weights @ values) / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
    valid = np.isfinite(prediction) & np.isfinite(values)
    return float(np.sqrt(np.mean((prediction[valid] - values[valid]) ** 2)))
