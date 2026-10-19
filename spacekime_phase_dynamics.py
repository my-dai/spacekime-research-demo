"""Smooth latent phase-trajectory baselines for repeated synthetic data.

This module advances beyond the constant-phase baseline while keeping the
identifiability assumptions explicit.  It does **not** implement official KPT.
Each repeat is assumed to follow one slowly varying circular path.  Sparse
phase anchors, when supplied, are declared observed labels; all remaining phase
cells are latent and estimated without access to synthetic truth.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.special import logsumexp

from spacekime_advanced import reconstruct_advanced_surface
from spacekime_blind import align_circular_phase, spectral_phase_initialization
from spacekime_synthetic import latent_kimesurface


DYNAMIC_REPRESENTATION = "synthetic_smooth_latent_phase_trajectories"


def simulate_smooth_phase_repeats(
    n_repeats: int = 48,
    n_time: int = 40,
    noise_sigma: float = 0.08,
    drift_scale: float = 0.45,
    rng: Optional[np.random.Generator] = None,
    phase_sampler: str = "uniform",
) -> Dict[str, np.ndarray]:
    """Simulate repeated trajectories with a smooth withheld phase path.

    The path is a low-dimensional combination of a repeat-specific intercept,
    one sinusoidal drift, and a small linear trend.  This is a controlled
    identifiability benchmark, not a claim about the true fMRI phase dynamics.
    """
    n_repeats, n_time = int(n_repeats), int(n_time)
    if n_repeats < 8 or n_time < 8:
        raise ValueError("n_repeats and n_time must both be at least eight")
    if noise_sigma < 0 or not np.isfinite(noise_sigma):
        raise ValueError("noise_sigma must be finite and non-negative")
    if drift_scale < 0 or not np.isfinite(drift_scale):
        raise ValueError("drift_scale must be finite and non-negative")
    generator = np.random.default_rng() if rng is None else rng
    time = np.linspace(0.0, 1.0, n_time, dtype=np.float64)
    sampler = str(phase_sampler).strip().lower()
    if sampler == "uniform":
        phase_intercept = generator.uniform(-np.pi, np.pi, size=n_repeats)
    elif sampler == "vonmises":
        phase_intercept = generator.vonmises(0.35, 1.5, size=n_repeats)
    else:
        raise ValueError("phase_sampler must be 'uniform' or 'vonmises'")

    amplitude = drift_scale * generator.uniform(0.65, 1.0, size=n_repeats)
    oscillation_offset = generator.uniform(-np.pi, np.pi, size=n_repeats)
    slope = generator.normal(0.0, 0.16 * max(drift_scale, 1e-12), size=n_repeats)
    phase = _wrap(
        phase_intercept[:, None]
        + amplitude[:, None]
        * np.sin(2.0 * np.pi * time[None, :] + oscillation_offset[:, None])
        + slope[:, None] * (time[None, :] - 0.5)
    )
    signal = latent_kimesurface(time[None, :], phase)
    observations = signal + generator.normal(0.0, noise_sigma, size=signal.shape)
    return {
        "representation": DYNAMIC_REPRESENTATION,
        "time": time,
        "phase": phase,
        "phase_intercept": phase_intercept,
        "signal": signal,
        "observations": observations,
    }


def nested_phase_anchor_masks(
    n_repeats: int,
    n_time: int,
    max_anchors_per_repeat: int,
    rng: Optional[np.random.Generator] = None,
) -> list[np.ndarray]:
    """Return nested masks for 0..K observed phase labels per repeat."""
    n_repeats, n_time = int(n_repeats), int(n_time)
    maximum = int(max_anchors_per_repeat)
    if n_repeats < 1 or n_time < 1 or not 0 <= maximum <= n_time:
        raise ValueError("invalid repeat/time/anchor counts")
    generator = np.random.default_rng(0) if rng is None else rng
    order = np.vstack([generator.permutation(n_time) for _ in range(n_repeats)])
    masks: list[np.ndarray] = []
    for count in range(maximum + 1):
        mask = np.zeros((n_repeats, n_time), dtype=bool)
        if count:
            rows = np.repeat(np.arange(n_repeats), count)
            mask[rows, order[:, :count].ravel()] = True
        masks.append(mask)
    return masks


def designed_phase_anchor_masks(
    n_repeats: int,
    n_time: int,
    max_anchors_per_repeat: int,
    strategy: str,
    *,
    rng: Optional[np.random.Generator] = None,
    ambiguity: Optional[np.ndarray] = None,
    downstream_variance: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Create nested phase-label designs without consulting unobserved truth.

    ``uniform`` is a nested temporal maximin design, ``endpoints`` starts at the
    two boundaries and then fills the largest gaps, and ``ambiguity`` ranks the
    observation-only pseudo-posterior circular variance returned by
    :func:`estimate_smooth_phase_trajectories`. ``surface_variance`` ranks the
    observation-only variance of the LOO surface value induced by the same
    path pseudo-posterior, making it a downstream-aware score rather than a
    coordinate-only uncertainty score. Selecting a cell and revealing its
    phase label are deliberately separate operations in the benchmark.
    """
    n_repeats, n_time = int(n_repeats), int(n_time)
    maximum = int(max_anchors_per_repeat)
    if n_repeats < 1 or n_time < 1 or not 0 <= maximum <= n_time:
        raise ValueError("invalid repeat/time/anchor counts")
    name = str(strategy).strip().lower()
    if name == "random":
        return nested_phase_anchor_masks(n_repeats, n_time, maximum, rng=rng)
    if name not in (
        "uniform",
        "endpoints",
        "ambiguity",
        "ambiguity_spread",
        "surface_variance",
        "surface_variance_spread",
    ):
        raise ValueError(
            "strategy must be random, uniform, endpoints, ambiguity, "
            "ambiguity_spread, surface_variance, or surface_variance_spread"
        )

    if name.startswith("ambiguity") or name.startswith("surface_variance"):
        supplied_score = (
            ambiguity if name.startswith("ambiguity") else downstream_variance
        )
        score = np.asarray(supplied_score, dtype=np.float64)
        if score.shape != (n_repeats, n_time) or np.any(~np.isfinite(score)):
            score_name = "ambiguity" if name.startswith("ambiguity") else "downstream_variance"
            raise ValueError(
                f"finite {score_name} matching (repeats,time) is required"
            )
        if not name.endswith("_spread"):
            order = np.argsort(-score, axis=1, kind="stable")
        else:
            order = np.vstack([_score_spread_order(row) for row in score])
    else:
        start = [n_time // 2] if name == "uniform" else [0, n_time - 1]
        temporal_order = _temporal_maximin_order(n_time, start)
        order = np.broadcast_to(temporal_order[None, :], (n_repeats, n_time))

    masks: list[np.ndarray] = []
    for count in range(maximum + 1):
        mask = np.zeros((n_repeats, n_time), dtype=bool)
        if count:
            rows = np.repeat(np.arange(n_repeats), count)
            mask[rows, order[:, :count].ravel()] = True
        masks.append(mask)
    return masks


def estimate_smooth_phase_trajectories(
    observations: np.ndarray,
    time: np.ndarray,
    *,
    anchor_phase: Optional[np.ndarray] = None,
    anchor_mask: Optional[np.ndarray] = None,
    initial_phase: Optional[np.ndarray] = None,
    n_phase_grid: int = 48,
    kappa: float = 8.0,
    transition_strength: float = 2.0,
    max_step_rad: float = 0.65,
    max_iterations: int = 3,
    tolerance: float = 1e-3,
) -> Dict[str, object]:
    """Estimate smooth phase paths by LOO surface fitting and circular Viterbi.

    Each iteration reconstructs a surface for one repeat from *other* repeats,
    then minimizes trajectory-wide squared observation error plus a circular
    first-difference penalty.  This avoids the trivial self-fitting solution.
    Sparse anchors are hard constraints at the nearest phase-grid coordinate.
    """
    values, time_arr = _validate_inputs(observations, time)
    grid_size = int(n_phase_grid)
    if grid_size < 16:
        raise ValueError("n_phase_grid must be at least 16")
    if kappa <= 0 or transition_strength < 0 or max_step_rad <= 0:
        raise ValueError("kappa/max_step_rad must be positive and transition_strength non-negative")
    if int(max_iterations) < 1 or tolerance <= 0:
        raise ValueError("max_iterations and tolerance must be positive")
    anchors, mask = _validate_anchors(anchor_phase, anchor_mask, values.shape)
    grid = np.linspace(-np.pi, np.pi, grid_size, endpoint=False, dtype=np.float64)

    if initial_phase is None:
        initial_repeat = spectral_phase_initialization(values)
        phase = np.broadcast_to(initial_repeat[:, None], values.shape).copy()
        initialization_method = "observation_only_spectral_constant_path"
    else:
        supplied = np.asarray(initial_phase, dtype=np.float64)
        if supplied.shape == (values.shape[0],):
            supplied = np.broadcast_to(supplied[:, None], values.shape)
        if supplied.shape != values.shape or np.any(~np.isfinite(supplied)):
            raise ValueError("initial_phase must match repeats or (repeats,time)")
        phase = _wrap(supplied.copy())
        initialization_method = "user_supplied"

    if np.any(mask):
        alignment = align_circular_phase(phase[mask], anchors[mask])
        phase = _wrap(float(alignment["sign"]) * phase + float(alignment["offset"]))
        phase[mask] = anchors[mask]
    initial_matrix = phase.copy()

    transition = transition_strength * (
        1.0 - np.cos(grid[:, None] - grid[None, :])
    )
    circular_step = np.abs(np.angle(np.exp(1j * (grid[:, None] - grid[None, :]))))
    transition[circular_step > max_step_rad] = np.inf
    observation_scale = max(float(np.nanvar(values)), 1e-8)
    history: list[float] = []
    ambiguity = np.full(values.shape, np.nan, dtype=np.float64)
    downstream_variance = np.full(values.shape, np.nan, dtype=np.float64)

    for _ in range(int(max_iterations)):
        previous = phase.copy()
        updated = np.empty_like(phase)
        for repeat_idx in range(values.shape[0]):
            train_mask = np.arange(values.shape[0]) != repeat_idx
            surface = reconstruct_advanced_surface(
                phase[train_mask],
                values[train_mask],
                time_arr,
                grid,
                "kernel",
                {"kappa": float(kappa)},
            )
            cost = (surface - values[repeat_idx, :, None]) ** 2 / observation_scale
            cost[~np.isfinite(cost)] = np.inf
            missing_time = ~np.isfinite(values[repeat_idx])
            cost[missing_time] = 0.0
            if np.any(mask[repeat_idx]):
                for time_idx in np.flatnonzero(mask[repeat_idx]):
                    nearest = int(
                        np.argmin(
                            np.abs(
                                np.angle(
                                    np.exp(1j * (grid - anchors[repeat_idx, time_idx]))
                                )
                            )
                        )
                    )
                    cost[time_idx] = np.inf
                    cost[time_idx, nearest] = 0.0
            updated[repeat_idx] = _circular_viterbi(cost, grid, transition)
            probability = _path_phase_marginals(cost, transition)
            ambiguity[repeat_idx] = _phase_marginal_circular_variance(
                probability, grid
            )
            posterior_mean = np.sum(probability * surface, axis=1)
            downstream_variance[repeat_idx] = np.sum(
                probability * (surface - posterior_mean[:, None]) ** 2,
                axis=1,
            )
        phase = _wrap(updated)
        phase[mask] = anchors[mask]
        change = float(
            np.sqrt(np.mean(np.angle(np.exp(1j * (phase - previous))) ** 2))
        )
        history.append(change)
        if change <= tolerance:
            break

    return {
        "representation": DYNAMIC_REPRESENTATION,
        "estimator": "loo_surface_plus_circular_viterbi",
        "assumption": "one_slowly_varying_latent_phase_path_per_repeat",
        "phase": phase,
        "initial_phase": initial_matrix,
        "anchor_mask": mask,
        "anchors_per_repeat_mean": float(np.mean(np.sum(mask, axis=1))),
        "supervised_anchor_values_used": bool(np.any(mask)),
        "phase_truth_used_beyond_declared_anchors": False,
        "is_official_kpt": False,
        "initialization_method": initialization_method,
        "iterations": len(history),
        "change_history": history,
        "phase_ambiguity": ambiguity,
        "ambiguity_kind": "observation_only_path_posterior_circular_variance",
        "ambiguity_uses_truth": False,
        "downstream_surface_variance": downstream_variance,
        "downstream_score_kind": (
            "observation_only_loo_surface_value_variance_under_path_pseudo_posterior"
        ),
        "downstream_score_uses_truth": False,
        "n_phase_grid": grid_size,
        "kappa": float(kappa),
        "transition_strength": float(transition_strength),
        "max_step_rad": float(max_step_rad),
    }


def phase_trajectory_alignment_metrics(
    estimated: np.ndarray, truth: np.ndarray
) -> Dict[str, object]:
    """Evaluate a phase matrix under unavoidable global O(2) ambiguity."""
    estimate = np.asarray(estimated, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if estimate.shape != target.shape or estimate.ndim != 2:
        raise ValueError("estimated and truth must be matching phase matrices")
    alignment = align_circular_phase(estimate.ravel(), target.ravel())
    aligned = np.asarray(alignment["aligned"]).reshape(target.shape)
    step_error = np.angle(
        np.exp(
            1j
            * (
                np.diff(aligned, axis=1)
                - np.diff(target, axis=1)
            )
        )
    )
    return {
        **alignment,
        "aligned": aligned,
        "step_rmse": float(np.sqrt(np.mean(step_error**2))),
    }


def _circular_viterbi(
    observation_cost: np.ndarray,
    grid: np.ndarray,
    transition_cost: np.ndarray,
) -> np.ndarray:
    n_time, n_grid = observation_cost.shape
    score = observation_cost[0].copy()
    backpointer = np.empty((n_time, n_grid), dtype=np.int32)
    backpointer[0] = -1
    for time_idx in range(1, n_time):
        candidate = score[:, None] + transition_cost
        predecessor = np.argmin(candidate, axis=0)
        score = observation_cost[time_idx] + candidate[predecessor, np.arange(n_grid)]
        backpointer[time_idx] = predecessor
    if not np.any(np.isfinite(score)):
        raise ValueError("no feasible phase path; relax max_step_rad or anchor spacing")
    state = int(np.nanargmin(score))
    states = np.empty(n_time, dtype=np.int32)
    states[-1] = state
    for time_idx in range(n_time - 1, 0, -1):
        states[time_idx - 1] = backpointer[time_idx, states[time_idx]]
    return grid[states]


def _path_phase_marginals(
    observation_cost: np.ndarray,
    transition_cost: np.ndarray,
) -> np.ndarray:
    """Return circular-state marginals from an energy-based path posterior."""
    n_time, n_grid = observation_cost.shape
    log_transition = -transition_cost
    forward = np.empty((n_time, n_grid), dtype=np.float64)
    forward[0] = -observation_cost[0]
    forward[0] -= logsumexp(forward[0])
    for time_idx in range(1, n_time):
        forward[time_idx] = -observation_cost[time_idx] + logsumexp(
            forward[time_idx - 1, :, None] + log_transition, axis=0
        )
        forward[time_idx] -= logsumexp(forward[time_idx])

    backward = np.zeros((n_time, n_grid), dtype=np.float64)
    for time_idx in range(n_time - 2, -1, -1):
        backward[time_idx] = logsumexp(
            log_transition
            - observation_cost[time_idx + 1][None, :]
            + backward[time_idx + 1][None, :],
            axis=1,
        )
        backward[time_idx] -= logsumexp(backward[time_idx])
    log_marginal = forward + backward
    log_marginal -= logsumexp(log_marginal, axis=1, keepdims=True)
    return np.exp(log_marginal)


def _phase_marginal_circular_variance(
    probability: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Return circular variance for supplied phase-grid marginals."""
    resultant = np.abs(probability @ np.exp(1j * grid))
    return np.clip(1.0 - resultant, 0.0, 1.0)


def _temporal_maximin_order(n_time: int, start: list[int]) -> np.ndarray:
    selected: list[int] = []
    for value in start:
        index = int(np.clip(value, 0, n_time - 1))
        if index not in selected:
            selected.append(index)
    while len(selected) < n_time:
        remaining = np.asarray([idx for idx in range(n_time) if idx not in selected])
        distance = np.min(
            np.abs(remaining[:, None] - np.asarray(selected)[None, :]), axis=1
        )
        selected.append(int(remaining[int(np.argmax(distance))]))
    return np.asarray(selected, dtype=int)


def _score_spread_order(score: np.ndarray) -> np.ndarray:
    """Greedily balance an observation-only score with temporal separation."""
    values = np.asarray(score, dtype=np.float64)
    n_time = values.size
    span = float(np.ptp(values))
    normalized = (values - float(np.min(values))) / max(span, 1e-12)
    selected = [int(np.argmax(normalized))]
    while len(selected) < n_time:
        remaining = np.asarray([idx for idx in range(n_time) if idx not in selected])
        distance = np.min(
            np.abs(remaining[:, None] - np.asarray(selected)[None, :]), axis=1
        ) / max(n_time - 1, 1)
        priority = normalized[remaining] + 0.75 * distance
        selected.append(int(remaining[int(np.argmax(priority))]))
    return np.asarray(selected, dtype=int)


def _validate_inputs(
    observations: np.ndarray, time: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(observations, dtype=np.float64)
    time_arr = np.asarray(time, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 8 or values.shape[1] < 8:
        raise ValueError("observations must have shape (repeats,time), at least (8,8)")
    if time_arr.shape != (values.shape[1],) or np.any(~np.isfinite(time_arr)):
        raise ValueError("time must be finite and match the observation time dimension")
    if np.any(np.diff(time_arr) <= 0):
        raise ValueError("time must be strictly increasing")
    return values, time_arr


def _validate_anchors(
    anchor_phase: Optional[np.ndarray],
    anchor_mask: Optional[np.ndarray],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    if anchor_mask is None:
        if anchor_phase is not None:
            raise ValueError("anchor_mask is required when anchor_phase is supplied")
        return np.full(shape, np.nan), np.zeros(shape, dtype=bool)
    mask = np.asarray(anchor_mask, dtype=bool)
    if mask.shape != shape:
        raise ValueError("anchor_mask must match observations")
    if anchor_phase is None:
        raise ValueError("anchor_phase is required when anchor_mask is supplied")
    anchors = np.asarray(anchor_phase, dtype=np.float64)
    if anchors.shape != shape or np.any(~np.isfinite(anchors[mask])):
        raise ValueError("finite anchor_phase values are required at every anchor")
    return _wrap(anchors), mask


def _wrap(theta: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * np.asarray(theta, dtype=np.float64)))
