"""Gauge-invariant observable-law tools for KPT-style inverse models.

For the marginal model ``Y = S(Theta) + epsilon``, a smooth change of latent
phase coordinate can alter both ``S`` and the law of ``Theta`` without changing
the law of ``Y``.  This module therefore works with the identifiable quotient
target: the pushforward observable law ``S#Phi`` in the noiseless model, or its
convolution with the noise law when only noisy observations are available.

The canonical quantile curve returned here is a statistical representative of
that law.  It is not a recovered circular phase coordinate or Kime surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, Optional

import numpy as np


QUOTIENT_REPRESENTATION = "observable_pushforward_law_quantile_representative"


def weighted_quantile(
    values: Sequence[float],
    probability_mass: Sequence[float],
    quantile_levels: Sequence[float],
) -> np.ndarray:
    """Evaluate the left-continuous quantile of a finite weighted law."""
    support, mass = _validate_weighted_law(values, probability_mass)
    levels = np.asarray(quantile_levels, dtype=np.float64)
    if levels.ndim != 1 or np.any(~np.isfinite(levels)):
        raise ValueError("quantile_levels must be a finite vector")
    if np.any(levels < 0.0) or np.any(levels > 1.0):
        raise ValueError("quantile_levels must lie in [0, 1]")

    order = np.argsort(support, kind="mergesort")
    sorted_support = support[order]
    cumulative_mass = np.cumsum(mass[order])
    cumulative_mass[-1] = 1.0
    indices = np.searchsorted(cumulative_mass, levels, side="left")
    return sorted_support[np.minimum(indices, sorted_support.size - 1)]


def observable_wasserstein_distance(
    values_a: Sequence[float],
    probability_mass_a: Sequence[float],
    values_b: Sequence[float],
    probability_mass_b: Sequence[float],
    *,
    order: float = 2.0,
) -> float:
    """Return the exact one-dimensional weighted Wasserstein-p distance.

    The implementation integrates the two discrete quantile functions over
    their merged cumulative-mass intervals.  It does not depend on the latent
    phase grid or on an arbitrary quantile discretization.
    """
    support_a, mass_a = _validate_weighted_law(values_a, probability_mass_a)
    support_b, mass_b = _validate_weighted_law(values_b, probability_mass_b)
    exponent = float(order)
    if not np.isfinite(exponent) or exponent < 1.0:
        raise ValueError("order must be finite and at least 1")

    sort_a = np.argsort(support_a, kind="mergesort")
    sort_b = np.argsort(support_b, kind="mergesort")
    support_a = support_a[sort_a]
    support_b = support_b[sort_b]
    mass_a = mass_a[sort_a].copy()
    mass_b = mass_b[sort_b].copy()

    index_a = 0
    index_b = 0
    remaining_a = float(mass_a[0])
    remaining_b = float(mass_b[0])
    transport_cost = 0.0
    tolerance = 32.0 * np.finfo(np.float64).eps
    while index_a < support_a.size and index_b < support_b.size:
        transported = min(remaining_a, remaining_b)
        transport_cost += transported * abs(
            float(support_a[index_a] - support_b[index_b])
        ) ** exponent
        remaining_a -= transported
        remaining_b -= transported
        if remaining_a <= tolerance:
            index_a += 1
            if index_a < support_a.size:
                remaining_a = float(mass_a[index_a])
        if remaining_b <= tolerance:
            index_b += 1
            if index_b < support_b.size:
                remaining_b = float(mass_b[index_b])

    return float(max(transport_cost, 0.0) ** (1.0 / exponent))


def canonical_observable_representative(
    values: Sequence[float],
    probability_mass: Sequence[float],
    *,
    n_quantiles: int = 256,
) -> Dict[str, object]:
    """Map an observable law to its monotone uniform-coordinate representative."""
    n_points = int(n_quantiles)
    if n_points < 2:
        raise ValueError("n_quantiles must be at least 2")
    support, mass = _validate_weighted_law(values, probability_mass)
    uniform_coordinate = (np.arange(n_points, dtype=np.float64) + 0.5) / n_points
    quantile_values = weighted_quantile(support, mass, uniform_coordinate)
    return {
        "representation": QUOTIENT_REPRESENTATION,
        "uniform_coordinate": uniform_coordinate,
        "quantile_values": quantile_values,
        "n_quantiles": n_points,
        "target": "noiseless_pushforward_law_if_input_is_noiseless",
        "recovers_latent_phase_coordinate": False,
        "is_official_kpt": False,
    }


def observable_law_signature(
    values: Sequence[float],
    probability_mass: Sequence[float],
    *,
    quantile_levels: Optional[Sequence[float]] = None,
    characteristic_frequencies: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Return invariant moments, quantiles, and characteristic-function values."""
    support, mass = _validate_weighted_law(values, probability_mass)
    levels = (
        np.linspace(0.05, 0.95, 19)
        if quantile_levels is None
        else np.asarray(quantile_levels, dtype=np.float64)
    )
    frequencies = (
        np.linspace(-4.0, 4.0, 33)
        if characteristic_frequencies is None
        else np.asarray(characteristic_frequencies, dtype=np.float64)
    )
    if frequencies.ndim != 1 or np.any(~np.isfinite(frequencies)):
        raise ValueError("characteristic_frequencies must be a finite vector")
    mean = float(np.sum(mass * support))
    centered = support - mean
    variance = float(np.sum(mass * centered**2))
    characteristic = np.exp(
        1j * frequencies[:, None] * support[None, :]
    ) @ mass
    return {
        "representation": QUOTIENT_REPRESENTATION,
        "mean": mean,
        "standard_deviation": float(np.sqrt(max(variance, 0.0))),
        "third_central_moment": float(np.sum(mass * centered**3)),
        "quantile_levels": levels,
        "quantile_values": weighted_quantile(support, mass, levels),
        "characteristic_frequencies": frequencies,
        "characteristic_function": characteristic,
        "positive_support_points": int(support.size),
        "recovers_latent_phase_coordinate": False,
        "is_official_kpt": False,
    }


def compare_observable_laws(
    values_a: Sequence[float],
    probability_mass_a: Sequence[float],
    values_b: Sequence[float],
    probability_mass_b: Sequence[float],
    *,
    n_quantiles: int = 256,
) -> Dict[str, object]:
    """Compare two quotient targets without aligning latent phase coordinates."""
    representative_a = canonical_observable_representative(
        values_a, probability_mass_a, n_quantiles=n_quantiles
    )
    representative_b = canonical_observable_representative(
        values_b, probability_mass_b, n_quantiles=n_quantiles
    )
    quantiles_a = np.asarray(representative_a["quantile_values"])
    quantiles_b = np.asarray(representative_b["quantile_values"])
    return {
        "representation": QUOTIENT_REPRESENTATION,
        "wasserstein_1": observable_wasserstein_distance(
            values_a, probability_mass_a, values_b, probability_mass_b, order=1
        ),
        "wasserstein_2": observable_wasserstein_distance(
            values_a, probability_mass_a, values_b, probability_mass_b, order=2
        ),
        "canonical_quantile_rmse": float(
            np.sqrt(np.mean((quantiles_a - quantiles_b) ** 2))
        ),
        "n_quantiles": int(n_quantiles),
        "requires_phase_alignment": False,
        "is_official_kpt": False,
    }


def weighted_gaussian_mmd_squared(
    samples_a: Sequence[Sequence[float]] | Sequence[float],
    probability_mass_a: Sequence[float],
    samples_b: Sequence[Sequence[float]] | Sequence[float],
    probability_mass_b: Sequence[float],
    *,
    bandwidth: float,
) -> float:
    """Return the population Gaussian-kernel MMD squared for weighted laws."""
    left = _validate_sample_matrix(samples_a, "samples_a")
    right = _validate_sample_matrix(samples_b, "samples_b")
    if left.shape[1] != right.shape[1]:
        raise ValueError("sample dimensions must match")
    mass_a = _validate_probability_mass(probability_mass_a, left.shape[0])
    mass_b = _validate_probability_mass(probability_mass_b, right.shape[0])
    scale = _validate_bandwidth(bandwidth)
    kernel_aa = _gaussian_kernel_matrix(left, left, scale)
    kernel_bb = _gaussian_kernel_matrix(right, right, scale)
    kernel_ab = _gaussian_kernel_matrix(left, right, scale)
    statistic = (
        mass_a @ kernel_aa @ mass_a
        + mass_b @ kernel_bb @ mass_b
        - 2.0 * (mass_a @ kernel_ab @ mass_b)
    )
    return float(max(statistic, 0.0))


def pooled_median_bandwidth(
    samples_a: Sequence[Sequence[float]] | Sequence[float],
    samples_b: Sequence[Sequence[float]] | Sequence[float],
) -> float:
    """Select an RBF bandwidth from pooled nonzero pairwise distances."""
    left = _validate_sample_matrix(samples_a, "samples_a")
    right = _validate_sample_matrix(samples_b, "samples_b")
    if left.shape[1] != right.shape[1]:
        raise ValueError("sample dimensions must match")
    pooled = np.vstack([left, right])
    squared = _pairwise_squared_distances(pooled, pooled)
    upper = squared[np.triu_indices(pooled.shape[0], k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return 1.0
    return float(np.sqrt(np.median(positive)))


def permutation_mmd_test(
    samples_a: Sequence[Sequence[float]] | Sequence[float],
    samples_b: Sequence[Sequence[float]] | Sequence[float],
    *,
    bandwidth: Optional[float] = None,
    n_permutations: int = 199,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, object]:
    """Run a label-permutation two-sample test with biased Gaussian MMD.

    If no bandwidth is supplied, the pooled median distance is computed once
    without labels and then held fixed for every permutation.  The Monte Carlo
    p-value uses the ``(1 + exceedances) / (B + 1)`` correction.
    """
    left = _validate_sample_matrix(samples_a, "samples_a")
    right = _validate_sample_matrix(samples_b, "samples_b")
    if left.shape[1] != right.shape[1]:
        raise ValueError("sample dimensions must match")
    if left.shape[0] < 2 or right.shape[0] < 2:
        raise ValueError("each sample must contain at least two observations")
    permutations = int(n_permutations)
    level = float(alpha)
    if permutations < 19:
        raise ValueError("n_permutations must be at least 19")
    if not np.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    scale = (
        pooled_median_bandwidth(left, right)
        if bandwidth is None
        else _validate_bandwidth(bandwidth)
    )

    pooled = np.vstack([left, right])
    kernel = _gaussian_kernel_matrix(pooled, pooled, scale)
    n_left = left.shape[0]
    n_right = right.shape[0]
    weights = np.empty((permutations + 1, pooled.shape[0]), dtype=np.float64)
    weights[0, :n_left] = 1.0 / n_left
    weights[0, n_left:] = -1.0 / n_right
    generator = np.random.default_rng() if rng is None else rng
    for index in range(1, permutations + 1):
        assignment = generator.permutation(pooled.shape[0])
        weights[index] = -1.0 / n_right
        weights[index, assignment[:n_left]] = 1.0 / n_left
    statistics = np.sum((weights @ kernel) * weights, axis=1)
    statistics = np.maximum(statistics, 0.0)
    observed = float(statistics[0])
    null_statistics = statistics[1:]
    comparison_tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0, abs(observed)
    )
    exceedances = int(
        np.sum(null_statistics >= observed - comparison_tolerance)
    )
    p_value = (1.0 + exceedances) / (permutations + 1.0)
    critical_index = min(
        permutations - 1,
        max(0, int(np.ceil((1.0 - level) * permutations)) - 1),
    )
    critical_value = float(np.sort(null_statistics)[critical_index])
    return {
        "representation": QUOTIENT_REPRESENTATION,
        "test": "pooled_fixed_bandwidth_permutation_gaussian_mmd",
        "mmd_squared_biased": observed,
        "bandwidth": scale,
        "bandwidth_rule": (
            "pooled_median_nonzero_pairwise_distance"
            if bandwidth is None
            else "prespecified"
        ),
        "bandwidth_reselected_within_permutations": False,
        "n_sample_a": n_left,
        "n_sample_b": n_right,
        "dimension": left.shape[1],
        "n_permutations": permutations,
        "alpha": level,
        "p_value": float(p_value),
        "reject": bool(p_value <= level),
        "permutation_critical_value": critical_value,
        "permutation_null_mean": float(np.mean(null_statistics)),
        "is_official_kpt": False,
    }


def empirical_observable_law(
    observations: Sequence[float],
    *,
    sample_weights: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    """Represent scalar observations as a normalized finite probability law."""
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("observations must be a non-empty finite vector")
    weights = (
        np.ones(values.size, dtype=np.float64)
        if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    support, mass = _validate_weighted_law(values, weights)
    return {"values": support, "probability_mass": mass}


def audit_repeated_measurement_design(
    observation_shape: Sequence[int],
    axis_labels: Sequence[str],
    *,
    repeat_axis: Optional[int] = None,
    minimum_repeats: int = 2,
    repeat_independence_documented: bool = False,
    controlled_time_alignment_documented: bool = False,
) -> Dict[str, object]:
    """Audit whether an array can estimate a fixed-time observable law.

    Axis size alone cannot establish independent controlled repetitions.  The
    two documentation flags are therefore required before this function marks
    a design suitable for inferential use.  Spatial voxels and successive time
    frames are not silently treated as experimental repeats.
    """
    try:
        shape = tuple(int(size) for size in observation_shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation_shape must contain integer sizes") from exc
    labels = tuple(str(label).strip() for label in axis_labels)
    required = int(minimum_repeats)
    if not shape or any(size < 1 for size in shape):
        raise ValueError("observation_shape must contain positive sizes")
    if len(labels) != len(shape) or any(not label for label in labels):
        raise ValueError("axis_labels must be non-empty and match observation_shape")
    if len({label.lower() for label in labels}) != len(labels):
        raise ValueError("axis_labels must be unique")
    if required < 2:
        raise ValueError("minimum_repeats must be at least 2")

    axis_index: Optional[int]
    if repeat_axis is None:
        axis_index = None
        repeat_count = 0
    else:
        axis_index = int(repeat_axis)
        if axis_index < 0:
            axis_index += len(shape)
        if axis_index < 0 or axis_index >= len(shape):
            raise ValueError("repeat_axis is out of range")
        repeat_count = shape[axis_index]

    count_sufficient = axis_index is not None and repeat_count >= required
    documentation_sufficient = bool(
        repeat_independence_documented and controlled_time_alignment_documented
    )
    suitable = bool(count_sufficient and documentation_sufficient)
    if suitable:
        status = "candidate_repeated_measurement_design"
    elif axis_index is None:
        status = "descriptive_proxy_only_no_repeat_axis"
    elif not count_sufficient:
        status = "insufficient_repeat_count"
    else:
        status = "candidate_axis_but_protocol_metadata_missing"

    limitations: list[str] = []
    if axis_index is None:
        limitations.append("no independent repeat axis is declared")
    elif not count_sufficient:
        limitations.append(
            f"declared repeat count {repeat_count} is below {required}"
        )
    if not repeat_independence_documented:
        limitations.append("repeat independence is not documented")
    if not controlled_time_alignment_documented:
        limitations.append("controlled time alignment is not documented")
    limitations.append(
        "spatial voxels and successive frames are not interchangeable with controlled repeats"
    )
    return {
        "observation_shape": list(shape),
        "axis_labels": list(labels),
        "repeat_axis": axis_index,
        "repeat_axis_label": labels[axis_index] if axis_index is not None else None,
        "repeat_count": repeat_count,
        "minimum_repeats": required,
        "repeat_count_sufficient": count_sufficient,
        "repeat_independence_documented": bool(repeat_independence_documented),
        "controlled_time_alignment_documented": bool(
            controlled_time_alignment_documented
        ),
        "supports_empirical_fixed_time_observable_law": suitable,
        "status": status,
        "limitations": limitations,
        "latent_phase_identified_by_this_audit": False,
    }


def _validate_weighted_law(
    values: Sequence[float], probability_mass: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    support = np.asarray(values, dtype=np.float64)
    mass = np.asarray(probability_mass, dtype=np.float64)
    if (
        support.ndim != 1
        or support.size == 0
        or mass.shape != support.shape
        or np.any(~np.isfinite(support))
        or np.any(~np.isfinite(mass))
        or np.any(mass < 0.0)
        or float(np.sum(mass)) <= 0.0
    ):
        raise ValueError(
            "values and probability_mass must define a non-empty finite law"
        )
    positive = mass > 0.0
    support = support[positive]
    mass = mass[positive]
    mass = mass / np.sum(mass)
    return support, mass


def _validate_sample_matrix(
    samples: Sequence[Sequence[float]] | Sequence[float], name: str
) -> np.ndarray:
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] == 0
        or np.any(~np.isfinite(matrix))
    ):
        raise ValueError(f"{name} must be a non-empty finite sample matrix")
    return matrix


def _validate_probability_mass(
    probability_mass: Sequence[float], n_samples: int
) -> np.ndarray:
    mass = np.asarray(probability_mass, dtype=np.float64)
    if (
        mass.shape != (n_samples,)
        or np.any(~np.isfinite(mass))
        or np.any(mass < 0.0)
        or float(np.sum(mass)) <= 0.0
    ):
        raise ValueError("probability_mass must be finite, nonnegative, and match")
    return mass / np.sum(mass)


def _validate_bandwidth(bandwidth: float) -> float:
    scale = float(bandwidth)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("bandwidth must be finite and positive")
    return scale


def _pairwise_squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left**2, axis=1)[:, None]
    right_norm = np.sum(right**2, axis=1)[None, :]
    squared = left_norm + right_norm - 2.0 * (left @ right.T)
    return np.maximum(squared, 0.0)


def _gaussian_kernel_matrix(
    left: np.ndarray, right: np.ndarray, bandwidth: float
) -> np.ndarray:
    squared = _pairwise_squared_distances(left, right)
    return np.exp(-squared / (2.0 * bandwidth**2))
