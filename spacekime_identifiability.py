"""Finite-basis identifiability laboratory for circular phase densities.

This module is a synthetic inverse-problem analog motivated by KPT.  It is not
an implementation of the official SOCR KPT algorithm.  It makes the forward
operator explicit so rank deficiency, ill-conditioning, and noise stability
can be measured instead of inferred from reconstruction plots.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize


IDENTIFIABILITY_REPRESENTATION = (
    "synthetic_finite_fourier_phase_density_measurements"
)


def harmonic_design_matrix(theta: Sequence[float], order: int) -> np.ndarray:
    """Return columns ``cos(k theta), sin(k theta)`` for ``k=1,...,order``."""
    angles = np.asarray(theta, dtype=np.float64)
    order = int(order)
    if angles.ndim != 1 or angles.size == 0 or np.any(~np.isfinite(angles)):
        raise ValueError("theta must be a non-empty finite vector")
    if order < 1:
        raise ValueError("order must be at least 1")
    harmonics = np.arange(1, order + 1, dtype=np.float64)
    phase = angles[:, None] * harmonics[None, :]
    matrix = np.empty((angles.size, 2 * order), dtype=np.float64)
    matrix[:, 0::2] = np.cos(phase)
    matrix[:, 1::2] = np.sin(phase)
    return matrix


def fourier_density(
    theta: Sequence[float], coefficients: Sequence[float]
) -> np.ndarray:
    """Evaluate a normalized, truncated real Fourier density.

    Coefficients are ordered ``a1,b1,a2,b2,...`` in

    ``p(theta) = (1 + 2 sum_k[a_k cos(k theta)+b_k sin(k theta)])/(2 pi)``.

    The function does not silently clip negative values; invalid truncated
    densities remain visible to callers and tests.
    """
    values = _validate_coefficients(coefficients)
    basis = harmonic_design_matrix(theta, values.size // 2)
    return (1.0 + 2.0 * basis @ values) / (2.0 * np.pi)


def random_positive_fourier_coefficients(
    order: int,
    *,
    rng: Optional[np.random.Generator] = None,
    positivity_margin: float = 0.8,
    harmonic_decay: float = 0.65,
) -> np.ndarray:
    """Generate coefficients whose density is positive for every angle.

    The sufficient bound ``2 sum_k sqrt(a_k^2+b_k^2) <= positivity_margin``
    leaves a known positive floor.  It is conservative but useful for clean
    identifiability experiments because positivity is guaranteed analytically.
    """
    order = int(order)
    if order < 1:
        raise ValueError("order must be at least 1")
    if not 0.0 < positivity_margin < 1.0:
        raise ValueError("positivity_margin must lie strictly between 0 and 1")
    if not 0.0 < harmonic_decay <= 1.0:
        raise ValueError("harmonic_decay must lie in (0, 1]")
    generator = np.random.default_rng() if rng is None else rng
    amplitudes = generator.uniform(0.35, 1.0, size=order)
    amplitudes *= harmonic_decay ** np.arange(order, dtype=np.float64)
    amplitudes *= positivity_margin / (2.0 * np.sum(amplitudes))
    phases = generator.uniform(-np.pi, np.pi, size=order)
    coefficients = np.empty(2 * order, dtype=np.float64)
    coefficients[0::2] = amplitudes * np.cos(phases)
    coefficients[1::2] = amplitudes * np.sin(phases)
    return coefficients


def quadrature_probe_operator(
    order: int,
    phase_offsets: Sequence[float],
    *,
    harmonics: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Build the forward operator for rotated harmonic expectation probes.

    A row for harmonic ``k`` and offset ``alpha`` observes
    ``E[cos(k theta - alpha)] = a_k cos(alpha) + b_k sin(alpha)``.
    Two linearly independent offsets per harmonic identify both quadratures;
    one offset leaves one null direction per harmonic.
    """
    order = int(order)
    if order < 1:
        raise ValueError("order must be at least 1")
    offsets = np.asarray(phase_offsets, dtype=np.float64)
    if offsets.ndim != 1 or offsets.size == 0 or np.any(~np.isfinite(offsets)):
        raise ValueError("phase_offsets must be a non-empty finite vector")
    selected = (
        np.arange(1, order + 1, dtype=int)
        if harmonics is None
        else np.asarray(harmonics, dtype=int)
    )
    if (
        selected.ndim != 1
        or selected.size == 0
        or np.any(selected < 1)
        or np.any(selected > order)
        or np.unique(selected).size != selected.size
    ):
        raise ValueError("harmonics must be unique integers between 1 and order")

    operator = np.zeros((selected.size * offsets.size, 2 * order), dtype=np.float64)
    row = 0
    for harmonic in selected:
        for offset in offsets:
            operator[row, 2 * (harmonic - 1)] = np.cos(offset)
            operator[row, 2 * (harmonic - 1) + 1] = np.sin(offset)
            row += 1
    return operator


def operator_diagnostics(
    operator: np.ndarray, *, rank_tolerance: Optional[float] = None
) -> Dict[str, object]:
    """Report rank, nullity, singular spectrum, and inversion stability."""
    matrix = _validate_operator(operator)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if rank_tolerance is None:
        tolerance = (
            max(matrix.shape)
            * np.finfo(np.float64).eps
            * float(singular_values[0])
        )
    else:
        tolerance = float(rank_tolerance)
        if tolerance < 0 or not np.isfinite(tolerance):
            raise ValueError("rank_tolerance must be finite and non-negative")
    rank = int(np.sum(singular_values > tolerance))
    n_parameters = int(matrix.shape[1])
    full_column_rank = rank == n_parameters
    smallest_nonzero = (
        float(singular_values[rank - 1]) if rank > 0 else 0.0
    )
    condition_number = (
        float(singular_values[0] / singular_values[n_parameters - 1])
        if full_column_rank
        else None
    )
    return {
        "n_measurements": int(matrix.shape[0]),
        "n_parameters": n_parameters,
        "rank": rank,
        "nullity": n_parameters - rank,
        "full_column_rank": full_column_rank,
        "rank_tolerance": tolerance,
        "largest_singular_value": float(singular_values[0]),
        "smallest_nonzero_singular_value": smallest_nonzero,
        "smallest_singular_value": (
            float(singular_values[n_parameters - 1])
            if full_column_rank
            else 0.0
        ),
        "condition_number": condition_number,
        "singular_values": singular_values,
    }


def simulate_probe_measurements(
    operator: np.ndarray,
    coefficients: Sequence[float],
    *,
    n_repeats: int = 1,
    noise_sigma: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Simulate repeated noisy measurements from an explicit operator."""
    matrix = _validate_operator(operator)
    values = _validate_coefficients(coefficients)
    n_repeats = int(n_repeats)
    noise_sigma = float(noise_sigma)
    if matrix.shape[1] != values.size:
        raise ValueError("operator columns must match coefficient count")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")
    if noise_sigma < 0 or not np.isfinite(noise_sigma):
        raise ValueError("noise_sigma must be finite and non-negative")
    generator = np.random.default_rng() if rng is None else rng
    expectation = matrix @ values
    observations = expectation[None, :] + generator.normal(
        0.0, noise_sigma, size=(n_repeats, matrix.shape[0])
    )
    return {
        "expectation": expectation,
        "observations": observations,
        "mean": np.mean(observations, axis=0),
    }


def recover_fourier_coefficients(
    operator: np.ndarray,
    measurements: Sequence[float],
    *,
    ridge: float = 1e-10,
    enforce_nonnegative: bool = True,
    constraint_grid_size: int = 256,
) -> Dict[str, object]:
    """Recover Fourier coefficients with an optional density constraint."""
    matrix = _validate_operator(operator)
    observed = np.asarray(measurements, dtype=np.float64)
    ridge = float(ridge)
    if observed.shape != (matrix.shape[0],) or np.any(~np.isfinite(observed)):
        raise ValueError("measurements must be finite and match operator rows")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    if int(constraint_grid_size) < 4 * (matrix.shape[1] // 2) + 1:
        raise ValueError("constraint_grid_size is too small for the Fourier order")

    if ridge > 0:
        augmented_operator = np.vstack(
            [matrix, np.sqrt(ridge) * np.eye(matrix.shape[1])]
        )
        augmented_observed = np.concatenate(
            [observed, np.zeros(matrix.shape[1], dtype=np.float64)]
        )
        initial = np.linalg.lstsq(
            augmented_operator, augmented_observed, rcond=None
        )[0]
    else:
        initial = np.linalg.lstsq(matrix, observed, rcond=None)[0]
    if not enforce_nonnegative:
        estimate = initial
        success = True
        message = "unconstrained ridge least squares"
    else:
        theta = np.linspace(
            -np.pi, np.pi, int(constraint_grid_size), endpoint=False
        )
        basis = harmonic_design_matrix(theta, matrix.shape[1] // 2)

        def objective(candidate: np.ndarray) -> float:
            residual = matrix @ candidate - observed
            return 0.5 * float(residual @ residual + ridge * (candidate @ candidate))

        def gradient(candidate: np.ndarray) -> np.ndarray:
            return matrix.T @ (matrix @ candidate - observed) + ridge * candidate

        result = minimize(
            objective,
            initial,
            jac=gradient,
            constraints={
                "type": "ineq",
                "fun": lambda candidate: 1.0 + 2.0 * basis @ candidate,
                "jac": lambda candidate: 2.0 * basis,
            },
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        estimate = np.asarray(result.x, dtype=np.float64)
        success = bool(result.success)
        message = str(result.message)

    fitted = matrix @ estimate
    return {
        "representation": IDENTIFIABILITY_REPRESENTATION,
        "coefficients": estimate,
        "fitted_measurements": fitted,
        "measurement_rmse": float(np.sqrt(np.mean((fitted - observed) ** 2))),
        "ridge": ridge,
        "nonnegative_constraint": bool(enforce_nonnegative),
        "optimizer_success": success,
        "optimizer_message": message,
        "is_official_kpt": False,
    }


def observable_characteristic_function(
    values: Sequence[float],
    probability_mass: Sequence[float],
    frequencies: Sequence[float],
) -> np.ndarray:
    """Evaluate the characteristic function of a discrete observable law."""
    observable = np.asarray(values, dtype=np.float64)
    mass = np.asarray(probability_mass, dtype=np.float64)
    query = np.asarray(frequencies, dtype=np.float64)
    if (
        observable.ndim != 1
        or mass.shape != observable.shape
        or observable.size == 0
        or np.any(~np.isfinite(observable))
        or np.any(~np.isfinite(mass))
        or np.any(mass < 0)
        or float(np.sum(mass)) <= 0
    ):
        raise ValueError("values and probability_mass must define a finite law")
    if query.ndim != 1 or np.any(~np.isfinite(query)):
        raise ValueError("frequencies must be a finite vector")
    normalized_mass = mass / np.sum(mass)
    return np.exp(1j * query[:, None] * observable[None, :]) @ normalized_mass


def phase_reparameterization_counterexample(
    theta: Sequence[float],
    surface: Sequence[float],
    phase_density: Sequence[float],
    *,
    warp_strength: float = 0.35,
    characteristic_frequencies: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Construct an observationally equivalent non-rotation phase coordinate.

    The orientation-preserving map ``h(theta)=theta+a sin(theta)`` has positive
    derivative for ``|a| < 1``.  Pairing ``S'(h(theta))=S(theta)`` with the
    pushforward phase law leaves the complete noiseless observable law exactly
    unchanged.  Independent additive noise therefore cannot distinguish the
    two parameterizations either.

    This counterexample applies to the marginal model
    ``Y=S(t,Theta)+epsilon`` at a fixed time.  It does not claim that every
    longitudinal model remains non-identifiable after adding known phase
    anchors or a specified cross-time phase dynamics law.
    """
    angles, signal, density = _validate_phase_grid_objects(
        theta, surface, phase_density
    )
    strength = float(warp_strength)
    if not np.isfinite(strength) or abs(strength) >= 1.0 or strength == 0.0:
        raise ValueError("warp_strength must be finite, nonzero, and have abs < 1")
    frequencies = (
        np.linspace(-8.0, 8.0, 65)
        if characteristic_frequencies is None
        else np.asarray(characteristic_frequencies, dtype=np.float64)
    )
    mass = density / np.sum(density)
    transformed_theta = angles + strength * np.sin(angles)
    jacobian = 1.0 + strength * np.cos(angles)
    transformed_density = density / jacobian
    transformed_mass = mass.copy()
    original_characteristic = observable_characteristic_function(
        signal, mass, frequencies
    )
    transformed_characteristic = observable_characteristic_function(
        signal, transformed_mass, frequencies
    )
    displacement = transformed_theta - angles
    centered_displacement = displacement - np.mean(displacement)
    return {
        "representation": IDENTIFIABILITY_REPRESENTATION,
        "counterexample": "smooth_phase_reparameterization",
        "theta": angles,
        "surface": signal,
        "phase_density": density,
        "phase_probability_mass": mass,
        "transformed_theta": transformed_theta,
        "transformed_surface_at_transformed_theta": signal.copy(),
        "transformed_phase_density_at_transformed_theta": transformed_density,
        "transformed_probability_mass": transformed_mass,
        "warp_jacobian": jacobian,
        "warp_strength": strength,
        "nonrotation_displacement_rms": float(
            np.sqrt(np.mean(centered_displacement**2))
        ),
        "observable_characteristic_max_abs_difference": float(
            np.max(np.abs(original_characteristic - transformed_characteristic))
        ),
        "complete_discrete_observable_law_preserved": True,
        "is_official_kpt": False,
    }


def spectral_gauge_counterexample(
    surface: Sequence[float],
    phase_density: Sequence[float],
    *,
    gauge_harmonic: int = 1,
    gauge_strength: float = 0.2,
) -> Dict[str, object]:
    """Construct a non-rotation factorization with the same spectral product.

    For a circular mixture with ``M_hat=F_hat*Phi_hat``, multiply one conjugate
    harmonic pair of ``F_hat`` by ``g`` and divide the same pair of ``Phi_hat``
    by ``g``.  Their product, and hence the observable circular mixture, is
    unchanged.  Positive gauge strengths shrink the corresponding phase-law
    harmonic and preserve positivity for the strictly positive densities made
    by :func:`random_positive_fourier_coefficients`.

    This proves non-uniqueness of the stated spectral product alone.  It does
    not by itself settle whether additional likelihood information and fully
    specified structural constraints identify a particular KPT model.
    """
    signal = np.asarray(surface, dtype=np.float64)
    density = np.asarray(phase_density, dtype=np.float64)
    if (
        signal.ndim != 1
        or density.shape != signal.shape
        or signal.size < 8
        or np.any(~np.isfinite(signal))
        or np.any(~np.isfinite(density))
        or np.any(density < 0)
        or float(np.sum(density)) <= 0
    ):
        raise ValueError("surface and phase_density must be matching finite vectors")
    harmonic = int(gauge_harmonic)
    strength = float(gauge_strength)
    if harmonic < 1 or harmonic >= signal.size // 2:
        raise ValueError("gauge_harmonic must lie between 1 and half the grid size")
    if not np.isfinite(strength) or strength == 0.0:
        raise ValueError("gauge_strength must be finite and nonzero")

    mass = density / np.sum(density)
    surface_spectrum = np.fft.fft(signal)
    phase_spectrum = np.fft.fft(mass)
    gauge = np.ones(signal.size, dtype=np.float64)
    factor = float(np.exp(strength))
    gauge[harmonic] = factor
    gauge[-harmonic] = factor
    transformed_surface_spectrum = surface_spectrum * gauge
    transformed_phase_spectrum = phase_spectrum / gauge
    transformed_surface = np.fft.ifft(transformed_surface_spectrum).real
    transformed_mass = np.fft.ifft(transformed_phase_spectrum).real
    if float(np.min(transformed_mass)) < -1e-12:
        raise ValueError(
            "gauge creates a negative phase law; use a smaller or positive strength"
        )
    transformed_mass = np.maximum(transformed_mass, 0.0)
    transformed_mass /= np.sum(transformed_mass)
    transformed_density = transformed_mass * np.sum(density)

    observable = np.fft.ifft(surface_spectrum * phase_spectrum).real
    transformed_observable = np.fft.ifft(
        np.fft.fft(transformed_surface) * np.fft.fft(transformed_mass)
    ).real
    magnitude_change = np.linalg.norm(
        np.abs(np.fft.fft(transformed_surface)) - np.abs(surface_spectrum)
    ) / max(float(np.linalg.norm(np.abs(surface_spectrum))), 1e-12)
    return {
        "representation": IDENTIFIABILITY_REPRESENTATION,
        "counterexample": "spectral_factorization_gauge",
        "surface": signal,
        "phase_density": density,
        "transformed_surface": transformed_surface,
        "transformed_phase_density": transformed_density,
        "observable": observable,
        "transformed_observable": transformed_observable,
        "gauge_harmonic": harmonic,
        "gauge_strength": strength,
        "gauge_factor": factor,
        "transformed_phase_density_min": float(np.min(transformed_density)),
        "surface_spectral_magnitude_relative_change": float(magnitude_change),
        "observable_max_abs_difference": float(
            np.max(np.abs(observable - transformed_observable))
        ),
        "not_explainable_by_rotation": bool(magnitude_change > 1e-10),
        "is_official_kpt": False,
    }


def known_surface_spectral_diagnostics(
    surface: Sequence[float],
    *,
    relative_tolerance: float = 1e-10,
    maximum_required_harmonic: Optional[int] = None,
) -> Dict[str, object]:
    """Audit transfer modes for the observed circular-convolution experiment.

    This criterion does not establish identifiability from scalar samples
    Y=S(Theta)+noise, which have a pushforward-mixture likelihood.
    """
    signal = np.asarray(surface, dtype=np.float64)
    tolerance_fraction = float(relative_tolerance)
    if signal.ndim != 1 or signal.size < 2 or np.any(~np.isfinite(signal)):
        raise ValueError("surface must be a finite vector")
    if tolerance_fraction < 0 or not np.isfinite(tolerance_fraction):
        raise ValueError("relative_tolerance must be finite and non-negative")
    magnitude = np.abs(np.fft.fft(signal))
    threshold = tolerance_fraction * max(float(np.max(magnitude)), 1.0)
    observed = magnitude > threshold
    nonzero = magnitude[observed]
    result: Dict[str, object] = {
        "n_modes": int(signal.size),
        "observable_modes": int(np.sum(observed)),
        "unobservable_modes": int(np.sum(~observed)),
        "all_modes_identifiable_with_known_surface": bool(np.all(observed)),
        "minimum_observable_magnitude": (
            float(np.min(nonzero)) if nonzero.size else 0.0
        ),
        "maximum_magnitude": float(np.max(magnitude)),
        "relative_tolerance": tolerance_fraction,
    }
    if maximum_required_harmonic is not None:
        maximum = int(maximum_required_harmonic)
        if maximum < 0 or maximum >= signal.size // 2:
            raise ValueError(
                "maximum_required_harmonic must lie between 0 and half the grid size"
            )
        required_indices = np.concatenate(
            [np.arange(0, maximum + 1), np.arange(signal.size - maximum, signal.size)]
        )
        required_indices = np.unique(required_indices)
        required_observed = observed[required_indices]
        result.update(
            {
                "maximum_required_harmonic": maximum,
                "required_modes": int(required_indices.size),
                "observable_required_modes": int(np.sum(required_observed)),
                "unobservable_required_modes": int(np.sum(~required_observed)),
                "all_required_modes_identifiable_with_known_surface": bool(
                    np.all(required_observed)
                ),
            }
        )
    return result


def circular_mixture_observable(
    surface: Sequence[float], phase_density: Sequence[float]
) -> np.ndarray:
    """Evaluate circular convolution on an observed offset grid.

    The full offset-indexed response is assumed observable. This is not the
    probability density of scalar observations Y=S(Theta)+noise.
    """
    signal = np.asarray(surface, dtype=np.float64)
    density = np.asarray(phase_density, dtype=np.float64)
    if (
        signal.ndim != 1
        or density.shape != signal.shape
        or signal.size < 2
        or np.any(~np.isfinite(signal))
        or np.any(~np.isfinite(density))
        or np.any(density < 0)
        or float(np.sum(density)) <= 0
    ):
        raise ValueError("surface and phase_density must be matching finite vectors")
    probability_mass = density / np.sum(density)
    return np.fft.ifft(
        np.fft.fft(signal) * np.fft.fft(probability_mass)
    ).real


def recover_phase_law_known_surface(
    observable: Sequence[float],
    surface: Sequence[float],
    *,
    maximum_harmonic: int,
    ridge: float = 0.0,
    relative_spectral_tolerance: float = 1e-10,
) -> Dict[str, object]:
    """Recover a band-limited phase law when the surface is calibrated.

    The zero-frequency phase coefficient is fixed by probability normalization.
    The input is the entire offset-indexed circular convolution response,
    not unlabelled scalar samples Y=S(Theta)+noise.
    Every requested nonzero harmonic additionally requires a nonzero calibrated
    surface coefficient.  The returned probability mass is projected onto the
    simplex after truncated spectral deconvolution.
    """
    mixture = np.asarray(observable, dtype=np.float64)
    signal = np.asarray(surface, dtype=np.float64)
    maximum = int(maximum_harmonic)
    ridge = float(ridge)
    tolerance_fraction = float(relative_spectral_tolerance)
    if (
        mixture.ndim != 1
        or signal.shape != mixture.shape
        or mixture.size < 4
        or np.any(~np.isfinite(mixture))
        or np.any(~np.isfinite(signal))
    ):
        raise ValueError("observable and surface must be matching finite vectors")
    if maximum < 1 or maximum >= mixture.size // 2:
        raise ValueError("maximum_harmonic must lie between 1 and half the grid size")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and non-negative")
    if tolerance_fraction < 0 or not np.isfinite(tolerance_fraction):
        raise ValueError("relative_spectral_tolerance must be non-negative")

    mixture_spectrum = np.fft.fft(mixture)
    surface_spectrum = np.fft.fft(signal)
    recovered_spectrum = np.zeros(mixture.size, dtype=np.complex128)
    recovered_spectrum[0] = 1.0
    spectral_threshold = tolerance_fraction * max(
        float(np.max(np.abs(surface_spectrum))), 1.0
    )
    unobservable_harmonics: list[int] = []
    transfer_magnitudes: list[float] = []
    for harmonic in range(1, maximum + 1):
        transfer = surface_spectrum[harmonic]
        transfer_magnitude = float(abs(transfer))
        transfer_magnitudes.append(transfer_magnitude)
        if transfer_magnitude <= spectral_threshold:
            unobservable_harmonics.append(harmonic)
            continue
        coefficient = (
            np.conj(transfer)
            * mixture_spectrum[harmonic]
            / (transfer_magnitude**2 + ridge)
        )
        recovered_spectrum[harmonic] = coefficient
        recovered_spectrum[-harmonic] = np.conj(coefficient)

    raw_mass = np.fft.ifft(recovered_spectrum).real
    probability_mass = _project_probability_simplex(raw_mass)
    fitted_observable = np.fft.ifft(
        surface_spectrum * np.fft.fft(probability_mass)
    ).real
    return {
        "representation": IDENTIFIABILITY_REPRESENTATION,
        "estimator": "known_surface_band_limited_spectral_deconvolution",
        "forward_model": "observed_offset_indexed_circular_convolution",
        "phase_probability_mass": probability_mass,
        "raw_phase_probability_mass": raw_mass,
        "fitted_observable": fitted_observable,
        "observable_rmse": float(
            np.sqrt(np.mean((fitted_observable - mixture) ** 2))
        ),
        "maximum_harmonic": maximum,
        "unobservable_harmonics": unobservable_harmonics,
        "all_requested_harmonics_observable": not unobservable_harmonics,
        "minimum_requested_transfer_magnitude": min(transfer_magnitudes),
        "ridge": ridge,
        "truth_used_during_recovery": False,
        "is_official_kpt": False,
    }


def phase_anchor_gauge_diagnostics(
    anchor_theta: Sequence[float],
    *,
    warp_order: int,
    witness_grid_size: int = 512,
    maximum_derivative_perturbation: float = 0.5,
) -> Dict[str, object]:
    """Measure residual harmonic reparameterization freedom after phase anchors.

    A warp displacement is represented by ``K`` sine/cosine pairs.  An anchor
    imposes one linear condition that its displacement is zero.  If the anchor
    design matrix has a null space, the function returns an orientation-
    preserving nontrivial witness that fixes every supplied anchor.
    """
    anchors = np.asarray(anchor_theta, dtype=np.float64)
    order = int(warp_order)
    grid_size = int(witness_grid_size)
    derivative_limit = float(maximum_derivative_perturbation)
    if anchors.ndim != 1 or np.any(~np.isfinite(anchors)):
        raise ValueError("anchor_theta must be a finite vector")
    if np.any(anchors < -np.pi) or np.any(anchors >= np.pi):
        raise ValueError("anchor_theta values must lie in [-pi, pi)")
    if np.unique(np.round(anchors, decimals=12)).size != anchors.size:
        raise ValueError("anchor_theta values must be unique")
    if order < 1:
        raise ValueError("warp_order must be at least 1")
    if grid_size < 8 * order:
        raise ValueError("witness_grid_size must be at least 8 * warp_order")
    if not 0.0 < derivative_limit < 1.0:
        raise ValueError("maximum_derivative_perturbation must lie in (0, 1)")

    n_parameters = 2 * order
    operator = (
        harmonic_design_matrix(anchors, order)
        if anchors.size
        else np.empty((0, n_parameters), dtype=np.float64)
    )
    if anchors.size:
        _, singular_values, right_vectors = np.linalg.svd(
            operator, full_matrices=True
        )
        tolerance = (
            max(operator.shape)
            * np.finfo(np.float64).eps
            * float(singular_values[0])
        )
        rank = int(np.sum(singular_values > tolerance))
    else:
        singular_values = np.empty(0, dtype=np.float64)
        right_vectors = np.eye(n_parameters)
        tolerance = 0.0
        rank = 0
    nullity = n_parameters - rank
    grid = np.linspace(-np.pi, np.pi, grid_size, endpoint=False)
    coefficients = np.zeros(n_parameters, dtype=np.float64)
    displacement = np.zeros(grid_size, dtype=np.float64)
    minimum_warp_derivative = 1.0
    maximum_anchor_displacement = 0.0
    if nullity > 0:
        null_vector = right_vectors[rank].copy()
        phase = grid[:, None] * np.arange(1, order + 1)[None, :]
        derivative_operator = np.empty((grid_size, n_parameters))
        derivative_operator[:, 0::2] = -np.sin(phase) * np.arange(
            1, order + 1
        )[None, :]
        derivative_operator[:, 1::2] = np.cos(phase) * np.arange(
            1, order + 1
        )[None, :]
        derivative = derivative_operator @ null_vector
        scale = derivative_limit / max(float(np.max(np.abs(derivative))), 1e-12)
        coefficients = null_vector * scale
        displacement = harmonic_design_matrix(grid, order) @ coefficients
        warp_derivative = 1.0 + derivative_operator @ coefficients
        minimum_warp_derivative = float(np.min(warp_derivative))
        maximum_anchor_displacement = (
            float(np.max(np.abs(operator @ coefficients)))
            if anchors.size
            else 0.0
        )
    return {
        "representation": IDENTIFIABILITY_REPRESENTATION,
        "n_anchors": int(anchors.size),
        "warp_order": order,
        "n_warp_parameters": n_parameters,
        "anchor_operator_rank": rank,
        "residual_gauge_nullity": nullity,
        "harmonic_warp_fully_fixed": nullity == 0,
        "rank_tolerance": tolerance,
        "singular_values": singular_values,
        "witness_coefficients": coefficients,
        "witness_grid": grid,
        "witness_displacement": displacement,
        "witness_displacement_rms": float(np.sqrt(np.mean(displacement**2))),
        "minimum_warp_derivative": minimum_warp_derivative,
        "maximum_anchor_displacement": maximum_anchor_displacement,
        "is_official_kpt": False,
    }


def _validate_coefficients(coefficients: Sequence[float]) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    if (
        values.ndim != 1
        or values.size < 2
        or values.size % 2 != 0
        or np.any(~np.isfinite(values))
    ):
        raise ValueError("coefficients must be a finite a1,b1,... vector")
    return values


def _validate_phase_grid_objects(
    theta: Sequence[float],
    surface: Sequence[float],
    phase_density: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    angles = np.asarray(theta, dtype=np.float64)
    signal = np.asarray(surface, dtype=np.float64)
    density = np.asarray(phase_density, dtype=np.float64)
    if (
        angles.ndim != 1
        or angles.size < 8
        or signal.shape != angles.shape
        or density.shape != angles.shape
        or np.any(~np.isfinite(angles))
        or np.any(~np.isfinite(signal))
        or np.any(~np.isfinite(density))
        or np.any(density < 0)
        or float(np.sum(density)) <= 0
    ):
        raise ValueError("theta, surface, and phase_density must be matching finite vectors")
    spacing = np.diff(angles)
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0]):
        raise ValueError("theta must be a strictly increasing uniform grid")
    if angles[0] < -np.pi - 1e-10 or angles[-1] >= np.pi + 1e-10:
        raise ValueError("theta must cover one cycle inside [-pi, pi)")
    return angles, signal, density


def _validate_operator(operator: np.ndarray) -> np.ndarray:
    matrix = np.asarray(operator, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] < 2
        or matrix.shape[1] % 2 != 0
        or np.any(~np.isfinite(matrix))
    ):
        raise ValueError("operator must be a finite non-empty matrix with 2K columns")
    return matrix


def _project_probability_simplex(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    sorted_values = np.sort(vector)[::-1]
    cumulative = np.cumsum(sorted_values) - 1.0
    indices = np.arange(1, vector.size + 1)
    valid = sorted_values - cumulative / indices > 0
    threshold = cumulative[np.nonzero(valid)[0][-1]] / np.sum(valid)
    projected = np.maximum(vector - threshold, 0.0)
    return projected / np.sum(projected)
