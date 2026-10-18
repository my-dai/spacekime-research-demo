import numpy as np
import pytest

from spacekime_identifiability import (
    circular_mixture_observable,
    fourier_density,
    known_surface_spectral_diagnostics,
    observable_characteristic_function,
    operator_diagnostics,
    phase_anchor_gauge_diagnostics,
    phase_reparameterization_counterexample,
    quadrature_probe_operator,
    random_positive_fourier_coefficients,
    recover_fourier_coefficients,
    recover_phase_law_known_surface,
    simulate_probe_measurements,
    spectral_gauge_counterexample,
)
from spacekime_quotient import (
    audit_repeated_measurement_design,
    canonical_observable_representative,
    compare_observable_laws,
    empirical_observable_law,
    observable_wasserstein_distance,
    permutation_mmd_test,
    pooled_median_bandwidth,
    weighted_gaussian_mmd_squared,
    weighted_quantile,
)


def test_positive_fourier_generator_is_normalized_and_positive():
    coefficients = random_positive_fourier_coefficients(
        5, rng=np.random.default_rng(101), positivity_margin=0.8
    )
    theta = np.linspace(-np.pi, np.pi, 4096, endpoint=False)
    density = fourier_density(theta, coefficients)
    assert np.min(density) > 0.0
    assert np.sum(density) * (2.0 * np.pi / theta.size) == pytest.approx(
        1.0, abs=1e-12
    )


def test_one_quadrature_is_rank_deficient_but_paired_is_complete():
    single = operator_diagnostics(quadrature_probe_operator(4, [0.0]))
    paired = operator_diagnostics(
        quadrature_probe_operator(4, [0.0, np.pi / 2.0])
    )
    assert single["rank"] == 4
    assert single["nullity"] == 4
    assert single["condition_number"] is None
    assert single["smallest_singular_value"] == 0.0
    assert paired["rank"] == 8
    assert paired["nullity"] == 0
    assert paired["condition_number"] == pytest.approx(1.0)


def test_nearly_redundant_quadratures_are_identifiable_but_unstable():
    stable = operator_diagnostics(
        quadrature_probe_operator(3, [0.0, np.pi / 2.0])
    )
    unstable = operator_diagnostics(
        quadrature_probe_operator(3, [0.0, 0.02])
    )
    assert unstable["full_column_rank"] is True
    assert float(unstable["condition_number"]) > 50.0
    assert float(unstable["condition_number"]) > float(stable["condition_number"])


def test_complete_noiseless_operator_recovers_coefficients():
    coefficients = random_positive_fourier_coefficients(
        4, rng=np.random.default_rng(102)
    )
    operator = quadrature_probe_operator(4, [0.0, np.pi / 2.0])
    simulation = simulate_probe_measurements(
        operator,
        coefficients,
        n_repeats=3,
        noise_sigma=0.0,
        rng=np.random.default_rng(103),
    )
    fit = recover_fourier_coefficients(
        operator, simulation["mean"], ridge=0.0
    )
    assert fit["optimizer_success"] is True
    assert fit["is_official_kpt"] is False
    assert np.allclose(fit["coefficients"], coefficients, atol=1e-8)


def test_probe_operator_rejects_missing_or_duplicate_harmonic_labels():
    with pytest.raises(ValueError, match="harmonics"):
        quadrature_probe_operator(3, [0.0, np.pi / 2.0], harmonics=[1, 1])
    with pytest.raises(ValueError, match="harmonics"):
        quadrature_probe_operator(3, [0.0, np.pi / 2.0], harmonics=[])


def test_smooth_phase_reparameterization_preserves_observable_law():
    theta = np.linspace(-np.pi, np.pi, 512, endpoint=False)
    coefficients = random_positive_fourier_coefficients(
        4, rng=np.random.default_rng(104)
    )
    density = fourier_density(theta, coefficients)
    surface = 0.3 + np.cos(theta + 0.2) + 0.25 * np.cos(3.0 * theta - 0.4)
    counterexample = phase_reparameterization_counterexample(
        theta, surface, density, warp_strength=0.45
    )
    assert counterexample["complete_discrete_observable_law_preserved"] is True
    assert counterexample["nonrotation_displacement_rms"] > 0.2
    assert counterexample["observable_characteristic_max_abs_difference"] < 1e-14
    assert np.min(counterexample["warp_jacobian"]) > 0.0
    assert np.min(counterexample["transformed_phase_density_at_transformed_theta"]) > 0.0


def test_spectral_gauge_changes_both_factors_but_not_their_product():
    theta = np.linspace(-np.pi, np.pi, 256, endpoint=False)
    coefficients = random_positive_fourier_coefficients(
        4, rng=np.random.default_rng(105)
    )
    density = fourier_density(theta, coefficients)
    surface = 0.2 + np.cos(theta + 0.3) + 0.4 * np.cos(2.0 * theta)
    counterexample = spectral_gauge_counterexample(
        surface, density, gauge_harmonic=1, gauge_strength=0.3
    )
    assert counterexample["not_explainable_by_rotation"] is True
    assert counterexample["surface_spectral_magnitude_relative_change"] > 0.1
    assert counterexample["transformed_phase_density_min"] > 0.0
    assert counterexample["observable_max_abs_difference"] < 1e-12
    assert not np.allclose(counterexample["surface"], counterexample["transformed_surface"])
    assert not np.allclose(
        counterexample["phase_density"], counterexample["transformed_phase_density"]
    )


def test_known_surface_with_spectral_zeros_cannot_recover_every_mode():
    theta = np.linspace(-np.pi, np.pi, 128, endpoint=False)
    diagnostics = known_surface_spectral_diagnostics(
        np.cos(theta), maximum_required_harmonic=1
    )
    assert diagnostics["observable_modes"] == 2
    assert diagnostics["unobservable_modes"] == 126
    assert diagnostics["all_modes_identifiable_with_known_surface"] is False
    assert diagnostics["required_modes"] == 3
    assert diagnostics["unobservable_required_modes"] == 1
    assert diagnostics["all_required_modes_identifiable_with_known_surface"] is False


def test_observable_characteristic_function_validates_probability_law():
    with pytest.raises(ValueError, match="probability_mass"):
        observable_characteristic_function([0.0, 1.0], [1.0, -0.1], [0.0])


def test_known_surface_recovers_band_limited_phase_law_without_noise():
    theta = np.linspace(-np.pi, np.pi, 256, endpoint=False)
    coefficients = random_positive_fourier_coefficients(
        3, rng=np.random.default_rng(106)
    )
    density = fourier_density(theta, coefficients)
    surface = (
        1.0
        + np.cos(theta + 0.2)
        + 0.7 * np.cos(2.0 * theta - 0.3)
        + 0.5 * np.cos(3.0 * theta + 0.4)
    )
    observable = circular_mixture_observable(surface, density)
    fit = recover_phase_law_known_surface(
        observable, surface, maximum_harmonic=3
    )
    assert fit["all_requested_harmonics_observable"] is True
    assert fit["truth_used_during_recovery"] is False
    assert np.allclose(
        fit["phase_probability_mass"], density / np.sum(density), atol=1e-10
    )


def test_missing_surface_harmonic_is_reported_during_recovery():
    theta = np.linspace(-np.pi, np.pi, 128, endpoint=False)
    density = fourier_density(
        theta,
        random_positive_fourier_coefficients(2, rng=np.random.default_rng(107)),
    )
    surface = 1.0 + np.cos(theta)
    observable = circular_mixture_observable(surface, density)
    fit = recover_phase_law_known_surface(
        observable, surface, maximum_harmonic=2
    )
    assert fit["all_requested_harmonics_observable"] is False
    assert fit["unobservable_harmonics"] == [2]


def test_anchor_rank_exposes_and_then_fixes_finite_warp_gauge():
    underanchored = phase_anchor_gauge_diagnostics(
        [-2.3, -0.4, 1.1], warp_order=2
    )
    fixed = phase_anchor_gauge_diagnostics(
        [-2.7, -1.2, 0.2, 1.8], warp_order=2
    )
    assert underanchored["residual_gauge_nullity"] == 1
    assert underanchored["witness_displacement_rms"] > 0.0
    assert underanchored["maximum_anchor_displacement"] < 1e-12
    assert underanchored["minimum_warp_derivative"] > 0.0
    assert fixed["residual_gauge_nullity"] == 0
    assert fixed["harmonic_warp_fully_fixed"] is True


def test_weighted_quantile_and_wasserstein_have_known_values():
    quantiles = weighted_quantile(
        [2.0, 0.0, 1.0], [0.25, 0.25, 0.5], [0.0, 0.25, 0.75, 1.0]
    )
    assert np.array_equal(quantiles, [0.0, 0.0, 1.0, 2.0])
    shifted = observable_wasserstein_distance(
        [0.0, 1.0], [0.4, 0.6], [2.0, 3.0], [0.4, 0.6], order=2
    )
    assert shifted == pytest.approx(2.0)


def test_quotient_representative_is_invariant_to_phase_reparameterization():
    theta = np.linspace(-np.pi, np.pi, 512, endpoint=False)
    density = fourier_density(
        theta,
        random_positive_fourier_coefficients(4, rng=np.random.default_rng(108)),
    )
    surface = 0.3 + np.cos(theta + 0.2) + 0.25 * np.cos(3.0 * theta - 0.4)
    counterexample = phase_reparameterization_counterexample(
        theta, surface, density, warp_strength=0.6
    )
    comparison = compare_observable_laws(
        surface,
        density,
        counterexample["transformed_surface_at_transformed_theta"],
        counterexample["transformed_probability_mass"],
    )
    original = canonical_observable_representative(surface, density)
    transformed = canonical_observable_representative(
        counterexample["transformed_surface_at_transformed_theta"],
        counterexample["transformed_probability_mass"],
    )
    assert comparison["wasserstein_2"] < 1e-14
    assert np.array_equal(
        original["quantile_values"], transformed["quantile_values"]
    )
    assert comparison["requires_phase_alignment"] is False


def test_empirical_quotient_detects_a_changed_observable_law():
    original = empirical_observable_law([-1.0, 0.0, 1.0])
    alternative = empirical_observable_law([-2.0, 0.0, 2.0])
    distance = observable_wasserstein_distance(
        original["values"],
        original["probability_mass"],
        alternative["values"],
        alternative["probability_mass"],
        order=2,
    )
    assert distance == pytest.approx(np.sqrt(2.0 / 3.0))


def test_data_design_audit_does_not_invent_repeats():
    descriptive = audit_repeated_measurement_design(
        [64, 64, 40, 160], ["X", "Y", "Z", "T"]
    )
    undocumented = audit_repeated_measurement_design(
        [64, 64, 40, 160], ["X", "Y", "Z", "T"], repeat_axis=3
    )
    documented = audit_repeated_measurement_design(
        [64, 64, 40, 12],
        ["X", "Y", "Z", "repeat"],
        repeat_axis=3,
        repeat_independence_documented=True,
        controlled_time_alignment_documented=True,
    )
    assert descriptive["status"] == "descriptive_proxy_only_no_repeat_axis"
    assert descriptive["supports_empirical_fixed_time_observable_law"] is False
    assert undocumented["status"] == "candidate_axis_but_protocol_metadata_missing"
    assert undocumented["supports_empirical_fixed_time_observable_law"] is False
    assert documented["supports_empirical_fixed_time_observable_law"] is True


def test_joint_mmd_detects_dependence_hidden_from_channel_marginals():
    theta = np.linspace(0.0, 2.0 * np.pi, 257, endpoint=False)
    mass = np.full(theta.size, 1.0 / theta.size)
    original = np.column_stack([np.cos(theta), np.sin(theta)])
    dependence_alternative = np.column_stack(
        [np.cos(theta), np.sin(2.0 * theta)]
    )
    first_marginal = observable_wasserstein_distance(
        original[:, 0], mass, dependence_alternative[:, 0], mass, order=2
    )
    second_marginal = observable_wasserstein_distance(
        original[:, 1], mass, dependence_alternative[:, 1], mass, order=2
    )
    joint_mmd = weighted_gaussian_mmd_squared(
        original,
        mass,
        dependence_alternative,
        mass,
        bandwidth=1.0,
    )
    assert first_marginal < 1e-14
    assert second_marginal < 1e-14
    assert joint_mmd > 1e-3
    assert weighted_gaussian_mmd_squared(
        original, mass, original, mass, bandwidth=1.0
    ) == pytest.approx(0.0, abs=1e-14)


def test_permutation_mmd_is_reproducible_and_detects_clear_shift():
    generator = np.random.default_rng(109)
    original = generator.normal(size=(40, 2))
    shifted = original + np.array([3.0, -2.0])
    bandwidth = pooled_median_bandwidth(original, shifted)
    first = permutation_mmd_test(
        original,
        shifted,
        bandwidth=bandwidth,
        n_permutations=49,
        rng=np.random.default_rng(110),
    )
    second = permutation_mmd_test(
        original,
        shifted,
        bandwidth=bandwidth,
        n_permutations=49,
        rng=np.random.default_rng(110),
    )
    null = permutation_mmd_test(
        original,
        original.copy(),
        n_permutations=49,
        rng=np.random.default_rng(111),
    )
    assert first["p_value"] == second["p_value"]
    assert first["p_value"] <= 0.05
    assert first["reject"] is True
    assert first["bandwidth_reselected_within_permutations"] is False
    assert null["p_value"] == pytest.approx(1.0)
    assert null["reject"] is False
