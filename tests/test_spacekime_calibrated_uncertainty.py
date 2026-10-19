import numpy as np
import pytest

from spacekime_advanced import (
    bootstrap_critical_value_diagnostics,
    bootstrap_surface_uncertainty_reselected,
    conformal_interval_metrics,
    reconstruction_support_diagnostics,
    simultaneous_bootstrap_surface_band,
    simultaneous_surface_coverage_metrics,
    split_conformal_prediction_intervals,
    synthetic_population_smoothed_surface,
    undersmooth_reconstruction_config,
)
from spacekime_synthetic import (
    latent_kimesurface,
    make_theta_grid,
    simulate_repeated_measurements,
)


def test_reselected_bootstrap_records_selection_instability():
    simulation = simulate_repeated_measurements(
        n_repeats=24,
        n_time=8,
        noise_sigma=0.05,
        rng=np.random.default_rng(7),
    )
    result = bootstrap_surface_uncertainty_reselected(
        simulation["theta"],
        simulation["observations"],
        simulation["time"],
        make_theta_grid(16),
        "kernel",
        [{"kappa": 4.0}, {"kappa": 8.0}],
        n_bootstrap=20,
        confidence=0.9,
        cv_folds=3,
        rng=np.random.default_rng(8),
    )
    assert result["truth_used_during_estimation"] is False
    assert sum(result["selection_frequency"].values()) == 20
    assert result["lower"].shape == (8, 16)
    assert np.all(result["upper"] >= result["lower"])


def test_split_conformal_uses_disjoint_repeat_units_and_scores_target():
    simulation = simulate_repeated_measurements(
        n_repeats=32,
        n_time=10,
        noise_sigma=0.06,
        rng=np.random.default_rng(11),
    )
    result = split_conformal_prediction_intervals(
        simulation["theta"],
        simulation["observations"],
        simulation["time"],
        "kernel",
        [{"kappa": 4.0}, {"kappa": 8.0}],
        confidence=0.9,
        split_seed=13,
        cv_seed=14,
    )
    train = set(np.asarray(result["train_indices"]).tolist())
    calibration = set(np.asarray(result["calibration_indices"]).tolist())
    test = set(np.asarray(result["test_indices"]).tolist())
    assert train.isdisjoint(calibration | test)
    assert calibration.isdisjoint(test)
    metrics = conformal_interval_metrics(
        simulation["observations"][np.asarray(result["test_indices"])], result
    )
    assert 0.0 <= metrics["pointwise_coverage"] <= 1.0
    assert 0.0 <= metrics["trajectory_coverage"] <= 1.0
    assert metrics["mean_interval_width"] > 0
    assert result["truth_used_during_estimation"] is False


def test_undersmoothing_transform_is_method_specific_and_prespecified():
    kernel = undersmooth_reconstruction_config("kernel", {"kappa": 4.0}, factor=2)
    joint = undersmooth_reconstruction_config(
        "joint_kernel", {"kappa": 8.0, "time_bandwidth": 0.1}, factor=4
    )
    stitching = undersmooth_reconstruction_config(
        "stitching", {"time_sigma": 1.0}, factor=2
    )
    assert kernel == {"kappa": 8.0}
    assert joint == {"kappa": 32.0, "time_bandwidth": 0.05}
    assert stitching == {"time_sigma": 0.5}


def test_simultaneous_band_is_truth_free_reproducible_and_auditable():
    simulation = simulate_repeated_measurements(
        n_repeats=24,
        n_time=8,
        noise_sigma=0.05,
        rng=np.random.default_rng(51),
    )
    grid = make_theta_grid(16)
    kwargs = dict(
        theta_samples=simulation["theta"],
        observations=simulation["observations"],
        time=simulation["time"],
        theta_grid=grid,
        method="kernel",
        config={"kappa": 8.0},
        n_bootstrap=20,
        confidence=0.9,
    )
    first = simultaneous_bootstrap_surface_band(
        **kwargs, rng=np.random.default_rng(52)
    )
    second = simultaneous_bootstrap_surface_band(
        **kwargs, rng=np.random.default_rng(52)
    )
    assert first["truth_used_during_estimation"] is False
    assert first["is_bayesian"] is False
    assert first["lower"].shape == (8, 16)
    assert np.all(first["upper"] >= first["lower"])
    assert np.allclose(first["lower"], second["lower"])
    assert first["critical_value"] >= 0
    truth = latent_kimesurface(simulation["time"], grid)
    metrics = simultaneous_surface_coverage_metrics(truth, first)
    assert 0.0 <= metrics["pointwise_coverage"] <= 1.0
    assert isinstance(metrics["whole_surface_covered"], bool)
    assert metrics["mean_band_width"] > 0
    assert metrics["max_error_to_critical_ratio"] >= 0


def test_population_smoothed_target_is_a_separate_synthetic_oracle():
    time = np.linspace(0.0, 1.0, 8)
    grid = make_theta_grid(16)
    truth = latent_kimesurface(time, grid)
    broad = synthetic_population_smoothed_surface(
        time, grid, "kernel", {"kappa": 2.0}, "uniform", integration_points=1024
    )
    narrow = synthetic_population_smoothed_surface(
        time, grid, "kernel", {"kappa": 32.0}, "uniform", integration_points=1024
    )
    joint = synthetic_population_smoothed_surface(
        time,
        grid,
        "joint_kernel",
        {"kappa": 8.0, "time_bandwidth": 0.08},
        "vonmises",
        integration_points=1024,
    )
    assert broad.shape == truth.shape == narrow.shape == joint.shape
    assert np.all(np.isfinite(joint))
    assert np.sqrt(np.mean((narrow - truth) ** 2)) < np.sqrt(
        np.mean((broad - truth) ** 2)
    )


def test_support_mask_uses_phase_design_only_and_tracks_clustered_sampling():
    uniform = simulate_repeated_measurements(
        n_repeats=72,
        n_time=10,
        noise_sigma=0.05,
        rng=np.random.default_rng(101),
        phase_sampler="uniform",
    )
    clustered = simulate_repeated_measurements(
        n_repeats=72,
        n_time=10,
        noise_sigma=0.05,
        rng=np.random.default_rng(102),
        phase_sampler="vonmises",
    )
    grid = make_theta_grid(32)
    kwargs = dict(
        time=uniform["time"],
        theta_grid=grid,
        method="kernel",
        config={"kappa": 16.0},
        relative_mass_threshold=0.25,
        min_repeat_ess=5.0,
    )
    uniform_support = reconstruction_support_diagnostics(
        uniform["theta"], **kwargs
    )
    clustered_support = reconstruction_support_diagnostics(
        clustered["theta"], **kwargs
    )
    assert uniform_support["truth_used_during_definition"] is False
    assert uniform_support["supported_mask"].shape == (10, 32)
    assert np.all(uniform_support["repeat_effective_sample_size"] >= 0)
    assert uniform_support["supported_fraction"] > clustered_support["supported_fraction"]


def test_supported_domain_band_is_paired_and_no_wider_than_full_domain():
    simulation = simulate_repeated_measurements(
        n_repeats=32,
        n_time=8,
        noise_sigma=0.05,
        rng=np.random.default_rng(111),
        phase_sampler="vonmises",
    )
    grid = make_theta_grid(20)
    config = {"kappa": 16.0}
    support = reconstruction_support_diagnostics(
        simulation["theta"],
        simulation["time"],
        grid,
        "kernel",
        config,
        relative_mass_threshold=0.4,
        min_repeat_ess=3.0,
    )
    common = dict(
        theta_samples=simulation["theta"],
        observations=simulation["observations"],
        time=simulation["time"],
        theta_grid=grid,
        method="kernel",
        config=config,
        n_bootstrap=30,
        confidence=0.9,
    )
    full = simultaneous_bootstrap_surface_band(
        **common,
        rng=np.random.default_rng(112),
        return_bootstrap_samples=True,
    )
    bootstrap_samples = full.pop("bootstrap_samples")
    supported = simultaneous_bootstrap_surface_band(
        **common,
        domain_mask=support["supported_mask"],
        precomputed_bootstrap_samples=bootstrap_samples,
    )
    independently_recomputed = simultaneous_bootstrap_surface_band(
        **common,
        domain_mask=support["supported_mask"],
        rng=np.random.default_rng(112),
    )
    assert supported["domain_kind"] == "prespecified_observation_only_supported_grid"
    assert supported["critical_value"] <= full["critical_value"]
    assert np.allclose(supported["lower"], independently_recomputed["lower"])
    assert np.allclose(supported["upper"], independently_recomputed["upper"])
    truth = latent_kimesurface(simulation["time"], grid)
    audit = simultaneous_surface_coverage_metrics(truth, supported)
    assert audit["evaluated_fraction"] == support["supported_fraction"]


def test_precomputed_bootstrap_surface_shape_is_validated():
    simulation = simulate_repeated_measurements(
        n_repeats=8,
        n_time=12,
        rng=np.random.default_rng(113),
    )
    grid = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    with pytest.raises(ValueError, match="precomputed_bootstrap_samples"):
        simultaneous_bootstrap_surface_band(
            simulation["theta"],
            simulation["observations"],
            simulation["time"],
            grid,
            "kernel",
            {"kappa": 2.0},
            n_bootstrap=20,
            precomputed_bootstrap_samples=np.zeros((19, 12, 16)),
        )


def test_critical_value_mc_resolution_improves_with_more_resamples():
    values = np.linspace(0.5, 4.0, 500)
    small = bootstrap_critical_value_diagnostics(values[:40], 0.9)
    large = bootstrap_critical_value_diagnostics(values, 0.9)
    assert large["dkw_probability_epsilon"] < small["dkw_probability_epsilon"]
    assert large["bootstrap_resamples"] == 500
