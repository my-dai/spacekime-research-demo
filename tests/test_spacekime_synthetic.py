import unittest

import numpy as np

from spacekime_advanced import (
    SYNTHETIC_REPRESENTATION,
    bootstrap_surface_uncertainty,
    default_method_configs,
    density_corrected_circular_predict,
    inverse_density_sample_weights,
    joint_time_phase_predict,
    nested_method_holdout_metrics,
    periodic_stitching_predict,
    reconstruct_advanced_surface,
    select_reconstruction_config,
    uncertainty_calibration_metrics,
)
from spacekime_synthetic import (
    circular_kernel_predict,
    latent_kimesurface,
    make_theta_grid,
    reconstruct_kimesurface,
    select_kernel_kappa,
    nested_holdout_prediction_metrics,
    phase_sampling_density,
    sampling_weighted_surface_metrics,
    simulate_repeated_measurements,
    smooth_kimesurface_time_phase,
    surface_recovery_metrics,
)


class SpacekimeSyntheticTests(unittest.TestCase):
    def test_latent_surface_is_positive_and_periodic(self) -> None:
        time = np.linspace(0, 1, 7)
        left = latent_kimesurface(time, np.full_like(time, -np.pi))
        right = latent_kimesurface(time, np.full_like(time, np.pi))

        self.assertTrue(np.all(left > 0))
        self.assertTrue(np.allclose(left, right, atol=1e-12))

    def test_circular_kernel_wraps_at_pi_boundary(self) -> None:
        prediction = circular_kernel_predict(
            np.asarray([-np.pi + 0.05, 0.0]),
            np.asarray([2.0, 0.0]),
            np.asarray([np.pi - 0.05]),
            kappa=10.0,
        )
        self.assertGreater(float(prediction[0]), 1.9)

    def test_circular_kernel_is_stable_when_all_raw_weights_are_tiny(self) -> None:
        prediction = circular_kernel_predict(
            np.asarray([-0.1, 0.0, 0.1]),
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([np.pi]),
            kappa=256.0,
        )
        self.assertTrue(np.isfinite(prediction[0]))
        self.assertAlmostEqual(float(prediction[0]), 2.0, places=8)

    def test_more_repeats_improve_ground_truth_recovery(self) -> None:
        simulation = simulate_repeated_measurements(
            n_repeats=128,
            n_time=24,
            noise_sigma=0.03,
            rng=np.random.default_rng(21),
        )
        grid = make_theta_grid(64)
        truth = latent_kimesurface(simulation["time"], grid)
        small = reconstruct_kimesurface(
            simulation["theta"][:8],
            simulation["observations"][:8],
            grid,
            kappa=4.0,
        )
        large = reconstruct_kimesurface(
            simulation["theta"],
            simulation["observations"],
            grid,
            kappa=4.0,
        )

        small_metrics = surface_recovery_metrics(truth, small)
        large_metrics = surface_recovery_metrics(truth, large)
        self.assertLess(large_metrics["nrmse_std"], small_metrics["nrmse_std"])
        self.assertGreater(large_metrics["correlation"], small_metrics["correlation"])

    def test_bandwidth_selection_uses_observations_and_returns_candidate_scores(self) -> None:
        simulation = simulate_repeated_measurements(
            n_repeats=40,
            n_time=16,
            noise_sigma=0.04,
            rng=np.random.default_rng(8),
        )
        result = select_kernel_kappa(
            simulation["theta"],
            simulation["observations"],
            candidates=(1.0, 4.0, 16.0),
            cv_folds=4,
            rng=np.random.default_rng(9),
        )
        self.assertIn(result["selected_kappa"], (1.0, 4.0, 16.0))
        self.assertEqual(set(result["candidate_rmse"]), {"1", "4", "16"})
        self.assertTrue(np.isfinite(result["cv_rmse"]))
        self.assertEqual(result["cv_scheme"], "repeat_kfold")
        self.assertEqual(result["cv_folds"], 4)

    def test_nested_holdout_has_no_truth_in_bandwidth_selection(self) -> None:
        simulation = simulate_repeated_measurements(
            n_repeats=48,
            n_time=18,
            noise_sigma=0.04,
            rng=np.random.default_rng(10),
        )
        metrics = nested_holdout_prediction_metrics(
            simulation["theta"],
            simulation["observations"],
            simulation["signal"],
            candidates=(1.0, 2.0, 4.0, 8.0),
            seed=11,
        )
        self.assertIn(metrics["selected_kappa"], (1.0, 2.0, 4.0, 8.0))
        self.assertGreater(metrics["time_mean_baseline_rmse"], 0.0)
        self.assertTrue(np.isfinite(metrics["kernel_rmse"]))

    def test_sampling_weighted_metrics_expose_low_coverage_failure(self) -> None:
        grid = make_theta_grid(64)
        time = np.linspace(0.0, 1.0, 6)
        truth = latent_kimesurface(time, grid)
        reconstruction = truth.copy()
        # Introduce error only around the poorly sampled opposite phase sector.
        reconstruction[:, :8] += 1.0
        reconstruction[:, -8:] += 1.0
        theta = np.zeros((32, time.size), dtype=np.float64)
        density = phase_sampling_density(theta, grid, kappa=8.0)
        metrics = sampling_weighted_surface_metrics(
            truth, reconstruction, theta, grid, density_kappa=8.0
        )
        self.assertEqual(density.shape, truth.shape)
        self.assertGreater(metrics["low_coverage_nrmse_std"], metrics["high_coverage_nrmse_std"])
        self.assertGreater(metrics["low_to_high_rmse_ratio"], 1.0)

    def test_time_phase_smoothing_wraps_phase_boundary(self) -> None:
        surface = np.zeros((3, 16), dtype=np.float64)
        surface[:, 0] = 1.0
        smoothed = smooth_kimesurface_time_phase(surface, theta_sigma=1.0)
        self.assertGreater(float(smoothed[1, -1]), 0.0)
        self.assertTrue(np.allclose(smoothed[0], smoothed[1]))

    def test_inverse_density_weights_upweight_isolated_phase(self) -> None:
        theta = np.asarray([0.0, 0.01, -0.01, 0.02, 2.8])
        weights = inverse_density_sample_weights(
            theta, density_kappa=8.0, max_weight_ratio=20.0
        )
        self.assertAlmostEqual(float(np.mean(weights)), 1.0, places=12)
        self.assertGreater(float(weights[-1]), float(np.max(weights[:-1])))

    def test_density_corrected_prediction_is_finite_and_periodic(self) -> None:
        theta = np.asarray([-np.pi + 0.05, -0.1, 0.0, 0.1, 1.0])
        values = np.cos(theta)
        prediction = density_corrected_circular_predict(
            theta,
            values,
            np.asarray([-np.pi, np.pi]),
            kappa=8.0,
            density_kappa=4.0,
        )
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertAlmostEqual(float(prediction[0]), float(prediction[1]), places=12)

    def test_periodic_stitching_connects_across_phase_seam(self) -> None:
        theta = np.asarray([-np.pi + 0.2, 0.0, np.pi - 0.2])
        values = np.asarray([2.0, 0.0, 2.0])
        prediction = periodic_stitching_predict(
            theta, values, np.asarray([-np.pi, np.pi, 0.0])
        )
        self.assertAlmostEqual(float(prediction[0]), 2.0, places=12)
        self.assertAlmostEqual(float(prediction[1]), 2.0, places=12)
        self.assertAlmostEqual(float(prediction[2]), 0.0, places=12)

    def test_joint_kernel_borrows_information_across_time(self) -> None:
        time = np.asarray([0.0, 0.5, 1.0])
        theta = np.zeros((4, 3), dtype=np.float64)
        values = np.asarray(
            [[0.0, np.nan, 2.0], [0.0, np.nan, 2.0],
             [0.0, np.nan, 2.0], [0.0, np.nan, 2.0]]
        )
        query = np.zeros((1, 3), dtype=np.float64)
        prediction = joint_time_phase_predict(
            theta, values, time, query, kappa=4.0, time_bandwidth=0.5
        )
        self.assertTrue(np.isfinite(prediction[0, 1]))
        self.assertAlmostEqual(float(prediction[0, 1]), 1.0, places=12)

    def test_advanced_model_selection_and_nested_holdout_use_observations(self) -> None:
        simulation = simulate_repeated_measurements(
            n_repeats=24,
            n_time=10,
            noise_sigma=0.04,
            rng=np.random.default_rng(101),
            phase_sampler="vonmises",
        )
        configs = default_method_configs(
            "joint_kernel", kappas=(4.0, 16.0), time_bandwidths=(0.05, 0.1)
        )
        selection = select_reconstruction_config(
            simulation["theta"],
            simulation["observations"],
            simulation["time"],
            "joint_kernel",
            configs,
            cv_folds=3,
            rng=np.random.default_rng(5),
        )
        self.assertIn(selection["selected_config"], configs)
        self.assertEqual(selection["cv_scheme"], "repeat_kfold_observations_only")
        nested = nested_method_holdout_metrics(
            simulation["theta"],
            simulation["observations"],
            simulation["signal"],
            simulation["time"],
            "joint_kernel",
            configs,
            split_seed=7,
            cv_seed=8,
        )
        self.assertTrue(np.isfinite(nested["outer_truth_rmse"]))
        self.assertTrue(np.isfinite(nested["outer_observed_rmse"]))

    def test_bootstrap_uncertainty_is_truth_free_and_reproducible(self) -> None:
        simulation = simulate_repeated_measurements(
            n_repeats=24,
            n_time=8,
            noise_sigma=0.03,
            rng=np.random.default_rng(44),
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
        )
        first = bootstrap_surface_uncertainty(
            **kwargs, rng=np.random.default_rng(9)
        )
        second = bootstrap_surface_uncertainty(
            **kwargs, rng=np.random.default_rng(9)
        )
        self.assertEqual(first["representation"], SYNTHETIC_REPRESENTATION)
        self.assertTrue(np.allclose(first["lower"], second["lower"]))
        self.assertTrue(np.all(first["lower"] <= first["upper"]))
        self.assertTrue(np.all(first["basic_lower"] <= first["basic_upper"]))

        truth = latent_kimesurface(simulation["time"], grid)
        calibration = uncertainty_calibration_metrics(truth, first)
        basic_calibration = uncertainty_calibration_metrics(
            truth, first, interval="basic", point_key="bias_corrected_point"
        )
        self.assertGreaterEqual(calibration["empirical_coverage"], 0.0)
        self.assertLessEqual(calibration["empirical_coverage"], 1.0)
        self.assertGreater(calibration["mean_interval_width"], 0.0)
        self.assertGreater(basic_calibration["mean_interval_width"], 0.0)

    def test_all_advanced_reconstructors_preserve_surface_shape(self) -> None:
        simulation = simulate_repeated_measurements(
            n_repeats=16,
            n_time=7,
            noise_sigma=0.02,
            rng=np.random.default_rng(55),
        )
        grid = make_theta_grid(24)
        configs = {
            "kernel": {"kappa": 8.0},
            "density_kernel": {
                "kappa": 8.0,
                "density_kappa": 4.0,
                "max_weight_ratio": 10.0,
            },
            "joint_kernel": {"kappa": 8.0, "time_bandwidth": 0.1},
            "stitching": {"time_sigma": 0.5},
        }
        for method, config in configs.items():
            with self.subTest(method=method):
                surface = reconstruct_advanced_surface(
                    simulation["theta"],
                    simulation["observations"],
                    simulation["time"],
                    grid,
                    method,
                    config,
                )
                self.assertEqual(surface.shape, (7, 24))
                self.assertTrue(np.all(np.isfinite(surface)))



if __name__ == "__main__":
    unittest.main()
