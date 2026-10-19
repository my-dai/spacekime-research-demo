import math
import unittest

import numpy as np

from spacekime_core import (
    apply_phase_noise_volume,
    brain_mask_from_magnitude,
    build_phase_distribution_surface,
    build_circular_stats_over_time,
    complex_phase_metrics,
    estimate_phase_noise_sigma,
    kime_distance_L2,
    kime_js_distance,
    kime_surface_summary,
    phase_difference_noise_diagnostics,
    phase_scope_mask,
    roi_center_box,
    snr_mean_over_std,
)


class SpacekimeCoreTests(unittest.TestCase):
    def test_apply_phase_noise_preserves_magnitude_and_respects_t_list(self) -> None:
        np.random.seed(7)
        phase = np.random.normal(0.0, 0.05, size=(6, 6, 2, 4))
        bigim = (2.5 * np.exp(1j * phase)).astype(np.complex64)

        np.random.seed(11)
        noisy = apply_phase_noise_volume(bigim, sigma=0.4, t_list=[0, 2])

        self.assertTrue(np.allclose(np.abs(noisy), np.abs(bigim), atol=1e-6))
        self.assertTrue(np.allclose(noisy[:, :, :, 1], bigim[:, :, :, 1]))
        self.assertTrue(np.allclose(noisy[:, :, :, 3], bigim[:, :, :, 3]))
        self.assertGreater(np.max(np.abs(noisy[:, :, :, 0] - bigim[:, :, :, 0])), 0.01)

    def test_kime_distances_are_zero_for_identical_stats(self) -> None:
        np.random.seed(3)
        bigim = np.exp(1j * np.random.normal(0.0, 0.2, size=(5, 5, 2, 6)))
        mask = np.ones((5, 5, 2), dtype=bool)

        stats = build_circular_stats_over_time(bigim, mask, n_theta=32, normalize_h=True)

        self.assertEqual(stats["H"].shape, (6, 32))
        self.assertEqual(stats["entropy_t"].shape, (6,))
        self.assertTrue(np.all((stats["entropy_normalized_t"] >= 0) & (stats["entropy_normalized_t"] <= 1)))
        self.assertAlmostEqual(kime_distance_L2(stats, stats), 0.0, places=8)
        self.assertAlmostEqual(kime_js_distance(stats, stats), 0.0, places=8)

    def test_kime_summary_tracks_phase_spread(self) -> None:
        focused = np.ones((8, 8, 1, 4), dtype=np.complex64)
        spread_phase = np.linspace(-np.pi, np.pi, 8 * 8 * 1 * 4).reshape(8, 8, 1, 4)
        spread = np.exp(1j * spread_phase).astype(np.complex64)
        mask = np.ones((8, 8, 1), dtype=bool)

        focused_stats = build_circular_stats_over_time(
            focused, mask, n_theta=32, normalize_h=True
        )
        spread_stats = build_circular_stats_over_time(spread, mask, n_theta=32, normalize_h=True)

        focused_summary = kime_surface_summary(focused_stats)
        spread_summary = kime_surface_summary(spread_stats)

        self.assertGreater(focused_summary["mean_R"], spread_summary["mean_R"])
        self.assertLess(focused_summary["mean_entropy"], spread_summary["mean_entropy"])

    def test_phase_noise_sigma_estimate_reports_per_frame_scale(self) -> None:
        rng = np.random.default_rng(123)
        true_sigma = 0.12
        phase_noise = rng.normal(0.0, true_sigma, size=(18, 18, 2, 40))
        bigim = np.exp(1j * phase_noise).astype(np.complex64)

        per_frame = estimate_phase_noise_sigma(
            bigim, z_idx=0, t_start=0, t_end=40, background_margin=4, per_frame=True
        )
        diff_scale = estimate_phase_noise_sigma(
            bigim, z_idx=0, t_start=0, t_end=40, background_margin=4, per_frame=False
        )

        self.assertAlmostEqual(per_frame, true_sigma, delta=0.02)
        self.assertAlmostEqual(diff_scale / math.sqrt(2.0), per_frame, delta=1e-9)

        diagnostics = phase_difference_noise_diagnostics(
            bigim, z_idx=0, t_start=0, t_end=40, background_margin=4
        )
        self.assertAlmostEqual(
            diagnostics["circular_per_frame_sigma"], true_sigma, delta=0.02
        )

    def test_paired_complex_metrics_retain_phase_correspondence(self) -> None:
        ref = np.ones((20, 20), dtype=np.complex64)
        constant_shift = ref * np.exp(0.5j)
        shifted = complex_phase_metrics(ref, constant_shift)

        self.assertAlmostEqual(shifted["phase_rmse_rad"], 0.5, places=6)
        self.assertAlmostEqual(shifted["phase_bias_rad"], 0.5, places=6)
        self.assertAlmostEqual(shifted["phase_coherence"], 1.0, places=6)
        self.assertAlmostEqual(shifted["magnitude_nrmse"], 0.0, places=6)

        rng = np.random.default_rng(9)
        random_shift = ref * np.exp(1j * rng.normal(0.0, 1.0, size=ref.shape))
        dispersed = complex_phase_metrics(ref, random_shift)
        self.assertLess(dispersed["phase_coherence"], shifted["phase_coherence"])
        self.assertGreater(dispersed["complex_nrmse"], 0.5)

    def test_streamed_phase_noise_surface_reports_paired_metrics(self) -> None:
        bigim = np.ones((12, 12, 2, 8), dtype=np.complex64)
        mask = np.ones((12, 12, 2), dtype=bool)
        stats = build_phase_distribution_surface(
            bigim,
            mask,
            n_theta=32,
            normalize_h=True,
            phase_noise_sigma=1.0,
            rng=np.random.default_rng(5),
        )
        summary = kime_surface_summary(stats)

        self.assertEqual(stats["representation"], "spatial_phase_distribution_proxy")
        self.assertAlmostEqual(summary["mean_phase_coherence"], math.exp(-0.5), delta=0.08)
        self.assertGreater(summary["mean_complex_nrmse"], 0.5)

    def test_relative_phase_mode_removes_static_spatial_phase_offsets(self) -> None:
        rng = np.random.default_rng(12)
        spatial_phase = rng.uniform(-np.pi, np.pi, size=(10, 10, 1, 1))
        bigim = np.repeat(np.exp(1j * spatial_phase), 4, axis=3).astype(np.complex64)
        mask = np.ones((10, 10, 1), dtype=bool)
        raw = build_phase_distribution_surface(bigim, mask, n_theta=32)
        relative = build_phase_distribution_surface(
            bigim, mask, n_theta=32, phase_mode="relative", reference_t=0
        )

        self.assertLess(kime_surface_summary(raw)["mean_R"], 0.25)
        self.assertGreater(kime_surface_summary(relative)["mean_R"], 0.99)

    def test_invalid_time_row_is_explicitly_marked(self) -> None:
        bigim = np.ones((4, 4, 1, 3), dtype=np.complex64)
        mask = np.ones_like(bigim, dtype=bool)
        mask[:, :, :, 1] = False
        stats = build_phase_distribution_surface(bigim, mask, n_theta=16, normalize_h=True)

        self.assertEqual(stats["row_valid"].tolist(), [True, False, True])
        self.assertTrue(np.isnan(stats["entropy_t"][1]))
        self.assertTrue(np.isnan(stats["R_t"][1]))
        self.assertEqual(int(stats["sample_count_t"][1]), 0)
        self.assertEqual(kime_surface_summary(stats)["valid_time_points"], 2.0)

    def test_phase_noise_diagnostics_empty_values_return_dict(self) -> None:
        bigim = np.full((4, 4, 1, 3), np.nan + 1j * np.nan, dtype=np.complex64)
        diagnostics = phase_difference_noise_diagnostics(
            bigim, z_idx=0, background_margin=1
        )
        self.assertIsInstance(diagnostics, dict)
        self.assertEqual(diagnostics["sample_count"], 0.0)
        self.assertTrue(np.isnan(diagnostics["linear_diff_sigma"]))

    def test_phase_scope_mask_honors_slice_and_roi(self) -> None:
        bigim = np.ones((10, 12, 3, 2), dtype=np.complex64)
        bigim[4:7, 4:8, 1, :] = 10

        slice_mask = phase_scope_mask(bigim, scope="slice", z_idx=1, percentile=50)
        roi_mask = phase_scope_mask(
            bigim, scope="roi", z_idx=1, roi_half=3, percentile=50
        )
        self.assertFalse(np.any(slice_mask[:, :, 0]))
        self.assertFalse(np.any(roi_mask[:, :, 2]))
        self.assertGreater(slice_mask[:, :, 1].sum(), 0)
        self.assertLessEqual(roi_mask.sum(), 6 * 6)

    def test_roi_helpers_and_mask_are_stable(self) -> None:
        img = np.arange(100, dtype=np.float32).reshape(10, 10)
        roi = roi_center_box(img, half=2)
        self.assertEqual(roi.shape, (4, 4))
        self.assertTrue(math.isfinite(snr_mean_over_std(roi)))

        bigim = np.ones((4, 4, 2, 3), dtype=np.complex64)
        bigim[2:, :, :, :] *= 5
        mask = brain_mask_from_magnitude(bigim, t_idx=99, percentile=50)
        self.assertEqual(mask.shape, (4, 4, 2))
        self.assertGreater(mask.sum(), 0)


if __name__ == "__main__":
    unittest.main()
