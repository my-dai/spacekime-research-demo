import numpy as np

from spacekime_blind import (
    align_circular_phase,
    estimate_blind_repeat_phase,
    simulate_constant_phase_repeats,
    truth_grid_to_estimated_coordinate,
)


def test_alignment_resolves_rotation_and_reflection():
    truth = np.linspace(-np.pi, np.pi, 30, endpoint=False)
    estimate = np.angle(np.exp(1j * (-(truth - 0.8))))
    aligned = align_circular_phase(estimate, truth)
    assert aligned["sign"] == -1.0
    assert aligned["circular_rmse"] < 1e-10


def test_blind_estimator_never_accepts_truth_and_returns_finite_phase():
    simulation = simulate_constant_phase_repeats(
        n_repeats=24,
        n_time=24,
        noise_sigma=0.03,
        rng=np.random.default_rng(12),
    )
    fit = estimate_blind_repeat_phase(
        simulation["observations"], n_phase_grid=48, max_iterations=3
    )
    assert fit["truth_used_during_estimation"] is False
    assert fit["is_official_kpt"] is False
    assert np.asarray(fit["phase"]).shape == (24,)
    assert np.all(np.isfinite(fit["phase"]))


def test_truth_grid_coordinate_mapping_inverts_alignment():
    grid = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    sign, offset = -1.0, 0.55
    estimated_grid = truth_grid_to_estimated_coordinate(grid, sign=sign, offset=offset)
    remapped = np.angle(np.exp(1j * (sign * estimated_grid + offset)))
    assert np.allclose(np.angle(np.exp(1j * (remapped - grid))), 0.0, atol=1e-12)
