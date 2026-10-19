import numpy as np

from spacekime_phase_dynamics import (
    DYNAMIC_REPRESENTATION,
    designed_phase_anchor_masks,
    estimate_smooth_phase_trajectories,
    nested_phase_anchor_masks,
    phase_trajectory_alignment_metrics,
    simulate_smooth_phase_repeats,
)


def test_smooth_phase_simulator_is_explicitly_synthetic_and_reproducible():
    first = simulate_smooth_phase_repeats(
        n_repeats=10,
        n_time=12,
        noise_sigma=0.0,
        rng=np.random.default_rng(12),
    )
    second = simulate_smooth_phase_repeats(
        n_repeats=10,
        n_time=12,
        noise_sigma=0.0,
        rng=np.random.default_rng(12),
    )
    assert first["representation"] == DYNAMIC_REPRESENTATION
    assert first["phase"].shape == (10, 12)
    assert np.allclose(first["observations"], first["signal"])
    assert np.allclose(first["phase"], second["phase"])
    step = np.abs(np.angle(np.exp(1j * np.diff(first["phase"], axis=1))))
    assert np.max(step) < 0.5


def test_anchor_masks_are_nested_and_have_exact_counts():
    masks = nested_phase_anchor_masks(9, 11, 4, rng=np.random.default_rng(4))
    assert len(masks) == 5
    for count, mask in enumerate(masks):
        assert mask.shape == (9, 11)
        assert np.all(mask.sum(axis=1) == count)
        if count:
            assert np.all(masks[count - 1] <= mask)


def test_observation_only_anchor_designs_are_nested_and_truth_independent():
    ambiguity = np.tile(np.linspace(0.0, 1.0, 11), (9, 1))
    downstream = np.tile(np.linspace(1.0, 0.0, 11), (9, 1))
    for strategy in (
        "uniform",
        "endpoints",
        "ambiguity",
        "ambiguity_spread",
        "surface_variance",
        "surface_variance_spread",
    ):
        masks = designed_phase_anchor_masks(
            9,
            11,
            4,
            strategy,
            ambiguity=ambiguity if strategy.startswith("ambiguity") else None,
            downstream_variance=(
                downstream if strategy.startswith("surface_variance") else None
            ),
        )
        for count, mask in enumerate(masks):
            assert np.all(mask.sum(axis=1) == count)
            if count:
                assert np.all(masks[count - 1] <= mask)
    ambiguity_masks = designed_phase_anchor_masks(
        9, 11, 1, "ambiguity", ambiguity=ambiguity
    )
    assert np.all(ambiguity_masks[1][:, -1])
    downstream_masks = designed_phase_anchor_masks(
        9,
        11,
        1,
        "surface_variance",
        downstream_variance=downstream,
    )
    assert np.all(downstream_masks[1][:, 0])


def test_dynamic_estimator_is_truth_free_without_declared_anchors():
    simulation = simulate_smooth_phase_repeats(
        n_repeats=10,
        n_time=10,
        rng=np.random.default_rng(21),
    )
    fit = estimate_smooth_phase_trajectories(
        simulation["observations"],
        simulation["time"],
        n_phase_grid=24,
        max_iterations=1,
    )
    assert fit["phase"].shape == simulation["phase"].shape
    assert fit["phase_truth_used_beyond_declared_anchors"] is False
    assert fit["ambiguity_uses_truth"] is False
    assert fit["phase_ambiguity"].shape == simulation["phase"].shape
    assert np.all((fit["phase_ambiguity"] >= 0) & (fit["phase_ambiguity"] <= 1))
    assert fit["downstream_score_uses_truth"] is False
    assert fit["downstream_surface_variance"].shape == simulation["phase"].shape
    assert np.all(np.asarray(fit["downstream_surface_variance"]) >= 0)
    assert fit["supervised_anchor_values_used"] is False
    assert fit["is_official_kpt"] is False


def test_fully_declared_anchor_grid_is_preserved_exactly():
    simulation = simulate_smooth_phase_repeats(
        n_repeats=10,
        n_time=10,
        rng=np.random.default_rng(31),
    )
    mask = np.ones_like(simulation["phase"], dtype=bool)
    fit = estimate_smooth_phase_trajectories(
        simulation["observations"],
        simulation["time"],
        anchor_phase=simulation["phase"],
        anchor_mask=mask,
        n_phase_grid=32,
        max_step_rad=1.0,
        max_iterations=1,
    )
    circular_error = np.angle(
        np.exp(1j * (np.asarray(fit["phase"]) - simulation["phase"]))
    )
    assert np.max(np.abs(circular_error)) < 1e-12
    assert fit["supervised_anchor_values_used"] is True


def test_unmasked_anchor_values_are_never_read():
    simulation = simulate_smooth_phase_repeats(
        n_repeats=10,
        n_time=10,
        rng=np.random.default_rng(37),
    )
    mask = nested_phase_anchor_masks(
        10, 10, 1, rng=np.random.default_rng(38)
    )[1]
    sparse = np.where(mask, simulation["phase"], np.nan)
    altered = np.where(mask, simulation["phase"], 12345.0)
    common = dict(
        observations=simulation["observations"],
        time=simulation["time"],
        anchor_mask=mask,
        n_phase_grid=24,
        max_iterations=1,
    )
    first = estimate_smooth_phase_trajectories(
        **common, anchor_phase=sparse
    )
    second = estimate_smooth_phase_trajectories(
        **common, anchor_phase=altered
    )
    assert np.allclose(first["phase"], second["phase"])


def test_phase_trajectory_metrics_remove_only_global_o2_ambiguity():
    truth = np.array(
        [[-2.1, -1.8, -1.5], [0.2, 0.5, 0.9], [2.0, 2.2, 2.4]],
        dtype=float,
    )
    estimate = np.angle(np.exp(1j * (-truth + 0.7)))
    metrics = phase_trajectory_alignment_metrics(estimate, truth)
    assert metrics["sign"] == -1.0
    assert metrics["circular_rmse"] < 1e-12
    assert metrics["step_rmse"] < 1e-12
