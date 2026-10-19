import csv
import json
from pathlib import Path

import numpy as np
import pytest

from spacekime_analytic import (
    closed_form_transform,
    compare_tciu_reference,
    conformance_cases,
    inverse_laplace_bromwich,
    laplace_kimesurface,
    numerical_laplace_transform,
    run_closed_form_conformance,
)


def test_closed_form_forward_conformance_is_precise():
    rows = run_closed_form_conformance()
    assert len(rows) == 45
    assert max(float(row["absolute_error"]) for row in rows) < 1e-7


def test_csv_conformance_does_not_claim_r_execution(tmp_path, monkeypatch):
    import run_tciu_conformance

    reference = Path(__file__).resolve().parents[1] / "results/tciu_conformance/tciu_lt_reference.csv"
    monkeypatch.setattr("sys.argv", ["run_tciu_conformance.py", "--output-dir", str(tmp_path), "--tciu-reference", str(reference)])
    assert run_tciu_conformance.main() == 0
    manifest = json.loads((tmp_path / "tciu_conformance_manifest.json").read_text())
    assert manifest["official_r_executed"] is False
    assert manifest["official_r_reference_compared"] is True


def test_laplace_surface_keeps_raw_complex_values():
    real = np.array([0.7, 1.2])
    imag = np.array([-1.0, 0.0, 1.0])
    surface = laplace_kimesurface(lambda t: np.exp(-0.5 * t), real, imag)
    expected = 1.0 / (real[:, None] + 0.5 + 1j * imag[None, :])
    assert surface.dtype == np.complex128
    assert np.allclose(surface, expected, atol=1e-8)


def test_bromwich_inverse_recovers_decay_away_from_zero():
    time = np.linspace(0.2, 1.5, 15)
    recovered = inverse_laplace_bromwich(
        lambda z: closed_form_transform("exp_decay", z, 0.7),
        time,
        omega_max=180.0,
        n_omega=12_001,
    )
    assert np.sqrt(np.mean((recovered - np.exp(-0.7 * time)) ** 2)) < 0.02


def test_numerical_lt_rejects_nonconvergent_half_plane():
    with pytest.raises(ValueError, match=r"Re\(z\)"):
        numerical_laplace_transform(lambda t: np.exp(-t), -0.1 + 1j)


def test_official_tciu_reference_requires_complete_conforming_grid(tmp_path):
    reference = tmp_path / "tciu_reference.csv"
    fields = [
        "function",
        "parameter",
        "real_x",
        "imag_y",
        "tciu_real",
        "tciu_imag",
        "tciu_version",
        "r_version",
    ]
    with reference.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in conformance_cases():
            value = closed_form_transform(
                str(case["function"]),
                complex(float(case["real_x"]), float(case["imag_y"])),
                float(case["parameter"]),
            )
            writer.writerow(
                {
                    **case,
                    "tciu_real": value.real,
                    "tciu_imag": value.imag,
                    "tciu_version": "test",
                    "r_version": "test",
                }
            )

    result = compare_tciu_reference(reference)

    assert result["cases"] == 45
    assert result["passed_at_tolerance"] is True
    assert result["max_absolute_error"] < 1e-7
