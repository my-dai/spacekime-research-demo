"""Analytic Laplace-domain kimesurfaces and TCIU conformance helpers.

This module implements the analytic-transform line described by SOCR/TCIU.  A
Laplace-domain surface ``F(x + i y)`` is not the same statistical object as the
latent repeated-measurement surface used in :mod:`spacekime_synthetic`, and it
is not the observed real-fMRI spatial phase-distribution proxy.

The numerical transform is deliberately kept independent of the experimental
SOCR Python port.  Closed-form transforms provide an executable oracle here;
``tools/generate_tciu_reference.R`` can additionally generate references from
the official R package on a machine with R and TCIU installed.
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.integrate import quad


ANALYTIC_REPRESENTATION = "analytic_laplace_kimesurface"
TCIU_CONFORMANCE_ABS_TOL = 5e-5


def numerical_laplace_transform(
    function: Callable[[float], float],
    z: complex,
    *,
    epsabs: float = 1e-10,
    epsrel: float = 1e-10,
    limit: int = 300,
) -> complex:
    """Numerically evaluate ``integral_0^inf f(t) exp(-z t) dt``.

    The real part of ``z`` must be positive.  Real and imaginary integrals are
    evaluated separately so the result can be compared directly with
    ``TCIU::LT(FUNCT, z)``.
    """
    z_value = complex(z)
    if not np.isfinite(z_value.real) or not np.isfinite(z_value.imag):
        raise ValueError("z must be finite")
    if z_value.real <= 0:
        raise ValueError("Re(z) must be positive for this numerical implementation")

    def real_integrand(t: float) -> float:
        value = float(function(t))
        return value * np.exp(-z_value.real * t) * np.cos(z_value.imag * t)

    def imag_integrand(t: float) -> float:
        value = float(function(t))
        return -value * np.exp(-z_value.real * t) * np.sin(z_value.imag * t)

    real, _ = quad(real_integrand, 0.0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)
    imag, _ = quad(imag_integrand, 0.0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)
    return complex(real, imag)


def laplace_kimesurface(
    function: Callable[[float], float],
    real_axis: Sequence[float],
    imag_axis: Sequence[float],
    **quad_options: float,
) -> np.ndarray:
    """Map a one-dimensional function onto a raw complex Laplace surface.

    Rows correspond to ``real_axis`` and columns to ``imag_axis``.  No plotting
    normalization is applied, which keeps the returned values mathematically
    auditable and avoids conflating visualization transforms with the LT.
    """
    real = np.asarray(real_axis, dtype=np.float64)
    imag = np.asarray(imag_axis, dtype=np.float64)
    if real.ndim != 1 or imag.ndim != 1 or real.size == 0 or imag.size == 0:
        raise ValueError("real_axis and imag_axis must be non-empty vectors")
    if np.any(~np.isfinite(real)) or np.any(real <= 0) or np.any(~np.isfinite(imag)):
        raise ValueError("real_axis must be finite/positive and imag_axis finite")
    surface = np.empty((real.size, imag.size), dtype=np.complex128)
    for real_idx, x_value in enumerate(real):
        for imag_idx, y_value in enumerate(imag):
            surface[real_idx, imag_idx] = numerical_laplace_transform(
                function, complex(x_value, y_value), **quad_options
            )
    return surface


def inverse_laplace_bromwich(
    transform: Callable[[np.ndarray], np.ndarray],
    time: Sequence[float],
    *,
    gamma: float = 1.0,
    omega_max: float = 180.0,
    n_omega: int = 12_001,
) -> np.ndarray:
    """Approximate an inverse LT with a truncated Fourier--Bromwich integral.

    This transparent baseline is useful for analytic validation, but it is not
    claimed to reproduce the contour-selection and fallback logic of
    ``TCIU::ILT``.  Values very close to discontinuities (notably ``t=0``) are
    sensitive to truncation and should not be used as a conformance oracle.
    """
    time_arr = np.asarray(time, dtype=np.float64)
    if time_arr.ndim != 1 or np.any(~np.isfinite(time_arr)) or np.any(time_arr <= 0):
        raise ValueError("time must be a finite vector with strictly positive values")
    if gamma <= 0 or omega_max <= 0 or int(n_omega) < 101:
        raise ValueError("gamma/omega_max must be positive and n_omega at least 101")
    omega = np.linspace(0.0, float(omega_max), int(n_omega), dtype=np.float64)
    z = float(gamma) + 1j * omega
    transformed = np.asarray(transform(z), dtype=np.complex128)
    if transformed.shape != z.shape:
        transformed = np.broadcast_to(transformed, z.shape)
    output = np.empty(time_arr.shape, dtype=np.float64)
    for idx, t_value in enumerate(time_arr):
        integrand = np.real(transformed * np.exp(1j * omega * t_value))
        output[idx] = np.exp(gamma * t_value) * np.trapezoid(integrand, omega) / np.pi
    return output


def closed_form_transform(name: str, z: np.ndarray | complex, parameter: float = 1.0):
    """Evaluate one of the closed-form LT oracles used by conformance tests."""
    key = str(name).strip().lower()
    z_arr = np.asarray(z, dtype=np.complex128)
    value = float(parameter)
    if value <= 0 or not np.isfinite(value):
        raise ValueError("parameter must be finite and positive")
    if key == "exp_decay":
        result = 1.0 / (z_arr + value)
    elif key == "sine":
        result = value / (z_arr**2 + value**2)
    elif key == "linear":
        result = 1.0 / z_arr**2
    else:
        raise ValueError("name must be 'exp_decay', 'sine', or 'linear'")
    return complex(result) if np.ndim(z) == 0 else result


def conformance_cases() -> list[dict[str, object]]:
    """Return deterministic analytic LT cases shared with the R harness."""
    cases: list[dict[str, object]] = []
    for function_name, parameter in (("exp_decay", 0.7), ("sine", 1.3), ("linear", 1.0)):
        for real in (0.6, 1.0, 1.8):
            for imag in (-2.0, -0.5, 0.0, 0.5, 2.0):
                cases.append(
                    {
                        "function": function_name,
                        "parameter": parameter,
                        "real_x": real,
                        "imag_y": imag,
                    }
                )
    return cases


def run_closed_form_conformance() -> list[dict[str, object]]:
    """Compare the numerical LT against exact analytic values."""
    functions: Mapping[str, Callable[[float, float], float]] = {
        "exp_decay": lambda t, p: float(np.exp(-p * t)),
        "sine": lambda t, p: float(np.sin(p * t)),
        "linear": lambda t, p: float(t),
    }
    rows: list[dict[str, object]] = []
    for case in conformance_cases():
        name = str(case["function"])
        parameter = float(case["parameter"])
        z = complex(float(case["real_x"]), float(case["imag_y"]))
        numerical = numerical_laplace_transform(lambda t: functions[name](t, parameter), z)
        expected = closed_form_transform(name, z, parameter)
        rows.append(
            {
                **case,
                "numeric_real": numerical.real,
                "numeric_imag": numerical.imag,
                "reference_real": expected.real,
                "reference_imag": expected.imag,
                "absolute_error": abs(numerical - expected),
                "reference_source": "closed_form_analytic",
            }
        )
    return rows


def compare_tciu_reference(
    reference_csv: Path | str,
    *,
    absolute_tolerance: float = TCIU_CONFORMANCE_ABS_TOL,
) -> Dict[str, object]:
    """Validate and compare an official-R ``TCIU::LT`` reference grid."""
    path = Path(reference_csv)
    if not np.isfinite(absolute_tolerance) or absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be finite and positive")

    required_columns = {
        "function",
        "parameter",
        "real_x",
        "imag_y",
        "tciu_real",
        "tciu_imag",
        "tciu_version",
    }
    expected_cases = {
        (
            str(case["function"]),
            float(case["parameter"]),
            float(case["real_x"]),
            float(case["imag_y"]),
        )
        for case in conformance_cases()
    }
    rows: list[dict[str, object]] = []
    observed_cases: set[tuple[str, float, float, float]] = set()
    versions: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            raise ValueError(
                "TCIU reference is missing columns: " + ", ".join(sorted(missing_columns))
            )
        for row_number, raw in enumerate(reader, start=2):
            name = raw["function"]
            parameter = float(raw["parameter"])
            z = complex(float(raw["real_x"]), float(raw["imag_y"]))
            official = complex(float(raw["tciu_real"]), float(raw["tciu_imag"]))
            version = raw["tciu_version"].strip()
            case = (name, parameter, z.real, z.imag)
            numeric_values = (parameter, z.real, z.imag, official.real, official.imag)
            if not all(np.isfinite(value) for value in numeric_values):
                raise ValueError(f"TCIU reference row {row_number} contains non-finite values")
            if not version:
                raise ValueError(f"TCIU reference row {row_number} has no package version")
            if case not in expected_cases:
                raise ValueError(f"TCIU reference row {row_number} is not an expected case: {case}")
            if case in observed_cases:
                raise ValueError(f"TCIU reference contains a duplicate case: {case}")
            observed_cases.add(case)
            versions.add(version)
            numeric = numerical_laplace_transform(
                {
                    "exp_decay": lambda t: float(np.exp(-parameter * t)),
                    "sine": lambda t: float(np.sin(parameter * t)),
                    "linear": lambda t: float(t),
                }[name],
                z,
            )
            rows.append(
                {
                    "function": name,
                    "parameter": parameter,
                    "real_x": z.real,
                    "imag_y": z.imag,
                    "absolute_error": abs(numeric - official),
                }
            )
    missing_cases = expected_cases.difference(observed_cases)
    if missing_cases:
        raise ValueError(f"TCIU reference is incomplete: {len(missing_cases)} cases missing")
    if len(versions) != 1:
        raise ValueError("TCIU reference must contain exactly one package version")

    errors = np.asarray([float(row["absolute_error"]) for row in rows])
    max_error = float(np.max(errors))
    return {
        "reference_source": "official_TCIU_R_LT",
        "reference_csv": str(path),
        "cases": len(rows),
        "tciu_version": next(iter(versions)),
        "absolute_tolerance": float(absolute_tolerance),
        "max_absolute_error": max_error,
        "mean_absolute_error": float(np.mean(errors)),
        "passed_at_tolerance": bool(max_error <= absolute_tolerance),
        "rows": rows,
    }
