"""Load 4D complex fMRI volumes from MATLAB .mat files."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import scipy.io as sio
except ImportError:  # pragma: no cover - exercised only in minimal environments
    sio = None


def _require_scipy() -> None:
    if sio is None:
        raise ImportError(
            "scipy is required to read MATLAB .mat files. Install project "
            "dependencies with: python -m pip install -r requirements.txt"
        )


def inspect_mat_keys(path: str | Path) -> List[str]:
    """Return user-visible variable names inside a .mat file without loading arrays."""
    return [item["name"] for item in inspect_mat_variables(path)]


def inspect_mat_variables(path: str | Path) -> List[Dict[str, Any]]:
    """Return MAT variable metadata without loading large array payloads."""
    _require_scipy()
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MAT file not found: {path}")

    variables = []
    for name, shape, matlab_class in sio.whosmat(path):
        if name.startswith("__"):
            continue
        variables.append(
            {
                "name": name,
                "shape": tuple(int(x) for x in shape),
                "matlab_class": matlab_class,
            }
        )
    return sorted(variables, key=lambda item: item["name"])


def load_bigim(path: str | Path) -> np.ndarray:
    """
    Load a .mat file containing a 4D complex array named ``bigim``.

    Expected shape: (X, Y, Z, T).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MAT file not found: {path}")

    _require_scipy()
    # The source file also contains a ~451 MiB ``bigk`` array.  Restricting the
    # read to ``bigim`` avoids loading both arrays and sharply lowers peak RAM.
    mat = sio.loadmat(path, variable_names=["bigim"])
    if "bigim" not in mat:
        keys = inspect_mat_keys(path)
        raise KeyError(f"'bigim' not found in {path}. Keys: {keys}")

    bigim = np.asarray(mat["bigim"])
    if bigim.ndim != 4:
        raise ValueError(f"bigim must be 4D (X,Y,Z,T), got shape {bigim.shape}")

    if not np.iscomplexobj(bigim):
        warnings.warn("bigim is not complex; casting to complex64 may be incorrect.")
        bigim = bigim.astype(np.complex64)
    else:
        bigim = bigim.astype(np.complex64, copy=False)

    return np.ascontiguousarray(bigim)


def bigim_metadata(bigim: np.ndarray) -> Dict[str, Any]:
    """Scalar summary of a loaded volume (no voxel data exposed)."""
    mag = np.abs(bigim)
    return {
        "shape": tuple(int(x) for x in bigim.shape),
        "dtype": str(bigim.dtype),
        "is_complex": bool(np.iscomplexobj(bigim)),
        "mag_min": float(mag.min()),
        "mag_max": float(mag.max()),
        "mag_mean": float(mag.mean()),
        "n_voxels": int(bigim.size),
    }
