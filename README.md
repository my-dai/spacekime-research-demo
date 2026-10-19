# SpaceKime Research Workbench

[![Test](https://github.com/my-dai/spacekime-research-demo/actions/workflows/test.yml/badge.svg)](https://github.com/my-dai/spacekime-research-demo/actions/workflows/test.yml)

A reproducible Python research workbench from my University of Michigan SOCR
work on complex fMRI signals, time-phase models, sampling, and reconstruction.
It includes the numerical modules, experiment runners, and tests I used to
study:

- complex-valued fMRI loading and phase-distribution analysis;
- synthetic 4D time-phase surfaces and repeated-measurement simulation;
- uniform and nonuniform sampling experiments;
- density-corrected and joint time-phase reconstruction;
- rank, null-space, conditioning, and non-identifiability analysis;
- calibrated recovery, bootstrap uncertainty, and split-conformal evaluation;
- rotation/reflection-aware comparisons and sparse phase-anchor experiments;
- PSNR, SSIM, Wasserstein distance, MMD, and reproducible JSON/CSV reporting.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python run_research_showcase.py --output-dir output/research_showcase
python -m pytest -q
```

The full test suite currently contains 74 passing tests plus 4 parameterized
subtests.

## Main modules

| Module | Purpose |
| --- | --- |
| `spacekime_core.py` | Complex signal metrics, phase noise, and phase-distribution surfaces |
| `spacekime_synthetic.py` | Known latent surfaces, sampling, reconstruction, PSNR, and SSIM |
| `spacekime_advanced.py` | Joint time-phase estimation, density correction, stitching, and uncertainty |
| `spacekime_analytic.py` | Complex Laplace-domain models and TCIU conformance checks |
| `spacekime_blind.py` | Observation-only latent-phase baselines and symmetry-aware evaluation |
| `spacekime_phase_dynamics.py` | Time-varying phase models and sparse-anchor recovery |
| `spacekime_identifiability.py` | Finite-basis measurement operators, rank diagnostics, and constrained recovery |
| `spacekime_quotient.py` | Gauge-invariant summaries, Wasserstein distance, and permutation MMD |

The `run_*.py` files expose the benchmark and audit workflows. `tests/` covers
the numerical primitives, sampling assumptions, recovery behavior, reporting,
and data-loading boundaries.

## Optional public fMRI data

The real-data path expects a MATLAB file containing a complex `bigim` array.
The code deliberately does not bundle participant data. A compatible public
SOCR data source can be supplied with:

```bash
set SPACEKIME_MAT_FILE=C:\path\to\subj1_run1_complex_all.mat
python run_real_bigim_demo.py
```

The synthetic experiments and complete test suite do not require the real data.

## Interpretation

Observed phase summaries and latent reconstructions are kept separate in the
code and reports. Reconstruction claims require repeated-measurement assumptions
or explicit constraints; the rank and non-identifiability tools make those
conditions visible. This repository is research code and is not the official
SpaceKime or Kime-Phase Tomography implementation.

