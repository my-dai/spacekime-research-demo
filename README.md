# SpaceKime Research Demo

This repository is a compact, synthetic-data portfolio snapshot of work I completed for a University of Michigan SOCR research project. It explores identifiability, sampling design, reconstruction, uncertainty, and evaluation for time-phase models inspired by SpaceKime and Kime-Phase Tomography.

The public demo focuses on the parts that can be reproduced without private research data:

- finite Fourier models for latent phase distributions;
- synthetic measurement and sampling operators;
- rank, null-space, conditioning, and noise-sensitivity analysis;
- phase-reparameterization counterexamples that expose non-identifiability;
- recovery when calibration assumptions are available;
- observable-law comparisons using Wasserstein distance and MMD;
- JSON and figure outputs for repeatable evaluation.

The repository does **not** contain participant data, private fMRI files, or claim to reproduce an official SpaceKime optimizer. It is a focused engineering and numerical-research demonstration.

## Run the showcase

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python run_research_showcase.py --output-dir output/research_showcase
```

The command writes a machine-readable summary and, when Matplotlib is available, a figure comparing the synthetic measurement and recovery cases.

## Test

```bash
python -m pytest -q
```

## Repository map

- `run_research_showcase.py` — end-to-end synthetic experiment and report.
- `spacekime_identifiability.py` — measurement operators, diagnostics, counterexamples, and calibrated recovery.
- `spacekime_quotient.py` — invariant observable summaries and distribution comparisons.
- `spacekime_paths.py` — portable output-path helpers.
- `tests/` — synthetic regression tests.

## Skills demonstrated

Python, NumPy, SciPy, numerical simulation, sampling design, inverse-problem diagnostics, statistical distance metrics, reproducible experiment reporting, and technical visualization.

