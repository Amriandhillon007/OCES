# OCES

OCES is a research codebase for the phased development of an exploratory compression and restructuring framework. The repository preserves the phase history from early mechanism prototypes through Phase 11.5, including the ASTRAEUS unified field experiments, failure-regime mapping, and causal ablation studies.

## Repository Layout

```text
OCES/
  src/phases/          Phase implementations and experiment scripts
  results/figures/     Published figures and generated plots
  results/json/        Selected structured result summaries
  docs/                Notes, theory drafts, and phase documentation
  scripts/             Utility and legacy runner scripts
  tests/               Lightweight verification space
```

## Main Phase 11.5 Scripts

- `src/phases/phase11.5_complete.py`  
  ASTRAEUS unified field framework with multi-dimensional runs, stabilized critical-slowing diagnostics, convergence curves, adaptive PCA, occupancy tracking, and failure-event logging.

- `src/phases/phase11.5b.py`  
  Failure-regime mapping with bounded Gamma normalization, corrected fragmentation logic, adaptive occupancy radius, and time-dependent scar memory.

- `src/phases/phase11.5_C.py`  
  Causal ablation framework with empirical baseline Gamma, adaptive mixing propagation, occupancy warmup exclusion, and accessibility-coupling ablation.

## Quick Start

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run a phase directly:

```bash
python src/phases/phase11.5_complete.py
```

Some runs are intentionally long. The Phase 11.5 canonical configuration uses 50 seeds across dimensions `[8, 16, 32, 64]`; the full run can take days depending on CPU resources.

## Results

Selected figures are in `results/figures/`. Selected JSON summaries are in `results/json/`. Large transient artifacts such as virtual environments, caches, and checkpoints are intentionally excluded from Git.

## Scientific Notes

The current 11.5 line treats "compressed exploratory regime" as an operational regime, not a universal closure claim. Gamma in Phase 11.5C is computed relative to an empirical canonical baseline rather than an artificial zero baseline. Some weighting choices remain operational heuristics and should be discussed as such in publication text.

## Status

This repository is organized as a research archive rather than a packaged library. Filenames preserve the phase numbering used during development.
