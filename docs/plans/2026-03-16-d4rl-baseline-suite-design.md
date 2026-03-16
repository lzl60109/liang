# D4RL Baseline Suite Design

**Date:** 2026-03-16

## Goal

Extend the VGKS repository into a paper-ready experiment scaffold that can run four standalone training entrypoints on D4RL tasks:

- BC
- KATS
- TGCVG-style value baseline
- VGKS

Each method must support switching between D4RL tasks from the command line, log metrics to Weights & Biases, save evaluation summaries locally, and export generated augmented datasets as `.npz`.

## Requirements

- Each baseline must have its own `train_*.py` entrypoint.
- Each entrypoint must accept `--env-name`, `--seed`, `--device`, and `--save-dir`.
- Training and evaluation should report D4RL normalized scores using `env.get_normalized_score(return) * 100`.
- Methods that generate synthetic data must save `augmented_dataset.npz`.
- Logging must go to `wandb`, with local JSON backups for evaluation and configuration.
- README must document the end-to-end server workflow, required packages, and example commands.

## Architecture

### Shared experiment utilities

Create shared modules under `vgks/` for:

- environment creation and D4RL dimension discovery,
- offline dataset loading and conversion,
- evaluation with normalized score reporting,
- wandb-backed logging,
- augmented dataset export.

### Baseline entrypoints

Expose four independent entrypoints:

- `vgks/train_bc.py`
- `vgks/train_kats.py`
- `vgks/train_tgcvg.py`
- `vgks/train_vgks.py`

All scripts share a common CLI shape and output layout.

### Output layout

Each run writes to:

`runs/<method>/<env_name>/seed_<seed>/`

The run directory contains:

- `config.json`
- `eval.json`
- `checkpoint.pt`
- `augmented_dataset.npz` when the method produces synthetic data

## Evaluation

For every method:

1. Train the method-specific model or policy.
2. Evaluate for a fixed number of episodes.
3. Save:
   - average raw return
   - normalized D4RL score
   - evaluation episode count

## Simplifying Assumptions

- The first implementation will use compact PyTorch models and the existing checkpoint adapters instead of reproducing every detail of the original KATS and TGCVG repositories.
- The baseline scripts prioritize comparable output format and reproducible evaluation over exact paper reproduction.
- D4RL loading remains optional so the code still supports `.npz` datasets for local smoke testing.

## Testing Strategy

Add tests for:

- shared CLI options and run directory layout,
- normalized score evaluation behavior with a fake D4RL-like environment,
- wandb-safe logger behavior when wandb is unavailable or disabled,
- `.npz` export of augmented datasets,
- all four training entrypoints running one small epoch on toy data.
