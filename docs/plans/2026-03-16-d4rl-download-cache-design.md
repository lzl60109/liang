# D4RL Download Cache Design

**Date:** 2026-03-16

## Goal

Add a dataset download-and-cache flow for VGKS so experiments can first save D4RL trajectories locally in a TGCVG-style format and then reuse the cached files for augmentation and training.

## Requirements

- Provide a standalone script to download and save D4RL trajectories.
- Cache format should follow the user's preference: one `dataset.pkl` plus per-array `.npy` files.
- VGKS should be able to read the cached trajectory folder directly instead of requiring live D4RL access during training.
- README should document both steps:
  1. download/cache trajectories
  2. run VGKS on cached trajectories

## Architecture

### New script

- `vgks/download_d4rl_dataset.py`

This script will:

1. resolve the D4RL environment name from either `--env-name` or `--task + --dataset-name`,
2. call `d4rl.qlearning_dataset(env)`,
3. save:
   - `dataset.pkl`
   - `observations.npy`
   - `actions.npy`
   - `next_observations.npy`
   - optional `rewards.npy`
   - optional `terminals.npy`

### Dataset loading changes

Extend dataset loading so that `load_offline_dataset()` accepts:

- `.npz`
- `.pt`
- cache directories containing `dataset.pkl`

VGKS will then treat cached D4RL directories as first-class inputs.

## Testing

Add tests for:

- saving a cache directory writes both `dataset.pkl` and `.npy` arrays,
- loading a cache directory reconstructs the expected arrays,
- VGKS training accepts a cached directory path as `dataset_path`.
