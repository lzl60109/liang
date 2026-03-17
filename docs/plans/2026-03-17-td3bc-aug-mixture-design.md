# TD3BC Augmented Data Mixture Design

## Goal

Add a simple dataset-source control mechanism to the second-stage TD3BC trainer so we can separate:

- trainer stability on raw D4RL data
- augmentation benefit or harm from VGKS-generated data

## Chosen Interface

Expose three dataset arguments for TD3BC:

- `raw_dataset_path`
- `aug_dataset_path`
- `mix_aug_ratio`

Behavior:

- if only `dataset_path` is provided, keep current behavior for backward compatibility
- if `raw_dataset_path` is provided and `mix_aug_ratio == 0`, train on raw data only
- if both `raw_dataset_path` and `aug_dataset_path` are provided with `mix_aug_ratio > 0`, sample a subset of augmented transitions and concatenate them onto the raw data before creating the loader

## Why This Helps

This lets us run the three most important debugging experiments without rewriting the pipeline:

- raw only
- raw + 10% aug
- raw + 20% aug

If raw only is stable but mixed runs are unstable, the main issue is likely augmentation quality rather than the TD3BC trainer itself.

## Scope

- Update `train_td3bc.py` config and CLI parsing.
- Add dataset concatenation helpers in `vgks/offline_rl.py`.
- Add tests for raw-only and raw+aug mixing behavior.
- Update `configs/offline_rl/td3bc.yaml` to expose the new options.
