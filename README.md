# Value-Guided Koopman Symmetry (VGKS)

VGKS is an offline RL data augmentation pipeline built around three stages:

1. prepare Koopman dynamics and value guidance checkpoints
2. generate augmented transitions with `VGKS`
3. validate the generated data with downstream offline RL or IL algorithms

This repository supports both a full "paper-style" pipeline and a lighter shortcut pipeline that skips pretrained checkpoints.

## Setup

```console
conda create -n vgks python=3.8
conda activate vgks
pip install -r requirements-gpu-cu121.txt
pip install -r requirements-d4rl.txt
```

Optional environment check:

```console
python check_env.py --env-name halfcheetah-medium-v2
```

## Data Layout

You can either point scripts directly at a D4RL-style cached dataset path, or use `env_name` and let the code read from D4RL.

Typical cached layout:

```text
data/
  d4rl/
    halfcheetah-medium-v2
    hopper-medium-expert-v2
    maze2d-medium-v1
    pen-human-v1
```

The loaders accept:

- directory with `dataset.pkl`
- `.pkl`
- `.npz`
- `.pt`

## Pipeline Overview

There are two ways to use this repo.

### Full Pipeline

Use this when you want the method to match the intended idea most closely.

1. Run `train_vgks.py` to train Koopman dynamics, inverse dynamics, a conservative critic, and the value-guided sigma model.
2. Reuse the produced `kats_checkpoint.pt` and `critic_checkpoint.pt`.
3. Run `generate_vgks.py` with those checkpoints to synthesize augmented transitions.
4. Train `TD3BC`, `IQL`, `CQL`, or `BC` on raw-only or raw-plus-augmented data.

In this mode:

- `kats_checkpoint` provides pretrained Koopman dynamics and inverse dynamics
- `critic_checkpoint` provides the conservative value guidance used during augmentation

### Shortcut Pipeline

Use this when you want to quickly test whether the current `VGKS` generator can produce useful data.

1. Run `generate_vgks.py` without checkpoints.
2. Train `TD3BC`, `IQL`, `CQL`, or `BC` on the generated data.

In this mode:

- the code still runs
- but the generator is not using pretrained `KATS` or pretrained value guidance
- this is a weaker version of the full method

This shortcut is the path you used for the earlier locomotion experiments.

## Stage 1: Prepare Checkpoints

### 1A. Unified Upstream Training

Use [train_vgks.py](/H:/codex_test/nips2026/vgks/train_vgks.py) to train the full upstream stack on the target dataset.

Important:

- for real experiments, always pass an environment-specific config with `--config`
- do not rely on `--dataset-path` alone
- `dataset_path` only tells the script where the data lives
- the config controls the training hyperparameters such as `epochs`, `batch_size`, `sigma_lr`, `lambda_q`, and output naming
- if you run the script without the matching config, you can easily end up using debug-like settings such as `epochs=1`

Example:

```console
python vgks/train_vgks.py ^
  --config configs/vgks.halfcheetah-medium-expert.yaml ^
  --dataset-path data/d4rl/halfcheetah-medium-expert-v2 ^
  --save-dir runs/kats/halfcheetah-medium-expert-v2/seed_0 ^
  --device cuda:0
```

What you need from this stage:

- `kats_checkpoint.pt`
- `critic_checkpoint.pt`
- `vgks_checkpoint.pt`
- `best_checkpoint.pt`, `last_checkpoint.pt`, `checkpoint.pt`

The first two checkpoints are later passed to `generate_vgks.py` as `kats_checkpoint` and `critic_checkpoint`.

## Stage 2: Generate Augmented Data

There are two related scripts here.

### `generate_vgks.py`

This is the lightweight augmentation entrypoint.

It:

- builds a `VGKS` trainer
- optionally loads `kats_checkpoint` and `critic_checkpoint`
- trains the sigma transformation
- exports augmented transitions

Example:

```console
python generate_vgks.py --config configs/vgks.halfcheetah-medium-expert.yaml
```

Generated outputs typically include:

```text
runs/vgks/<env>/seed_0/
  <run_name>.npz
  <run_name>.pkl
  <run_name>.npy
  eval.json
```

### `train_vgks.py`

This is the fuller training-style `VGKS` script.

It does more than just export data:

- trains sigma across epochs
- trains an auxiliary BC policy on raw plus augmented data
- evaluates during training
- saves `best_checkpoint.pt`, `last_checkpoint.pt`, `checkpoint.pt`
- exports `augmented_dataset.npz`

Use this when you want:

- a more complete `VGKS` training record
- `VGKS` checkpoints
- epoch-by-epoch evaluation history

Example:

```console
python vgks/train_vgks.py ^
  --config configs/vgks.halfcheetah-medium-expert.yaml ^
  --dataset-path data/d4rl/halfcheetah-medium-expert-v2 ^
  --env-name halfcheetah-medium-expert-v2 ^
  --save-dir runs/vgks_train/halfcheetah-medium-expert-v2/seed_0
```

### Which one should I use?

- use `generate_vgks.py` if your immediate goal is to create augmented data for `TD3BC/IQL/CQL/BC`
- use `train_vgks.py` if you also want `VGKS` checkpoints and the full training history

## Stage 3: Validate with Offline RL or IL

After augmentation, use the generated dataset with downstream algorithms.

### TD3+BC

```console
python train_td3bc.py ^
  --config configs/offline_rl/td3bc.yaml ^
  --raw-dataset-path data/d4rl/halfcheetah-medium-expert-v2 ^
  --aug-dataset-path runs/vgks/halfcheetah-medium-expert-v2/seed_0/halfcheetah-medium-expert-seed0.npz ^
  --mix-aug-ratio 0.1 ^
  --save-dir runs/td3bc/halfcheetah-medium-expert-v2/ratio_0p1
```

Mixed-mode `TD3BC` uses:

- raw data for critic updates
- mixed raw plus augmented data for actor and BC updates

### IQL

```console
python train_iql.py ^
  --config configs/offline_rl/iql.yaml ^
  --raw-dataset-path data/d4rl/halfcheetah-medium-expert-v2 ^
  --aug-dataset-path runs/vgks/halfcheetah-medium-expert-v2/seed_0/halfcheetah-medium-expert-seed0.npz ^
  --mix-aug-ratio 0.1 ^
  --save-dir runs/iql/halfcheetah-medium-expert-v2/ratio_0p1
```

`IQL` now supports the same raw-plus-augmented mixing interface as `TD3BC`.

### CQL

```console
python train_cql.py --config configs/offline_rl/cql.yaml
```

`CQL` supports the same raw-plus-augmented mixing interface as `TD3BC/IQL`.

### BC

```console
python vgks/train_bc.py ^
  --raw-dataset-path data/d4rl/halfcheetah-medium-expert-v2 ^
  --aug-dataset-path runs/vgks/halfcheetah-medium-expert-v2/seed_0/halfcheetah-medium-expert-seed0.npz ^
  --mix-aug-ratio 0.1 ^
  --env-name halfcheetah-medium-expert-v2 ^
  --save-dir runs/bc/halfcheetah-medium-expert-v2/ratio_0p1
```

### CORL-style BC

```console
python vgks/train_corl_bc.py ^
  --raw-dataset-path data/d4rl/halfcheetah-medium-expert-v2 ^
  --aug-dataset-path runs/vgks/halfcheetah-medium-expert-v2/seed_0/halfcheetah-medium-expert-seed0.npz ^
  --mix-aug-ratio 0.1 ^
  --env-name halfcheetah-medium-expert-v2 ^
  --device cuda ^
  --max-timesteps 1000000 ^
  --eval-freq 5000 ^
  --log-every 1000 ^
  --save-dir runs/corl_bc/halfcheetah-medium-expert-v2/ratio_0p1
```

## Configuration Notes

### Default config vs environment-specific configs

[configs/vgks/config.yaml](/H:/codex_test/nips2026/configs/vgks/config.yaml) is only a default example.

When training checkpoints, the safest pattern is:

1. choose the config that matches the dataset
2. pass it explicitly with `--config`
3. optionally override `dataset_path` or `save_dir` on the command line

For example, if you are training on `halfcheetah-medium-expert-v2`, prefer:

```console
python vgks/train_vgks.py ^
  --config configs/vgks.halfcheetah-medium-expert.yaml ^
  --dataset-path data/d4rl/halfcheetah-medium-expert-v2 ^
  --env-name halfcheetah-medium-expert-v2 ^
  --save-dir runs/vgks_train/halfcheetah-medium-expert-v2/seed_0
```

Do not assume that changing only `dataset_path` is enough. In this repository:

- the dataset-specific config sets the intended hyperparameters for that family of tasks
- `configs/vgks.mujoco.base.yaml` is the shared base for MuJoCo tasks
- `configs/vgks.base.yaml` is the lowest-level global default
- `configs/vgks.yaml` is only the default entry config and should not be your main way to switch experiments

In practice:

- edit the dataset-specific config first if you are changing one experiment
- edit the family base such as `vgks.mujoco.base.yaml` only if you want the same change to affect all tasks in that family
- avoid editing `vgks.base.yaml` unless you want to change the global default for every domain
- MuJoCo presets now also carry `env_name`, so using the matching dataset config avoids silently falling back to the toy evaluator

For the current stability defaults, the repository now uses:

- `sigma_tau: 0.01` to avoid exponential weight blow-up in sigma training
- `sigma_warmup_steps: 1000` so value guidance does not dominate too early
- `lambda_q: 0.02` for a more conservative start on MuJoCo datasets
- a CQL-style critic stage with target critics, behavior-policy backups, and conservative logsumexp regularization

When checking critic health during `train_vgks.py`, the most useful metrics are:

- `critic/target_q_mean`
- `critic/data_q_mean`
- `critic/ood_q_mean`
- `critic/cql_gap`

If `critic/data_q_mean` immediately collapses to very large negative values and `sigma/mean_conservative_q` stays pinned at the clip floor, the value guidance is still not healthy.

For actual experiments, prefer the environment-specific configs already provided, such as:

- [configs/vgks.halfcheetah-medium-expert.yaml](/H:/codex_test/nips2026/configs/vgks.halfcheetah-medium-expert.yaml)
- [configs/vgks.maze2d-medium.yaml](/H:/codex_test/nips2026/configs/vgks.maze2d-medium.yaml)
- [configs/vgks.pen-human.yaml](/H:/codex_test/nips2026/configs/vgks.pen-human.yaml)

These configs inherit from base files such as:

- [configs/vgks.mujoco.base.yaml](/H:/codex_test/nips2026/configs/vgks.mujoco.base.yaml)
- [configs/vgks.maze2d.base.yaml](/H:/codex_test/nips2026/configs/vgks.maze2d.base.yaml)
- [configs/vgks.adroit.base.yaml](/H:/codex_test/nips2026/configs/vgks.adroit.base.yaml)

### Paper Baseline Configs

If you are validating augmentation with the stronger reference-style `TD3BC` and `IQL` scripts, use:

- [configs/offline_rl/td3bc_base.yaml](/H:/codex_test/nips2026/configs/offline_rl/td3bc_base.yaml)
- [configs/offline_rl/iql_base.yaml](/H:/codex_test/nips2026/configs/offline_rl/iql_base.yaml)

These are intended for baseline verification with:

- `raw_dataset_path`
- `aug_dataset_path`
- `mix_aug_ratio`
- `aug_take_topq`

The expectation is that the script itself preserves its original algorithm logic, while the dataset path layer is switched to your raw-plus-augmented trajectory pipeline.

### `kats_checkpoint` and `critic_checkpoint`

These are optional in code, but important in the full method.

- if left empty, `VGKS` will still run
- but it will not be using pretrained Koopman dynamics or pretrained value guidance

So:

- empty checkpoint fields are acceptable for quick debugging
- filled checkpoint fields are preferred for serious experiments

### Generated data contents

The generated augmented dataset intentionally stores:

- `observations`
- `actions`
- `next_observations`
- optional `q_values`

It does not copy raw `rewards` and `terminals` onto synthetic transitions.

## Recommended Experiment Order

If you are starting a new environment, the most reliable order is:

1. verify the raw dataset path works
2. train or prepare `kats_checkpoint`
3. train or prepare `critic_checkpoint`
4. run `generate_vgks.py`
5. run `TD3BC` on `raw_only`
6. run `TD3BC` on `ratio=0.1`
7. run `TD3BC` on `ratio=0.2`
8. if promising, repeat with `IQL`

For quick diagnostics, a shortcut order is:

1. run `generate_vgks.py` without checkpoints
2. run `TD3BC raw_only`
3. run `TD3BC ratio=0.1`
4. run `TD3BC ratio=0.2`

## Common Confusions

### "Why did `generate_vgks.py` run even though checkpoints were empty?"

Because checkpoint loading is optional. Empty checkpoint fields do not stop the script.

### "Do `train_td3bc.py` or `train_iql.py` produce `kats_checkpoint` and `critic_checkpoint`?"

No.

- `kats_checkpoint` should come from `KATS` training
- `critic_checkpoint` should come from a critic trained before augmentation
- `TD3BC/IQL/CQL/BC` are downstream evaluation stages

### "Do I need `train_vgks.py` to use VGKS?"

Not always.

- no, if you only want augmented data and will use `generate_vgks.py`
- yes, if you want the fuller `VGKS` training workflow and checkpoints

## Convenience Script

```console
bash run.sh
```
