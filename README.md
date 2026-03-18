# Value-Guided Koopman Symmetry (VGKS)

VGKS is a two-stage offline RL pipeline that generates Koopman-based augmented transitions and then trains a downstream offline RL backbone such as TD3+BC, IQL, or CQL on raw or mixed data.

## Setup

```console
conda create -n vgks python=3.8
conda activate vgks
pip install -r requirements-gpu-cu121.txt
pip install -r requirements-d4rl.txt
```

Check the environment stack before long runs:

```console
python check_env.py --env-name halfcheetah-medium-v2
```

## Data Layout

Download a D4RL dataset into the local `data/` cache:

```console
python -m vgks.download_d4rl_dataset --env-name halfcheetah-medium-v2 --output-dir data
```

Expected files:

```text
data/
  halfcheetah-medium-v2.pkl
  halfcheetah-medium-v2.npy
  halfcheetah-medium-v2.json
  halfcheetah-medium-v2/
    dataset.pkl
    meta.json
```

## Stage 1: Generate VGKS Augmented Data

Edit `configs/vgks/config.yaml` or pass overrides on the command line, then run:

```console
python generate_vgks.py --config configs/vgks/config.yaml
```

Generated files are written to:

```text
data/aug/<env_name>/<env_name>.pkl
data/aug/<env_name>/<env_name>.npy
data/aug/<env_name>/<env_name>.npz
```

Important notes:

- `kats_checkpoint` should include both Koopman dynamics weights and the trained `inverse_model` weights. VGKS now restores both when available.
- `critic_checkpoint` should point to a trained conservative critic before enabling value guidance.
- The generated augmented dataset intentionally stores synthetic `observations`, `actions`, `next_observations`, and `q_values`. It does not copy raw `rewards` or `terminals` onto synthetic transitions.

## Stage 2: Train Offline RL Backbones

For pure augmented-data training, point `dataset_path` at the generated dataset in the chosen config:

```console
python train_td3bc.py --config configs/offline_rl/td3bc.yaml
python train_iql.py --config configs/offline_rl/iql.yaml
python train_cql.py --config configs/offline_rl/cql.yaml
```

For safer diagnosis and ablations with TD3+BC, prefer mixing raw and augmented data:

```console
python train_td3bc.py ^
  --config configs/offline_rl/td3bc.yaml ^
  --raw-dataset-path data/halfcheetah-medium-v2.pkl ^
  --aug-dataset-path data/aug/halfcheetah-medium-v2/halfcheetah-medium-v2.npz ^
  --mix-aug-ratio 0.1 ^
  --save-dir runs/td3bc/halfcheetah-medium-v2/ratio_0p1
```

Each training script writes `eval.json` with the normalized D4RL score and a checkpoint under the configured `save_dir`.

## Recommended Debugging Workflow

When augmented data hurts performance:

1. Run `raw_only` with `--raw-dataset-path` and `--mix-aug-ratio 0.0`.
2. Add a small amount of augmented data such as `--mix-aug-ratio 0.05` or `0.1`.
3. Verify that the stage-1 run used a real `kats_checkpoint` and `critic_checkpoint`.
4. Compare `critic_loss`, `q_mean`, and final `normalized_score` across runs.

## Convenience Script

```console
bash run.sh
```
