# Value-Guided Koopman Symmetry (VGKS)

## Requirements

```console
conda create -n vgks python=3.8
conda activate vgks
pip install -r requirements-gpu-cu121.txt
```

## Datasets

Datasets are stored in the `data` directory. Run the following command to download the datasets and save them in TGCVG-style trajectory format:

```console
python -m vgks.download_d4rl_dataset --env-name halfcheetah-medium-v2 --output-dir data
```

The downloader saves:

```text
data/
  halfcheetah-medium-v2.pkl
  halfcheetah-medium-v2.npy
  halfcheetah-medium-v2.json
  halfcheetah-medium-v2/
    dataset.pkl
    meta.json
```

## Example

The VGKS training pipeline consists of two main stages:

#### 1. Generate Augmented Trajectory-Level Data

Edit `configs/vgks/config.yaml`, then run:

```console
python generate_vgks.py --config configs/vgks/config.yaml
```

This stage saves augmented trajectory data in:

```text
data/aug/<env_name>/<env_name>.pkl
data/aug/<env_name>/<env_name>.npy
data/aug/<env_name>/<env_name>.npz
```

#### 2. Train Offline RL Algorithm

Choose one algorithm config and run:

```console
python train_td3bc.py --config configs/offline_rl/td3bc.yaml
python train_iql.py --config configs/offline_rl/iql.yaml
python train_cql.py --config configs/offline_rl/cql.yaml
```

Each training script reads the generated augmented dataset, trains the selected offline RL backbone, and writes `eval.json` with the normalized D4RL score.

## Run Script

```console
bash run.sh
```
