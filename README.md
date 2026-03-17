# VGKS Run Commands

## Install

```bash
pip install -r requirements-gpu-cu121.txt
wandb login
```

## Train

1. Edit [configs/vgks.yaml](/H:/codex_test/nips2026/configs/vgks.yaml)
2. Set:

```yaml
base_config: your-preset.yaml
```

3. Run:

```bash
python train_vgks.py
```

## Download D4RL Datasets

### Mujoco

```bash
python -m vgks.download_d4rl_dataset --env-name halfcheetah-medium-v2 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name halfcheetah-medium-replay-v2 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name halfcheetah-medium-expert-v2 --output-dir data/d4rl

python -m vgks.download_d4rl_dataset --env-name hopper-medium-v2 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name hopper-medium-replay-v2 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name hopper-medium-expert-v2 --output-dir data/d4rl

python -m vgks.download_d4rl_dataset --env-name walker2d-medium-v2 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name walker2d-medium-replay-v2 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name walker2d-medium-expert-v2 --output-dir data/d4rl
```

### Maze2D

```bash
python -m vgks.download_d4rl_dataset --env-name maze2d-umaze-v1 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name maze2d-medium-v1 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name maze2d-large-v1 --output-dir data/d4rl
```

### AntMaze

```bash
python -m vgks.download_d4rl_dataset --env-name antmaze-umaze-diverse-v2 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name antmaze-umaze-play-v2 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name antmaze-medium-diverse-v2 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name antmaze-medium-play-v2 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name antmaze-large-diverse-v2 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name antmaze-large-play-v2 --output-dir data/d4rl
```

### Adroit

```bash
python -m vgks.download_d4rl_dataset --env-name pen-human-v1 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name hammer-human-v1 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name door-human-v1 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name relocate-human-v1 --output-dir data/d4rl
```

### Kitchen

```bash
python -m vgks.download_d4rl_dataset --env-name kitchen-complete-v0 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name kitchen-partial-v0 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name kitchen-mixed-v0 --output-dir data/d4rl
python -m vgks.download_d4rl_dataset --env-name kitchen-undirected-v0 --output-dir data/d4rl
```

## Preset Names For `configs/vgks.yaml`

### Mujoco

```yaml
base_config: vgks.halfcheetah-medium.yaml
base_config: vgks.halfcheetah-medium-replay.yaml
base_config: vgks.halfcheetah-medium-expert.yaml
base_config: vgks.hopper-medium.yaml
base_config: vgks.hopper-medium-replay.yaml
base_config: vgks.hopper-medium-expert.yaml
base_config: vgks.walker2d-medium.yaml
base_config: vgks.walker2d-medium-replay.yaml
base_config: vgks.walker2d-medium-expert.yaml
```

### Maze2D

```yaml
base_config: vgks.maze2d-umaze.yaml
base_config: vgks.maze2d-medium.yaml
base_config: vgks.maze2d-large.yaml
```

### AntMaze

```yaml
base_config: vgks.antmaze-umaze-diverse.yaml
base_config: vgks.antmaze-umaze-play.yaml
base_config: vgks.antmaze-medium-diverse.yaml
base_config: vgks.antmaze-medium-play.yaml
base_config: vgks.antmaze-large-diverse.yaml
base_config: vgks.antmaze-large-play.yaml
```

### Adroit

```yaml
base_config: vgks.pen-human.yaml
base_config: vgks.hammer-human.yaml
base_config: vgks.door-human.yaml
base_config: vgks.relocate-human.yaml
```

### Kitchen

```yaml
base_config: vgks.kitchen-complete.yaml
base_config: vgks.kitchen-partial.yaml
base_config: vgks.kitchen-mixed.yaml
base_config: vgks.kitchen-undirected.yaml
```
