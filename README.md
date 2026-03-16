# Value-Guided Koopman Symmetry

This repository contains a runnable prototype of `Value-Guided Koopman Symmetry (VGKS)`, a fusion of:

- `KATS`: Koopman-assisted trajectory synthesis
- `TGCVG`: conservative value guidance

The core idea is to learn a `value-aware symmetry operator` in latent Koopman space. Instead of training the symmetry operator only with a dynamics consistency objective, VGKS adds a conservative Q-guidance term so that generated transitions are both:

- dynamically plausible
- biased toward higher-value behavior

## What Is Implemented

This codebase currently provides a clean research prototype with:

- Koopman-style encoder, decoder, and latent transition model
- latent symmetry operator `sigma`
- inverse dynamics model for recovering actions from transformed latent pairs
- conservative double-critic wrapper using `min(Q1, Q2)`
- checkpoint adapters for KATS-style sysmodel weights and TGCVG-style critic weights
- value-aware sigma loss with:
  - Koopman commutation loss
  - conservative value loss
  - state anchor loss
  - latent anchor loss
- augmentation utilities with optional Q-threshold filtering
- epoch-level sigma training over replay-style batches
- tests that verify the fusion logic

This version is implemented in `PyTorch` and is intended as a clean method scaffold and ablation-friendly reference implementation. It is not yet a full D4RL benchmark runner wired into the original KATS/TGCVG code, but the core value-guided sigma training logic is now differentiable and trainable.

## Repository Layout

- `vgks/models.py`
  Koopman dynamics model, sigma model, inverse dynamics model, and conservative critic.
- `vgks/trainer.py`
  The main VGKS logic: sigma loss computation, sigma optimization, and epoch training.
- `vgks/integration.py`
  Helpers for loading original KATS and TGCVG style checkpoints.
- `vgks/cli.py`
  CLI parser for value-guidance hyperparameters.
- `examples/run_demo.py`
  A minimal script showing how to build the trainer and run one forward pass.
- `tests/`
  Regression tests for the method behavior.
- `docs/plans/`
  Design and implementation plan documents for this fusion method.

## Method

Let:

- `z_t = E(s_t)`
- `z_{t+1} = E(s_{t+1})`
- `hat_z_t = sigma(z_t)`
- `hat_z_{t+1} = sigma(z_{t+1})`
- `hat_s_t = D(hat_z_t)`
- `hat_s_{t+1} = D(hat_z_{t+1})`
- `hat_a_t = g(hat_z_t, hat_z_{t+1})`

The sigma operator is trained with:

`L_total = L_comm + lambda_q * L_value + lambda_state * L_state_anchor + lambda_latent * L_latent_anchor`

Where:

- `L_comm`
  Encourages Koopman commutation consistency.
- `L_value`
  Maximizes the conservative lower-bound value `min(Q1, Q2)`.
- `L_state_anchor`
  Keeps decoded augmented states near the original observations.
- `L_latent_anchor`
  Keeps sigma from drifting too far away from the original latent states.

## Requirements

The current prototype needs:

- Python with `torch`
- `unittest` for running tests, which is included with Python

If your environment does not already have PyTorch:

```bash
pip install torch
```

## How To Run

### 1. Run the demo

From the repository root:

```bash
python examples/run_demo.py
```

Expected output:

- a block of sigma loss metrics
- the shapes of the generated augmented batch

### 2. Run the test suite

```bash
python -m unittest discover -s tests -v
```

This verifies:

- conservative Q uses the lower critic value
- sigma metrics include value and anchor terms
- changing `lambda_q` changes the total loss
- augmentation can be filtered by Q threshold
- the CLI exposes the value-guidance flags

### 3. Use the CLI parser

The parser is exposed from `vgks.cli.build_parser()` and currently supports:

- `--lambda-q`
- `--lambda-state-anchor`
- `--lambda-latent-anchor`
- `--q-clip-min`
- `--q-clip-max`
- `--q-threshold`
- `--sigma-warmup-steps`
- `--kats-checkpoint`
- `--critic-checkpoint`

Example:

```bash
python -c "from vgks.cli import build_parser; print(build_parser().parse_args(['--lambda-q','0.2','--q-threshold','1.5']))"
```

## Minimal Python Example

```python
import torch

from vgks import ConservativeCritic, KoopmanDynamicsModel, SigmaModel, ValueGuidedKoopmanTrainer

dynamics = KoopmanDynamicsModel(state_dim=3, action_dim=2, latent_dim=4, hidden_dim=16)
sigma = SigmaModel(latent_dim=4)
critic = ConservativeCritic(state_dim=3, action_dim=2, hidden_dim=16)

trainer = ValueGuidedKoopmanTrainer(
    dynamics=dynamics,
    sigma_model=sigma,
    critic=critic,
    action_dim=2,
    lambda_q=0.1,
    lambda_state_anchor=1.0,
    lambda_latent_anchor=0.1,
)

batch = {
    "observations": torch.tensor([[0.2, -0.1, 0.3]], dtype=torch.float32),
    "next_observations": torch.tensor([[0.25, -0.05, 0.35]], dtype=torch.float32),
}

metrics = trainer.compute_sigma_loss(batch)
augmented = trainer.augment_batch(batch)

print(metrics)
print(augmented["q_values"])
```

## Loading Original Checkpoints

The repo now includes adapters for the two source projects:

```python
from vgks import (
    ConservativeCritic,
    KoopmanDynamicsModel,
    load_kats_checkpoint,
    load_tgcvg_critic_checkpoint,
)

dynamics = KoopmanDynamicsModel(state_dim=17, action_dim=6, latent_dim=32, hidden_dim=512)
critic = ConservativeCritic(state_dim=17, action_dim=6, hidden_dim=256)

load_kats_checkpoint(dynamics, "path/to/kats_sysmodel.pth")
load_tgcvg_critic_checkpoint(critic, "path/to/tgcvg_checkpoint.pt")
```

Expected formats:

- KATS checkpoint:
  contains `layer1`, `layer2`, `layer3`, `layerK`, `layer3inv`, `layer2inv`, `layer1inv`
- TGCVG checkpoint:
  contains `critic1` and `critic2`, or `critic_1` and `critic_2`

## Running Epoch-Level Sigma Training

You can now optimize `sigma` over a stream of replay-style batches:

```python
batches = [
    {
        "observations": torch.randn(256, 17),
        "next_observations": torch.randn(256, 17),
    }
    for _ in range(100)
]

epoch_metrics = trainer.train_sigma_epoch(batches)
print(epoch_metrics)
```

The returned dictionary contains averaged:

- `total_loss`
- `commutation_loss`
- `value_loss`
- `state_anchor_loss`
- `latent_anchor_loss`
- `mean_conservative_q`
- `step_count`

## Current Limitations

- This is not yet the original full KATS or TGCVG training code.
- The current implementation is a prototype method scaffold, not a full D4RL benchmark runner.
- The current implementation uses simplified wrappers with KATS/TGCVG-compatible parameter names instead of directly importing the original training stacks.
- Replay loading, D4RL environment setup, and full end-to-end experiment scripts are not yet wired in.

## Recommended Next Step

If you want to continue, the next upgrade should be:

1. Reuse the original KATS encoder/decoder and inverse model
2. Load TGCVG CQL critic checkpoints directly
3. Replace the prototype replay flow with real D4RL batches
4. Train `sigma` with the value-aware objective on real offline RL datasets
