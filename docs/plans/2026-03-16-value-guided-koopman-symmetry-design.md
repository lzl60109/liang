# Value-Guided Koopman Symmetry Design

**Date:** 2026-03-16

## Goal

Build a new offline imitation learning augmentation method that keeps KATS's Koopman-space symmetry learning while adding TGCVG-style conservative value guidance during symmetry operator training.

## Problem

KATS learns a symmetry operator by prioritizing regions with larger Koopman prediction error. The generated trajectories can be dynamically plausible, but they are not explicitly encouraged to improve return. TGCVG, on the other hand, provides a conservative Q estimate that can score state-action pairs, but it does not shape KATS's latent symmetry operator.

## Proposed Method

The new method is `Value-Guided Koopman Symmetry (VGKS)`.

The symmetry operator is no longer trained only with a Koopman commutation objective. Instead, it is trained with a joint objective that:

1. preserves Koopman consistency in latent space,
2. increases conservative value after decoding transformed latent states, and
3. keeps transformed states near the original data manifold.

## Architecture

### Modules

- `KoopmanDynamicsModel`
  - Encodes observations into a latent Koopman space.
  - Applies a learned latent Koopman operator.
  - Decodes latent states back to observation space.
- `InverseDynamicsModel`
  - Predicts an action from two consecutive latent states.
- `SigmaModel`
  - Learns a linear latent transformation representing a symmetry-like operator.
- `ConservativeCritic`
  - Uses two Q networks and returns the conservative value `min(Q1, Q2)`.
- `ValueGuidedKoopmanTrainer`
  - Coordinates sigma training, critic loading, and trajectory augmentation.

### Training Stages

1. Train Koopman dynamics model.
2. Train inverse dynamics model.
3. Train or load conservative critic.
4. Freeze dynamics, inverse model, and critic.
5. Train `SigmaModel` with value-aware loss.
6. Generate augmented transitions and train a policy on mixed data.

## Losses

Let:

- `z_t = E(s_t)`
- `z_{t+1} = E(s_{t+1})`
- `hat_z_t = sigma(z_t)`
- `hat_z_{t+1} = sigma(z_{t+1})`
- `hat_s_t = D(hat_z_t)`
- `hat_s_{t+1} = D(hat_z_{t+1})`
- `hat_a_t = g(hat_z_t, hat_z_{t+1})`

The total sigma loss is:

`L_total = L_comm + lambda_q * L_value + lambda_state * L_state_anchor + lambda_latent * L_latent_anchor`

### Koopman commutation loss

`L_comm = E[w_t ||K sigma(z_t) - sigma(z_{t+1})||^2]`

where

`w_t = exp(tau ||z_{t+1} - K z_t||^2)`

### Conservative value loss

`L_value = -E[min(Q1(hat_s_t, hat_a_t), Q2(hat_s_t, hat_a_t))]`

### State anchor loss

`L_state_anchor = E[||hat_s_t - s_t||^2 + ||hat_s_{t+1} - s_{t+1}||^2]`

### Latent anchor loss

`L_latent_anchor = E[||hat_z_t - z_t||^2 + ||hat_z_{t+1} - z_{t+1}||^2]`

## Stability Constraints

- Freeze critic weights during sigma training.
- Use `min(Q1, Q2)` instead of the average Q estimate.
- Clip conservative Q values before they enter the loss.
- Warm up sigma training with commutation loss only before enabling value guidance.
- Keep anchor losses enabled by default.

## Why This Is More Than A+B

This is not a pipeline where KATS generates data and TGCVG scores it afterward.

The conservative critic directly shapes the symmetry operator's optimization objective. The learned operator becomes a value-aware latent transformation that jointly satisfies dynamics constraints and conservative return improvement.

## Expected Outputs

- A reusable fusion codebase rooted in the KATS training flow.
- A value-aware sigma trainer.
- Conservative critic integration with checkpoint loading.
- Augmentation utilities that export Q-scored synthetic transitions.
- Tests covering the new loss and augmentation path.
