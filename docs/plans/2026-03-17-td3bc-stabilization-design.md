# TD3BC Stabilization Design

## Goal

Replace the current lightweight second-stage TD3BC trainer with a more stable implementation that is closer to the TGCVG/CORL-style reference and suitable for evaluating VGKS-generated datasets.

## Root Cause Summary

The current TD3BC stage is diverging because it omits several stability-critical pieces from the standard algorithm:

- no state normalization
- no target policy smoothing noise
- no delayed actor updates
- no `alpha / mean(|Q|)` scaling for the BC-vs-Q actor objective
- no explicit action range handling

This is why the run produced exploding critic losses and meaningless normalized scores.

## Chosen Approach

Implement a stabilized TD3BC training path inside the existing project structure rather than importing the entire external CORL stack.

This preserves the current two-stage VGKS workflow while fixing the main source of instability. The stabilized trainer will:

- normalize observations using dataset mean/std
- keep a `max_action` parameter and clamp actions to valid range
- use twin critics plus target critics
- add target policy noise and noise clipping
- update the actor with TD3 delay
- use `lambda = alpha / mean(abs(Q))` in the actor loss
- preserve current console progress, JSON logging, and evaluation output

## Scope

- Update `vgks/offline_rl.py` with stable TD3BC components.
- Update `vgks/train_td3bc.py` config surface as needed.
- Update `configs/offline_rl/td3bc.yaml` with algorithm parameters closer to the reference implementation.
- Add tests for the stabilized update path.

## Non-Goals

- Do not rewrite IQL or CQL in this change.
- Do not import the full external CORL framework.
- Do not change the VGKS generation stage.
