# Action-Consistency Guidance Design

## Goal

Reduce the mismatch between high-advantage generated samples and the behavior distribution by adding explicit action-consistency constraints and action-deviation filtering to VGKS.

## Problem Summary

Current VGKS training produces augmented samples with strong conservative-value scores, but downstream BC, TD3BC, and IQL results show that these samples do not preserve behavior well enough. The current trainer constrains latent commutation and state proximity, but it does not directly constrain the decoded action relative to the original offline action. This leaves a gap: generated transitions can look high-value to the critic while still being poor supervision for behavior-sensitive downstream learners.

## Approach Options

### Option 1: Generation-only filtering

Add `max_action_deviation` filtering during `generate_vgks.py` and keep training unchanged.

Pros:
- Lowest engineering risk
- Easy ablation

Cons:
- Does not improve the sigma optimization objective itself
- Can only discard bad samples after they are already generated

### Option 2: Training loss + generation filtering

Add an action-consistency loss during sigma training and also expose action-deviation filtering during generation.

Pros:
- Aligns training and generation behavior
- Directly targets the observed failure mode
- Preserves current project structure

Cons:
- Adds another tradeoff knob to tune

### Option 3: Replace inverse-dynamics supervision with behavior-policy actions

Generate actions from a separate behavior policy and constrain sigma against that policy instead of the inverse model.

Pros:
- Closer to BC-style supervision

Cons:
- Larger architectural change
- Harder to attribute improvements cleanly

## Recommendation

Use Option 2.

It is the most defensible next iteration: it adds an explicit behavior-preserving term to sigma training and a matching filter at generation time, without rewriting the project around a new actor. This directly addresses the current empirical failure mode while keeping the method coherent.

## Design

### Trainer changes

In `vgks/trainer.py`, add:

- `lambda_action_anchor`: weight for an action consistency term
- `max_action_deviation`: optional filtering threshold

During sigma loss computation:

- If batch actions are available, compare `augmented_actions` against raw batch actions
- Add `action_anchor_loss = mse(augmented_actions, raw_actions)`
- Include it in `total_loss`
- Log `action_anchor_loss` and `mean_action_deviation`

### Generation changes

In `augment_batch()` and `generate_vgks.py`, add:

- `max_action_deviation`

Generated samples are kept only if they satisfy all active filters:

- `q_threshold`
- `q_delta`
- `max_state_shift`
- `max_action_deviation`

### Config changes

In `configs/vgks.base.yaml`, add:

- `lambda_action_anchor`
- `max_action_deviation`

Defaults should be conservative so the new constraint is enabled but not dominant.

### Testing

Add tests that verify:

- sigma metrics include `action_anchor_loss` and `mean_action_deviation`
- `augment_batch()` can filter by action deviation
- config defaults include the new keys

## Success Criteria

- All existing tests still pass
- New tests for action consistency pass
- `generate_vgks.py` reports filtered samples with action deviation statistics
- The code path is ready for new downstream evaluation without manual patching
