# Action Consistency Guidance Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add behavior-preserving action-consistency guidance to VGKS training and generation so augmented samples stay closer to the original action manifold.

**Architecture:** Extend `ValueGuidedKoopmanTrainer` with an action anchor term and action-deviation metrics, then expose matching generation filters through the CLI and config. Keep the current critic and sigma structure unchanged; only add behavior-preserving constraints around generated actions.

**Tech Stack:** Python, PyTorch, YAML configs, unittest

---

### Task 1: Add failing trainer tests

**Files:**
- Modify: `tests/test_vgks_trainer.py`

**Step 1: Write failing tests**

Add tests asserting:
- `compute_sigma_loss_tensors()` returns `action_anchor_loss` and `mean_action_deviation`
- `augment_batch(..., max_action_deviation=...)` can filter all samples

**Step 2: Run targeted test**

Run: `python -m unittest tests.test_vgks_trainer`

Expected: FAIL because the new metrics and filter do not exist yet.

**Step 3: Commit**

Commit after implementation, not after the red step.

### Task 2: Add trainer implementation

**Files:**
- Modify: `vgks/trainer.py`

**Step 1: Add config fields**

Extend `ValueGuidedKoopmanTrainer` with:
- `lambda_action_anchor`
- `max_action_deviation`

**Step 2: Add action consistency computations**

Use raw batch actions when available:
- `action_anchor_loss = mse(augmented_actions, raw_actions)`
- `mean_action_deviation = norm(augmented_actions - raw_actions, dim=1).mean()`

Include `lambda_action_anchor * action_anchor_loss` in `total_loss`.

**Step 3: Extend metrics and filtering**

Return:
- `action_anchor_loss`
- `mean_action_deviation`

Filter in `augment_batch()` using `max_action_deviation` when supplied.

**Step 4: Run targeted test**

Run: `python -m unittest tests.test_vgks_trainer`

Expected: PASS

### Task 3: Thread generation/config inputs

**Files:**
- Modify: `vgks/cli.py`
- Modify: `vgks/train_vgks.py`
- Modify: `vgks/generate_vgks.py`
- Modify: `configs/vgks.base.yaml`

**Step 1: Add CLI/config keys**

Add:
- `lambda_action_anchor`
- `max_action_deviation`

**Step 2: Pass arguments through**

Ensure both `train_vgks.py` and `generate_vgks.py` pass the new fields into `build_trainer_from_args()` and generation filtering.

**Step 3: Include generation metrics**

`generate_vgks.py` should report:
- `mean_action_deviation`

### Task 4: Run regression tests

**Files:**
- Reuse existing tests

**Step 1: Run targeted suite**

Run: `python -m unittest tests.test_vgks_trainer tests.test_train_methods`

Expected: PASS

**Step 2: Run full suite**

Run: `python -m unittest discover -s tests`

Expected: PASS

### Task 5: Commit and push

**Files:**
- Stage only files related to this feature

**Step 1: Commit**

Run:

```bash
git add docs/plans/2026-03-25-action-consistency-guidance-design.md docs/plans/2026-03-25-action-consistency-guidance.md tests/test_vgks_trainer.py vgks/trainer.py vgks/cli.py vgks/train_vgks.py vgks/generate_vgks.py configs/vgks.base.yaml
git commit -m "feat: add action consistency guidance to vgks"
```

**Step 2: Push**

Run:

```bash
git push origin master
git push origin master:main
```
