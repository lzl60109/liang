# TD3BC Stabilization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stabilize the second-stage TD3BC trainer so VGKS-generated datasets can be evaluated with a TD3BC variant that is much closer to the TGCVG/CORL-style reference.

**Architecture:** Keep the existing two-stage VGKS pipeline, but replace the current simplified TD3BC update logic with a normalized, delayed-update, target-smoothed variant. Reuse the current logging and evaluation shell so the user-facing workflow stays the same.

**Tech Stack:** Python, PyTorch, unittest, YAML config, existing VGKS evaluation and logging utilities

---

### Task 1: Add failing tests for stable TD3BC utilities

**Files:**
- Modify: `H:/codex_test/nips2026/tests/test_train_methods.py`
- Modify: `H:/codex_test/nips2026/vgks/offline_rl.py`

**Step 1: Write the failing test**

Add tests that expect:
- observation normalization stats can be computed from offline data
- a stabilized TD3BC train step returns `actor_loss`, `critic_loss`, and `q_mean`
- actor updates are delayed according to `policy_freq`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_train_methods.MethodTrainingTests.test_stable_td3bc_train_step_uses_policy_delay -v`

Expected: FAIL because the helper does not exist yet.

**Step 3: Write minimal implementation**

Add normalization helpers and a stateful TD3BC trainer object in `vgks/offline_rl.py`.

**Step 4: Run test to verify it passes**

Run the same targeted test and confirm PASS.

### Task 2: Switch `train_td3bc.py` to the stabilized trainer

**Files:**
- Modify: `H:/codex_test/nips2026/vgks/train_td3bc.py`
- Modify: `H:/codex_test/nips2026/vgks/offline_rl.py`
- Modify: `H:/codex_test/nips2026/tests/test_train_methods.py`

**Step 1: Write the failing test**

Add a short end-to-end test that expects `run_td3bc_training(...)` to:
- print progress
- save eval output
- include `normalized_score`
- expose stable train metrics including `q_mean`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_train_methods.MethodTrainingTests.test_run_td3bc_training_reports_stable_metrics -v`

Expected: FAIL until the new trainer is wired in.

**Step 3: Write minimal implementation**

Update `run_td3bc_training(...)` to use the stabilized trainer and normalized observations.

**Step 4: Run test to verify it passes**

Run the targeted test again and confirm PASS.

### Task 3: Update TD3BC config defaults

**Files:**
- Modify: `H:/codex_test/nips2026/configs/offline_rl/td3bc.yaml`
- Modify: `H:/codex_test/nips2026/tests/test_train_methods.py`

**Step 1: Write the failing test**

Add a config-level test that expects TD3BC config to include:
- `discount`
- `tau`
- `policy_noise`
- `noise_clip`
- `policy_freq`
- `alpha`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_train_methods.MethodTrainingTests.test_td3bc_config_includes_stability_hyperparameters -v`

Expected: FAIL if config is still too minimal.

**Step 3: Write minimal implementation**

Add the missing parameters to the YAML config and plumb them through the trainer.

**Step 4: Run test to verify it passes**

Run the targeted config test and confirm PASS.

### Task 4: Full verification

**Files:**
- Modify as needed from prior tasks

**Step 1: Run targeted tests**

Run:
- `python -m unittest tests.test_train_methods -v`

Expected: PASS

**Step 2: Run the full suite**

Run:
- `python -m unittest discover -s tests -v`

Expected: PASS

**Step 3: Commit**

```bash
git add docs/plans/2026-03-17-td3bc-stabilization-design.md docs/plans/2026-03-17-td3bc-stabilization.md tests/test_train_methods.py vgks/offline_rl.py vgks/train_td3bc.py configs/offline_rl/td3bc.yaml
git commit -m "feat: stabilize td3bc offline training"
```
