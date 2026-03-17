# Offline RL Progress Logging Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add TGCVG-style terminal progress output to the stage-two offline RL training scripts so users can monitor long experiments in real time.

**Architecture:** Introduce shared progress-formatting helpers in the offline RL utility layer, then call them from each second-stage training script at `log_every` and `eval_freq` boundaries. Keep JSON logging and checkpoint saving unchanged so this is a behavior addition rather than a storage refactor.

**Tech Stack:** Python, PyTorch, unittest, existing VGKS logging/evaluation helpers

---

### Task 1: Add failing tests for progress output helpers

**Files:**
- Modify: `H:/codex_test/nips2026/tests/test_train_methods.py`
- Test: `H:/codex_test/nips2026/tests/test_train_methods.py`

**Step 1: Write the failing test**

Add tests that expect:
- a shared train-progress formatter to include step and losses
- a shared eval-progress formatter to include step and normalized score

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_train_methods.TrainMethodsTests.test_format_train_progress_includes_step_and_losses -v`

Expected: FAIL because formatter does not exist yet.

**Step 3: Write minimal implementation**

Add shared helper functions in `vgks/offline_rl.py`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_train_methods.TrainMethodsTests.test_format_train_progress_includes_step_and_losses -v`

Expected: PASS

### Task 2: Add failing stdout test for TD3BC

**Files:**
- Modify: `H:/codex_test/nips2026/tests/test_train_methods.py`
- Modify: `H:/codex_test/nips2026/vgks/train_td3bc.py`

**Step 1: Write the failing test**

Add a short-run test that captures stdout and expects both `[Train]` and `[Eval]` lines from `run_td3bc_training(...)`.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_train_methods.TrainMethodsTests.test_run_td3bc_training_prints_progress -v`

Expected: FAIL because the function currently prints nothing during training.

**Step 3: Write minimal implementation**

Print formatted train progress at `log_every` intervals and eval progress at `eval_freq` intervals.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_train_methods.TrainMethodsTests.test_run_td3bc_training_prints_progress -v`

Expected: PASS

### Task 3: Apply shared logging to IQL and CQL

**Files:**
- Modify: `H:/codex_test/nips2026/vgks/train_iql.py`
- Modify: `H:/codex_test/nips2026/vgks/train_cql.py`
- Modify: `H:/codex_test/nips2026/tests/test_train_methods.py`

**Step 1: Write the failing test**

Add short-run stdout tests for IQL and CQL that expect `[Train]` and `[Eval]`.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_train_methods.TrainMethodsTests.test_run_iql_training_prints_progress tests.test_train_methods.TrainMethodsTests.test_run_cql_training_prints_progress -v`

Expected: FAIL because those functions currently print nothing.

**Step 3: Write minimal implementation**

Use the same shared progress helper in `train_iql.py` and `train_cql.py`.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_train_methods.TrainMethodsTests.test_run_iql_training_prints_progress tests.test_train_methods.TrainMethodsTests.test_run_cql_training_prints_progress -v`

Expected: PASS

### Task 4: Quiet the default generation config

**Files:**
- Modify: `H:/codex_test/nips2026/configs/vgks/config.yaml`

**Step 1: Write the failing test**

Add a config test that expects `use_wandb: false` in the shipped VGKS generation config.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_vgks_config.VGKSConfigTests.test_default_generation_config_disables_wandb -v`

Expected: FAIL if the config still enables wandb.

**Step 3: Write minimal implementation**

Set `use_wandb: false` in the default generation config.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_vgks_config.VGKSConfigTests.test_default_generation_config_disables_wandb -v`

Expected: PASS

### Task 5: Verify the full suite

**Files:**
- Modify as needed from prior tasks

**Step 1: Run targeted tests**

Run:
- `python -m unittest tests.test_train_methods -v`
- `python -m unittest tests.test_vgks_config -v`

Expected: PASS

**Step 2: Run the full suite**

Run: `python -m unittest discover -s tests -v`

Expected: PASS

**Step 3: Commit**

```bash
git add docs/plans tests/test_train_methods.py tests/test_vgks_config.py vgks/offline_rl.py vgks/train_td3bc.py vgks/train_iql.py vgks/train_cql.py configs/vgks/config.yaml
git commit -m "feat: print offline rl training progress"
```
