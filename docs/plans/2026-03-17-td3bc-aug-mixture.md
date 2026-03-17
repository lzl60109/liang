# TD3BC Augmented Data Mixture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add raw/augmented dataset mixing controls to TD3BC so we can quickly diagnose whether instability comes from the trainer or from VGKS-generated data.

**Architecture:** Keep the current TD3BC training loop, but let it build the replay dataset from either a single source or a controlled mixture of raw and augmented transitions. Use deterministic subsampling of the augmented data based on `mix_aug_ratio`.

**Tech Stack:** Python, NumPy, PyTorch, unittest, YAML config

---

### Task 1: Add failing tests for dataset mixing

**Files:**
- Modify: `H:/codex_test/nips2026/tests/test_train_methods.py`
- Modify: `H:/codex_test/nips2026/vgks/offline_rl.py`

**Step 1: Write the failing test**

Add tests that expect:
- raw-only mode returns a loader built only from raw samples
- mixed mode appends a deterministic subset of augmented samples

**Step 2: Run test to verify it fails**

Run:
- `python -m unittest tests.test_train_methods.MethodTrainingTests.test_build_td3bc_dataset_uses_raw_only_when_mix_ratio_zero -v`

Expected: FAIL because helper does not exist yet.

**Step 3: Write minimal implementation**

Add helper functions to load raw and augmented datasets and concatenate them by ratio.

**Step 4: Run test to verify it passes**

Run the targeted test again and confirm PASS.

### Task 2: Wire the new inputs into `train_td3bc.py`

**Files:**
- Modify: `H:/codex_test/nips2026/vgks/train_td3bc.py`
- Modify: `H:/codex_test/nips2026/tests/test_train_methods.py`

**Step 1: Write the failing test**

Add a short training test that:
- passes both `raw_dataset_path` and `aug_dataset_path`
- sets `mix_aug_ratio`
- confirms the run still completes and writes eval output

**Step 2: Run test to verify it fails**

Run:
- `python -m unittest tests.test_train_methods.MethodTrainingTests.test_run_td3bc_training_accepts_raw_and_aug_dataset_paths -v`

Expected: FAIL until the trainer accepts the new arguments.

**Step 3: Write minimal implementation**

Update the TD3BC entrypoint and runtime config plumbing.

**Step 4: Run test to verify it passes**

Run the targeted test and confirm PASS.

### Task 3: Update config defaults

**Files:**
- Modify: `H:/codex_test/nips2026/configs/offline_rl/td3bc.yaml`
- Modify: `H:/codex_test/nips2026/tests/test_train_methods.py`

**Step 1: Write the failing test**

Add a config test that expects:
- `raw_dataset_path`
- `aug_dataset_path`
- `mix_aug_ratio`

**Step 2: Run test to verify it fails**

Run:
- `python -m unittest tests.test_train_methods.MethodTrainingTests.test_td3bc_config_includes_aug_mixture_fields -v`

Expected: FAIL if config is missing the fields.

**Step 3: Write minimal implementation**

Add the new fields to the YAML config.

**Step 4: Run test to verify it passes**

Run the targeted test again and confirm PASS.

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
git add docs/plans/2026-03-17-td3bc-aug-mixture-design.md docs/plans/2026-03-17-td3bc-aug-mixture.md tests/test_train_methods.py vgks/offline_rl.py vgks/train_td3bc.py configs/offline_rl/td3bc.yaml
git commit -m "feat: add td3bc raw aug mixture controls"
```
