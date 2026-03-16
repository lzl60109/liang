# D4RL Baseline Suite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add standalone BC, KATS, TGCVG, and VGKS training scripts with shared D4RL evaluation, wandb logging, and `.npz` augmentation export.

**Architecture:** Build shared experiment utilities for environments, evaluation, logging, and export, then wire each baseline entrypoint to those utilities so every method produces the same output artifacts and normalized score metrics. Keep the first version compact and compatible with the current VGKS prototype while preserving room for later upgrades.

**Tech Stack:** Python, PyTorch, D4RL, Gym, wandb, unittest

---

### Task 1: Add shared experiment utilities

**Files:**
- Create: `H:\codex_test\nips2026\vgks\envs.py`
- Create: `H:\codex_test\nips2026\vgks\eval.py`
- Create: `H:\codex_test\nips2026\vgks\logging.py`
- Create: `H:\codex_test\nips2026\vgks\export.py`
- Test: `H:\codex_test\nips2026\tests\test_experiment_utils.py`

**Step 1: Write the failing test**

Add tests covering:
- normalized score evaluation with a fake env,
- config/eval JSON writing,
- augmentation `.npz` export shape and keys.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_experiment_utils -v`
Expected: FAIL because shared utility modules do not exist.

**Step 3: Write minimal implementation**

Implement:
- D4RL environment helper and dimension inference,
- rollout evaluator that computes normalized score,
- wandb-backed logger with safe disabled mode,
- `.npz` export helper.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_experiment_utils -v`
Expected: PASS

### Task 2: Add BC training entrypoint

**Files:**
- Create: `H:\codex_test\nips2026\vgks\train_bc.py`
- Create: `H:\codex_test\nips2026\tests\test_train_bc.py`

**Step 1: Write the failing test**

Add a test that runs one short BC training cycle on a toy dataset and verifies:
- `eval.json` is written,
- normalized score is present,
- local config is saved.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_train_bc -v`
Expected: FAIL because BC entrypoint does not exist.

**Step 3: Write minimal implementation**

Implement standalone BC training with shared logging and evaluation.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_train_bc -v`
Expected: PASS

### Task 3: Add KATS and TGCVG standalone scripts

**Files:**
- Create: `H:\codex_test\nips2026\vgks\train_kats.py`
- Create: `H:\codex_test\nips2026\vgks\train_tgcvg.py`
- Create: `H:\codex_test\nips2026\tests\test_train_methods.py`

**Step 1: Write the failing test**

Add tests that verify:
- KATS run saves `augmented_dataset.npz`,
- TGCVG run saves `augmented_dataset.npz`,
- both save `eval.json`.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_train_methods -v`
Expected: FAIL because the method scripts do not exist.

**Step 3: Write minimal implementation**

Implement compact KATS and TGCVG-style training flows that:
- load offline data,
- generate synthetic data,
- export augmentation,
- evaluate a learned policy.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_train_methods -v`
Expected: PASS

### Task 4: Upgrade VGKS entrypoint to unified experiment mode

**Files:**
- Modify: `H:\codex_test\nips2026\vgks\train_vgks.py`
- Modify: `H:\codex_test\nips2026\tests\test_training_entry.py`

**Step 1: Write the failing test**

Add a test that verifies unified VGKS training writes:
- `eval.json`
- `checkpoint.pt`
- `augmented_dataset.npz`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_training_entry -v`
Expected: FAIL until unified output handling is added.

**Step 3: Write minimal implementation**

Refactor VGKS training to share:
- logger setup,
- evaluation,
- augmentation export,
- checkpoint saving.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_training_entry -v`
Expected: PASS

### Task 5: Update docs and installation guidance

**Files:**
- Modify: `H:\codex_test\nips2026\README.md`
- Modify: `H:\codex_test\nips2026\requirements.txt`
- Modify: `H:\codex_test\nips2026\requirements-gpu-cu121.txt`

**Step 1: Write the failing test**

Add a lightweight test that verifies the documented CLI flags exist for the four scripts if needed.

**Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -v`
Expected: FAIL until new CLI/documented requirements are present.

**Step 3: Write minimal implementation**

Document:
- wandb setup,
- server install steps,
- example commands for medium / medium-replay / medium-expert tasks,
- output file locations.

**Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -v`
Expected: PASS
