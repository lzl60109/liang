# Value-Guided Koopman Symmetry Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a clean VGKS codebase that integrates Koopman symmetry learning with conservative value guidance and provides a tested augmentation pipeline.

**Architecture:** Create a small standalone Python package under `vgks/` that reuses KATS ideas for Koopman dynamics and inverse dynamics while importing the conservative critic design from TGCVG. Keep the value-aware sigma training logic isolated in a trainer class so the method can be tested without requiring full D4RL training runs.

**Tech Stack:** Python, PyTorch, pytest

---

### Task 1: Scaffold the new package

**Files:**
- Create: `H:\codex_test\nips2026\vgks\__init__.py`
- Create: `H:\codex_test\nips2026\vgks\models.py`
- Create: `H:\codex_test\nips2026\vgks\trainer.py`
- Create: `H:\codex_test\nips2026\vgks\cli.py`

**Step 1: Write the failing test**

Create a test that imports `vgks` modules and fails because the package does not exist.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vgks_trainer.py::test_sigma_metrics_include_value_guidance_terms -v`
Expected: FAIL with import error for `vgks`

**Step 3: Write minimal implementation**

Create the package files with empty but importable classes and functions.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vgks_trainer.py::test_sigma_metrics_include_value_guidance_terms -v`
Expected: PASS import stage, then fail on missing behavior

### Task 2: Add sigma loss tests

**Files:**
- Create: `H:\codex_test\nips2026\tests\test_vgks_trainer.py`
- Modify: `H:\codex_test\nips2026\vgks\trainer.py`
- Modify: `H:\codex_test\nips2026\vgks\models.py`

**Step 1: Write the failing test**

Add tests that verify:
- sigma metrics include commutation, value, and anchor terms,
- conservative Q uses the minimum of two critics,
- total loss changes when `lambda_q` changes.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vgks_trainer.py -v`
Expected: FAIL on missing trainer behavior

**Step 3: Write minimal implementation**

Implement:
- Koopman dynamics model with encoder/decoder helpers,
- inverse dynamics model,
- sigma model,
- conservative critic wrapper,
- value-aware sigma loss computation.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vgks_trainer.py -v`
Expected: PASS for sigma loss tests

### Task 3: Add augmentation tests

**Files:**
- Modify: `H:\codex_test\nips2026\tests\test_vgks_trainer.py`
- Modify: `H:\codex_test\nips2026\vgks\trainer.py`

**Step 1: Write the failing test**

Add tests that verify augmentation returns:
- observations,
- actions,
- next observations,
- q values,
- optional threshold filtering.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vgks_trainer.py::test_augment_batch_can_filter_by_q_threshold -v`
Expected: FAIL on missing augmentation behavior

**Step 3: Write minimal implementation**

Implement augmentation generation and Q-threshold filtering in the trainer.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vgks_trainer.py -v`
Expected: PASS

### Task 4: Add CLI wiring

**Files:**
- Modify: `H:\codex_test\nips2026\vgks\cli.py`
- Modify: `H:\codex_test\nips2026\vgks\trainer.py`

**Step 1: Write the failing test**

Add a test that verifies CLI argument parsing exposes the value-guidance knobs with the expected defaults.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL on missing CLI parser

**Step 3: Write minimal implementation**

Implement an argument parser with:
- `--lambda-q`
- `--lambda-state-anchor`
- `--lambda-latent-anchor`
- `--q-clip-min`
- `--q-clip-max`
- `--q-threshold`
- `--sigma-warmup-steps`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

### Task 5: Verify and document usage

**Files:**
- Modify: `H:\codex_test\nips2026\vgks\__init__.py`
- Create: `H:\codex_test\nips2026\README.md`

**Step 1: Write the failing test**

Add a smoke test that imports the public API and builds a trainer from simple dimensions.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_smoke.py -v`
Expected: FAIL until exports are wired

**Step 3: Write minimal implementation**

Export the public symbols and add a short README with the method summary and package layout.

**Step 4: Run test to verify it passes**

Run: `pytest tests -v`
Expected: PASS
