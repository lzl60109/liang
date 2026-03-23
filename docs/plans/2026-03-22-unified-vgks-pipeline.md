# Unified VGKS Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a complete three-stage VGKS project that trains KATS-compatible dynamics and a conservative critic, generates value-guided augmented data, and validates with robust mixed-data offline RL baselines.

**Architecture:** Keep the user-facing workflow to three steps: `train_vgks.py` for upstream training and checkpoint production, `generate_vgks.py` for augmentation-only generation, and downstream train scripts for validation. Internally, `train_vgks.py` will run staged training for Koopman dynamics, inverse dynamics, conservative critic, and value-guided sigma while saving unified checkpoints and metrics in a stable format.

**Tech Stack:** Python, PyTorch, NumPy, pytest, YAML configs, D4RL-style offline datasets

---

### Task 1: Add regression tests for unified checkpoint production

**Files:**
- Modify: `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\tests\test_train_vgks_full.py`
- Modify: `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\tests\test_integration.py`

**Step 1: Write the failing test**

Add tests that require `run_training(...)` to save:
- `kats_checkpoint.pt`
- `critic_checkpoint.pt`
- `vgks_checkpoint.pt`

Add integration coverage that `load_kats_checkpoint(...)` and `load_tgcvg_critic_checkpoint(...)` can restore the new unified formats.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_vgks_full.py tests/test_integration.py -q`

Expected: collection is currently blocked by missing `torch`; once environment is present, new assertions should fail until implementation is added.

**Step 3: Write minimal implementation**

Update checkpoint save/load logic in:
- `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\vgks\train_vgks.py`
- `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\vgks\integration.py`
- `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\vgks\models.py`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_vgks_full.py tests/test_integration.py -q`

Expected: pass in a Python environment with `torch` installed.

### Task 2: Add regression tests for staged upstream training in `train_vgks.py`

**Files:**
- Modify: `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\tests\test_train_vgks_full.py`
- Modify: `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\tests\test_vgks_trainer.py`

**Step 1: Write the failing test**

Add tests that require:
- Koopman/inverse-dynamics pretraining metrics to be returned
- conservative critic pretraining metrics to be returned
- sigma metrics to remain available
- `build_trainer_from_args(...)` to support initializing from newly produced checkpoints

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_vgks_full.py tests/test_vgks_trainer.py -q`

Expected: fail after environment setup because current code does not expose the full staged metrics/checkpoint behavior.

**Step 3: Write minimal implementation**

Refactor `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\vgks\train_vgks.py` to:
- train Koopman dynamics and inverse dynamics first
- train a conservative twin critic second
- freeze both and train sigma third
- return structured histories for all three stages

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_vgks_full.py tests/test_vgks_trainer.py -q`

Expected: pass in a Python environment with `torch` installed.

### Task 3: Add regression tests for augmentation filtering and generation behavior

**Files:**
- Modify: `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\tests\test_train_methods.py`
- Modify: `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\tests\test_vgks_trainer.py`

**Step 1: Write the failing test**

Add tests that require:
- `q_threshold` to flow from config/CLI into generation
- generation to optionally export only high-value synthetic transitions
- generation metadata to include produced checkpoint paths and kept sample counts

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_methods.py tests/test_vgks_trainer.py -q`

Expected: fail after environment setup because generation currently ignores `q_threshold`.

**Step 3: Write minimal implementation**

Update:
- `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\vgks\generate_vgks.py`
- `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\vgks\trainer.py`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_methods.py tests/test_vgks_trainer.py -q`

Expected: pass in a Python environment with `torch` installed.

### Task 4: Add regression tests for downstream mixed-data training coverage

**Files:**
- Modify: `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\tests\test_train_methods.py`
- Modify: `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\tests\test_train_bc.py`
- Modify: `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\tests\test_train_corl_bc.py`

**Step 1: Write the failing test**

Add tests that require:
- `TD3BC`, `IQL`, `CQL`, and `BC` to all accept raw-only and raw-plus-augmented workflows
- metrics and saved artifacts to expose whether raw-only or mixed mode was used
- `CQL` to gain the same mixed-data path style as `TD3BC/IQL`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_methods.py tests/test_train_bc.py tests/test_train_corl_bc.py -q`

Expected: fail after environment setup because `CQL` mixed-data support and some reporting are incomplete.

**Step 3: Write minimal implementation**

Update:
- `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\vgks\offline_rl.py`
- `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\vgks\train_td3bc.py`
- `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\vgks\train_iql.py`
- `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\vgks\train_cql.py`
- `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\vgks\train_bc.py`
- `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\vgks\train_corl_bc.py`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_train_methods.py tests/test_train_bc.py tests/test_train_corl_bc.py -q`

Expected: pass in a Python environment with `torch` installed.

### Task 5: Update configs and docs for the three-stage workflow

**Files:**
- Modify: `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\README.md`
- Modify: `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\configs\vgks.base.yaml`
- Modify: environment preset files under `H:\codex_test\nips2026\.worktrees\codex-vgks-unified\configs\`

**Step 1: Write the failing test**

Add or extend config/docs tests to require:
- documented three-step workflow
- staged training hyperparameters for Koopman, critic, sigma
- explicit checkpoint outputs

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_vgks_config.py tests/test_vgks_presets.py -q`

Expected: fail after environment setup if required config keys are missing.

**Step 3: Write minimal implementation**

Add config keys and documentation for:
- Koopman pretrain epochs/lr
- inverse dynamics epochs/lr
- critic epochs/lr and CQL settings
- sigma training settings
- generation filtering settings
- downstream experiment commands

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_vgks_config.py tests/test_vgks_presets.py -q`

Expected: pass in a Python environment with `torch` installed.

### Task 6: Full verification pass

**Files:**
- Modify only if verification reveals issues

**Step 1: Run targeted verification**

Run:
- `pytest tests/test_train_vgks_full.py tests/test_integration.py tests/test_vgks_trainer.py tests/test_train_methods.py tests/test_train_bc.py tests/test_train_corl_bc.py tests/test_vgks_config.py tests/test_vgks_presets.py -q`

**Step 2: Run broad verification**

Run:
- `pytest -q`

**Step 3: Record actual status**

If `torch` is still unavailable in the environment, report that code changes are complete but runtime verification remains blocked by missing dependency. If `torch` is available, report exact pass/fail counts and any remaining gaps.
