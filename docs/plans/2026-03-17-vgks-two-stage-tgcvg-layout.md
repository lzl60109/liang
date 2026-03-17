# VGKS Two-Stage TGCVG-Style Layout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor VGKS into a two-stage pipeline that matches the user's intended paper workflow: stage 1 generates augmented trajectories, stage 2 trains downstream offline RL algorithms such as TD3+BC, IQL, and CQL on the augmented dataset.

**Architecture:** Keep the existing VGKS core trainer and data utilities, but split the workflow into a generation entrypoint and offline-RL training entrypoints. Reorganize configs, outputs, and README so they resemble TGCVG: configs grouped by stage, data directories for raw and augmented trajectories, and command-oriented docs.

**Tech Stack:** Python, PyTorch, PyYAML, unittest, existing `vgks/` package, extracted TGCVG algorithm references in `_extract_tgcvg/TGCVG-main/corl/algorithms`

---

### Task 1: Add failing tests for two-stage dataset generation

**Files:**
- Modify: `H:\codex_test\nips2026\tests\test_training_entry.py`
- Modify: `H:\codex_test\nips2026\tests\test_download_dataset.py`

**Step 1: Write the failing test**

Add tests that assert:
- generation entrypoints save augmented trajectory datasets without training/evaluating a policy
- generated datasets can be read back as replay dicts from TGCVG-style `.pkl` and `.npy`

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_training_entry -v`
Expected: FAIL because generation-only helpers do not exist yet.

**Step 3: Write minimal implementation**

Add generation helpers and file outputs.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_training_entry -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_training_entry.py tests/test_download_dataset.py vgks
git commit -m "feat: split vgks generation stage"
```

### Task 2: Add VGKS generation entrypoint and output layout

**Files:**
- Create: `H:\codex_test\nips2026\vgks\generate_vgks.py`
- Modify: `H:\codex_test\nips2026\vgks\train_vgks.py`
- Modify: `H:\codex_test\nips2026\vgks\data.py`
- Modify: `H:\codex_test\nips2026\vgks\export.py`

**Step 1: Write the failing test**

Covered by Task 1.

**Step 2: Run test to verify it fails**

Covered by Task 1.

**Step 3: Write minimal implementation**

Make stage 1 do only:
- load raw offline dataset
- train VGKS sigma
- generate augmented transitions
- save outputs under a TGCVG-style directory layout

Do not train downstream policy inside the generation entrypoint.

**Step 4: Run test to verify it passes**

Covered by Task 1.

**Step 5: Commit**

Included in Task 1 commit.

### Task 3: Add downstream offline RL training entrypoints

**Files:**
- Create: `H:\codex_test\nips2026\vgks\train_td3bc.py`
- Create: `H:\codex_test\nips2026\vgks\train_iql.py`
- Create: `H:\codex_test\nips2026\vgks\train_cql.py`
- Modify: `H:\codex_test\nips2026\tests\test_train_methods.py`

**Step 1: Write the failing test**

Add tests that assert the new entrypoints can:
- load generated augmented datasets
- write checkpoints and eval files
- report normalized score

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_train_methods -v`
Expected: FAIL because the new entrypoints do not exist yet.

**Step 3: Write minimal implementation**

Wire stage-2 trainers to read augmented datasets and run algorithm-specific loops. Reuse or adapt logic from extracted TGCVG files where it is practical.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_train_methods -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_train_methods.py vgks
git commit -m "feat: add vgks offline rl stage entrypoints"
```

### Task 4: Reorganize configs and docs to match TGCVG-style usage

**Files:**
- Create: `H:\codex_test\nips2026\configs\vgks\`
- Create: `H:\codex_test\nips2026\configs\offline_rl\`
- Modify: `H:\codex_test\nips2026\README.md`
- Create: `H:\codex_test\nips2026\run.sh`

**Step 1: Write the failing test**

Add assertions for any new config paths only if needed; rely mainly on command and integration tests.

**Step 2: Run test to verify it fails**

Not required if doc-only.

**Step 3: Write minimal implementation**

Make the user-facing structure resemble TGCVG:
- stage-specific config directories
- command examples for `generate_vgks.py`
- command examples for `train_td3bc.py`, `train_iql.py`, `train_cql.py`

**Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -v`
Expected: PASS

**Step 5: Commit**

```bash
git add configs README.md run.sh
git commit -m "docs: align vgks workflow with tgcvg layout"
```
