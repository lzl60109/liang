# VGKS Multi-Domain Config Expansion Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand VGKS configuration coverage from Mujoco locomotion to Maze2D, AntMaze, Adroit, and Kitchen using family-specific base configs plus per-environment presets.

**Architecture:** Keep one shared global VGKS base for universal knobs, then layer task-family YAML bases on top for domain-specific defaults. Individual environment presets only override dataset path, output path, and run naming so the training entrypoint remains unchanged.

**Tech Stack:** Python, PyYAML, unittest, YAML config inheritance in `vgks/train_vgks.py`

---

### Task 1: Add failing tests for config families and presets

**Files:**
- Modify: `H:\codex_test\nips2026\tests\test_vgks_presets.py`

**Step 1: Write the failing test**

Add assertions that:
- family bases exist for `mujoco`, `maze2d`, `antmaze`, `adroit`, `kitchen`
- Maze2D, AntMaze, Adroit, and Kitchen preset files exist
- representative presets inherit expected values from their family base

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_vgks_presets -v`
Expected: FAIL because the new family files and presets do not exist yet.

**Step 3: Write minimal implementation**

Create the missing YAML files and any necessary README/config loading support.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_vgks_presets -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_vgks_presets.py configs README.md vgks/train_vgks.py
git commit -m "feat: add vgks multi-domain presets"
```

### Task 2: Add family base configs

**Files:**
- Create: `H:\codex_test\nips2026\configs\vgks.mujoco.base.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.maze2d.base.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.antmaze.base.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.adroit.base.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.kitchen.base.yaml`

**Step 1: Write the failing test**

Covered by Task 1.

**Step 2: Run test to verify it fails**

Covered by Task 1.

**Step 3: Write minimal implementation**

Set conservative domain defaults:
- `mujoco`: current locomotion defaults
- `maze2d`: moderate anchors, slightly longer epochs
- `antmaze`: stronger anchors, lower `lambda_q`, warmup, smaller LR
- `adroit`: stronger anchors, smaller LR, larger hidden dim
- `kitchen`: conservative `lambda_q`, longer warmup, stronger state anchor

**Step 4: Run test to verify it passes**

Covered by Task 1.

**Step 5: Commit**

Included in Task 1 commit.

### Task 3: Add per-environment presets

**Files:**
- Create: `H:\codex_test\nips2026\configs\vgks.maze2d-umaze.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.maze2d-medium.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.maze2d-large.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.antmaze-umaze-diverse.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.antmaze-umaze-play.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.antmaze-medium-diverse.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.antmaze-medium-play.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.antmaze-large-diverse.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.antmaze-large-play.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.pen-human.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.hammer-human.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.door-human.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.relocate-human.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.kitchen-complete.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.kitchen-partial.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.kitchen-mixed.yaml`
- Create: `H:\codex_test\nips2026\configs\vgks.kitchen-undirected.yaml`

**Step 1: Write the failing test**

Covered by Task 1.

**Step 2: Run test to verify it fails**

Covered by Task 1.

**Step 3: Write minimal implementation**

Each preset should:
- inherit from its family base
- set `dataset_path`
- set `save_dir`
- set `wandb_name`
- set `run_name`

**Step 4: Run test to verify it passes**

Covered by Task 1.

**Step 5: Commit**

Included in Task 1 commit.

### Task 4: Update documentation and default config behavior

**Files:**
- Modify: `H:\codex_test\nips2026\README.md`
- Modify: `H:\codex_test\nips2026\configs\vgks.yaml`

**Step 1: Write the failing test**

No direct test needed; rely on preset loading tests plus manual README inspection.

**Step 2: Run test to verify it fails**

Not applicable.

**Step 3: Write minimal implementation**

Document:
- which domains are now covered
- why family-specific bases are used
- how to switch presets by editing `configs/vgks.yaml`

**Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -v`
Expected: PASS

**Step 5: Commit**

Included in Task 1 commit.
