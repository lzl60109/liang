# D4RL Download Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a standalone D4RL downloader that saves `pkl + npy` trajectory caches and wire VGKS to train from those cached trajectories.

**Architecture:** Build one shared cache save/load layer in `vgks/data.py`, expose a downloader script for D4RL datasets, and update the README plus VGKS commands so cached trajectory folders become the preferred training input. Keep the file format simple and explicit.

**Tech Stack:** Python, PyTorch, D4RL, pickle, NumPy, unittest

---

### Task 1: Add cache save/load tests

**Files:**
- Modify: `H:\codex_test\nips2026\tests\test_training_entry.py`
- Create: `H:\codex_test\nips2026\tests\test_download_dataset.py`

**Step 1: Write the failing test**

Add tests that verify:
- saving a cache directory writes `dataset.pkl` and `.npy` files,
- loading a cache directory reconstructs the dataset,
- VGKS training can read a cached directory path.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_download_dataset -v`
Expected: FAIL because cache helpers do not exist.

**Step 3: Write minimal implementation**

Add cache helpers in the data layer.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_download_dataset -v`
Expected: PASS

### Task 2: Add D4RL downloader script

**Files:**
- Create: `H:\codex_test\nips2026\vgks\download_d4rl_dataset.py`

**Step 1: Write the failing test**

Add a test for the downloader helper that uses a toy dataset dictionary and writes the expected cache files.

**Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_download_dataset -v`
Expected: FAIL on missing downloader module.

**Step 3: Write minimal implementation**

Implement the script and shared save helper.

**Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_download_dataset -v`
Expected: PASS

### Task 3: Update VGKS docs and commands

**Files:**
- Modify: `H:\codex_test\nips2026\README.md`

**Step 1: Write the failing test**

If needed, add or update a CLI test to confirm the downloader supports task/split arguments.

**Step 2: Run test to verify it fails**

Run: `python -m unittest discover -s tests -v`
Expected: FAIL until docs and CLI are aligned.

**Step 3: Write minimal implementation**

Document:
- how to download `medium`, `medium-replay`, and `medium-expert`,
- how to train VGKS from the cached trajectory folder.

**Step 4: Run test to verify it passes**

Run: `python -m unittest discover -s tests -v`
Expected: PASS
