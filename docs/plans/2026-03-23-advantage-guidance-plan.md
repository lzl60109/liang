# Advantage-Guided VGKS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve VGKS data quality by replacing absolute-Q sigma guidance with relative advantage guidance, filtering generated data by both value and state shift, and selecting higher-quality augmented samples for downstream TD3BC mixing.

**Architecture:** Keep the current paper-grade critic, but change how sigma consumes critic outputs and how augmented samples enter downstream training. The key idea is to optimize and select for "better than the original transition" rather than merely "large absolute Q", then bias downstream mixing toward higher-scoring synthetic samples.

**Tech Stack:** Python, PyTorch, unittest, existing VGKS configs and offline RL utilities.

---

### Task 1: Add failing tests for advantage-guided sigma metrics

**Files:**
- Modify: `tests/test_vgks_trainer.py`

**Step 1: Write failing tests**

Add tests asserting:
- sigma loss metrics expose `mean_advantage` and `mean_state_shift`
- `augment_batch()` can filter using a relative Q margin and state-distance threshold

**Step 2: Run targeted tests to verify failure**

Run: `python -m unittest tests.test_vgks_trainer`

Expected: FAIL because the current implementation only supports absolute `q_threshold`.

### Task 2: Add failing tests for top-k augmented mixing

**Files:**
- Modify: `tests/test_train_methods.py`

**Step 1: Write failing tests**

Add tests asserting:
- TD3BC mixed-data construction prefers the top-scoring augmented samples when `q_values` exist
- the mixed-data builder returns metadata describing `raw_size`, `aug_size`, and selected augmented count

**Step 2: Run targeted tests to verify failure**

Run: `python -m unittest tests.test_train_methods`

Expected: FAIL because current selection is random and does not expose selection stats.

### Task 3: Implement trainer-side quality controls

**Files:**
- Modify: `vgks/trainer.py`
- Modify: `configs/vgks.base.yaml`

**Step 1: Replace absolute-Q sigma loss with advantage guidance**

Compute:
- raw conservative Q on original `(s, a)`
- augmented conservative Q on `(s_hat, a_hat)`
- normalized advantage `aug_q - raw_q`

Use that signal in sigma loss instead of absolute clipped Q.

**Step 2: Add generation filtering**

Support:
- `q_delta`
- `max_state_shift`
- `commute_horizon`

### Task 4: Implement downstream top-k augmented mixing

**Files:**
- Modify: `vgks/offline_rl.py`
- Modify: `vgks/train_td3bc.py`

**Step 1: Replace random augmented subset with top-k by `q_values` when available**

If augmented data includes `q_values`, select the highest-scoring subset first.

**Step 2: Expose dataset-selection stats**

Record and print:
- `raw_size`
- `aug_size`
- `taken_aug_size`
- `actor_dataset_size`

### Task 5: Wire generation CLI and docs

**Files:**
- Modify: `vgks/generate_vgks.py`
- Modify: `README.md`

**Step 1: Thread new filtering parameters through generation entrypoint**

**Step 2: Document recommended usage**

### Task 6: Verify and ship

**Files:**
- No new files unless debugging output is needed

**Step 1: Run targeted tests**

Run: `python -m unittest tests.test_vgks_trainer tests.test_train_methods`

Expected: PASS

**Step 2: Run full suite**

Run: `python -m unittest discover -s tests`

Expected: PASS

**Step 3: Commit**

```bash
git add docs/plans/2026-03-23-advantage-guidance-plan.md tests/test_vgks_trainer.py tests/test_train_methods.py vgks/trainer.py vgks/generate_vgks.py vgks/offline_rl.py vgks/train_td3bc.py configs/vgks.base.yaml README.md
git commit -m "feat: improve vgks data selection and guidance"
git push origin main
```
