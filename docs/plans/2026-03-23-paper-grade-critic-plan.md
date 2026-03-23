# Paper-Grade Critic Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the VGKS critic from a simplified placeholder into a paper-defensible conservative critic that can provide stable value guidance for sigma training.

**Architecture:** Keep the existing three-stage `train_vgks.py` pipeline, but replace the critic stage with a medium-fidelity CQL-style implementation. Add target critics, a dedicated behavior/value policy for next-action estimation, conservative logsumexp regularization, soft target updates, and richer diagnostics so sigma receives a meaningful conservative Q signal.

**Tech Stack:** Python, PyTorch, unittest, existing VGKS config/CLI system.

---

### Task 1: Add failing tests for critic stability signals

**Files:**
- Modify: `tests/test_train_vgks_full.py`
- Modify: `tests/test_vgks_trainer.py`

**Step 1: Write failing tests**

Add tests that assert:
- critic metrics now include target/data/OOD diagnostics
- target critic weights are saved in critic checkpoint payload
- sigma guidance can consume unclipped critic values without all guidance collapsing to the clip floor in toy settings

**Step 2: Run targeted tests to verify failure**

Run: `python -m unittest tests.test_train_vgks_full tests.test_vgks_trainer`

Expected: FAIL because the current critic implementation does not expose the new metrics or checkpoint fields.

**Step 3: Commit checkpoint**

Defer commit until implementation and verification are complete.

### Task 2: Implement paper-grade critic primitives

**Files:**
- Modify: `vgks/models.py`
- Modify: `vgks/train_vgks.py`

**Step 1: Add minimal actor for critic backup**

Implement a small Gaussian/Tanh policy used only for critic backup and CQL sampling inside `train_vgks.py` or `vgks/models.py`.

**Step 2: Add target critic handling**

Create target copies for both critics, soft-update helpers, and checkpoint save/load support for target weights.

**Step 3: Replace simplified critic loss**

Implement:
- Bellman target using target critics and sampled next actions
- logsumexp conservative penalty over random/current/next policy actions
- gradient clipping and richer logging

**Step 4: Keep sigma integration stable**

Ensure sigma still reads `critic.conservative_value(...)`, but now from the improved critic training stage.

### Task 3: Tune config surface for the upgraded critic

**Files:**
- Modify: `configs/vgks.base.yaml`
- Modify: `README.md`

**Step 1: Add critic hyperparameters**

Expose the upgraded critic knobs in config:
- `critic_tau`
- `critic_policy_lr`
- `critic_samples`
- `cql_temp`
- `cql_min_q_weight`

**Step 2: Document expected behavior**

Update README notes so users know the critic is now a CQL-style conservative critic and what metrics to watch.

### Task 4: Verify end-to-end behavior

**Files:**
- No new files unless debugging output is needed

**Step 1: Run targeted tests**

Run: `python -m unittest tests.test_train_vgks_full tests.test_vgks_trainer tests.test_integration`

Expected: PASS

**Step 2: Run full suite**

Run: `python -m unittest discover -s tests`

Expected: PASS

**Step 3: Commit**

```bash
git add docs/plans/2026-03-23-paper-grade-critic-plan.md tests/test_train_vgks_full.py tests/test_vgks_trainer.py vgks/models.py vgks/train_vgks.py configs/vgks.base.yaml README.md
git commit -m "feat: upgrade vgks critic to paper-grade cql"
git push origin main
```
