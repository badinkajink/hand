# In-hand reorientation — research state & handoff (2026-06-03)

Living handoff doc for a FRESH session. Full chronological log: `docs/rl/reorientation.md`.
Task: flat-laying `screwdriver_medium` cylinder → vertical, finger-only (9 DOF), in-hand.

## TL;DR — what's true now
- **Best reorientation policy = `signed+critic`** (held-vertical cos **0.978**, beats v1's 0.96,
  fixes slip-back): `results/rl/20260602-1636-policyB_abl_signed/tensorboard/model_405.pt`.
  Recipe = Policy B v1 + signed `target_axis_progress` + **critic warmstart**.
- **THE bug that wasted v2:** the warmstart loaded only the actor and **discarded the critic**;
  a fresh value function knocks the converged actor off its optimum. Fixed: `--warmstart-critic`
  (default ON). Always warmstart the critic.
- **Judge on deterministic behavior, not training reward sums.** Reward sums conflate
  how-long×how-aligned and hid a verticality regression. Use
  `scripts/rl_eval_reorient_metrics.py` (held_cos / peak / obj_jerk / min_z / drop).
- **Open problems (each unsolved, all documented):**
  - *Smoothness:* jerk-penalties are counterproductive — the corrective finger jerk IS the
    stabilization; penalizing it makes the hold slip/wobble. Use a non-reward lever
    (action low-pass at deploy, or motor-delay/obs-noise DR), not a reward.
  - *De-centering (REAL, 5.1 cm lateral for signed+critic vs v1 2.9):* the `object_lateral_drift`
    penalty exists but **stacking it with handoff-DR + warmstart diverges** (grip collapses).
    Apply ONE constraint at a time.
  - *Bracing:* signed+critic reaches cos 0.99 with its top end only **~3 cm below the palm**
    (vertically; 8 cm 3D incl. de-centering) — "almost," no contact. Needs the de-centering
    closed + a gentle upward nudge; reward-only attempts so far degraded the reorient.
  - *Seamless A→B handoff:* the no-reset single-env mechanism works
    (`rl_demo_handoff_continuous.py`) but B (skip-lift trained) is OOD at the seam and drops
    the cylinder. The in-progress runs below are the fix attempts.

## IN-PROGRESS runs (started 2026-06-03, ~40M ts, 3072 envs each, parallel)
All warmstart `signed+critic` (model_405), skip-lift, signed progress, critic ON, no smoothness.
Single-variable, to fix the handoff/de-centering without the stacking instability:
- **P1 `policyB_p1_handoff_dronly`** — handoff curriculum-DR ALONE (spawn tilt 0→0.20 + z 0→0.03
  ramped over 200 iters via `--handoff-dr-curriculum-iters`), no lateral. Log `p1_handoff_dronly.log`.
- **P2 `policyB_p2_lateral_only`** — gentle de-centering penalty `--lateral-drift-weight=-8` ALONE
  (no DR). Tests if it curbs xy-drift without collapse. Log `p2_lateral_only.log`.
- **P3 `policyB_p3_statebank`** — train-the-handoff: spawn from A's recorded terminal states
  (`--handoff-state-bank results/rl/handoff_state_bank.npz`, 1024 states, A's obj-z≈0.111).
  Log `p3_statebank.log`. Early: drop 2.6/iter (holds A's real grips best).
Early health (all): NO NaN, drops 2.6–5.9, alignment 40–53 and rising.

## How to EVALUATE (the honest metrics)
```
# held-cos / jerk / min_z / drop  (deterministic) — the authoritative comparison:
WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python \
  scripts/rl_eval_reorient_metrics.py "name=<run_dir>:model_<N>.pt" ...
# seamless A→B handoff (no reset); reports object-z at handoff + min-z (hold = min-z>0.05):
WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl uv run --extra rl --extra gpu python \
  scripts/rl_demo_handoff_continuous.py --policy-b <ckpt> --output <mp4> --handoff-step 45 --total-steps 240
# de-centering (palm-frame lateral excursion): see /tmp/diag2.py pattern in reorientation.md.
# single-policy reorient video: scripts/rl_render_reorient.py --run <dir> --checkpoint model_N.pt --output <mp4>
```
Reference numbers: v1 held 0.96/jerk 41; signed+critic 0.978/52; de-center w40 cut drift but on a
degraded base. **min_z<0.05 ⇒ floor contact/drop.**

## Tooling / knobs (all in `scripts/rl_train_cube.py`, env in `src/morphohand/rl/env_cfg.py`)
- `--warmstart-critic` (default ON), `--init-actor-checkpoint`.
- reorient: `--enable-target-axis-reward --target-axis-weight --target-axis-progress-weight`
  (signed by default), `--skip-lift-phase`, `--reorient-start-step`.
- de-centering: `--lateral-drift-weight/-deadband/-power` (palm-frame, quadratic past deadband).
- handoff DR: `--skip-lift-spawn-tilt-jitter --skip-lift-spawn-z-jitter --handoff-dr-curriculum-iters`.
- train-the-handoff: `--handoff-state-bank <npz>` (record via `scripts/rl_record_handoff_states.py`).
- bracing (built, geometry-limited): `--brace-force-weight --brace-distance-weight --grip-force-weight`.
- smoothness (don't rely on it): `--action-rate-weight --object-ang-acc-weight --*-final --smoothness-curriculum-*`.
- finger-residual gating: `--finger-residual-active-from-step` (zero residual during scripted lift).

## GOTCHAS (do not relearn these)
1. **Always `--warmstart-critic`** (default ON) — actor-only warmstart wrecks finetunes.
2. **Parallel training:** give each run its OWN Warp cache `WARP_CACHE_PATH=$(mktemp -d)` — a shared
   cache races and NaNs. VRAM is cheap: 3×3072-env runs ≈ 11 GB / 50% on the 16 GB GPU; push more
   (4096 envs and/or 4 parallel). Stagger launches ~60-90 s so kernel compiles don't pile up.
3. **Judge on `rl_eval_reorient_metrics.py` (deterministic held-cos), never reward sums.**
4. **Stacking objectives diverges** — the finger-only reorient is fragile; add ONE new
   constraint (DR, or lateral, or brace) at a time; warmstart from a stable base.
5. **Launch training DETACHED** (`nohup setsid bash … >log 2>&1 </dev/null & disown`) — SSH/laptop
   sleep otherwise kills it (and the harness-tracked jobs).
6. **The "revert gremlin":** SSH reconnects have reverted tracked files (env_cfg.py / rl_train_cube.py)
   to stale versions mid-session. **Commit after every change**, and if a file looks short/old,
   `git checkout HEAD -- <file>` to restore. HEAD is the source of truth.

## Reproduce the in-progress launches
The exact P1/P2/P3 commands are in `scripts/queue_reorient_handoff_dr.sh` / the run dirs'
`config.yaml`. All three: `--num-envs 3072 --total-timesteps 40000000 --init-actor-checkpoint
results/rl/20260602-1636-policyB_abl_signed/tensorboard/model_405.pt` + the per-path knobs above.
