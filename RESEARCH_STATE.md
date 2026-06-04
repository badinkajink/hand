# In-hand reorientation — research state & handoff (2026-06-04)

Living handoff doc for a FRESH session. Full chronological log: `docs/rl/reorientation.md`.
Task: flat-laying `screwdriver_medium` cylinder → vertical, finger-only (9 DOF), in-hand.

## TL;DR — what's true now
- **Best reorientation policy = `p2_lateral`** (held-vertical cos **0.988**, peak **0.999**,
  obj_jerk **25.8** = HALF the prior best, no drop):
  `results/rl/20260603-1746-policyB_p2_lateral_only/tensorboard/model_541.pt`.
  Recipe = signed+critic + `--lateral-drift-weight=-8` ALONE (one constraint at a time).
  Surprise: the lateral penalty did NOT reduce de-centering (it actually drifted ~1 cm more);
  what it DID do was act as a **smoothing regularizer** — halving object jerk while pushing
  verticality up. (Prior best was `signed+critic` model_405 at 0.978/jerk 51.6.)
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
  - *De-centering (REAL):* still open. P2's `--lateral-drift-weight=-8` did NOT curb it
    (5.6/4.8 cm world vs baseline 3.8/5.2). **The best de-centering lever found is the
    statebank** (P3: 3.0/3.1 cm ≈ 4.3 cm 3D) — training on A's real centered grips keeps it
    centered — but P3 reorients worse (0.930). Still: stacking diverges; add ONE at a time.
  - *Bracing:* unchanged; reaches cos 0.99 ~3 cm below palm (vertically), no contact.
  - *Seamless A→B handoff:* **DIAGNOSED, still OPEN (2026-06-04).** Seam drop is an
    OBSERVATION-DISCONTINUITY shock (skip-lift B is OOD on A's normal-lift delivery), not a grip
    problem. Tried fix = train B in the normal-lift env with a **grace window** (hold-first,
    reorient later). v2 (no grace) collapsed; v3 grace held reward flat ~10 but NaN'd at 60/750;
    **v3b reran grace to completion (NaN-resilient) and BOTH variants COLLAPSED** — held-cos
    −0.035/−0.078, 100% drop, handoff min-z 0.004/0.007 ≪ 0.05. The grace window prevented the
    v2 *training* collapse but locked B in a hold-during-grace / drop-at-reorient-onset local
    optimum (reward flat ~9.3, never learned). **A grace window does NOT make the skip-lift P2
    warmstart robust to the seam in 40M ts.** Best untried next step: **warmstart the *hold-only*
    control (proven to survive the takeover), not P2** — see "Normal-lift B history" below.

## P1/P2/P3 — DONE (2026-06-03, 40M ts / 3072 envs each). Authoritative deterministic eval:
| policy | held_cos | peak | obj_jerk | min_z | drop | world Δlat |
|---|---|---|---|---|---|---|
| baseline signed+critic (405) | 0.979 | 0.988 | 51.6 | 0.109 | 0 | 3.8/5.2 |
| P1 handoff-DR alone (541) | 0.954 | 0.994 | 59.1 | 0.115 | 0 | — |
| **P2 lateral-only (541)** | **0.988** | **0.999** | **25.8** | 0.117 | 0 | 5.6/4.8 |
| P3 statebank (541) | 0.930 | 0.944 | **8.4** | 0.114 | 0 | **3.0/3.1** |
- **P1 (handoff-DR alone): worse, discard.** DR-alone destabilized the grip (training term
  stats: drop 16.4, floor 11.25) — confirms gotcha #4.
- **P2: new best reorienter** (verticality + smoothness). De-centering unchanged.
- **P3: best de-centering + smoothest, but weakest reorienter.**

## HANDOFF DIAGNOSIS (the one remaining open problem) + the fix in training
Instrumented the continuous A→B rollout (z every step). The drop is **instantaneous at the seam**,
identical for P2/P3/baseline (all skip-lift trained):
```
step 40 z=0.111 (A holding) → 45 z=0.094 (handoff) → 46 z=0.073 → 48 z=0.022 → 50 z=0.010 (floor)
```
B collapses the grip within **3–5 steps** of taking over → it's an OOD shock, not grip weakness.
That's why neither DR (P1) nor statebank (P3) helped: both still trained B in the **skip-lift**
env, whose lift-command phase / `ref_object_pose` schedule differs from the **normal-lift** env
used at deploy. B never saw the seam obs in training. **PROOF:** the (undertrained, 15M/1024)
`20260603-1315-policyB_normallift` B — trained in the normal-lift env with residual gated to
activate at step 35 — held **z≈0.09 for ~10–15 steps past the seam** (vs instant collapse), then
dropped (undertrained). So normal-lift training removes the shock; it just needs convergence.

### Normal-lift B history: v2 collapsed → v3 grace (NaN'd) → **v3b ran to completion, BOTH COLLAPSED**
- **v2_fromP2 (normal-lift, warmstart P2): COLLAPSED** — held-cos 0.029, 100% drop, handoff
  min-z 0.005. Step-35 fired residual+terminations+full-reorient at once; OOD warmstart fumbles,
  terminations kill episodes → reward 12→3 → never learns.
- **v3 grace window: looked promising, NaN'd before it could be judged** — B takes over (residual)
  at step 35 but only HOLDS until step 50, when terminations+reorient engage. Reward stayed
  flat (~10) for 60 iters (no v2 *training* collapse), then **NaN-crashed at iter ~60/750**.
  Only model_50 (undertrained, drops).
- **v3 hold-only control (reorient OFF): completed, PROVES B survives the handoff** — tip_lost
  humped to ~44 then recovered to ~1–4. (65-dim, not deployable; isolation control only.)
  **→ this is the warmstart to use next.**
- **v3b (`policyB_normallift_v3b_{repro,soft}`, 40M/3072, completed 2026-06-04): BOTH COLLAPSED.**
  NaN-resilience worked (both ran the full 542 iters, no crash), but that *revealed* the v3 "flat
  ~10 reward" was a degenerate plateau, not learning: reward stayed flat ~9.3 (R)/~9.0 (S) the
  whole run. Deterministic held-cos **−0.035 (R) / −0.078 (S)**, **100% drop**; handoff min-z
  **0.0037 / 0.0069** (≪ 0.05). Training at convergence: floor_proximity term 95.9/58.9,
  object_height 0.012/0.015 m, success 0. **Mechanism:** grace window stopped the v2 collapse but
  locked B in a *hold-during-grace, drop-at-reorient-onset* local optimum — it can't cross from
  "hold" into a working post-seam reorient from the OOD skip-lift P2 prior. Soft onset (S) bought
  nothing. Videos `handoff_v3b_{repro,soft}.mp4` (show the drop). Full writeup: reorientation.md
  "v3b OUTCOME".
- **Best untried next experiment (NOT run, budget spent):** finetune the **v3 hold-only**
  checkpoint toward reorient (it already holds A's delivery → grace→reorient transition is
  in-distribution), instead of the OOD skip-lift P2. Backup: add P3's handoff **state-bank** to
  the normal-lift env (train on A's real seam states). Longer training is least promising (reward
  was *flat*, i.e. a local optimum, not undertraining).

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
