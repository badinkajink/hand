# Training Phases

A chronological log of the RL training journey for the morphohand grasp
task. Each phase entry captures: the **goal**, what **changed** vs the
previous phase, the **result**, and the **takeaway**. New attempts append
a section here; do not edit prior entries except for factual corrections
(this is the historical record, not the current state).

For the *current* configuration see [Architecture](architecture.md) +
[Training](training.md). For the cross-phase scoreboard see
[Results](results.md).

---

## Phase 0 — Plan (witty-tumbling-flamingo)

Plan locked: PPO on a single object (cube), single morphology
(`run18_final` candidate 0), warm-start from CEM `best_finger_ctrl`,
scripted palm + 9-d finger residual policy.

Out-of-scope: cross-morphology transfer, in-hand reorientation,
distillation. See the plan file for the full charter.

---

## Phase 1 — first MVP, `cube_mvp_scripted_palm`

**Goal:** verify the env builds, PPO converges to *something*, video
recording works end-to-end.

**Setup:**
- Finger action: 9-d residual atop CEM `best_finger_ctrl` (static offset),
  `init_noise_std=1.0` (PPO default).
- Palm: `ScriptedPalmAction` (settle 240 sim steps → lift_pz over 80
  sim steps).
- Rewards: tracking (finger_qpos / object_pos / object_quat) + contact
  rewards + lift_height + drop penalty.
- DR: none (cube spawned at keyframe).

**Result:** ≈ −9 mean reward after 20 iters. Cube spawn bug (cube
`object_pose_range.z = (lift_target, +0.02)`) made the cube spawn at
z=0.06–0.07 — fingers slammed shut on a *floating* cube, knocking it
across the workspace.

**Takeaway:** `LiftingCommand.object_pose_range` is the SPAWN range, not
the lift target. Fix: pin `z=(cube_size, cube_size)` so the cube sits on
the floor.

---

## Phase 2 — caged lift, `cube_full_lownoise` / `cube_full_warmstart_frozenstd`

**Goal:** match CEM in the RL env. CEM achieves +29.45 on
`Phase1GraspEvaluator` with a caged lift (cube wedged between proximal
phalanges + palm, not gripped by fingertips).

**Setup:**
- Static finger offset = CEM `best_finger_ctrl` (open-loop closing slams
  shut in one sim step at kp=4000).
- `init_noise_std=0.3`, then `--freeze-actor-std`, `--entropy-coef 0` for
  the `warmstart_frozenstd` variant.

**Result:**
- `cube_full_lownoise/iter120`: mean reward +9.8, peak cube z = 0.110 m,
  but `contact_min = 0` (fingertips never touched the cube). The cube
  floated mid-air, held by the proximal phalanges.
- `cube_full_warmstart_frozenstd/model_400`: mean +9.10, same caged
  behavior, no fingertip contact. Frozen std prevented exploration noise
  but didn't change the qualitative behavior.

**Takeaway:** CEM is structurally not a grasp — it's a wedge-lift.
Tracking CEM as a reference rewards the cage. To get a real grasp, the
finger setpoint has to *sweep* through contact, not jump to it.

---

## Phase 3 — MVP breakthrough, `cube_lerp_grasp`

**Goal:** real fingertip pinch grasp.

**Change (the one that mattered):** replaced the static finger offset
with `LerpFingerAction` — a time-varying setpoint that linearly
interpolates from `open_finger_qpos` (thumb mcp=3.14, index/middle
mcp=0) to `target_ctrl=best_finger_ctrl` over `settle_sim_steps=80`.
With a uniform sweep, the position controller crosses the cube smoothly
instead of teleporting fingers to the grip pose.

**Other supporting changes:**
- Open-pose convention fixed (thumb mcp=3.14 means *open*, not closed —
  this had silently been pushing the thumb across the palm at reset).
- Contact reward weights boosted (`contact_mean=10`, `contact_min=30`)
  so PPO can't ignore fingertip contact in favor of cage lift.
- Cube friction (2.4, 0.2, 0.02), mass 16 g — held constant.

**Result:** `cube_lerp_grasp/model_150`:
- Mean reward: +57.69
- `contact_min` reward sum: 24.81 (~59 % of max)
- Peak cube z: 0.069 m (CEM is 0.07 m, so we match within 1 mm)
- Drop rate: 0 % across 64 deterministic envs
- Visually: all 3 fingertips touch, real pinch grasp + lift

**Takeaway:** the entire MVP unlocked once the finger close became a
*sweep* rather than a step. PPO refines residuals on a working open-loop
controller; without the working open-loop, PPO can't find the basin.

---

## Phase 4 — domain randomization (cube spawn pose)

**Goal:** policy that grips a cube at unknown xy / yaw within the hand's
reachable envelope.

### Reachable region sweep

Before DR can be set, the reachable region must be measured. Open-loop
sweep (`/tmp/sweep_object.py`) over a 9×9 xy grid with the cube at each
cell records `contact_min` from the scripted (no-policy) rollout.

Per-object envelope (≥ 0.8 contact_min):

| Object | Reachable region | DR range |
|---|---|---|
| cube | x: [−20, +20] mm, y: [−5, +5] mm | x ±20mm, y ±5mm |
| prism (22×67×18 mm) | x: [0, +13] mm, y: [−19, +13] mm (centered +6, −3) | x ±6mm, y ±10mm offset to center |
| screwdriver_medium (vertical cylinder) | only the ~3.7 % near (0, 0) | x ±2mm, y ±2mm |

### `cube_dr_v1` / `v2` / `objpose` — DR from iter 0

**Setup:** full DR active from the very first iteration. Added
`object_pose_actual` obs (cube pose in palm frame) so the policy *can*
condition on the perturbation.

**Result:** peak mean reward +43 to +51 around iter 100–150, then
gradual decay through ~300 iters. Best checkpoints are pre-decay.

**Takeaway:** dropping the policy into the full DR distribution from
iter 0 lets it find a *coarse* policy quickly, but it never refines
because the noise floor of the DR distribution dominates the gradient
signal. The decay pattern was reproducible across three seeds (v1, v2,
objpose).

### `cube_dr_curriculum` — DR anneal

**Setup:** `dr_anneal_iters=200`. `Curriculum/dr_anneal` ramps the
spawn jitter linearly from 0 to (x ±20mm, y ±5mm, yaw ±0.52 rad) over
the first 200 PPO iters. Implementation: `anneal_cube_spawn_jitter`
curriculum term rewrites `LiftingCommand.object_pose_range` each step.

**Result:** `cube_dr_curriculum/model_200`:
- Mean reward +48.15 (vs +51 peak no-curriculum)
- `contact_min` reward sum 20.86 (~50 % of max)
- Median peak z 0.063 m, lift >6cm: 67 % across 64 deterministic envs
- Decay still present, but smoother peak

**Takeaway:** curriculum reduces variance and smooths the peak but does
not eliminate steady-state decay. The decay's root cause is upstream of
the schedule.

---

## Phase 5 — multi-object env

**Goal:** train one policy per object (cube, prism, screwdriver_medium)
using the same env scaffolding.

**Change:**
- `make_object_spec_from_frozen(xml, body_name, rename_to="cube")`
  extracts whatever object body lives in the frozen scene and exposes
  it under the canonical "cube" name to the rest of the env.
- Per-object DR center (`cube_spawn_x_center`, `cube_spawn_y_center`)
  to recenter the jitter window on each object's reachable region.
- Per-env `_spawn_pose()` helper replaced a buggy global cache that
  captured the spawn pose once on first call and never refreshed under
  DR. Symptom: `xy_drift` from "spawn" was actually drift from the
  *first ever episode*. Fix: snapshot per env at `episode_length_buf <= 1`.

**Results** (deterministic eval, 64 envs, full DR active):

| Tag | Median peak z | Lift > target | contact_min hold |
|---|---|---|---|
| `cube_dr_curriculum/model_200` | 0.063 m | 67 % (>6cm) | 0.93 |
| `prism_dr/model_150` | 0.059 m | 100 % (>4cm) | 0.91 |
| `screwdriver_vertical_dr/model_250` | 0.099 m | 100 % (>6cm) | 1.00 |

**Takeaway:** the same env machinery works across three objects with
minimal per-object tuning — only DR ranges and centers change.

---

## Phase 6 — stability rewards, `prism_stable`

**Goal:** stop grasps from being "loose" — cube/prism shifts a lot
during the squeeze, then drifts during lift.

**Change:** added three stability reward terms (all per-step,
ungated):
- `object_xy_drift` (weight −50): penalty per metre of cube xy
  displacement from spawn.
- `object_orientation_drift` (weight −20): per radian of quat geodesic
  drift from spawn orientation.
- `finger_drift_from_grip` (weight −15): per radian of finger qpos L2
  drift from the `LerpFinger` setpoint (the grip ctrl).

(Defaults: −3 / −3 / −2. The `prism_stable` run used 3–7× bumps.)

**Result:** `prism_stable/model_200` vs `prism_dr/model_150` (same DR):

| Metric | prism_dr | prism_stable | Δ |
|---|---|---|---|
| Median peak prism z | 0.058 m | 0.062 m | +4 mm |
| Lift success ≥6cm | 14 % | 80 % | +66 pp |
| contact_min hold | 0.97 | 0.94 | −3 pp |
| xy drift hold | 0.4 mm | 0.8 mm | +0.4 mm |
| orientation drift hold | 1.3° | 2.3° | +1° |

**Takeaway:** stability rewards markedly improved *lift success* (the
real bar) at a slight cost in contact persistence and drift metrics.
Drift went up because the stable policy holds a fully-lifted prism vs
the baseline that barely lifts it — drift is computed during the hold
window, which is now actually a hold instead of a flop.

---

## Phase 7 — open questions diagnosed (in progress)

Recent eval videos show two persistent issues:

1. **Grasps look like CEM ref playback**, regardless of cube position.
   The policy might be ignoring the cube-pose obs and outputting
   near-zero residuals.
2. **Objects still skew on contact**, and the squeeze looks "loose".

**Hypotheses being tested:**

| Hypothesis | Mechanism | Test |
|---|---|---|
| Policy is pose-invariant | `track_finger_ctrl_anchor` (w=1) penalizes deviation; `finger_residual_scale=0.2` clamps deviation small. PPO finds local optimum at action≈0. | `scripts/rl_diagnose_policy.py` — runs N envs at different cube poses, measures cross-env std of actions in contact window. |
| Linear close slams into contact | `LerpFinger` uses uniform alpha=t lerp. Fingers cross the air and contact at the same velocity. | `scripts/rl_train_cube.py --finger-close-easing ease_out_quad` — fast approach, slow contact. |
| Initial-contact perturbations are over-penalized | Stability rewards are ungated; they fire during approach. | `--contact-gate-stability-rewards` — drift penalties only fire once ≥ `--contact-gate-min` fraction of tips touch. |
| PPO has no sharp signal for slip during lift | Only sparse `object_drop` indicator. | `--enable-lift-terminations` — episodes terminate (GAE bootstrap cut) on slip / drop / tip-loss / finger-slip during lift hold phase. |

**Tooling added in Phase 7:**

- `scripts/rl_diagnose_policy.py` — pose-grid rollout; reports whether
  residuals vary across cube poses or sit at zero. Verdict labels:
  `POSE-INVARIANT` / `WEAKLY POSE-ADAPTIVE` / `POSE-ADAPTIVE`.
- `scripts/rl_plot_training.py` — multi-panel PNG per run (mean reward,
  lift, contact, drift, tracking, std, DR anneal, episode metrics) plus
  `_overlay.png` to compare runs head-to-head. Annotates the iter at
  which DR anneal hit 1.0.
- `scripts/rl_play_policy.py` — single-env continuous play across N
  episodes with random cube spawns each reset; one long video + a
  per-episode summary.

**Status:** all four hypotheses' fixes shipped, no full training run
yet. Next: train cube with `--finger-close-easing ease_out_quad
--contact-gate-stability-rewards --enable-lift-terminations
--finger-residual-scale 0.5 --object-xy-drift-weight -30
--object-orientation-drift-weight -20 --finger-drift-weight -10` and
compare to `cube_dr_curriculum/model_200`.

---

## Cross-phase lessons (the actually-load-bearing ones)

These are the ones that, if forgotten, will cost us days again:

1. **`LiftingCommand.object_pose_range` = spawn range, not target.**
2. **Thumb mcp `qpos=3.14` is OPEN**, not closed (only for thumb).
3. **Per-env spawn pose must be refreshed at `episode_length_buf<=1`** —
   a global cache is silently wrong under DR.
4. **CEM is a cage, not a grasp.** Any "track CEM" reward by itself
   rewards the cage. Need an explicit fingertip-contact reward, weighted
   high enough to dominate the lift reward.
5. **Finger close must be a sweep**, not a step. Static-offset finger
   action slams shut in 1 sim step at kp=4000.
6. **PPO's learnable std drifts up with `entropy_coef > 0`** when the
   optimum is near-deterministic. Pin with `--freeze-actor-std
   --entropy-coef 0`.
7. **DR curriculum doesn't fix steady-state decay**, only smooths the
   peak. Best checkpoints are still pre-decay.
8. **`reset_base` event** (default asset="robot") is what spreads
   parallel envs across env_origins. Don't remove it.
