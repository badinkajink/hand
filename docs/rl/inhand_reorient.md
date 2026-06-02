# In-Hand Reorientation (medium_flat → vertical)

How do we take the trained `screwdriver_medium_flat_short_proximal_stable_v1`
grasp policy and extend it to a *two-behavior* policy that: (1) picks up
the screwdriver lying flat on the ground, (2) lifts it, then (3) reorients
the cylinder from horizontal to vertical while maintaining grip?

This is harder than the lift-only tasks we've done — it requires the hand
to apply a coordinated finger + wrist motion to roll the cylinder ~90°
around a horizontal axis without dropping it.

## Problem decomposition

The base scene: `scene_screwdriver_medium_flat_short_proximal.xml`. Cylinder
(R=12.5 mm, L=8 cm) lying flat on the floor with body quat
≈(0.707, 0.707, 0, 0) — a 90° X-rotation that puts the cylinder's long axis
along world Y. Goal: cylinder's body-local +Z (long axis) aligned with
world +Z (vertical).

The current `medium_flat_stable_v1` policy outputs 9-dim residuals on the
finger setpoint. It cannot reorient the cylinder because:

1. The palm's rotation (rx, ry, rz) is scripted — held constant at the
   keyframe values throughout the episode.
2. Three smooth fingertips on a smooth cylinder have very limited
   rotational moment arm; even with finger motion, the cylinder mostly
   slips rather than rolls.

So the new capability we need is **policy-controlled wrist rotation**.

## Why not screwdriver_vertical_stable_v1?

That policy:
- Uses a different scene file (`scene_screwdriver_medium_vertical_short_proximal.xml`)
- Its CEM `best_finger_ctrl` is tuned for a vertically-standing cylinder.
  The fingers grip in a *different configuration* — typically tripod
  around the vertical axis rather than wrapped around a horizontal axis.
- It learned to pick a cylinder that was already vertical. Reusing it
  for *reorientation* would require the policy to transition through a
  flat-grip → vertical-grip swap mid-episode, which is a different
  capability than what was trained.

So vertical_stable_v1 is not a useful warmstart for this task.

## Architectural extensions

### 1. Action space: 9 → 12 dims

Extend `ScriptedPalmAction` (now used as `ScriptedPalmActionCfg`) to
optionally accept **3 policy residuals** on palm rx/ry/rz:
- `palm_rotation_residual_scale` (rad): typical 0.3 (~17°)
- `rotation_active_from_sim_step`: gates residuals to fire after the
  scripted lift completes, so warmstarted finger behavior runs cleanly
  during grasp/lift before wrist control kicks in.

Total action dim: 9 (finger residuals) + 3 (palm rotation residuals).

### 2. New reward: `target_axis_alignment`

```python
cos_theta = (R(quat) @ object_axis_local) · target_axis_world
reward = exp(-alpha * (1 - cos_theta)^2)
```

Defaults:
- `object_axis_local = (0, 0, 1)` (cylinder long axis, MuJoCo convention)
- `target_axis_world = (0, 0, 1)` (vertical)
- `alpha = 4` (reward is ~1.0 at 0°, ~0.37 at 60°, ~0.02 at 90°)

Gated by `reorient_start_step` (default 30) so the reward only fires
after the scripted lift ramp completes. Weight 50 makes it competitive
with the existing `lift_height` reward (weight 80).

### 3. New observation: `target_axis_misalignment`

The current-axis-to-target angle (rad), single scalar. Without an
explicit "how off" signal, the exp-shaped reward is too sparse for
gradient flow until the policy is already close to vertical.

### 4. Stricter termination

`enable_lift_terminations` + `strict_tip_lost_termination`: terminate
on a *single-step* tip loss during the lift/manipulation phase (the
default uses a 3-consecutive-step grace). Per the user's note: "probably
need to maintain contact the entire time, can terminate if we lose any
contacts."

### 5. Longer episode

`episode_length_s = 2.6` (was 1.4) — gives the policy ~70 PPO steps
after the scripted lift to actually reorient the cylinder.

## Bootstrap strategy: partial warmstart

`scripts/rl_train_cube.py` now does a *partial state_dict copy* when
`--init-actor-checkpoint` is set and the target action_dim ≠ source:

- MLP middle layers (shape unchanged): full copy.
- MLP input layer (input grew by `target_axis_misalignment` + extended
  `last_action`): partial copy of the leading `(out, source_in)` slice;
  new input cols at fresh init (small random weights for the new obs
  dims).
- MLP output layer (output grew by 3 for palm rotation): partial copy
  of `(source_out, in)`; new rows at fresh init. Critically, the
  output of the new rows is dominated by the bias term — initialized
  to **zero** — so on iter 0 the policy outputs ~0 palm rotation
  residuals = same behavior as the source policy.
- `distribution.std_param` (size grew by 3): leading 9 entries copied;
  trailing 3 at the default `init_noise_std` (we set 0.1 to avoid NaN).

This means iter 0 of the reorient policy reproduces medium_flat's
grasp+lift bit-for-bit, while the new palm rotation dims add small
exploration that the policy can learn to coordinate with the existing
finger control.

## NaN mitigation

Two distinct NaN failure modes hit during setup:

**(A) Finger residual std too high on small cylinders.** Empirically: with
`init_noise_std=0.3` (PPO default), random Gaussian finger residuals push
small-radius cylinders into degenerate physics states (penetration →
mjwarp NaN). At `init_noise_std=0.1`, 100 steps × 4096 envs survive
cleanly. Use 0.1 across both small_flat and reorient runs.

**(B) Palm rotation residuals during grip destabilize physics.** Even
small wrist rotations (`scale=0.3` ≈ 17°) applied while the fingers are
gripping the cylinder cause the leveraged contact to spike — by PPO step
30 (~6 steps after `palm_rotation_active_from_sim_step` default of 240
sim steps = 24 PPO steps), `obs` goes NaN on 36 of 1024 envs.

Confirmed isolation:
- `palm_rotation_residual_scale=0.3`, activate@step 24 → NaN at step 30
- Same config but `enable_palm_rotation_residual=False` → 80 steps clean
- `scale=0.1`, activate@step 50 → 120 steps clean

Final knob values for stability:
- `palm_rotation_residual_scale=0.1` (~5.7° max wrist delta per dim)
- `palm_rotation_active_from_sim_step=500` (PPO step 50 — well past the
  lift completion at PPO step 32, gives the grip time to settle before
  wrist motion begins)
- `reorient_start_step=50` (target_axis reward fires when the wrist can
  actually move)
- `init_noise_std=0.1`

## Will RL "figure it out"? Open questions

Honest answer: **uncertain.** This task has features that should make
it learnable:

1. The grasp+lift is already solved (warmstarted policy).
2. Cylinder + capsule fingertips + small palm rotation = mechanically
   reasonable to roll.
3. Dense reward signal: `target_axis_alignment` grows smoothly from
   ~0.02 (flat) to 1.0 (vertical).
4. Strict tip-lost termination + GAE bootstrap cut should give a
   sharp negative signal for any drop.

But also concerning:

1. The medium_flat finger ctrl (CEM-optimized for a flat grip) may
   force the fingers into a configuration where roll is hard. The
   policy may need to slip the cylinder through the grip and re-grasp
   in a different config — which contradicts "strict tip-lost
   termination."
2. The 3-finger morphology has limited dexterity vs. 5-fingered hands
   that handle in-hand reorientation in the literature
   (Shadow Hand, Allegro Hand).
3. With `strict_tip_lost_termination=True`, the policy gets cut off
   early — possibly before learning to recover from brief slips.

Network size is currently 512→256→128. May or may not be sufficient;
the existing grasp tasks all converged at this size. Bumping to
512→512→256 would require retraining from scratch (no warmstart) and
isn't justified until we see this size plateau.

## First run config

| Knob | Value | Why |
|---|---|---|
| `--num-envs` | 2048 | matches medium_flat (4 GB VRAM, 4 s/iter) |
| `--total-timesteps` | 100_000_000 | 2× the lift-only budget — harder task |
| `--init-noise-std` | 0.1 | NaN mitigation (mode A) |
| `--init-actor-checkpoint` | `medium_flat/model_500.pt` | warmstart grasp |
| `--enable-palm-rotation-residual` | on | wrist control |
| `--palm-rotation-residual-scale` | 0.1 | NaN mitigation (mode B); ~5.7° max delta |
| `--palm-rotation-active-from-sim-step` | 500 | NaN mitigation (mode B); rotation begins at PPO step 50 |
| `--enable-target-axis-reward` | on | reorientation signal |
| `--target-axis-weight` | 50.0 | competitive with lift reward |
| `--target-axis-alpha` | 4.0 | reward shaping sharpness |
| `--reorient-start-step` | 50 | matches palm activation step |
| `--strict-tip-lost-termination` | on | hard contact requirement |
| `--episode-length-s` | 2.6 | ~80 PPO steps post-rotation for reorient |
| `--lift-phase-start-step` | 40 | terminations engage after lift |

Expected wall-time: ~3-4 h on the RTX 4070 Ti at 2048 envs.

## Pre-planned follow-ups (if first run plateaus)

- **Lower alpha (softer shaping):** alpha=2 gives more gradient at
  large angles (reward 0.135 at 90° instead of 0.018).
- **Curriculum on target axis:** start with a 30° target tilt, ramp
  to 90° over training.
- **Larger network:** 512→512→256 + train from scratch (no warmstart).
- **Relax tip-lost:** allow 2-3 consecutive lost-steps so the policy
  can attempt a re-grip.
- **Random target axis:** train on multiple target axes (vertical,
  upside-down, sideways) to learn a general reorient skill.

## v2 / v3 / v4 outcomes (post-hoc, 2026-05-30 → 31)

The first-run config above (v1, 12-dim action with palm rotation) was
abandoned per user feedback: the palm should not rotate beyond its
scripted motion; all reorient must come from the 9 finger DOFs.
Subsequent runs are finger-only.

**v2 — plateaued, no rotation.** Root cause discovered later:
`object_orientation_drift_weight=-20` was actively penalizing any
rotation. Set to 0 in v3+.

**v3 — episodes died at step 40, reward gated at step 50, never fired.**
The lift-task stability terminations (`finger_slip` drift>0.3 rad,
`object_orientation_slip` drift>0.5 rad, `object_slip` xy>1.5 cm) are
fundamentally hostile to a reorient task and killed every episode
before reward could fire. `Episode_Reward/target_axis_alignment = 0.0`
for 2034 iters straight.

**v4 — mechanism fixed (disabled hostile terminations), partial success,
unintended floor-bracing emerged.** Episodes survive ~150 steps,
target_axis_progress went POSITIVE (+0.125 peak iter 941). The trained
policy reorients by **bracing the cylinder against the floor** — using
ground contact as a pivot to assist rolling. Creative but violates
in-hand intent. The lift target (0.05 m above init = effectively at
floor for a flat-laying 8 cm cylinder) gives the policy a usable floor
surface.

**v4 knob values that worked** (preserve for any rerun):

```
--term-finger-slip 100.0          (was 0.3 — disable)
--term-object-slip-xy 0.5         (was 0.015 — disable)
--term-object-slip-yaw 10.0       (was 0.5 — disable)
--reorient-start-step 35          (was 50 — fire reward pre-death)
--episode-length-s 8.0            (was 2.6)
--finger-drift-weight=-0.3        (was -2.0 — allow articulation)
--init-noise-std 0.15             (recover from std collapse)
--target-axis-weight 100.0
--target-axis-progress-weight 300.0
--target-axis-alpha-curriculum-iters 300
--target-axis-alpha-start 0.5
--target-axis-alpha 4.0
--contact-min-weight 15.0
--object-orientation-drift-weight=0   (CRITICAL — non-zero kills rotation)
--term-tip-lost-steps 10
```

**Next steps if revisiting (priority order):**

1. **Force lift-to-clearance before reorient.** Raise
   `--lift-target-z-above-init` to 0.10 m and add a "no-floor-contact"
   termination during reorient phase. Kills the floor-bracing
   strategy and forces true in-hand rotation.
2. Larger network from scratch (512→512→256, no warmstart, 200M+ steps).
3. Commit-bonus reward at `cos_theta > 0.5` to escape partial-rotation
   local optimum.
4. Target-axis curriculum (anneal goal from 30° → 90°, separate from
   alpha basin curriculum).

Run dirs: `results/rl/20260530-2159-inhand_reorient_v4/` (best so far),
best checkpoint `model_950.pt` (iter 941 peak). The final eval video
(`step-48000.mp4`) shows the floor-bracing behavior — informative even
if unintended.
