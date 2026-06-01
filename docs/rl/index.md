# Reinforcement Learning Pipeline

## Why

Phase 1 grasp synthesis via CEM has hit a quality ceiling on the harder
objects in `run18_final` (drill, small flat screwdriver) and is generally
inefficient. The next move is a **learning-based policy** that can match
— and ideally exceed — the per-task CEM scores on the *easier* end of the
run18 distribution first, providing a foothold for later cross-morphology
transfer and in-hand manipulation work.

## Status

| Milestone | Status | Notes |
|---|---|---|
| MVP env wired (mjlab + PPO + warm-start) | ✅ | `cube_lerp_grasp` |
| Replicate CEM caged-lift in RL env | ✅ | `cube_full_warmstart_frozenstd/model_400` |
| Replicate CEM lift with **fingertip contact** | ✅ | `cube_lerp_grasp/model_150` — `contact_min` reward 24.81 (~59% of max); peak cube z 0.069 m |
| Robust pickup under cube pose noise | ✅ | `cube_stable_v1/model_1400` — 100% lift ≥6cm, contact_min hold 0.999, sub-mm xy drift, 1.3° orient drift |
| Stable recipe ported to other objects | ✅ | `prism_stable_v1/model_800` (98 % lift ≥6cm, 0.45mm drift) + `screwdriver_vertical_stable_v1/model_550` (100 % lift ≥6cm, 0.37mm drift) |
| Cross-morphology transfer | 🚧 | see [Multi-Morphology](multimorphology.md) — eval-and-finetune approach in progress |
| In-hand reorientation (flat → vertical) | 🚧 | see [Reorientation](reorientation.md) — Policy B (two-policy split) achieves +0.253 target_axis_progress vs v4's +0.125 peak; sim-to-real smoothness work remaining |
| Robust pickup under friction / mass noise | 🔲 | deferred — needs custom raw-warp event |
| New objects (drill, more screwdrivers) | 🔲 | gated on per-object foundational CEM (drill blocked on keyframe issue) |

Reading order:
1. [Architecture](architecture.md) — modules, env composition, what each
   ActionTerm / RewardTerm does, the reference-trajectory loader.
2. [Training](training.md) — env setup with `uv`, launch command, all CLI
   flags, troubleshooting.
3. [Results](results.md) — scoreboard, eval videos, deterministic rollout
   stats per checkpoint.
4. [Phases](phases.md) — chronological log of the grasp work
   (Phase 1 MVP → Phase 9 stable recipe).
5. [Reorientation](reorientation.md) — chronological log of the in-hand
   reorientation work (v2 → v3 → v4 floor-bracing → v5 → Policy B).

## What we actually learned getting to MVP

Most of the time spent on the cube task went into env-setup bugs that
looked like RL convergence issues. Future work should not repeat them:

1. **CEM doesn't actually grasp.** The CEM `best_finger_ctrl` produces a
   *caged* lift: the cube is held by the proximal phalanges and palm,
   not by the fingertips. Playback metrics on mjwarp confirm
   `mean_contacts = 0` for the whole rollout — fingertips never touch
   the cube. Any "track CEM reference" reward structurally rewards the
   cage, not a fingertip grasp.
2. **`LiftingCommand.object_pose_range.z` is the cube SPAWN range**, not
   the target. Setting it to `(0.05, 0.07)` (the lift target) writes the
   cube to z=0.05–0.07 on every reset, making it look like it floats
   above the floor. Set this to `(cube_size, cube_size)` to spawn on the
   floor.
3. **Thumb mcp convention is inverted** from index/middle. Open thumb is
   `qpos = 3.14`; open index/middle is `qpos = 0`. A naive
   `factor * grip_ctrl` scaling sets the thumb closed across the palm,
   knocking the cube sideways before fingers close around it. See
   `MorphoHandEnvCfg.open_finger_qpos` for the canonical open pose.
4. **`reset_base` event (default asset = "robot") is what shifts the
   hand per env_origin.** The robot is fixed-base mocap; that event
   writes its world pose. Remove it and all parallel envs collapse onto
   the same world coords.
5. **PPO's learnable std drifts upward** with `entropy_coef > 0`. For
   tasks where the optimum is near-deterministic (action ≈ 0 reproduces
   the open-loop scripted behavior), this causes slow policy decay over
   ~1000 iters. Pin via `--freeze-actor-std` + `--entropy-coef 0`.
6. **Open-loop scripted behavior is most of the win.** With a
   time-varying finger setpoint (open → grip via `LerpFingerAction`) and
   scripted palm lift (`ScriptedPalmAction`), zero policy action already
   achieves `contact_min ≈ 1.0` and lifts the cube to z = 0.069. PPO
   only needs to refine residuals. This shapes how to think about the
   policy: it's a corrective layer on top of a working open-loop
   controller, not a from-scratch grasp solver.

## Resolved open question

**Palm: scripted vs learned.** The original plan was ambiguous — the
first implementation tried a learnable palm residual at `scale=0.1`. In
the final implementation, the palm is fully scripted via
`ScriptedPalmAction` (`action_dim = 0`): px/py/rx/ry/rz hold at the
keyframe ctrl, palm_pz ramps by `lift_delta_z` over `lift_ramp_steps`
substeps starting at `settle_steps`. The policy has zero control over
palm. This isolates credit-assignment to the 9 finger ctrls, which is
where the grasp behavior actually lives.

## Out of scope (do **not** expand without a new conversation)

- Cross-morphology transfer / Pollard-style fine-tuning across the run18
  set — gated on robustness work below.
- In-hand reorientation, pivot, regrasping.
- Multi-task RL (drill, screwdrivers). Drill is also blocked by the
  keyframe issue documented in the CEM phase results.
- Asymmetric actor-critic, DAgger/AWAC, RL-from-CEM distillation.
