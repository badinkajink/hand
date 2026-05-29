# Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   results/phase1/run18_final/foundational/cube/run_*/                    │
│     ├─ best_rollout.npz   (T × {qpos, qvel, cube_z, contacts}, 600 × …) │
│     ├─ summary.json       (best_finger_ctrl, eval_config, best_score)   │
│     └─ frozen_scene.xml   (rigid-morphology MJCF, self-contained)       │
│                                                                          │
│           ReferenceTrajectory  (src/morphohand/rl/reference_trajectory.py)
│                          │                                               │
│                          ▼                                               │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │  MorphoHandEnvCfg  (src/morphohand/rl/env_cfg.py)             │    │
│   │     ├─ entities                                                │    │
│   │     │     - robot (hand MjSpec, fixed-base mocap)             │    │
│   │     │     - cube  (4cm box, freejoint, init_state.pos=(0,0,0.02))
│   │     │     - terrain (mjlab plane at z=0)                      │    │
│   │     │                                                          │    │
│   │     ├─ actions  (9-d total — palm is scripted)                │    │
│   │     │     - finger_ctrl: LerpFingerAction(action_dim=9)       │    │
│   │     │         setpoint = lerp(open_finger_qpos,               │    │
│   │     │                          finger_default_ctrl,            │    │
│   │     │                          sim_step / finger_close_sim_steps)
│   │     │         + residual_scale=0.2 * policy_action            │    │
│   │     │     - palm_ctrl:   ScriptedPalmAction(action_dim=0)     │    │
│   │     │         palm_pz ramps by lift_delta_z over              │    │
│   │     │         lift_ramp_steps starting at settle_steps         │    │
│   │     │                                                          │    │
│   │     ├─ events (per-reset)                                     │    │
│   │     │     - reset_base:        writes robot mocap pose        │    │
│   │     │                          (REQUIRED for env_origin shift)│    │
│   │     │     - reset_cube:        writes cube freejoint qpos     │    │
│   │     │                          from default_root_state         │    │
│   │     │     - reset_robot_joints: zeros joint qpos/qvel         │    │
│   │     │                                                          │    │
│   │     ├─ commands                                               │    │
│   │     │     - lift_height (LiftingCommand)                      │    │
│   │     │         object_pose_range.z = (cube_size, cube_size)    │    │
│   │     │         — this is the CUBE SPAWN range, not the target  │    │
│   │     │                                                          │    │
│   │     ├─ observations (actor + critic, both = full)             │    │
│   │     │     - joint_pos_rel (15)                                │    │
│   │     │     - joint_vel_rel (15)                                │    │
│   │     │     - object_pos rel to palm (3)                        │    │
│   │     │     - ref_finger_qpos (9)  — from ReferenceTrajectory   │    │
│   │     │     - ref_object_pose (7)  — from ReferenceTrajectory   │    │
│   │     │     - last_action (9)                                   │    │
│   │     │     → 58-dim total (`obs_mode='full'`)                  │    │
│   │     │                                                          │    │
│   │     ├─ rewards (see table below)                              │    │
│   │     │                                                          │    │
│   │     └─ sensors                                                │    │
│   │           - fingertip_cube_contact: ContactSensor              │    │
│   │             {thumb_tip, index_tip, middle_tip} ↔ {cube}       │    │
│   │             reduce="none", num_slots=1                         │    │
│   └────────────────────────────────────────────────────────────────┘    │
│                          │                                               │
│                          ▼                                               │
│   mjlab.envs.ManagerBasedRlEnv  ←  mjlab/MuJoCo Warp (GPU envs ×N)      │
│                          │                                               │
│                          ▼                                               │
│   rsl_rl.OnPolicyRunner (PPO)                                            │
│                          │                                               │
│                          ▼                                               │
│   results/rl/<tag>/{checkpoints, tensorboard, eval_videos, rollouts}    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Module map (`src/morphohand/rl/`)

| File | Purpose |
|---|---|
| `reference_trajectory.py` | Load `best_rollout.npz`; sample qpos/palm/obj/finger at any continuous `t`. SLERP for quat. |
| `scene_loader.py` | Freeze a scene + resolve actuator/joint/body ids; mirrors Phase 1 ordering. |
| `actions.py` | Custom mjlab `ActionTerm`s: `LerpFingerAction` (time-varying setpoint + policy residual) and `ScriptedPalmAction` (zero-d, fully scripted palm lift). |
| `mjlab_terms.py` | RewTerm and ObsTerm adapters for mjlab's manager API. Wraps the pure functions in `reward.py` / `observations.py`. |
| `reward.py` | Pure-function reward terms (kept for offline scoring + tests). |
| `observations.py` | Pure-function obs extractors + composite assembler (kept for offline + tests; mjlab env uses simpler subset). |
| `env_cfg.py` | mjlab `ManagerBasedRlEnvCfg` assembly: entities, actions, obs, rewards, commands, events, sensors. |
| `ppo_config.py` | PPO hyperparameter dataclass. |
| `ppo_runner.py` | Build the mjlab env + RSL-RL config; dump `config.yaml`. |

## Reference-trajectory schema

`best_rollout.npz` columns:

| key | shape | meaning |
|---|---|---|
| `qpos` | `(T, nq)` | full state per sim step (object freejoint, palm, fingers) |
| `qvel` | `(T, nv)` | full velocity per sim step |
| `cube_z` | `(T,)` | object z (m) — convenience |
| `contacts` | `(T,)` | per-step fingertip-to-object contact count (CEM defines this) |
| `best_finger_ctrl` | `(9,)` | finger ctrl held **constant** across the rollout |

Default cube scene has `nq=22` (6 palm joints + 9 finger joints + 7 cube freejoint). Column indices are resolved at load time from `model.jnt_qposadr` — do **not** hardcode them.

> **The reference does not grasp.** mjwarp playback metrics on this rollout: `mean_contacts = 0`, `min_finger_contact_persistence = 0`, `mean_tip_distance = 0.007`. The cage-lift CEM found is what the reference tracks. To get fingertip contact you have to either (a) re-run CEM with a contact-min constraint, or (b) bypass the reference and shape rewards directly. The current pipeline does (b) — the `LerpFingerAction` open→close motion, combined with `contact_min` reward weight 30, produces real fingertip contact even though the *reference* doesn't.

## Action space (9-d total)

`LerpFingerAction` (`finger_ctrl`):

```
setpoint(t) = lerp(open_finger_qpos, finger_default_ctrl, clamp(sim_step/finger_close_sim_steps, 0, 1))
target      = setpoint(t) + residual_scale * policy_action
```

- `action_dim = 9` (one per finger joint).
- `residual_scale = 0.2` so even with `init_noise_std = 0.05` the policy noise on the actual ctrl is ~0.01 rad.
- `open_finger_qpos = (0, 3.14, 0,  0, 0, 0,  0, 0, 0)` — thumb mcp is the **3.14 end** of its range (inverted from index/middle which open at 0).

`ScriptedPalmAction` (`palm_ctrl`):

```
palm_px,py,rx,ry,rz = keyframe key_ctrl values  (constant)
palm_pz             = keyframe key_ctrl + lift_delta_z * clamp((sim_step - settle_steps + 1) / lift_ramp_steps, 0, 1)
```

- `action_dim = 0` — palm consumes no policy output.
- `settle_steps = 240`, `lift_ramp_steps = 80`, `lift_delta_z = 0.05` (matches `Phase1EvalConfig` defaults).
- Per-env `sim_step` counter resets on episode reset.

## Reward terms (current weights — `env_cfg.py`)

`T` = tracking `exp(−α‖·‖²)`; `L` = linear in clipped value; `P` = penalty.

| Term | Type | Weight | Notes |
|---|---|---|---|
| `track_finger_qpos` (α=20) | T | 4.0 | Tracks ref finger qpos (which is keyframe-constant) |
| `track_object_pos` (α=200) | T | 6.0 | Tracks ref object pos. Often **near 0** because alpha is tight and our cube is mid-air during lift |
| `track_object_quat` (α=10) | T | 2.0 | Geodesic quat tracking |
| `track_finger_ctrl_anchor` (α=4) | T | 1.0 | Penalises raw policy action magnitude (equivalent here since `default_offset = grip ctrl`) |
| `contact_mean` | L | **10.0** | Mean of per-tip contact flag — boosted from plan's 2.0 |
| `contact_min` | L | **30.0** | Min over tips of contact flag — boosted from 3.0, the main driver of 3-fingertip grips |
| `lift_height` | L | 80.0 | Cube z above settle_z, clipped to `lift_target_z_above_init` |
| `object_drop` | P (sparse) | -12.0 | 1 if cube fell `drop_threshold = 0.02` below settle_z |
| `object_xy_drift` | P | -3.0 | L2 of cube xy from initial xy |
| `fingertip_to_object` | P | **-3.0** | Sum of fingertip-to-cube distances — boosted from -0.5 for closer shaping |
| `action_rate_l2` | P | -0.005 | Standard PPO smoothness |
| `joint_pos_limits` | P | -2.0 | Soft penalty near joint limits |

Settle_z is captured per-env on the first call to `object_lift_height` / `object_drop_indicator` and cached on the env. Don't reset it between episodes within the same env — that would corrupt the drop indicator.

## Observation vector (58-d, `obs_mode = "full"`)

Per-step actor inputs:

| Term | dim | source |
|---|---|---|
| `joint_pos` (rel to default) | 15 | `velocity_mdp.joint_pos_rel` |
| `joint_vel` | 15 | `velocity_mdp.joint_vel_rel` |
| `object_pos` (palm → cube) | 3 | `manipulation_mdp.ee_to_object_distance` |
| `ref_finger_qpos` | 9 | `mjlab_terms.ref_finger_qpos` (lookahead from ReferenceTrajectory) |
| `ref_object_pose` | 7 | `mjlab_terms.ref_object_pose` (pos+quat) |
| `actions` (previous) | 9 | `velocity_mdp.last_action` |

Critic uses the same set (no asymmetric observations yet). Uniform noise on actor obs (corruption enabled).

> The pure-function `observations.py` defines a richer 72-dim vector (with contact features, time phase, lookahead). The mjlab env doesn't use it — that module is exercised only by unit tests. If you need a feature for training, add an `ObservationTermCfg` to `env_cfg.py`.

## Termination

| Term | Trigger | `time_out` flag |
|---|---|---|
| `time_out` | `episode_length_buf > episode_length_s / step_dt` (= 70 steps at 50 Hz / 1.4 s) | True |

No drop or escape termination is wired. The `object_drop` *reward* fires on drops but does not terminate the episode. Adding a drop termination is a known follow-up.

## Bug catalogue (don't repeat)

| Symptom | Root cause | Fix |
|---|---|---|
| Cube spawns at z=0.06, not z=0.02 | `LiftingCommand.object_pose_range.z` set to `(lift_target, lift_target+0.02)` — that's the cube SPAWN range, not the lift target | Set to `(cube_size, cube_size)` |
| Closing motion knocks cube sideways | Used `factor * grip_ctrl` as open pose; thumb mcp convention is inverted | Explicit `open_finger_qpos = (0, 3.14, 0, 0, 0, 0, 0, 0, 0)` |
| All 1024 parallel hands stack at world origin | Removed `reset_base` event thinking it was robot-only | Keep `reset_base` (it does env_origin shift for the mocap fixed-base robot) |
| `contact_min` stuck at 0 across all runs | CEM reference itself has zero fingertip contact (proximal-phalange cage lift); imitating it can't produce a grasp | LerpFingerAction (active closing motion) + boosted `contact_min`/`contact_mean` weights |
| Policy reward decays slowly (~5%/1000 iters) after peaking | PPO learnable std drifts up with entropy_coef > 0 | `--freeze-actor-std --entropy-coef 0` |
| Policy outputs noise that breaks the grip during finger closing | `init_noise_std` too high relative to position-controller kp | `init_noise_std ≤ 0.05` with `residual_scale = 0.2` on LerpFingerAction |
