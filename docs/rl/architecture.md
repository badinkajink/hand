# Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│   results/phase1/run18_final/foundational/cube/run_*/                    │
│     ├─ best_rollout.npz   (T × {qpos, qvel, cube_z, contacts}, 600 × …) │
│     ├─ summary.json       (best_finger_ctrl, eval_config, best_score)  │
│     └─ frozen_scene.xml   (rigid-morphology MJCF, self-contained)       │
│                                                                          │
│                                  │                                       │
│            ReferenceTrajectory  ◄┘  (src/morphohand/rl/reference_trajectory.py)
│             │                                                            │
│             ▼                                                            │
│   ┌────────────────────────────────────────────────────────────────┐    │
│   │  MorphoHandEnvCfg   (src/morphohand/rl/env_cfg.py)            │    │
│   │     ├─ entities: robot (hand MjSpec) + cube (MjSpec)          │    │
│   │     ├─ actions:  finger_ctrl (9d) + palm_ctrl (6d residual)   │    │
│   │     ├─ obs:      joint_pos/vel + ee_to_obj + last_action      │    │
│   │     ├─ rewards:  lift + lift_precise + action_rate + limits   │    │
│   │     │            (Opt2Skill tracking terms = follow-up patch) │    │
│   │     ├─ commands: LiftingCommand (sample target object pose)   │    │
│   │     └─ curriculum: tracking-vs-task anneal (planned)          │    │
│   └────────────────────────────────────────────────────────────────┘    │
│             │                                                            │
│             ▼                                                            │
│   mjlab.envs.ManagerBasedRlEnv  ←  mjlab/MuJoCo Warp (GPU envs ×N)      │
│             │                                                            │
│             ▼                                                            │
│   rsl_rl.OnPolicyRunner (PPO)                                            │
│             │                                                            │
│             ▼                                                            │
│   results/rl/<tag>/{checkpoints, tensorboard, eval_videos, rollouts}    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Module map (`src/morphohand/rl/`)

| File | Purpose | Depends on |
|---|---|---|
| `reference_trajectory.py` | Load `best_rollout.npz`; sample qpos/palm/obj/finger at any continuous `t`. SLERP for quat. | `numpy`, `mujoco` (for `jnt_qposadr`) |
| `scene_loader.py` | Freeze a scene + resolve actuator/joint/body ids; mirror Phase1 ordering. | `mujoco`, `morphohand.sampling.scene` |
| `reward.py` | Pure-function reward terms (tracking + task + penalties). Composite scorer for tests. | `numpy` |
| `observations.py` | Pure-function obs extractors + composite assembler. | `numpy` |
| `env_cfg.py` | mjlab `ManagerBasedRlEnvCfg` assembly: entities, actions, obs, rewards, commands. | `mjlab` |
| `ppo_config.py` | PPO hyperparameter dataclass. | none |
| `ppo_runner.py` | Build the mjlab env + RSL-RL config; dump `config.yaml`. | `mjlab`, `yaml` |

## Reference-trajectory schema

`best_rollout.npz` columns:

| key | shape | meaning |
|---|---|---|
| `qpos` | `(T, nq)` | full state per sim step (object freejoint, palm, fingers) |
| `qvel` | `(T, nv)` | full velocity per sim step |
| `cube_z` | `(T,)` | object z (m) — convenience column |
| `contacts` | `(T,)` | per-step fingertip-to-object contact count |
| `best_finger_ctrl` | `(9,)` | finger ctrl held constant across the rollout |

The default cube scene has `nq=22` (7 freejoint + 6 palm + 9 finger).
Column indices are resolved at load time from `model.jnt_qposadr` — do
**not** hardcode them.

## Reward terms (Phase 3 plan table)

`T` = tracking `exp(−α‖·‖²)`; `P` = clipped penalty; `L` = linear.

| Term | Type | Default `w` | Source |
|---|---|---|---|
| `track_finger_qpos` (9d, α=20) | T | 4.0 | Opt2Skill |
| `track_object_pos` (3d, α=200) | T | 6.0 | Opt2Skill |
| `track_object_quat` (geodesic, α=10) | T | 2.0 | Opt2Skill |
| `track_finger_ctrl_anchor` (α=4) | T | 1.0 | inherits `Phase1EvalConfig.finger_ctrl_anchor` |
| `task_contact_persistence` (α=1) | T | 0.8 | inherits |
| `task_min_finger_persist` (α=1) | T | 2.0 | inherits |
| `task_lift_height` | L | 35.0 | inherits |
| `task_distance_palm_to_obj` (α=15) | T | 2.0 | inherits |
| `pen_object_velocity` | P | 0.15 | inherits |
| `pen_xy_drift` | P | 6.0 | inherits |
| `pen_drop` | P (sparse) | 12.0 | inherits |
| `pen_finger_yaw_drift` | P | 0.8 | inherits |
| `pen_finger_flex_drift` | P | 0.4 | inherits |
| `pen_cube_yaw_drift` | P | 4.0 | inherits |
| `pen_cube_axis_tilt` | P | 6.0 | inherits |
| `pen_cube_ang_drift` | P | 2.0 | inherits |
| `pen_action_rate` | P | 0.05 | new for RL |
| `pen_action_l2` | P | 0.01 | new for RL |
| `bonus_alive` | const | 0.1 | new for RL |

Defaults live in `morphohand.rl.reward.DEFAULT_REWARD_WEIGHTS`. The
composite scorer `compute_total_reward(state)` runs every term and
returns `(total, per_term_dict)` — used by `test_rl_reward_on_reference`
to assert the open-loop reference replay scores near the maximum.

## Observation vector (~85 dim)

```
[ finger_joint_pos                  9   range-normalized
  finger_joint_vel                  9   clip ±10
  palm_joint_pos                    6   range-normalized
  object_pos rel to palm            3   clip ±5
  object_quat (wxyz)                4
  object_lin_vel                    3   clip ±5
  object_ang_vel                    3   clip ±5
  fingertip features (3 tips ×3)    9   {distance, contact_flag, normal_z}
  time_phase                        1   t / episode_length
  previous_action                   9   clip ±1.1
  finger_qpos_ref @ t+50ms          9
  object_pose_ref @ t+50ms          7   pos + quat
] -> 72-dim
```

`OBS_DIM` in `observations.py` is the source of truth.

## Action space

- 9 finger ctrls, normalized `[-1, 1]`, rescaled to per-actuator
  `(ctrl_lo, ctrl_hi)` from the frozen MJCF.
- 6 palm ctrls with `scale=0.1` (residual policy by default).
  See [open question in `index.md`](index.md).

## Termination

- `episode_length_s` reached (default 1.4 s ≈ settle 0.48 + lift 0.44 + hold 0.28 + slack).
- Drop: `object_z < settle_z − 0.02` (configured via `pen_drop`).
- Object escape: object exits a 0.5 m sphere around the palm base
  (configured via mjlab built-in termination — wiring TBD).
