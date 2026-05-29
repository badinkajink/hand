# Training

## Environment setup

All deps are managed via `uv`. The repository has three install profiles:

```bash
# baseline CPU-only — runs existing Phase 1 CEM scripts
uv sync

# adds MuJoCo Warp + ComFree Warp (editable from external/) + jax CUDA
uv sync --extra gpu

# adds mjlab + RSL-RL + torch (cu128 wheels) + gymnasium for RL training
uv sync --extra gpu --extra rl

# add dev tools (pytest, ruff, mypy, mkdocs)
uv sync --extra dev
```

### Verify the install

```bash
uv run python -c "
import torch, mjlab, rsl_rl, mujoco_warp, comfree_warp
print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())
print('mjlab:', mjlab.__file__)
print('mujoco_warp:', mujoco_warp.__version__)
"
```

Expected: `cuda: True` + a GPU name, mujoco_warp `3.6.0` (editable),
torch `2.11.0+cu128`.

## Launching the canonical run

The MVP that reproduces the CEM lift with **fingertip contact**:

```bash
uv run python scripts/rl_train_cube.py \
    --morphology-run results/phase1/run18_final/foundational/cube/run_20260521_161817 \
    --tag cube_lerp_grasp \
    --num-envs 1024 \
    --init-noise-std 0.05 \
    --entropy-coef 0.0 \
    --freeze-actor-std \
    --total-timesteps 50000000 \
    --wandb-project morphohand-rl \
    --eval-video-interval 25 \
    --eval-video-length 70
```

~2 hr on RTX 4070 Ti Super. Peak reward ~57 by iter 125; rest of the run
is just stability checking.

## CLI flags on `scripts/rl_train_cube.py`

| Flag | Default | Role |
|---|---|---|
| `--morphology-run PATH` | required | Phase 1 foundational dir (provides best_rollout, summary.json, frozen_scene.xml) |
| `--tag STR` | `cube_mvp_v1` | output subdir under `results/rl/` |
| `--num-envs INT` | 1024 | parallel envs; drop to 256 on <16 GB VRAM |
| `--seed INT` | 42 | |
| `--wandb` / `--no-wandb` | enabled | wandb logging vs tensorboard-only |
| `--wandb-project STR` | `morphohand-rl` | |
| `--wandb-tags STR ...` | `()` | |
| `--upload-model` / `--no-upload-model` | off | push checkpoints to wandb |
| `--record-videos` / `--no-record-videos` | enabled | eval video recording during training |
| `--eval-video-interval INT` | 50 | iters between eval videos |
| `--eval-video-length INT` | 70 | frames per eval video |
| `--init-noise-std FLOAT` | None | overrides PPO `init_noise_std` (default 0.3 in PPOConfig; **set to 0.01–0.05 for cube**) |
| `--entropy-coef FLOAT` | None | overrides PPO entropy bonus (**set to 0 for cube** — the optimum is near-deterministic) |
| `--total-timesteps INT` | None | overrides PPOConfig (default 200M) |
| `--lift-target-z-above-init FLOAT` | 0.05 | clip ceiling for `lift_height` reward |
| `--reward-mode {full,tracking_only}` | full | `tracking_only` zeros all task rewards |
| `--obs-mode {full,ref_only}` | full | `ref_only` is just `ref_finger_qpos` (9-d) |
| `--init-actor-checkpoint PATH` | None | warm-start actor weights from a prior `model_*.pt` |
| `--freeze-actor-std` | off | `requires_grad=False` on actor's `distribution.std_param` — pins exploration noise |

For the cube task the trio `--init-noise-std 0.05 --entropy-coef 0.0
--freeze-actor-std` is what stops the slow-decay pathology where PPO's
learnable std drifts upward and eventually breaks the grip.

## Key `MorphoHandEnvCfg` fields

These don't have CLI flags yet; edit `src/morphohand/rl/env_cfg.py` or
construct a custom cfg in your launcher:

| Field | Default | Role |
|---|---|---|
| `frozen_scene_xml` | required | frozen MJCF (must include `keyframe_name`) |
| `keyframe_name` | `"open_short_manual"` | provides palm pose + finger keyframe + cube initial qpos |
| `foundational_run_dir` | required | for ReferenceTrajectory |
| `finger_default_ctrl` | run18 cube grip ctrl | position-controller target at the end of the lerp; passed by `rl_train_cube.py` from the npz |
| `num_envs` | 1024 | |
| `sim_timestep` | 0.002 (500 Hz) | |
| `decimation` | 10 → 50 Hz policy | |
| `episode_length_s` | 1.4 → 70 policy steps | |
| `object_size` | 0.02 | cube half-extent |
| `object_mass` | 0.016 | density 500 → kg |
| `object_friction` | (2.4, 0.2, 0.02) | |
| `lift_target_z_above_init` | 0.05 | |
| `settle_steps` | 240 | sim steps before palm starts lifting |
| `lift_ramp_steps` | 80 | duration of palm_pz ramp |
| `lift_delta_z` | 0.05 | total palm_pz rise |
| `finger_close_sim_steps` | 80 | duration of the open→grip finger lerp; ≤ settle_steps |
| `open_finger_qpos` | `(0, 3.14, 0, 0, 0, 0, 0, 0, 0)` | open-hand pose; thumb mcp is inverted |
| `reward_mode` | `"full"` | |
| `obs_mode` | `"full"` | |
| `viewer_distance` | 0.6 | viewer cam zoom |

## Outputs

```
results/rl/<tag>/
  config.yaml          # MorphoHandEnvCfg + PPOConfig snapshot
  rsl_rl_cfg.json      # RSL-RL hyperparameters
  tensorboard/
    model_<iter>.pt    # save_interval = 50 iters
    events.out.tfevents.*
  eval_videos/
    <tag>-step-<N>.mp4 # one per eval_video_interval iters (stochastic policy)
```

## Bring-up checklist

1. **Dry-run** — `--dry-run` produces `config.yaml` + `rsl_rl_cfg.json`. Skips PPO launch.
2. **Pre-flight zero-action sanity** — run `/tmp/preflight_lerp.py` (or equivalent: 16 envs, zero action, 70 steps). Expect cube z=0.02 at step 0, `contact_min` reaching 1.0 by step 15, cube z=0.069 at step 60. If contact_min stays 0 or cube floats off the floor, the env is broken — fix before training. The most common breakage is one of the entries in the bug catalogue in `architecture.md`.
3. **PPO actually steps** — tensorboard shows `iter > 0`. If it dies on step 0, usually VRAM OOM (drop `--num-envs`).
4. **Reward signal is real** — `Mean reward` should rise to ~50+ within the first 25 iters (well above the `bonus_alive = 0.1 × 70 ≈ 7` floor). If it stays under 15, the contact rewards aren't firing — re-run pre-flight to find the env bug.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `CUDA initialization: driver too old` | torch installed cu130 wheels but driver is 12.8 | confirm `pyproject.toml [tool.uv.sources]` routes torch through `pytorch-cu128`; re-sync |
| `actuator '...' not found` | scene_loader's `FINGER_ACTUATOR_NAMES` ≠ scene XML | edit one to match |
| `VRAM OOM` at env init | `num_envs` too high | drop to 512 or 256 |
| Cube spawns at z=0.06, fingers never grip | `LiftingCommand.object_pose_range.z` set to lift target instead of spawn | set to `(cube_size, cube_size)` |
| Closing motion sweeps cube sideways | thumb mcp set to qpos=0 (closed across palm) | use `open_finger_qpos = (0, 3.14, 0, …)` |
| All parallel hands stacked at world origin | `reset_base` event was removed | keep `reset_base` (mocap shift) |
| Mean reward decays slowly after peaking | learnable std drifting up | `--freeze-actor-std --entropy-coef 0` |
| `contact_min` stuck at 0 even with grip | grip ctrl positions fingertips past the cube (cage, not pinch) | use bigger cube OR LerpFingerAction with `open_finger_qpos` — see architecture.md |
| `Mean reward` huge but cube doesn't lift in video | the cube is being lifted but the contact sensor is firing on `body=cube` against the wrong body — check `ContactSensorCfg.secondary.pattern` |

## Quick deterministic eval

```bash
# minimal eval — replace CHECKPOINT and tag, runs 64 envs det, saves env-0 video
python /tmp/eval_model400.py
```

The template script is at `/tmp/eval_model400.py` (treat as starting
point; eventually live in `scripts/rl_eval_cube.py`).
