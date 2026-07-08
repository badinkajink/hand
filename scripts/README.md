# scripts/ — active tooling map

36 active scripts (2026-07-08 consolidation; ~90 superseded ones live in `archive/` with their
own README). Everything runs via `uv run --extra rl --extra gpu python scripts/<x>.py`
(`MUJOCO_GL=egl` headless; every Warp process needs its own `WARP_CACHE_PATH=$(mktemp -d)` —
the `.sh` launchers handle both).

## Morphology + grasp synthesis (phase 1)

| script | role |
|---|---|
| `generate_morphology_xml.py` | bake a 9-param finger design into fixed scene geometry (frozen morphology) |
| `retarget_keyframe_ik.py` | IK-retarget the grasp keyframe across morphologies (world-frame fingertips → `open_ik` keyframe). REQUIRED before CEM/RL on a new design |
| `phase1_optimize_grasp.py` | CEM grasp synthesis on a (frozen) scene; writes `best_rollout.npz` + `frozen_scene.xml` — the "morphology run dir" every RL script consumes |
| `generate_primitive_meshes.py`, `build_morphohand_urdf.py` | asset generators (meshes; URDF for GraspGenX) |

## RL training (A = lift/deliver, B = reorient)

| script | role |
|---|---|
| `rl_train_cube.py` | THE trainer (PPO via mjlab/rsl_rl). All env knobs = CLI flags (`--compliance-dr`, `--imitation-ref-npz`, `--live-a-checkpoint`, …) |
| `train_A_on_morph.sh` | launcher: Policy A on a morphology run dir (from scratch per design — never warmstart A), baked-in collapse watchdog |
| `train_handoff_liveA_reset.sh` | launcher: Policy B via the LIVE-A reset (frozen A drives 0..onset organically; pre-onset PPO-masked). The validated B recipe |
| `train_reorient_on_morph.sh` | launcher: standalone reorient B on a new morphology |
| `train_corefine_BtoA.sh` | launcher: B→A co-refinement (train A on B's downstream reorient reward) |

## Evaluation + diagnostics (judge on these, never on reward sums)

| script | role |
|---|---|
| `rl_demo_handoff_continuous.py` | THE deploy eval: continuous A→B handoff, one env, video + trajectory-health scorecard + per-finger forces; `--record-fingertip-traj` dumps the imitation reference |
| `policy_healthcheck.py` | standalone trajectory-health scorecard on any policy |
| `rl_eval_reorient_metrics.py` | deterministic reorient metrics (held-cos etc.) from a run dir |
| `probe_grip_balance.py`, `probe_grip_force.py` | per-finger force probes (degenerate-pinch detection) |
| `rl_eval_object.py`, `rl_play_policy.py`, `rl_diagnose_policy.py` | pose-grid eval / interactive playback / obs-action dump |
| `rl_plot_training.py`, `rl_plot_reorient.py`, `rl_render_reorient.py` | training curves / reorient plots / rollout render |
| `rl_record_handoff_states.py`, `rl_record_initiation_bank.py`, `rl_record_reorient_schedule.py` | state-bank / schedule recorders (consumed by rl_train_cube flags) |

## Studies + pipelines (resumable, detached; logs → `logs/`, summaries → `docs/experiments/`)

| script | role |
|---|---|
| `morph_pipeline_sweep.py` (+`_plots`) | co-design sweep: per design XML→IK→CEM→A→B→handoff eval, health-gated. Runbook: `docs/rl/morph_sweep_STATUS.md` |
| `morph_landscape_sweep.py` (+`_plots`) | earlier grasp-only morphology landscape (kept: source of landscape figures) |
| `ik_recem_landscape.py` | IK-retarget + re-CEM over the landscape designs |
| `reorient_variance_study.py` | seed-variance study: fixed-A × {self, shared, imit} B warm-starts |
| `compliance_robustness_sweep.py` | eval-only: fixed policies × solimp range → compliance-response curves |
| `compliance_dr_pipeline.py` | compliance-DR retrain: DR-A → 2× imitation-B → robustness sweep |

## GraspGenX (zero-shot grasp baseline, branch `graspgenx-morphology-eval`)

`graspgenx_make_morphohand.py` (URDF+config), `graspgenx_eval_phase1.py`, `graspgenx_view_grasps.py`.

## Conventions

- Results registry: `rename_results_bids.sh` is the single source of truth for aNN/bNN policy IDs.
- Launch long jobs detached: `nohup setsid bash scripts/<x>.sh > logs/<x>.run.log 2>&1 </dev/null & disown`.
- Reference policies: **a10** (m05 lift) → **b33** (m05 reorient).
