# scripts/ — active tooling map

36 active scripts (2026-07-08 consolidation; ~90 superseded ones live in `archive/` with their
own README). Everything runs via `uv run --extra rl --extra gpu python scripts/<x>.py`
(`MUJOCO_GL=egl` headless; every Warp process needs its own `WARP_CACHE_PATH=$(mktemp -d)` —
the `.sh` launchers handle both).

## Morphology + grasp synthesis (phase 1)

| script | role |
|---|---|
| `generate_morphology_xml.py` | bake a 9-param finger design into fixed scene geometry (frozen morphology). Needs a hand/scene PAIR — baseline `hand.xml`, perp `perp_hand.xml`; never a `*_morphology_actuated.xml` |
| `morph_selfcollision_gate.py` | is a design PHYSICALLY REAL? bake → settle → ask the contact solver. The morph ranges are mount rails and the rails run through the palm; nothing else checks. Always `--retarget` |
| `retarget_keyframe_ik.py` | IK-retarget the grasp keyframe across morphologies (world-frame fingertips → `open_ik` keyframe). REQUIRED before CEM/RL on a new design |
| `phase1_optimize_grasp.py` | CEM grasp synthesis on a (frozen) scene; writes `best_rollout.npz` + `frozen_scene.xml` — the "morphology run dir" every RL script consumes |
| `generate_primitive_meshes.py`, `build_morphohand_urdf.py` | asset generators (meshes; URDF for GraspGenX) |

## The `real_v1` hardware hand (2026-08-27)

The CAD-matched hand: 33.45/33.45/37.16 mm links that OVERLAP by 12.70 mm, CAD ROM, and real XY
gantry travel. It is a separate topology from the m05/baseline lineage and **nothing transfers
onto it** — a10/b33 were trained on a 117 mm finger with coincident yaw/MCP axes.

| script | role |
|---|---|
| `build_real_v1_scenes.py` | emit the base pair + actuated explorer from the CAD spec. `--check` fails if the shipped MJCFs drifted; `--fit-palm-z` prints the palm-height utilisation table |
| `probe_real_v1_carry.py` | the reorient mechanism on `real_v1`: drives the shaft from horizontal to vertical OPEN-LOOP by rotating the contacts as a rigid body about a raised pivot, and reports the per-finger radial slack (`--workspace`) that decides whether a design can do it at all |
| `probe_real_v1_slip.py` | why a `real_v1` grip that should work does not: pad elevation, compliance and friction on one axis, held far longer than an RL episode |
| `probe_real_v1_vertical_hold.py` | can the hand REACH the vertical hold, and is that pose inside `finger_residual_scale` of the CEM grip |
| `probe_real_v1_pivot.py` | breakaway torque about the pinch axis vs straddle and grip offset — the grasp as a rotational lock |
| `fit_real_v1_pose.py` | per-design grasp keyframe: solves palm x/y/z AND the 9 finger angles onto fingertip targets ringing the shaft, writes `open_ik`. Replaces `retarget_keyframe_ik.py` here — new topology, and palm height is not shared across designs |
| `real_v1_pipeline.py` | the per-design chain: generate → pose → CEM → A → B → handoff eval. Resumable, stage-selectable (`--stages grasp` is the cheap prefix) |
| `real_v1_deploy_envelope.py` | how much perturbation the open-loop carry survives: one-at-a-time axes, everything-wrong-at-once ensembles, and the cell sweep that picks a design's operating point |
| `real_v1_export_plan.py` | a chosen cell → bench sheets: mounts, a 50 Hz joint trajectory, the four set-points it really is, and `<design>_plan.json` for the driver |
| `real_v1_hand_commands.py` | the sim design space against what the gantries actually reach, and an exported plan → literal `MOVEMM`/servo commands. `--travel` re-asks the question under a hypothetical rail length. Driver half = `manta_hand.plan` |
| `replay_real_v1_hardware_log.py` | replay a control-station JSONL command stream in its exported MuJoCo scene; writes an NPZ plus object-motion and endpoint servo-tracking summary. This replays commands, not unmeasured real object motion |
| `real_v1_trajectory_clearance.py` | finger-to-finger clearance ALONG a plan's path — the gate nobody had. `morph_selfcollision_gate.py` checks mount rails at one static pose; this one walks the trajectory. Trust `d.contact`, not `mj_geomDistance`: it returns exactly 0.0 on some box-box pairs |

### On the bench (workstation → CB1, in this order)

**Run a whole sitting through `real_v1_bench_session.py`** — it calls the three scripts below
in order, picks the clearance-safe truncation for the design, sets per-finger grip targets
from the plan's own driver/holder split, and captures the operator observations that no log
contains (there is no object-pose sensor on this hand). Protocol:
`docs/experiments/20260830-real_v1_bench_suite/README.md`.

| script | role |
|---|---|
| `real_v1_bench_session.py` | 0. the session driver — one design, one directory, self-describing |
| `real_v1_bench_report.py` | 4. read the sessions back: driver-yaw loaded vs free-air (= grip load), and the load-200 overload plateau. Excludes runs the operator flagged `slipped` (a shaft that turned because the grasp released is the opposite result) and whole sessions carrying an `EXCLUDED.txt` |
| `probe_hold_convergence.py` | does a plan's "held" verdict survive a longer hold? Re-runs saved plans at several `--hold-steps` and at several hold-phase force targets. Built because the Sobol-128 screen measured 1.6 s after the turn and every finalist drops by 3.2-8.0 s — `docs/experiments/20260830-real_v1-sobol128/HOLD_REVIEW.md` |
| `real_v1_vane_angle.py` | 5. the turn angle out of a bench video — an AprilTag on a vane whose face normal is the pinch axis, read as an IN-PLANE image rotation against a fixed reference tag. Needs `opencv-python-headless`; validated on synthetic footage only (0.33 deg rms) |

`mh.py` is the shared HTTP client (`10.99.99.2:8765`; GETs are unauthenticated, POSTs carry the
token). Every read must assert `servo_polling_suspended == False` and a fresh `servo_age_s` —
telemetry stops while a writer owns the bus, so a careless probe reads one stale sample forever.

| script | role |
|---|---|
| `real_v1_bench_grip.py` | 1. seat the grip — ramp open → grip over ~1 s and report the seated pose and loads |
| `real_v1_bench_regrip.py` | 2. relieve the over-clamp to a target load band, anchored at the MEASURED stall angle. The exported grip is a POSITION the sim reaches with the object already contacted; on hardware the leftover travel becomes clamping force. Per-finger targets — the fingers do not have the same job |
| `real_v1_bench_stepped_run.py` | 3. run the turn, one step at a time, each commanded-and-verified with a fresh servo sample. Arrival gating (`--gate`), load abort/settle (`--load-delta`, `--load-settle`, which require a STALL as well as a load rise), `--regrip` to start from a relieved pose |


## RL training (A = lift/deliver, B = reorient)

| script | role |
|---|---|
| `rl_train_cube.py` | THE trainer (PPO via mjlab/rsl_rl). All env knobs = CLI flags; `--recipe <name>` loads a pinned block from `configs/recipes/*.yaml` (a_lift, b_liveA, b_liveA_imit — the parity knobs live there); trainer-side collapse watchdog via `--watchdog-collapse-z` |
| `train_A_on_morph.sh` | launcher: Policy A on a morphology run dir (from scratch per design — never warmstart A); recipe `a_lift` + trainer-side collapse watchdog |
| `train_handoff_liveA_reset.sh` | launcher: Policy B via the LIVE-A reset (frozen A drives 0..onset organically; pre-onset PPO-masked). Recipe `b_liveA` (RECIPE=b_liveA_imit adds the imitation prior) |
| `train_reorient_on_morph.sh` | launcher: standalone reorient B on a new morphology |
| `train_corefine_BtoA.sh` | launcher: B→A co-refinement (train A on B's downstream reorient reward) |
| `train_blind_actor_2x2.sh` | launcher: the blind-actor 2x2 (sighted/blind x nominal/jittered), all four warmstarted from b33 with a parity gate against b33's own config. `--actor-blind-terms` zeroes the actor's object observations while the critic keeps them = genuine asymmetric actor-critic |
| `eval_blind_actor_2x2.sh` | scores that 2x2 on both test distributions; evaluates the blind arms WITH their blinding applied (a blind-trained actor read out sighted is gotcha #13 in an observation coordinate) |

## Evaluation + diagnostics (judge on these, never on reward sums)

| script | role |
|---|---|
| `rl_demo_handoff_continuous.py` | THE deploy eval: continuous A→B handoff, one env, video + trajectory-health scorecard + per-finger forces; `--record-fingertip-traj` dumps the imitation reference |
| `policy_healthcheck.py` | standalone trajectory-health scorecard on any policy |
| `rl_eval_reorient_metrics.py` | deterministic reorient metrics (held-cos etc.) from a run dir. Read held-cos WITH ITS SIGN; `min_z`/`drop` are whole-rollout and meaningless for runs that include the lift |
| `policy_filmstrip.py` | run/video → one labelled PNG of phase-aligned frames (skill `policy-eyes`). Look at a policy before explaining its reward table |
| `probe_grip_balance.py`, `probe_grip_force.py` | per-finger force probes (degenerate-pinch detection) |
| `probe_thumb_reach.py` | does the reoriented shaft land inside the thumb's reach shell? (perp) |
| `real_v1_render_deploy_plan.py` | render the trajectory the HARDWARE replays (the exported chord/CSV, not the dense carry), with per-frame finger clearance; `--physics` steps the CSV in the deploy scene and reports whether the tool is actually carried. All four plans reorient (cos 0.74-0.78); only g12 keeps its fingers apart |
| `probe_obs_ablation.py` | closed-loop observation ablation: which observation blocks does a trained policy actually steer on? Four interventions (zero/freeze/shuffle/replay) because `zero` conflates missing information with an off-manifold value and gives the OPPOSITE verdict; prints an across-env variance report first, since an intervention can only destroy variance that exists. Found b33 to be feed-forward (`docs/experiments/20260830-obs_ablation/`) |
| `sweep_perp_thumb.py` | perp thumb-morphology sweep: self-collision gate -> stow -> swing -> thumb press -> axial load |
| `rl_eval_object.py`, `rl_play_policy.py`, `rl_diagnose_policy.py` | pose-grid eval / interactive playback / obs-action dump |
| `rl_plot_training.py`, `rl_plot_reorient.py`, `rl_render_reorient.py` | training curves / reorient plots / rollout render |
| `rl_record_handoff_states.py`, `rl_record_initiation_bank.py`, `rl_record_reorient_schedule.py` | state-bank / schedule recorders (consumed by rl_train_cube flags) |

## Studies + pipelines (resumable, detached; logs → `logs/`, summaries → `docs/experiments/`)

| script | role |
|---|---|
| `morph_pipeline_sweep.py` (+`_plots`) | co-design sweep: per design XML→IK→CEM→A→B→handoff eval, health-gated. Runbook: `docs/rl/morph_sweep_STATUS.md` |
| `probe_queue.sh` | policy-bottleneck probe queue (P1 rescue → P2 avar): A best-of-N + paired imit/self B. Plan+decision tree: `morph_sweep_STATUS.md` §probes |
| `make_sweep_video_grids.py` | labeled ffmpeg comparison grids from a sweep tag (best replica/design): lift-phase grid, reorient-phase grid, optional highlights row |
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
