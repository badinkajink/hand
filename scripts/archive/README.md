# scripts/archive/ — superseded experiment tooling (2026-07-08 consolidation)

Everything here is **kept for the historical record but no longer on any active path** — the
engineering logs (`docs/rl/reorientation.md`, `RESEARCH_STATE.md`) reference these scripts by
name, so they stay browsable instead of deleted. Nothing in `scripts/`, `src/`, or `tests/`
imports from this directory (verified at move time). If you resurrect one, move it back out —
don't run it from here (relative paths assume `scripts/`).

Eras, newest first:

- **Unreferenced scripts moved in the docs reorganization (2026-10-10):** scripts that no page, builder, note,
  README, test or other script named and that had not changed since before 2026-09-12 (reference scan:
  `logs/20261009-docs_reorg/script_refs2.py`, one rg pass per kind, worktree copies excluded). m05/b33 era:
  `eval_b33_seed_band.py`, `probe_b33_noise_stability.py` (their summaries stay in
  `docs/experiments/B33_SEED_BAND.*` and `B33_NOISE_STABILITY.*`). Opposed-pair (perp) runs r6-r9:
  `train_perp_antislip.sh`, `eval_perp_antislip.sh`, `train_perp_thumbbrace.sh`, `train_perp_sp25_chuckpose.sh`.
  real_v1 analysis one-offs of 2026-08-29 to 09-02: `real_v1_deploy_report.py`, `real_v1_select_finalists.py`,
  `real_v1_select_retention.py`, `real_v1_transfer_table.py`. Assets and renders: `generate_primitive_meshes.py`
  (lightning-grasp object meshes), `fingertip_shape_render.py`.
- **Seam-closing era B-side variants (June 2026, superseded by the live-A reset):**
  `train_normallift_B_*.sh`, `train_handoff_iter2.sh`, `train_handoff_branchB_*.sh`,
  `train_handoff_adaptB_to_A.sh`, `train_handoff_onset_inject.sh`, `sweep_branchB_unfreezeA.sh`,
  `rl_demo_handoff.py` (two-env demo; superseded by `rl_demo_handoff_continuous.py`).
- **Force/gentleness chase (June 2026, concluded — the "grip force" target was a phantom):**
  `train_handoff_b33_forcereg.sh`, `train_gentle_lowforce_B.sh`, `train_lowforce_A.sh`,
  `sweep_b34_thresh.sh`, `b34_eval_on_done.sh`, `eval_lowforce_on_done.sh`.
- **One-shot orchestration / watchers (spent):** `overnight_*`, `queue_*`, `*_eval_trigger.sh`,
  `_relaunch_B12.sh`, `watch_training.sh`, `run18_launch.sh`.
- **Pre-RL grasp-synthesis era (April–May 2026, runs 6–18):** `run6_*`–`run18_*`, `RUN13.md`,
  `RUN18.md`, `EVAL_SET_KEYFRAMES.md`, `phase1_pollard_*`, `lightning_grasp_*`, `eval_suite.py`,
  `compare_methods.py`, `fit_synergy_basis.py`, `multimorph_*`, scene/keyframe one-offs
  (`generate_prism_scene.py`, `generate_capsule_tip_scene.py`, `make_short_proximal_scenes.py`,
  `build_short_proximal_keyframes.py`, `smoke_*`), `morphology_gui.py`, `rl_bc_pretrain.py`,
  `rl_reference_playback.py`, env setup one-offs (`setup_*.sh`, `run_phase1_gpu_sweep.sh`,
  `test.sh`).
