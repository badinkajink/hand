# Codebase audit & consolidation plan (2026-07-08)

Goal: make the (largely session-generated) codebase readable, deduplicated, and cohesive without
breaking the active research pipelines. This file records what was done, what was found, and the
prioritized plan for the rest.

## Done in this pass

- **Workspace layout**: top-level run logs / sentinels / pids → `logs/` (gitignored, archive under
  `logs/archive/`); experiment summaries → `docs/experiments/` (tracked); stray notes →
  `docs/notes/`. All writer scripts + `.sh` launcher LOG defaults repointed, so the root stays clean.
- **scripts/ consolidation**: 124 → 36 active files. ~90 superseded scripts moved to
  `scripts/archive/` (see its README for the era map). Verified nothing active imports them.
  Active map: `scripts/README.md`.

## src/ assessment (8.7k lines, mostly sound modularity)

Layout: `morphohand/{rl, optimization, sampling, tools, backends, driver, config}`. Tests: 9 files,
CPU-only. The structure is right; the debt is concentrated:

1. **`rl/env_cfg.py` (1519)** — one dataclass with ~120 fields + one 800-line `to_mjlab_cfg()`
   that wires scene, actions, rewards, events, terminations, curricula. Split target:
   `env_cfg.py` (config dataclass only) + `env_build.py` with per-manager builders
   (`_build_rewards(cfg)`, `_build_events(cfg)`, …). Pure mechanical move, high readability win.
2. **`rl/mjlab_terms.py` (1240)** — reward terms + event terms + obs helpers + quat math in one
   file. Split target: `terms_reward.py` / `terms_event.py` / `terms_obs.py`; move `_quat_mul`/
   `_quat_rotate` to a shared `math.py` (they're re-implemented in ≥2 places).
3. **Script→script imports (inverted layering)**: `policy_healthcheck.py` imports env/actor
   builders from `rl_demo_handoff_continuous.py`; `morph_pipeline_sweep.py` + `ik_recem_landscape.py`
   import IK internals (`_has_joint`, `_inject_keyframe`) from `retarget_keyframe_ik.py`.
   Fix: promote `make_env_cfg`/`build_actor`/`act*` into `morphohand/rl/deploy.py` and the IK
   helpers into `morphohand/tools/keyframe_ik.py`; scripts become thin CLIs.
4. **Copy-pasted study plumbing**: `latest()/final_ckpt()/mktemp-warp-cache/subprocess-with-env`
   re-implemented in `compliance_robustness_sweep.py`, `reorient_variance_study.py`,
   `morph_pipeline_sweep.py`, `compliance_dr_pipeline.py`. Fix: `morphohand/studies/runlib.py`
   (checkpoint lookup, resumable-JSON state, detached launch, log-watchdog helpers).
5. **23 `sys.path` hacks** across scripts — disappear naturally as logic moves into the package.
6. **`optimization/phase1_common.py` (1035)** — grasp-synthesis era; lower priority (stable, tested).

## Robustness notes (for the RL debugging agenda)

- `rl_train_cube.py` has `seed` (default 42) but rollouts are Warp/GPU-nondeterministic, so
  "seed variance" in the studies is really *training-run* variance; the variance study + imitation
  prior is the correct mitigation (already adopted).
- The trainer's config surface (~120 flags) is the real fragility: recipes live in `.sh` launchers
  and study scripts as flag soup, and train/deploy parity bugs (gotcha #13,
  `finger_residual_scale`) recur. Worth adding a `--recipe <name>.yaml` layer (named, versioned
  recipes: `a_lift`, `b_liveA_imit`, …) that scripts and studies share — single place to pin parity.
- Watchdog logic (object-height collapse) exists only in `train_A_on_morph.sh` (bash/grep). Moving
  it into the trainer as a first-class termination/abort callback would cover every launcher.

## Prioritized next steps

⚠️ Do NOT refactor `src/morphohand/rl/` or the active launchers while a training pipeline is
running — subprocess stages import whatever is on disk. Land refactors between runs, gated by:
`uv run python -m pytest tests` + a SMOKE A-train + `policy_healthcheck.py` on a10/b33.

1. (small, safe) `morphohand/studies/runlib.py`; deduplicate the 4 study scripts onto it.
2. `rl/deploy.py` (env/actor builders out of the demo script); `tools/keyframe_ik.py`.
3. Split `env_cfg.py` and `mjlab_terms.py` as above (mechanical; do in one sitting with tests).
4. Recipe YAML layer for the trainer; port the two blessed recipes (A-lift, B-liveA-imit).
5. Trainer-side collapse watchdog; retire the bash watchdogs.
