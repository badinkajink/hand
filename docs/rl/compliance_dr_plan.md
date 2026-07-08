# Compliance domain randomization — implementation spec (handoff, 2026-07-08)

Concise, actionable plan for training a **stiffness-robust** grasp+reorient policy by randomizing
the contact compliance during training. Written for a fresh session; read this + `CLAUDE.md` first.

## Why (the motivation, already measured)

The sim-to-real hardening pass found that policies trained at a **single** contact stiffness are
**fragile** to it: the compliance-robustness sweep (`scripts/compliance_robustness_sweep.py`,
`docs/rl/img/compliance_robustness.png`) showed `soft_b33` reorients at `solimp` dmax 0.995 **and**
0.998 but **drops at 0.997 and fails at 0.999** — a non-monotonic, contact-mode-sensitive response.
Training at one stiffness overfits (the hard-retrained policy dropped even at soft). Full write-up:
`reorientation.md` → "Compliance-robustness sweep". **Fix: show the policy a *range* of stiffnesses
during training so it can't overfit one.**

## What to build

Randomize the geom contact compliance — primarily `solimp` (dmin, dmax), optionally `solref` and
friction — per episode, over a range spanning soft→hard. Target range (from the sweep, where the
task is at least sometimes feasible): **`solimp` dmax ∈ [0.995, 0.999]`, dmin ∈ [0.97, 0.985]`**
(keep `solref = 0.006`; 0.003 hits the 2·dt stability edge). Sample jointly (dmin,dmax correlated).

### Approach A — per-env solimp DR (preferred, true DR)

Add an `EventTermCfg` (mode `"reset"`) that, per env at reset, samples a compliance and writes it
to that env's geom solimp. **First check whether mjwarp exposes `geom_solimp` per-world** (nworld
dim) — the env only does cube-spawn DR today (`env_cfg.py` `reset_cube`/`dr_anneal`), so there is no
precedent for per-env *physics-param* randomization here. If `model.geom_solimp` is per-world in the
mjwarp `Model`/`Data`, write the sampled value at reset (mirror how `reset_cube` writes per-env
state). If it is a *shared* model param (not per-world), per-env DR is not directly possible → use
Approach B.

Wire it like the existing DR: a config field (e.g. `compliance_dr: bool`, `compliance_dr_range`),
build the `EventTermCfg` in `to_mjlab_cfg` alongside `reset_cube`, and expose CLI flags in
`rl_train_cube.py` (mirror `--dr-*`). Reference pattern: `env_cfg.py` reset events (~L1208) +
`dr_anneal` curriculum (~L1377).

### Approach B — contact-stiffness CURRICULUM (simpler fallback)

If per-env geom params aren't writable, anneal the **whole-scene** solimp soft→hard over training
(all envs share the current stiffness, which ramps). Reuse the curriculum machinery
(`anneal_smoothness_weights` is the generic pattern; `target_axis_alpha_curriculum` shows a param
anneal). Needs a curriculum term that updates the model's `geom_solimp` each iter — verify the
mjwarp model param can be mutated at runtime (may require recompiling or writing the warp array).
Weaker than per-env DR (no within-batch diversity) but a valid first step ("learn soft, harden").

### Approach C — round-robin scene variants (crudest, zero new physics-write code)

Pre-generate K hardened `frozen_scene.xml` variants (as in `assets/mjcf/experimental/sim2real/`,
via the `solimp` sed used in `compliance_robustness_sweep.py`) and cycle the training env's
`--frozen-scene-xml` across them between runs/segments. No per-env diversity; use only if A/B blocked.

## Training recipe (use the validated tools)

- Train **A** from scratch with compliance DR (the lift transferred cleanly to harder contact, so it
  should tolerate DR): `train_A_on_morph.sh` on m05, add the compliance-DR flags.
- Train **B** via the **imitation prior** (it degraded most gracefully vs stiffness — the best base
  for robustness) with the *same* compliance DR: `train_handoff_liveA_reset.sh` + the imitation
  `EXTRA_ARGS` (see `reorient_variance_study.py` `imit` mode) + compliance-DR flags.
- **n ≥ 2 seeds** — per-design reorient training is seed-noisy (held-cos sd ~0.4 from scratch;
  see the variance study). Judge on the sweep below, not a single run.

## Evaluate (the success test — already built)

Re-run `scripts/compliance_robustness_sweep.py` with the DR-trained A/B added to `policies()`.
**Success = the DR policy's held-cos curve is FLAT and high across dmax 0.995→0.999** (holds + reorients
everywhere, no drops), vs the fragile non-monotonic curves of the single-stiffness policies. For
rigor, extend the sweep to N stochastic rollouts per point → success-*rate* curves (currently one
deterministic rollout per point).

## Pointers

- Contact params: geom default `solimp/solref` in `frozen_scene.xml` (m05:
  `results/phase1/landscape/m05_ik_cem/`); env also sets `impratio=10`/elliptic (`env_cfg.py` ~L1399).
- DR precedent: `env_cfg.py` `reset_cube`, `dr_anneal` (spawn DR only — no physics DR yet).
- Robustness sweep + curves: `scripts/compliance_robustness_sweep.py`, `COMPLIANCE_ROBUSTNESS.txt`,
  `docs/rl/img/compliance_robustness.png`.
- Reference policies: a10 (lift) / b33 (reorient) on m05; imitation prior
  `results/reorient_ref/m05_a10b33_fingertip_obj.npz` + `src/morphohand/rl/imitation.py`.

## One-line summary

Single-stiffness training overfits the contact model (fragile, non-monotonic robustness); randomize
`solimp` over [soft, hard] during training (per-env DR if mjwarp allows, else a stiffness curriculum),
retrain A + imitation-B with ≥2 seeds, and confirm a flat compliance-robustness curve.
