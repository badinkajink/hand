# Handoff to Claude: real-v1 partial-observation transfer and morphology sidequests

**Date:** 2026-08-30  
**Branch:** `partial-obs-transfer`  
**Status:** the four seed-42 training runs and their 32- and 128-environment evaluations are complete. No process from this queue remains active.

## Executive summary

The earlier observation-ablation experiment used the legacy simulation-only `m05_ik_cem` hand. That result is useful only as a historical control and must not be presented as evidence about the current physical platform.

I corrected the experiment onto two morphologies generated from the real-v1 hardware model:

- `rv03_narrowy_sp40`: a marginal but genuine learned reorienter.
- `rv05_manual_stored`: the strongest existing real-v1 learned reorienter.

For each design I warmstarted its own best Policy B and trained a sighted and an object-state-blind actor under +/-5 mm object-XY and +/-5 degree yaw randomization. All arms passed config parity against their own reference run before training.

The result is strongly morphology-dependent:

- On **rv03**, the blind checkpoint is the best result: under jitter it retains 75% versus the sighted policy's 50%, and on retained objects reaches held cosine 0.538 versus 0.366.
- On **rv05**, the sighted checkpoint works, retaining 60% under jitter with held cosine 0.884. The blind run collapses and retains only 2%.

This is interesting, but it is **not yet evidence that an 18-D hardware policy works**. The current "blind" actor only masks object state. It still receives joint velocity, reference trajectories, and previous action; simulated servo load has not been added. There is one training seed. The rv03 sighted actor also catastrophically loses its nominal behavior, so optimization variance/regularization is confounded with information availability.

## 1. Hardware-provenance correction

I added a durable rule at `CLAUDE.md` under "Critical lessons": new manipulation studies must use the `real_v1` hardware geometry/workspace family, must record provenance and deployment status, and must not silently treat `m05`, `perp`, a10, or b33 as hardware-deployable hands/policies.

The distinction to preserve is:

- **Hardware-family eligible for simulation:** generated from `assets/mjcf/real_v1/real_hand.xml`, within `REAL_V1_WORKSPACE` / `REAL_V1_MOUNTS`.
- **Cleared for a hardware-transfer claim:** additionally has an exported plan and passes trajectory-clearance/collision gates.

`rv03` and `rv05` meet the first criterion and can be physically configured on the gantry platform. This experiment does not claim they already have collision-cleared deployment trajectories. Among the currently exported candidates, only g12 has a cleared plan; g12 has no compatible learned A/B checkpoint pair.

## 2. Why these designs were selected

### rv05_manual

- Morphology: `results/phase1/real_v1/rv05_manual_stored`
- Policy A: `results/rl/20260828-0550-policyA_rv05_manual_t0/tensorboard/model_609.pt`
- Policy B reference: `results/rl/20260828-1215-policyB_rv05_anchor_t0/tensorboard/model_270.pt`
- Historical measured result: held cosine approximately 0.848 with very low repeat spread and 3/3 retained.

### rv03_narrowy

- Morphology: `results/phase1/real_v1/rv03_narrowy_sp40`
- Policy A: `results/rl/20260828-0000-policyA_rv03_narrowy_t0/tensorboard/model_609.pt`
- Policy B reference: `results/rl/20260828-1327-policyB_rv03_gated_t0/tensorboard/model_150.pt`
- Iteration 150 was deliberately used instead of 270: the recorded handoff study found iteration 150 retained 3/3 at held cosine approximately 0.645, while the later checkpoint regressed.
- Its reference config includes `target_axis_min_lift=0.06`; parity preserved this.

### Designs not trained

- `rv00_wide` was screened but excluded. It retains reliably yet reaches only final cosine approximately 0.23, consistent with its existing failed-reorienter classification. Training an observation student on a teacher that does not solve the task would be uninformative.
- g12 was not substituted merely because it has the best deployment plan. It has no compatible learned Policy A/B pair, so doing so would be a different experiment requiring a fresh RL pipeline.

## 3. Experiment definition

Launcher: `scripts/train_real_v1_blind_pair.sh`  
Evaluator: `scripts/eval_real_v1_blind_pair.sh`

Only the informative jittered pair was trained:

- `S1_sighted_jitter`: full simulated actor observation.
- `B1_blind_jitter`: actor terms `object_pos`, `object_pose_actual`, and `target_axis_misalign` are forced to zero; the critic keeps the complete privileged observation.

Common settings:

- seed 42
- 3,072 parallel environments
- 5,000,000 timesteps, 67 PPO iterations
- +/-0.005 m object XY jitter
- +/-0.087 rad object yaw jitter
- domain-randomization curriculum over 50 iterations
- each arm warmstarts both actor and critic from that morphology's own Policy B
- live Policy A drives the lift before the B handoff
- the `hold_ik` anchor begins at simulator step 600 and sweeps for 550 steps
- rendered evaluation video at iteration 50
- parity checked with `scripts/assert_config_parity.py` before GPU work

The nominal S0/B0 cells were omitted because the pre-training replay audit showed that retained nominal turns are mostly trajectory-driven. Compute was spent on the distribution where feedback/recovery could matter.

### Critical observation-interface limitation

The actor tensor remains 66 columns wide; blinded columns are zero rather than deleted so the privileged checkpoint can be warmstarted. Effective available information is still much larger than the requested hardware interface:

- `joint_pos`: 15 values, including scripted palm joints rather than only the nine servo joints
- `joint_vel`: 15 values, not directly measured by the platform
- `ref_finger_qpos`: 9 values
- `ref_object_pose`: 7 open-loop reference values
- previous `actions`: 9 values
- object-derived blocks are the only terms masked in B1

There is currently no nine-value simulated servo-load observation. Do not call B1 an 18-D policy or a hardware-ready student.

## 4. Pre-training replay audit

Before spending PPO compute, each frozen reference policy was evaluated over 32 environments using on-manifold replay interventions. "Replay hidden" supplies another environment's real hidden-object observation at the same timestep; this is more trustworthy than zero ablation, which can fail solely because zero is off-manifold.

| design | test | condition | held rate | cosine on held rollouts |
|---|---|---|---:|---:|
| rv03 | nominal | none | 0.59 | 0.859 |
| rv03 | nominal | replay hidden | 0.50 | 0.857 |
| rv03 | jitter | none | 0.53 | 0.398 |
| rv03 | jitter | replay hidden | 0.44 | 0.420 |
| rv05 | nominal | none | 0.56 | 0.970 |
| rv05 | nominal | replay hidden | 0.44 | 0.953 |
| rv05 | jitter | none | 0.34 | 0.851 |
| rv05 | jitter | replay hidden | 0.19 | 0.853 |

Interpretation: after the shaft is retained, destroying object-state correspondence barely changes alignment. The main perturbation failure is retention. Therefore the training comparison asks whether privileged state helps grip recovery, not whether it is required to replay the nominal turn.

Audit JSONs are `AUDIT_*.json` in this directory.

## 5. Training outcomes

| design | arm | terminal checkpoint | status | final training object height |
|---|---|---|---|---:|
| rv03 | S1 sighted | `model_66.pt` | completed | 0.0984 m |
| rv03 | B1 blind | `model_66.pt` | completed | 0.0922 m |
| rv05 | S1 sighted | `model_66.pt` | completed | 0.0887 m |
| rv05 | B1 blind | `model_50.pt` | **watchdog collapse at iteration 62** | 0.0278 m |

The rv05 blind run was correctly stopped when `Metrics/lift_height/object_height < 0.03`. Its queue sentinel says `DONE` because the launcher records a collapse as a completed experimental outcome; `rv05_manual-B1_blind_jitter_s42.trainer.log.COLLAPSED` is the authoritative failure marker. There is no model 66 for this arm, and the evaluator correctly selected model 50 as the latest available checkpoint.

Training logs:

- `logs/real_v1_blind_pair/rv03_narrowy-S1_sighted_jitter_s42.trainer.log`
- `logs/real_v1_blind_pair/rv03_narrowy-B1_blind_jitter_s42.trainer.log`
- `logs/real_v1_blind_pair/rv05_manual-S1_sighted_jitter_s42.trainer.log`
- `logs/real_v1_blind_pair/rv05_manual-B1_blind_jitter_s42.trainer.log`

## 6. Final 128-environment evaluation

The automatic evaluation initially used 32 environments. Because rv03 produced a surprising nominal/jitter reversal, I repeated all four policies on both distributions with 128 environments. The larger evaluation reproduced the result, including 0/128 retained for rv03 sighted nominal and 126/128 retained for rv03 blind nominal.

| design | actor | test | held | mean min-z | final cosine, all | cosine on held | peak cosine |
|---|---|---|---:|---:|---:|---:|---:|
| rv03 | sighted | nominal | 0.00 | 0.008 | 0.440 +/- 0.173 | n/a | 0.859 |
| rv03 | sighted | jitter | 0.50 | 0.064 | 0.336 +/- 0.346 | 0.366 | 0.733 |
| rv03 | blind | nominal | **0.98** | **0.110** | **0.888 +/- 0.018** | **0.888** | 0.914 |
| rv03 | blind | jitter | **0.75** | **0.088** | **0.542 +/- 0.350** | **0.538** | 0.788 |
| rv05 | sighted | nominal | **0.77** | **0.091** | 0.729 +/- 0.370 | **0.918** | 0.963 |
| rv05 | sighted | jitter | **0.60** | **0.072** | 0.572 +/- 0.435 | **0.884** | 0.937 |
| rv05 | blind | nominal | 0.00 | 0.007 | 0.009 +/- 0.037 | n/a | 0.877 |
| rv05 | blind | jitter | 0.02 | 0.010 | 0.017 +/- 0.079 | 0.016 | 0.862 |

Files: `EVAL128_*.json` in this directory. The earlier automatic 32-env results are `EVAL_*.json`.

The evaluator prints a generic message that jitter is "OOD for a policy trained at 0." Ignore that wording for these new checkpoints: all four new policies were trained with this jitter distribution. The message comes from `probe_obs_ablation.py`, which cannot infer the checkpoint's training config.

## 7. Rendered videos

- rv03 blind, successful retention: `results/rl/20260830-rv03_narrowy-B1_blind_jitter_s42/eval_videos/20260830-rv03_narrowy-B1_blind_jitter_s42-step-1200.mp4`
- rv03 sighted, visible drop: `results/rl/20260830-rv03_narrowy-S1_sighted_jitter_s42/eval_videos/20260830-rv03_narrowy-S1_sighted_jitter_s42-step-1200.mp4`
- rv05 sighted, successful: `results/rl/20260830-rv05_manual-S1_sighted_jitter_s42/eval_videos/20260830-rv05_manual-S1_sighted_jitter_s42-step-1200.mp4`
- rv05 blind, pre-collapse iteration 50: `results/rl/20260830-rv05_manual-B1_blind_jitter_s42/eval_videos/20260830-rv05_manual-B1_blind_jitter_s42-step-1200.mp4`

The visual check agrees with the scorecard on the key rv03 contrast: the sighted checkpoint drops the cylinder and the blind checkpoint retains it.

## 8. What can and cannot be concluded

### Supported

1. Removing direct object state is not universally fatal on the real-v1 geometry family. rv03 learns a strong object-blind policy under the tested distribution.
2. The same training recipe is not morphology-agnostic. rv05 blind collapses while rv03 blind succeeds.
3. Retention, not held-only alignment, is the central failure mode. Once retained, the old reference policies were already largely trajectory-driven.
4. `peak_cos` remains a dangerous metric: rv05 blind reaches peak approximately 0.86 while retaining almost nothing. The object rotates while falling.

### Not supported yet

1. Do not claim the blind actor outperforms a sighted oracle in general. A sighted policy has more information in principle; rv03's result can arise from PPO instability, privileged-feature overfitting, or regularization from masking.
2. Do not claim 18-D deployment. The actor is not using the real `(9 position, 9 load)` interface.
3. Do not rank morphologies from one policy seed. The project has repeatedly measured 0.3--0.5 across-seed spread for from-scratch reorientation, and even warmstarted policies can bifurcate.
4. Do not infer hardware transfer until the chosen morphology also passes plan export, self-collision, clearance, and real load calibration.

## 9. Recommended next work, in order

### A. Establish whether the rv03 result replicates

Run seeds 43 and 44 for both rv03 S1 and B1 with the launcher unchanged. This is the minimum experiment needed before interpreting the masking effect. Evaluate model 0, 50, and final checkpoints on 128 environments so that catastrophic forgetting is separated from learned improvement.

Suggested launch:

```bash
SEED=43 DESIGNS=rv03_narrowy \
  nohup setsid bash scripts/train_real_v1_blind_pair.sh \
  > logs/real_v1_blind_pair_s43.log 2>&1 </dev/null &
```

Repeat with seed 44. The launcher's sentinels include the seed, so this does not collide with seed 42.

### B. Build the actual hardware observation interface

Add an explicit actor observation mode containing exactly:

- nine finger-servo positions, in hardware joint order
- nine simulated actuator-load proxies

Keep the critic privileged. Do not simply expose all 15 robot positions or finite-difference velocity and call that 18-D.

The load proxy needs domain randomization for the hardware's known defects: per-servo scale and bias, sign convention, deadband, saturation, noise, latency, protection-mode plateaus, and possibly hysteresis. The SCS0009 readout is unitless and is not a calibrated torque sensor.

### C. Replace pure PPO masking with teacher supervision

The rv05 collapse and rv03 sighted regression show that warmstarting plus PPO is brittle. The safer student curriculum is:

1. Collect teacher rollouts over randomized state/physics.
2. Train the hardware-observation actor with action imitation or KL loss while the critic remains privileged.
3. Use DAgger-style aggregation so the teacher labels states reached by the student, not only the teacher manifold.
4. Anneal teacher loss while adding PPO return optimization.
5. If instantaneous position/load is insufficient, add a short history encoder or RMA-style latent estimator. Do not jump to an RNN before testing fixed history.

### D. Keep morphology in the experimental design

At minimum, treat morphology as a stratification variable. rv03 and rv05 already show opposite learning outcomes under the same masking recipe. A single-hand result cannot establish observability requirements for the platform.

## 10. Catalogued sidequests -- not started

The detailed note is `docs/notes/20260830-real-v1-sampling-and-gaiting.md`.

### Denser real-v1 morphology sampling -- higher priority

The paper currently says 108 morphologies were searched, but that set contains only 48 full six-dimensional uniform random draws; the rest are known anchors, one-axis sweeps, and a 5x5 compact-family plane. Counts are:

- 108 evaluated
- 80 graspable
- 58 classified as pinch-roll
- 49 pass the simulated reorientation threshold
- 97 lie within gantry travel
- only g12 currently has a collision-safe exported plan

Therefore the evidence supports a **narrow discovered behavior family**, not yet a mathematically tiny successful morphology volume. A larger Sobol/Latin-hypercube sample should use staged workspace, IK, CEM grasp, self-collision-through-trajectory, open-loop carry, robustness, and diversity gates before RL. This work was deliberately not started.

### Continued screwdriver rotation / finger gaiting -- much lower priority

After standing the cylinder up, require continued axial rotation with release, reposition, and re-contact. This could force richer morphology-dependent behavior than the present middle-finger-dominated one-shot pinch-roll. It should follow denser sampling so task complexity and sampling density are not changed simultaneously. This work was also deliberately not started.

## 11. Files created or changed in this handoff

Created:

- `scripts/train_real_v1_blind_pair.sh`
- `scripts/eval_real_v1_blind_pair.sh`
- `docs/experiments/20260830-real_v1-obs-transfer/README.md`
- `docs/experiments/20260830-real_v1-obs-transfer/HANDOFF_TO_CLAUDE.md`
- `docs/experiments/20260830-real_v1-obs-transfer/AUDIT_*.json`
- `docs/experiments/20260830-real_v1-obs-transfer/EVAL_*.json`
- `docs/experiments/20260830-real_v1-obs-transfer/EVAL128_*.json`
- `docs/notes/20260830-real-v1-sampling-and-gaiting.md`

Changed:

- `CLAUDE.md`: added the hardware-provenance gate.

The worktree contains unrelated pre-existing modifications and deletions, including `.gitignore`, a generated search scene, the `external/mujoco_warp` submodule state, `hand_paper/*`, earlier legacy observation-ablation JSONs, and hardware bench-suite output. Do not reset or bundle those into this work without checking ownership. No commit was made for the files above.

## 12. Fast orientation commands

```bash
# Read this handoff and the short experiment summary.
sed -n '1,260p' docs/experiments/20260830-real_v1-obs-transfer/HANDOFF_TO_CLAUDE.md
sed -n '1,220p' docs/experiments/20260830-real_v1-obs-transfer/README.md

# Inspect completion/collapse state.
find logs/real_v1_blind_pair -maxdepth 1 -type f \
  \( -name '*.DONE' -o -name '*.COLLAPSED' \) -printf '%f\n' | sort

# Inspect the exact configs that ran.
diff -u \
  results/rl/20260830-rv03_narrowy-S1_sighted_jitter_s42/config.yaml \
  results/rl/20260830-rv03_narrowy-B1_blind_jitter_s42/config.yaml

# Find final evaluation artifacts.
find docs/experiments/20260830-real_v1-obs-transfer \
  -maxdepth 1 -name 'EVAL128_*.json' -print | sort
```

