# Immediate plan: broaden the real-v1 open-loop reorientation search

**Decision date:** 2026-08-30
**Status:** 128-hand transient metric rejected; 4,096-design retention-aware screen complete; see
[`../experiments/20260830-real_v1-sobol128/REPORT.md`](../experiments/20260830-real_v1-sobol128/REPORT.md)
**Final report:**
[`../experiments/20260830-real_v1-sobol4096/REPORT.md`](../experiments/20260830-real_v1-sobol4096/REPORT.md)
**Priority:** immediate, ahead of further partial-observation RL work
**Baseline to beat:** g12, the only current design with a collision-safe exported plan

## Decision

Stop the current observation-masking thread as a product-path experiment. Its scientifically
correct continuation would be more seeds on rv03, but that does not close the actual deployment
gap: the learned policies are on rv03/rv05, while the working hardware path is g12 open-loop.

The next program step is to sample substantially more **hardware-configurable real-v1 hands** and
rank them for robust, collision-free, open-loop reorientation. After that search produces one or
more deployable winners, use 18-D position/load feedback as a residual correction layer around the
winning trajectory rather than asking RL to rediscover the entire maneuver.

This ordering connects morphology search, simulation, and hardware:

```text
real-v1 workspace
    -> dense hardware-valid sampling
    -> grasp + open-loop carry
    -> robustness + trajectory clearance
    -> a small set of deployable hands
    -> hardware trials
    -> 18-D load-aware residual control on the winning hand
```

## Why sampling is the immediate bottleneck

The existing paper statement that 108 morphologies were searched is true, but the population is
not dense coverage of the six movable XY coordinates. It consists of five known anchors,
one-coordinate sweeps, a 5x5 compact-family plane, and only 48 full-dimensional uniform random
draws. Current counts are:

- 108 evaluated
- 80 with a viable grasp
- 58 classified as pinch-roll
- 49 above the simulated held-reorientation threshold
- 97 inside measured gantry travel
- four promoted into the deployment comparison
- only g12 with a collision-safe exported trajectory

Those numbers do **not** prove that successful morphology occupies a tiny volume. They show that
the current objective elicits a narrow behavior family, and that deployment constraints remove
most nominal winners. Sparse sampling, the grasp fitter, the carry family, robustness, and
trajectory collision are all selection operators.

The existing open-loop pipeline is already the right low-cost evaluator. There is no reason to
pay for per-design PPO before exploiting it more thoroughly.

## Scope

The search includes only designs that:

1. originate from `assets/mjcf/real_v1/real_hand.xml`;
2. fit `REAL_V1_WORKSPACE` and `REAL_V1_MOUNTS`;
3. use the fixed physical links and actual joint ranges;
4. can be expressed as real gantry/mount coordinates;
5. pass collision checks over the complete grasp-and-carry trajectory before promotion.

The first search keeps the current task fixed: grasp the staged horizontal screwdriver, lift it
clear of the floor, and execute the existing one-shot open-loop reorientation family. Continued
axial screwdriver rotation and finger gaiting remain a later task; changing task complexity now
would confound morphology coverage with a new controller and objective.

## Search design

### Parameterization

Sample the six physical slide coordinates:

```text
thumb_x, thumb_y, index_x, index_y, middle_x, middle_y
```

Use a scrambled Sobol sequence, with a fixed recorded seed, rather than another small independent
uniform sample. Sobol gives useful coverage at every prefix, so the search can stop or extend
without discarding its design.

Retain the known hands and g12 as explicit anchors. Preserve asymmetric designs: the earlier
compact family and most one-axis studies cannot express them, while `rv05_manual` already showed
that asymmetry matters.

The fitter recenters the palm, so some raw mount translations can be behaviorally redundant.
Record both raw mount coordinates and derived relative geometry, then deduplicate near-identical
hands in derived space rather than silently counting equivalent samples as coverage.

### Proposed funnel

Counts below are compute budgets, not success quotas. Each stage keeps all clear successes plus a
diversity sample near the boundary; it must not retain only the highest scalar score.

| stage | approximate budget | work | output |
|---|---:|---|---|
| 0. Pipeline pilot | 128 new Sobol hands | exercise every gate and measure wall time | validated resumable pipeline |
| 1. Geometry population | 4,096 raw hands | workspace, joint-range, static collision, deduplication | hardware-valid geometry manifest |
| 2. Grasp fitting | 512--1,024 survivors | per-design pose fit, several straddles/depths, CEM grasp, held-lift test | graspable population |
| 3. Open-loop carry | 128--256 diverse survivors | pivot-height, direction, angle, grip-relief sweeps; repeated rollouts | nominal Pareto set and behavior labels |
| 4. Robustness | 24--48 finalists | at least 200 careful-bench and full-error draws per operating point | robustness-ranked set |
| 5. Deployment gate | 8--12 finalists | export plan, full-trajectory self-collision/clearance, servo excursion/load checks | genuinely deployable candidates |
| 6. Hardware | 3--5 finalists | repeated staged trials with measured object angle and load traces | real morphology ordering |

The 128-hand pilot proved that its stage outputs were resumable and uniquely identified, but it
also triggered the explicit stop condition below: a transiently upright object could pass and
then fall. That evaluator has been retired. The 4,096-design population proceeds only through a
replacement retention gate: torque-capped turn, position/load-only capture, 60 mm proof lift, and
five-second free hold.

## Stage gates and metrics

### Geometry and plan validity

- within measured gantry travel, not merely XML joint limits;
- no palm/finger or finger/finger interpenetration at reset, grasp, or any carry waypoint;
- no dependence on links, mounts, or proximal-length changes absent from the physical platform;
- exportable mount coordinates and servo commands;
- useful clearance margin, not a zero-margin binary pass.

### Grasp gate

- judge **held lift**, not peak lift;
- retain the shaft after settling with a meaningful minimum object height;
- record per-finger contact persistence and force;
- reject grasps whose success depends on deep geometric penetration;
- fit more than one straddle/depth operating point where reachable, because the old fitter often
  selected the grasp that most strongly resisted the desired rotation.

Graspability is only a filter. Prior work showed that grasp balance did not predict reorientation,
so do not rank the surviving hands primarily by grasp score.

### Reorientation gate

A rollout counts only if the object remains held. Record:

- final and tail alignment on retained rollouts;
- minimum object height through and after the turn;
- peak alignment only as a diagnostic, never as the success metric;
- object translation and floor clearance;
- pad-contact fractions and per-finger force;
- command excursion and margin to joint limits;
- driver identity and motion share;
- contact style: pinch-roll, tripod, palm-pin, single-contact, or new behavior;
- full-trajectory self-collision and minimum clearance.

The search should report both success density and behavior diversity. Fifty near-duplicates of the
same middle-finger pinch-roll are one mechanism, not fifty independent morphology discoveries.

### Hardware-relevance gate

Add metrics for the failure already observed on g12:

- predicted actuator force/torque utilization;
- per-finger force imbalance;
- commanded-versus-achieved excursion under torque limits;
- time spent near overload/protection thresholds;
- how much grip relief is required before the driver can advance;
- robustness to servo gain, torque ceiling, friction, placement, mass, and latency.

The MuJoCo actuator force is not the SCS0009 unitless load register. Use it as a ranking feature,
not a calibrated hardware prediction, until hardware traces provide the mapping.

## Ranking: use a Pareto set, not one score

The current strongest cheap feature, thumb torque capacity, has useful but imperfect discrimination
(reported AUC 0.821). The fixed-contact ceiling, mount separation, and sweep scores were much
weaker. None should become a hard surrogate for open-loop evaluation.

Promote a Pareto/diverse set spanning:

- retained alignment;
- careful-bench success rate;
- full-error success rate;
- trajectory clearance;
- actuator/load margin;
- low placement sensitivity;
- contact-strategy diversity;
- low total motion and simple hardware execution.

Nominal cosine alone is explicitly insufficient: `rv04_mid` was the nominal leader and the worst
of the four robustness candidates, while g12 became the practical winner.

## Existing code to reuse

- `scripts/real_v1_design_search.py`: design generation, grasp scores, open-loop carry, style
  classification, and current named sets.
- `scripts/fit_real_v1_pose.py`: per-design palm/grasp pose fitting.
- `scripts/probe_real_v1_carry.py`: carry-cell and contact-mechanics evaluation.
- `scripts/morph_selfcollision_gate.py`: morphology collision validation.
- `scripts/real_v1_deploy_envelope.py`: careful-bench/full-error robustness evaluation.
- `scripts/real_v1_export_plan.py`: build and trajectory export.
- `scripts/real_v1_trajectory_clearance.py`: full-path clearance check.
- `docs/experiments/20260828-real_v1_search/REPORT.md`: 108-hand baseline and behavior taxonomy.
- `docs/experiments/20260829-real_v1_deploy/`: current g12/g23/g24/rv04 deployment evidence.

Likely implementation work is modest but important:

1. add a parameterized, seeded Sobol design set and manifest to
   `real_v1_design_search.py`;
2. make every stage resumable and shardable by stable design ID;
3. add trajectory self-collision/clearance before robustness promotion;
4. preserve stage-specific rejection reasons rather than dropping failed rows;
5. export a single joined table from raw mount coordinates through robustness and plan status;
6. render representative successes, failures, and distinct contact styles automatically.

## Required artifacts

The search is not complete without:

- a manifest containing seed, Sobol index, raw mounts, derived geometry, scene path, and generator
  version for every candidate;
- counts and rejection reasons at every stage;
- a joined per-hand metric table;
- coverage plots in raw and derived morphology space;
- nominal-versus-robustness plots;
- a contact-style map, including within-style diversity;
- exported plans and clearance reports for every deployment finalist;
- rendered videos for finalists and representative failure modes;
- direct comparison against g12 under the identical evaluator.

## Stop conditions

Pause and repair the pipeline rather than scaling if the 128-hand pilot shows any of the following:

- generated designs lack traceable real-v1 mount coordinates;
- collision status changes between search and export;
- duplicate/equivalent designs dominate the sample;
- grasp success depends on object penetration inconsistent with hardware;
- a dropped object can pass the primary score;
- g12 cannot reproduce its existing ranking under the new joined evaluator;
- stage outputs cannot be resumed without recomputing finished candidates.

## Follow-on: the 18-D controller

Once the search produces a collision-safe open-loop winner, build the controller around that hand
and trajectory. The likely useful form is not an end-to-end memoryless policy. It is:

```text
nominal open-loop trajectory
    + history of 9 servo positions and 9 unitless loads
    -> latent contact/motor-state estimator
    -> small per-finger residual, advance/hold/regrip decisions
```

The controller can use recurrent state and known command history without adding physical sensors.
Train a privileged simulator teacher to correct placement/contact disturbances, then distill with
action supervision or DAgger into the history-based student. Randomize load scale, bias, deadband,
latency, saturation, and protection behavior using recorded hardware traces.

That is the plausible path to proprioceptive reorientation. The immediate search comes first
because the controller should be developed on the morphology and trajectory that can actually be
deployed, not on rv03/rv05 policies disconnected from the cleared hardware plan.
