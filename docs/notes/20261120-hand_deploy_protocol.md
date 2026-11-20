<!-- The repository copy of the local skill .claude/skills/hand-deploy/SKILL.md (.claude is not in git); change both. -->

# Deploying SR2 hand layouts to the bench and back

Owner, 2026-11-19 23:3x: "develop protocols / automated skills to more quickly vet a hand and prepare a morphology and its
associated controls for real hand deployment, good or bad, as well as the ensuing evaluation, analysis, and real2sim
calibration". This skill is that protocol. Its first use was the sphere twist of 2026-11-19 to 11-20
(`docs/experiments/20261119-twist_layout_ranking/`, page `20261119-sphere_twist_layout_ranking.html`); the commands below
default to that study's files.

The driver is `scripts/hom/hand_deploy.py` (`vet`, `plan`, `brief`, `score`). It calls the study's tested scripts in
`docs/experiments/20261119-twist_layout_ranking/sim/`; read their docstrings for the details.

## The prediction model

Every prediction comes from the simulation whose blind predictions came nearest the hand on 2026-11-17 (0.437 against
the hand's 0.447 at (70, 110) grasp 0): `sim/20261119-twist_ranking_search.py`, which loads
`docs/experiments/20261117-sr2_system_identification/sim/20261117-g13_open_pose_replan.py`.

- Servos at 7.5 V: kp 0.825 N m/rad, damping 0.011, torque limit 0.27 N m; the index and middle yaws 0.525 N m/rad.
- Gears the load cannot backdrive; the measured MCP stops (index -11.4 deg, middle -10.0 deg).
- The stem scene with a table until the lift; impratio 1 at friction 1.0.
- HOM converting at the server's 0.55 N m/rad under the torque budget 0.56, from the layout's own open pose.

Run it with `HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011` and `CUDA_VISIBLE_DEVICES=`. A score is the tracking
`max(0, 1 - RMS error / RMS commanded rotation)` over the task window, 0 when the sphere drops. Two contact models (point
and the sphere pads) and stage D's four plants (stiff pads, friction 0.6 under each model, servos at 5 V) give the
prediction's spread.

## 1. Choose layouts

- A full ranking of a task over the lattice runs as lanes: `sim/20261119-twist_ranking_search.py run --stage A|B|C|D|P`,
  then `sim/20261119-twist_ranking_report.py` (summary, candidates, proposal). On 20 CPU workers stages A-D take about 3 h
  for 482 layouts x 16 grasps x 2 models. Launch through `resguard.sh run` with `taskset` and fsynced rows (see the
  lanes in `logs/20261119-twist_layout_ranking/lane*.sh`).
- For a correspondence test, spread the layouts over the whole predicted range, the bad end included (owner, 23:2x):
  `sim/20261119-dense_proposal.py --per-bin 3 --clear 15 --drops 3`. It fills 0.05-wide bins of predicted tracking,
  clear layouts first and far apart on the lattice. Most predicted-bad layouts touch (below a predicted 0.35, 2 of 95
  fitting layouts keep 15 mm), so the bad end is mostly touching layouts; that is a property of the hand.
- Never take only the top of a ranking. The 148 best twist layouts spread by a standard deviation of 0.031, and there the
  plants reorder them (rho 0.08-0.28).

## 2. Vet each layout

```bash
HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 CUDA_VISIBLE_DEVICES= ~/.claude/bin/resguard.sh run --mem 5G -- taskset -c 2-8 \
    .venv/bin/python scripts/hom/hand_deploy.py vet 90,135 75,140:5 --workers 6
```

Per layout (`X,Y` takes the ranking's best grasp, `X,Y:K` grasp K), it checks:

1. **Fit.** The mounts lie in the gantries' travel boxes (`sr2_layouts.layout_report`).
2. **Prediction.** The twist at seeds 0-2 under point contact, the pads and the four plants. Runs already in
   `logs/20261119-twist_layout_ranking/rows_*.jsonl` are reused, and new ones go to `rows_V.jsonl`.
3. **Drops and HOM's errors.** These are counted under both models.
4. **Finger clearance.** This is the least gap between fingers on the cad_v3 collision spheres plus the tip pad, tip end
   and PIP balls, along the simulated twist at seed 0 under both models.

The verdict is clear at 15 mm or more, marginal at 10-15 mm, close at 2-10 mm and touching below 2 mm, with "drops" or
"unfit" added where they apply.

The clearance check replaced the ranking's three-ball test on 2026-11-19. The balls miss the DIP link, which reaches 23 mm
to the side: at (70, 90) the thumb's and index's DIP links met with the balls 27 mm apart. On the hand's measured joints
the CAD check read -2 to -5 mm where the index and middle met, at (105, 70) g13 on 11-17. It read 4-9 mm at (70, 110) g0,
which ran cleanly and which simulation puts at 17 mm. The hand's index and middle come 8-13 mm closer than the
simulation's, so a gap under about 13 mm in simulation is marginal on the hand. Static poses are not enough: the twist
moves the fingers beyond the grip pose, and the touching links meet mid-ramp.

## 3. Build the plans

```bash
.venv/bin/python scripts/hom/hand_deploy.py plan docs/experiments/<study>/data/<date>-vet.json [--verdicts clear marginal]
```

The plans are built as `logs/20261116-hom_setup/film_plan.py` builds them, through `sim/20261119-trial_plans.py`, into
`docs/experiments/20261115-hom_bench_trials/plans/`. The open pose's index and middle MCPs are raised to -11.0 and -9.5
deg where lower. Each plan's `meta.ranking` gets the prediction, written before any trial. The HOM server's load check
(`HomBench.from_plan` at budget 0.56) runs in a fresh process, and the plan list gets the vet fields.

Rules:

- **A sent plan is frozen.** The lab checks md5s. Never rebuild a plan file after sending it; write a new name instead.
- **Controls.** A plan with other HOM forces (stage E, `sim/20261119-twist_force_refine.py`) or a CEM pick
  (`sim/20261119-twist_cem_refine.py`, then `sim/20261119-cem_plans.py`) is a second plan of the same layout. Check its
  clearance too (`sim/20261119-cad_finger_gaps.py --plans ... --no-hand --merge`), because CEM moves the grasp. On
  2026-11-19 the lower forces (1.0-1.7 N in place of 2.5 N) raised the predicted tracking from 0.44 to 0.61, and the CEM
  of grasp and forces added nothing on average.

## 4. Brief the lab

```bash
.venv/bin/python scripts/hom/hand_deploy.py brief --trials 2 [--plans NAME ...]
```

Commit and push the plans and the list first, then send the brief to the lab session ('Physical model vs GP
calibration'). The hand, the UR5 and the HOM servers are the lab's: never start a server, a trial or a motion from the
workstation.

The lab's series (`lab/hom_series.py`, `lab/series_plan.py`) has these limits:

- **Trial time.** About 4.5 min a trial, plus a gantry move per layout.
- **Lights.** The lights go off near 21:10; with no camera tracking after that, trials run from 07:00.
- **Servers.** The servers run at `--torque-budget 0.56` with no tips or filter flags.
- **Drops.** A drop ends a trial at the 30 mm drift guard. A sphere that rolls away stops the series until someone
  restarts it.
- **Touching layouts.** These may trip a servo's overload guard, which stops the series with the sphere held.

Order blocks so that one failure stops the least: clear layouts first, shuffled, then the bad end, then the droppers,
then the control refinements. Relay the owner's words with the time when they change an order.

## 5. Score the trials

```bash
.venv/bin/python scripts/hom/hand_deploy.py score --since 20261120-060000
```

This wraps `sim/20261120-trials_vs_prediction.py`, which scores each HOM server record of the plan list with
`scripts/hardware/hom_trial_tracking.py`. It is the same score as the simulation's, and 0 when the trial did not reach
the schedule's end. It reports Spearman's rho of measured against predicted tracking over the default-force layouts, with
an interval from resampling each plan's trials, per contact model and per plant (`data/20261119-plant_predictions.json`).
Report the touching layouts and the droppers apart from the clear ones. Report a negative result in the same voice as a
positive one.

## 6. Calibrate the simulation from the trials

Each check below measures one part of the gap; rerun them on every new series.

- **Short-window gap** (`sim/20261119-segment_gap.py`). Every 0.25 s, 0.5 s windows start the plant from the hand's
  measured joints and tracked sphere and replay the hand's targets. It gives the twist error per plant. On 11-17 it was
  10.5 deg RMS at friction 1.0 under every plant. The hand slipped 25-74 deg in single windows mid-ramp, where the
  simulation turned the sphere a median of 2 deg.
- **Fingertip forces** (`sim/20261119-tip_forces_vs_sim.py`). The boards against the simulation's contact forces per
  finger. At (70, 110) g0 the thumb read 2.5 N against 2.1 N; the index read 0.62 N against 1.82 N, in contact about
  half the task; the middle was in contact 62 % of the task against 100 %. The boards also sense finger-to-finger contact
  (the lab's `sim/20261119-thumb_middle_contact.py`).
- **Finger gaps on the hand's joints** (`sim/20261119-cad_finger_gaps.py`, its hand part). The CAD clearance on the
  measured joints of every trial, against the simulated twist of the same plan.
- **Which plant predicts the hand.** Rho per plant (step 5). A plant whose rho beats the current one's replaces it in the
  next search.
- **Board calibration** (proposed, owner 21:3x). Use the window replays' simulated contact forces as labels for each
  board's gain per contact place. Check the fitted gains against the UR5 presses before trusting them, because the labels
  are the model's own.

## 7. Write it up

Result pages follow the repository's rules (`CLAUDE.md`): `scripts/hom/twist_ranking_page.py` is the builder of the first
use, with sections for each step above. Its trials section appears once `data/20261120-trials_vs_prediction.json` exists.
Keep the hand-off note (`docs/handoff/20261119-twist_layout_ranking.md`) current.

## Future work

- **Generalize the calibration scripts.** `segment_gap.py` and `tip_forces_vs_sim.py` should take any plan and any
  trials, as `cad_finger_gaps.py` does.
- **Close the loop on force.** Regulate each finger's normal force from the boards around HOM's impedance offset, once the
  boards' gains are trusted.
- **A new hand.** A new hardware iteration needs new cell scenes, an identification of its servos and gears (as on
  11-17), new CAD collision spheres, and a rerun of step 1.
