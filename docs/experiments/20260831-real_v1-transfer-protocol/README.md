# The morphology-ranking transfer study — protocol

Written 2026-08-31, before any trial is run. Everything below (the plan set, the measurement, the
exclusions, the statistics, the figures) is fixed in advance so that the result cannot be a
story assembled after the numbers arrive. If a step has to change once the bench disagrees with
this document, change it here, in a commit, with the reason.

## 1. The claim

The platform's thesis is that **a hand ranking obtained in simulation predicts the physical
ranking**, so morphologies can be searched cheaply in simulation and verified sparsely on
hardware. Its falsifiable form:

> Over the 16 exported, buildable, clearance-safe plans, the bench ordering by axis alignment
> agrees with the simulated ordering.

The study is powered to answer that with a Spearman correlation and a bootstrap interval. **A low
correlation is a publishable result**, not a failed session — it is the measurement that says
this platform's simulator, as it stands today, does not rank hands. Stage 3 exists to say
*which* of the two candidate reasons it is.

Two things this study is not. It is not a claim about learned policies: the deployed controller
is the open-loop geometric carry, and no RL policy is in the loop. And it is not a claim about
absolute performance: the hand is already known to arrive at 0.44-0.90 of its commanded yaw, so
the bench alignment will be systematically below the simulated one. Ranking is the question.

## 2. What is fixed

| | value | why it is pinned |
|---|---|---|
| object | 100 mm x 25 mm cylinder, 24 g, with the id0 vane | the plans were screened against this object |
| post | 144 mm on the bench floor = 100 mm in sim | `BENCH_POST_HEIGHT_SIM_MM`, `SIM_TO_BENCH_Z_MM` |
| trajectory | the exported plan, unmodified, at its own clip | the clip is a *property of the plan*, not a knob |
| `speed_ratio` | 1.0 | the yaw shortfall is torque, not speed, so slowing does not help and changes nothing but the confound |
| `rate_hz` | 50 | the rate every prior bench run used |
| grip | the plan's own per-finger driver/holder split | position control over-clamps a symmetric grip |
| servo protection | as configured, unchanged mid-study | the load-200 plateau is `protective_torque` 20% |
| camera | fixed exposure, `--exposure 4000 --gain 64` | a session that compares decision margins needs a fixed transfer function |
| shaft axis | `--shaft-axis=-x --tag-end top` | measured 2026-08-31: with the vane end up and the tool standing, `x` reads cos −0.999. The axial offset must run centre→tag or the centre lands 2×71×cos on the wrong side |
| pole | `--symmetric-object` | the 100×25 cylinder has no distinguished end, so which pole ends up on top is how the operator seated it, not what the hand did. **Off for the screwdriver** |
| heading | `--heading-deg 90` | see below |

**On the heading.** Nothing this study measures needs bench x/y. The primary metric is an
angle, which comes from the reference tag's observed up vector and the shaft direction — no
horizontal frame is involved at all — and `radial_mm`, the horizontal distance from the reference
tag, already gives a rotation-free slip proxy. The mounting inference refuses on this rig because
the object measures ~36 mm horizontally from the reference tag where `REF_TAG_BENCH_MM` predicts
~130, so the tape-measured x/y does not describe the rig and the refusal is correct. For the study
that is settled by declaring `--heading-deg 90`: the frame is then anchored on the reference tag
with a chosen azimuth, x/y is populated and **internally consistent across every trial**, and the
only thing given up is that `+x` need not point at the index gantry. Nothing downstream cares.
Re-derive the true azimuth later with `--calibrate-heading` if a claim ever needs it.

The camera exposure is the one place this protocol overrides the tool's default. The tracker's
default (settle auto-exposure, then lock it) adapts each run to the room, which is right for a
one-off. For a study that pools 60+ trials, pin it.

## 3. The plan set

Sixteen plans, eleven distinct morphologies. Sorted by the simulator's prediction; the `(sd)` is
over that plan's own four simulated replicates at its own clip.

| plan | morphology | clip b | sim cos (sd) | kept | clearance mm | max cmd b |
|---|---|---|---|---|---|---|
| `sv1_w6689_b060` | sv1_w6689 | 0.60 | 0.827 (0.001) | 4/4 | 8.5 | 0.60 |
| `sv1_w2360_b075` | sv1_w2360 | 0.75 | 0.726 (0.034) | 4/4 | 10.4 | 1.10 |
| `sv1_u1364_b080` | sv1_u1364 | 0.80 | 0.711 (0.001) | 4/4 | 5.0 | 2.00 |
| `g12_b095` | g12 | 0.95 | 0.627 (0.042) | 4/4 | 8.7 | 1.30 |
| `sv1_u0060_b75` | sv1_u0060 | 0.75 | 0.597 (0.019) | 4/4 | 9.9 | 2.00 |
| `sv1_u0308_b050` | sv1_u0308 | 0.50 | 0.585 (0.004) | 4/4 | 9.2 | 2.00 |
| `rv05_manual_b85` | rv05_manual_b85 | 0.85 | 0.568 (0.026) | 4/4 | 10.3 | 1.30 |
| `sv1_w0099_b100` | sv1_w0099 | 1.00 | 0.501 (0.020) | 3/4 | 8.2 | 1.30 |
| `g12w08` | g12w08 | 0.80 | 0.484 (0.059) | 4/4 | 8.7 | 1.30 |
| `sv1_u0060_b100` | sv1_u0060 | 1.00 | 0.441 (0.055) | 4/4 | 9.9 | 2.00 |
| `sv1_u7952_b065` | sv1_u7952 | 0.65 | 0.416 (0.044) | 4/4 | 5.4 | 2.00 |
| `sv1_w0116_b100` | sv1_w0116 | 1.00 | 0.372 (0.027) | 2/4 | 11.6 | 1.30 |
| `g12w11` | g12w11 | 1.10 | 0.267 (0.000) | 1/4 | 8.7 | 1.30 |
| `sv1_u0100_b70` | sv1_u0100 | 0.70 | 0.231 (0.007) | 4/4 | 7.5 | 1.30 |
| `g12` | g12 | 0.50 | 0.000 (0.000) | 0/4 | 8.7 | 1.30 |
| `sv1_u0100_b100` | sv1_u0100 | 1.00 | 0.000 (0.000) | 0/4 | 9.8 | 1.30 |

Three exported plans are **excluded before the study starts**, by the clearance gate, not by
their scores: `g23` (0.8 mm), `rv04_mid` (-2.6 mm) and `g24` (-5.2 mm) interpenetrate their own
fingers somewhere along the trajectory. `g24` is worth naming because the simulator scores it
0.750 — it would enter this table in second place. A hand the search likes and the body cannot
hold is exactly the failure mode a reconfigurable platform exists to catch, and it is a sentence
in the paper, not a trial on the bench.

The eleven morphologies are `sv1_w6689`, `sv1_w2360`, `sv1_u1364`, `g12`, `sv1_u0060`,
`sv1_u0308`, `rv05_manual`, `sv1_w0099`, `sv1_u7952`, `sv1_w0116`, `sv1_u0100`. Four of the
plans are `g12` at four clips (0.50 / 0.80 / 0.95 / 1.10), two are `sv1_u0060` (0.75 / 1.00) and
two are `sv1_u0100` (0.70 / 1.00). **Those repeats are not redundancy, they are the control**:
within a family the hand is identical and only the controller's residual clip moves, so any
bench difference inside a family is control, and any bench difference across families that is
not larger than it is not morphology.

Note what the simulator claims about those families. It says `g12` at 0.50 drops the tool and
`g12` at 0.95 turns it to 0.627 — the same steel, a different number in one config field. If the
bench does not reproduce a within-family ordering, the cross-family ordering means nothing, and
the study answers that before it answers anything else.

## 4. The measurement

Automatic, from the two AprilTags, at 0.017 deg / 0.030 mm rms:

- **primary — `cos_up`, averaged over the last 1.0 s of the hold.** Not the peak. A peak without
  a height check scores a dropped shaft as a perfect turn. The window is anchored to when
  **recording stopped**, and the station keeps the camera running `--post-roll 2.5` s past the
  end of the trajectory so that a hold window exists at all. Both of those were wrong on
  2026-08-31: the trace ended on the last instant of the motion, so "hold" was the last second
  of the *turn*, and on a run that lost its tag early the window slid back to whenever the tag
  was last seen and quoted the middle of the turn as the ending (two trials scored 0.359 and
  0.386 that way). The servos hold their last commanded position after a plan runs, so the
  dwell costs nothing but the seconds.
- `deg_from_up`, unfolded to [0, 180], so a turn to the wrong pole cannot score as a good one.
- **retention** — `dropped`, from a sustained fall (>= 25 mm for >= 0.4 s), never one low sample.
- `turned_deg` (first reading to last), horizontal `slip_mm`, `z` fall, tag visibility fraction.
- `u_px, v_px`, where the tag sat in the image. Pose alone cannot tell an occluded tool from one
  that swung out of the camera's view, and that was the open question after the first five
  trials.
- **the tape** — one IR frame every 0.25 s into `<trace>_frames/`, ~1 MB a run, assembled by
  `scripts/real_v1_tracker_replay.py` into an annotated mp4 and a phase-aligned filmstrip. The
  trace says the shaft turned 55.9 deg; only the pictures say whether it pivoted about a contact
  or about the floor, whether a finger walked off its pad, and which of two equal cosines was
  carried and which was flung.
- the operator's by-eye score, still recorded. It is the check on the instrument, and where the
  two disagree that disagreement is a figure, not an embarrassment.

**Parity with the simulator matters more than any single definition.** A bench trial that
released the tool contributes `cos_up = 0` and `kept = 0`, which is what the simulator's own
`ok = False` replicates contribute to `held_cos`. Score the two the same way or the correlation
is between two different quantities.

## 5. Stages

### Stage 0 — preflight, every sitting, ~5 min

1. `scripts/real_v1_tag_tracker.py --probe --exposure 4000 --gain 64 --shaft-axis=-x
   --tag-end top --symmetric-object --heading-deg 90`. Require: both tags found,
   **id6 decision margin >= 30 on the raw image**, frame mean in 20..235, and the sign line
   reading `--shaft-axis -x is CORRECT`.
2. Stand the tool upright (in the grip, or on the bench beside it) and re-probe. **A flat shaft
   cannot check the axis sign** — cos ≈ 0 puts the 71 mm offset horizontal and the height reads
   correctly either way — which is why the error survived until the first run stood the tool up.
3. `scripts/real_v1_trajectory_clearance.py` for the plans in this sitting.
4. Keep the probe PNG in the session directory. It is the record of what the instrument could
   see that day.

The id6 margin gate is not paperwork. On 2026-08-31 the same tag decoded raw at margin 62 at
14:11, needed histogram equalization to reach 35 at 15:20, was undecodable under every
preprocessing and every detector setting at 16:57, and was back at 62.6 at 17:06 — at unchanged
contrast and sharpness. Something bumps it. Probe every sitting, and re-probe after any
disturbance.

### Stage 1 — repeatability, one hand, n = 10, ~25 min

`sv1_w6689_b060`, whose simulated replicate sd is 0.001. Anything measured here is hardware.

This produces the number the rest of the study needs: **sigma_bench, within-plan**. It is also a
figure in its own right, because the reader's first question about a 16-point scatter is what
one point would have done if you had run it again.

Ten trials, full re-stage between each (open the hand, remove the tool, re-seat it), because a
re-grip is part of what a trial is.

### Stage 2 — the ranking, 16 plans x 4 trials = 64, ~3.5 h

**Two blocks. Within a block, all 16 plans in a randomised order, two consecutive trials at each
plan visit.** Randomise the order independently for each block; record the order.

Blocking is the load-bearing choice. Running four trials of one plan and moving on confounds the
ranking with anything that drifts over a sitting — servo temperature, grip staging habit, the
tag's mounting. Blocking makes drift orthogonal to design at the cost of 32 morphology changes
(~51 min of gantry motion) instead of 16. Pay it.

The two trials inside one visit share a staging, so they understate the true spread. That is
fine and it is why Stage 1 exists: report the within-visit and between-visit components
separately, and use the Stage 1 number for the error bars.

### Stage 3 — the diagnosis, ~45 min of new trials

- **(a) Commanded vs achieved. No new trials.** Every run's JSONL already carries commanded sim
  joint angles and servo telemetry. Extract the achieved/commanded ratio per joint at the turn
  end. The prior is 0.44-0.90 on yaw and ~1.00 on pip.
- **(b) Re-simulate each plan with the achieved trajectory** and repeat the correlation. This is
  the study's real payload. If the bench ordering is uncorrelated with the commanded-trajectory
  simulation but correlated with the achieved-trajectory one, then the simulator ranks hands
  correctly and the actuator is the entire sim-to-real gap — a result that names its own fix and
  is far more useful than a high correlation would have been.
- **(c) Clip band on hardware.** Two hands, `sv1_w2360` (band 0.50-0.80) and `sv1_u0100` (band
  0.60-0.90), at 5 clips each spanning inside and outside the simulated band, 2 trials each =
  20 trials. Does the band exist on the bench, and is it in the same place?
- **(d) Retention.** Already in every trial; no extra cost.

## 6. Exclusions, declared now

A trial is **void and re-run** (not a failure) when:
- the cylinder tag is lost for more than 0.5 s during the turn, or visibility is below 90%;
- the ending was never observed — no detection anywhere in the hold window, so `cos_hold` is
  `None`. **These are counted and attributed to their plan**, never quietly dropped: a tool that
  is flung out of the camera's view produces exactly this trace, so a void rule applied without
  bookkeeping deletes failures from the plans that fail. If a plan's unobserved endings outnumber
  its scored trials, that plan's number is the exclusion rate, not its cosine — read the tape
  before calling it a void.
- the grip released the tool before motion started, seen by the operator or by a fall in the
  pre-motion samples;
- the station refused the run (tracker failed to arm, servo gate, clearance gate).

A plan is excluded only by the clearance gate, decided before the study.

**No trial is ever excluded for its result.** A drop is data.

## 7. Pre-registered analysis

Implemented in `scripts/real_v1_transfer_study.py`, which runs on partial data so the analysis
can be exercised on Stage 1 before Stage 2 exists.

1. **Primary.** Spearman rho between simulated `held_cos` and bench mean `cos_up`, over the 16
   plans, with a BCa bootstrap 95% interval over plans. Report rho and the interval whatever
   they are. With n = 16 the two-sided 5% critical value is rho ~ 0.50, so this is powered for a
   strong ranking correlation and honestly underpowered for a weak one — which is stated here
   rather than discovered later.
2. **Secondary.** The same over the 11 distinct morphologies, taking each family's best plan, to
   check that the answer is not carried by the clip duplicates.
3. **Within-family contrasts.** `g12` across four clips, `sv1_u0060` across two, `sv1_u0100`
   across two. Same hand, so a bench difference here is control. Report each family's bench
   ordering against its simulated ordering.
4. **Retention agreement.** Simulated kept vs bench kept, 2x2 and Cohen's kappa.
5. **Achieved-trajectory correlation**, as Stage 3(b).
6. **Instrument vs operator.** Bland-Altman of the measured turn against the by-eye score.

Stage 2 runs all 64 trials regardless of what the interim numbers look like. No stopping early
in either direction.

## 8. The figures

- **Fig. 5, full width.** Simulated `held_cos` (x) against bench mean `cos_up` (y), 16 points,
  error bars on both axes (sim replicate sd; bench Stage 1 sigma), marker filled/open by
  retention, the three multi-clip families joined by thin lines, Spearman rho and its interval
  annotated. One panel, and it either supports the thesis or it does not.
- **Fig. 6, 2x2.** (a) achieved/commanded per joint over all trials; (b) Fig. 5 recomputed
  against the achieved-trajectory simulation; (c) clip band, simulated and measured, two hands;
  (d) retention 2x2, with the instrument's 0.017 deg noise floor as an inset so the reader can
  see the measurement is three orders below the effect.

## 9. Running it

The web UI's 1 (open) -> 2 (grip) -> 3 (reorient) is sufficient and is the recommended path. With
the CB1 started with `--tracker-url` and the workstation companion running, pressing *Run
reorientation* creates the run id, arms the camera on that exact id, waits for the reference
latch and the first delivered sample, moves, and finalizes the trace. A run that cannot arm is
refused rather than run blind.

Start the workstation companion once per session with the study's settings baked in, so they
cannot drift between trials:

```bash
MANTA_TOKEN="$MANTA_TOKEN" ~/miniconda3/bin/python scripts/real_v1_tracker_service.py \
  --host 10.99.99.50 --port 8770 \
  --tracker-arg --shaft-axis=-x \
  --tracker-arg --symmetric-object \
  --tracker-arg --heading-deg --tracker-arg 90 \
  --tracker-arg --exposure --tracker-arg 4000 \
  --tracker-arg --gain --tracker-arg 64 \
  --tracker-arg --video-hz --tracker-arg 4 \
  --post-roll 2.5
```

(Note the `--tracker-arg=--shaft-axis=-x` form for any value that starts with a dash; argparse
reads the spaced form as an option of its own.)

Afterwards, render what was taped:

```bash
~/miniconda3/bin/python scripts/real_v1_tracker_replay.py --all
```

`scripts/real_v1_bench_session.py` remains the tool for a sitting that needs its free-air control
arm, its grip-window arm, `max_u` truncation, or its self-describing manifest. Stage 2 needs none
of those.

Session directory: `docs/experiments/20260831-real_v1-transfer-protocol/sessions/YYYYMMDD-HHMM/`,
holding the probe PNG, the block orders, and a `NOTES.md`. The run JSONLs stay on the CB1 under
`logs/hardware/`; the tag traces stay on the workstation under `logs/tracker/`. Pull both into
the session directory at the end of the sitting — a trace whose run has been rotated away is
not evidence.
