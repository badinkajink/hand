# The reorient was blocked by the grasp, not the hand

2026-08-28. Follow-on to `SETUP.md` (the study) and `SLIP.md` (the grasp-elevation fix).
Data in this folder: `workspace.json`, `carry_*.json`, `hold_anchors.json`, `pivot_rv04_mid.json`,
`vertical_hold.json`. Narrative: `docs/rl/reorientation.md`, 2026-08-28 entry.

Four designs finished the A→B pipeline. All four hold the shaft through the continuous handoff
(min-z 0.104–0.116, three tips touching 100% of the time) and all four turn it by under four
degrees. Their mounts differ by 60 mm. **A failure that survives that much change of hand is not
a morphology result, and at four draws it is not the seed lottery.**

## The bound

Rotate the shaft by θ about the pinch axis with the pads on fixed material points. The pair
contact on the descending side moves down by `straddle · sin θ`; the only way its finger follows
is to **extend**. The mount-to-pad chain is 68.11 mm straight (yaw link 20.75 + 20.75 + 26.61), so

    ceiling = asin(extension_left / straddle)

`scripts/probe_real_v1_carry.py --workspace`:

    design         reach   extend  retract  half-straddle  ceiling   trained peak
    rv04_mid       66.8mm   +1.3    +46.8       30 mm       2.5 deg   0.015  (0.9 deg)
    rv00_wide      63.6     +4.5    +43.6       30          8.6       0.019  (1.1)
    rv05_manual    63.6     +4.1    +44.0       24.7        9.6       0.030  (1.7)
    rv03_narrowy   61.0     +7.1    +41.0       40         10.2       0.069  (4.0)

Every policy is at or under its own ceiling and the ceiling orders all four correctly. n=4, one
draw each — treat the ordering as suggestive; the bound itself is not statistical.

Both halves were ours. `fit_real_v1_pose` takes the **deepest reachable palm** (grip depth is
what clears the shaft's upper half when it stands up), which parks every design at 95% extension;
and it picks the straddle by scoring candidates on resisting *"the shaft levering about the pinch
axis"*, which is the reorientation itself. The obvious repair does not exist on this hand: below
~60 mm of grip depth the MCP pins against its −15° hyperextension stop and no pose is reachable.

    grip depth   max IK residual   joint margin   pinned            feasible
    65.0 mm         1.15 mm          0.201 rad    —                 yes
    60.0            1.59             0.000        thumb_mcp         no
    55.0            5.66             0.000        all three MCPs    no
    50.0            8.82             0.000        all three MCPs    no

Last night's reading — the vertical hold sits outside the ±0.5 rad residual budget — was true and
secondary: `rv04_mid` was inside its budget at 0.465 and turned 0.9°.

## The fix

Put the rotation axis **h above** the contact plane. Both pair contacts gain `h(1 − cos θ)`, so
the descending one needs no extension and the ascending one spends retraction, of which there is
38–47 mm. Physically the shaft slides down in the grip as it stands up. Open-loop, no policy, at
the full 0.10 m handoff height, floor 60 mm below the shaft's lowest point:

    design                          axis h   peak    final   obj z   contacts   force
    rv05_manual (the user's hand)    8.2mm   0.996   0.996   0.113      3       12.0 N
    rv00_wide_sp40                  15.5     0.995   0.989   0.097      2       20.6
    rv03_narrowy_sp40                5.8     0.936   0.909   0.110      2        5.1
    rv04_mid                        no cell holds — the design with 1.3 mm of extension

**The program's first floor-free reorient.** `REORIENT_PRIMITIVE.txt` measured 46–69% of both
reference policies' alignment as work done while the shaft still touched the floor.

Three traps, each of which gave a convincing wrong answer first:

* **Direction is not free.** The other rotation stands the shaft up at cos **−0.82**, and
  `target_axis_alignment` pays for cos → +1, not |cos|. The first training smoke of this schedule
  read `target_axis_progress −0.93` for that reason alone.
* **A straight line in joint space reproduces the IK'd carry** (0.82 vs 0.76 on rv03_narrowy), so
  the schedule collapses to one extra set-point — `LerpFingerActionCfg.hold_target_ctrl`, already
  in the repo, written for the opposed hand, never used here.
* **The anchor move must be bounded** — ±0.5 rad holds, ±1.5 drops the shaft in every cell.
* And `probe_real_v1_pivot.py`'s external-torque "breakaway" sweep measures **drops**: cos 0.82 at
  object z 0.048 with 0.2 N is a shaft standing on the table. Peak cosine without height and
  contact count is not a result, which `REORIENT_PRIMITIVE.txt` already says.

## Training

`hold_target_ctrl` latched on *alignment* — circular when the grasp is a rotational lock — so
`hold_switch_from_sim_step` (new) opens it on schedule and the reorient becomes a residual
problem. Recipe `b_liveA` plus `--hold-ctrl-from-keyframe hold_ik --hold-switch-from-sim-step 600
--hold-switch-steps 550 --hold-switch-min-z 0.08 --object-orientation-drift-weight 0.0`; tip-loss
termination 3 → 15 control steps (the carry runs on two contacts for most of the turn, as r4 does
on perp), drop termination and the collapse watchdog untouched.

**The anchor is env state, not policy state.** A B trained with it and evaluated without it is out
of distribution exactly as a residual-scale mismatch is (gotcha #13), so it is plumbed through
`deploy.make_env_cfg` and `rl_demo_handoff_continuous.py`. `--zero-b` is the control — the
scripted carry riding Policy A's real delivery — and `scripts/eval_real_v1_anchor.py` runs both.

## Universalisation, measured

`reorient_primitive.py execute`, retargeting the recorded primitives onto this hand:

    primitive     on              peak    final    final z   contacts   force
    r4 (perp)     rv03_narrowy    0.862   0.848    0.115       3.00     7.4 N
    r4 (perp)     rv05_manual     0.134   dropped  0.012       0        0
    b33 (m05)     rv03_narrowy    0.995   dropped  0.012       0        0
    b33 (m05)     rv05_manual     0.012   dropped  0.012       0        0
    computed here rv03_narrowy    0.936   0.909    0.110       2        5.1

The perp schedule transfers — 0.848 held on three fingers on a hand it has never seen, picking
out the same design the geometric carry does. b33's rolling schedule does not (its 0.995 comes
with final z 0.0124, i.e. on the table). But the transferred schedule does not beat one
*computed* from the object's own rigid-body kinematics, which is what `REORIENT_PRIMITIVE.txt`
§6 predicts: what transfers is the description, not the generator, because a third to a half of
each reference behaviour is authored by the floor and by gravity on an off-centre grip. Here
there is neither, so the schedule has to be generated.

## A design metric

`ceiling = asin(extension_left / straddle)` is one forward-kinematics call.
`scripts/real_v1_reorient_landscape.py` sweeps it over the two dimensions the workspace has and
scores each cell against the open-loop carry it claims to stand in for:

    X sep   Y sep        extend      ceiling        carry
    100     110/80/50    7.1-7.2mm   10.2-10.4 deg  HELD (final cos 0.52-0.79)
     70     110/80/50    1.4-1.6      2.0-2.3       drops
     40     110/80/50    0.0          0.0           drops

Nine of nine. The budget is set almost entirely by **thumb-to-pair X separation**; Y barely moves
it. A wide hand reaches inward and slants, keeping reach in hand; a compact one hangs its fingers
straight down at full extension. That inverts the intuition that compact is the manipulable
layout, and it is a hardware statement: **keep the X gantries wide**.

Two caveats. The landscape's carries use the fitter's geometric grip and only three pivot
heights, so they are a floor (X100/Y50 reads 0.575 here and 0.909 tuned with its CEM grip). And a
ceiling can only ever say a design *cannot* — clearing it does not mean a policy will find the
behaviour.

## Robustness of the scripted carry (rv05_manual, open-loop)

`carry_robustness_rv05.json`. This is what the RL residual has to absorb, not the policy's own
robustness. Read the rows against the baseline row: CPU contact solves still vary with the settle,
and the baseline reads 0.901 here against 0.996 in `probe_real_v1_carry` itself.

**Delivery height does not matter at all.** 0.995–0.997, three contacts, 10.9–13.0 N at every
lift from 0.06 to 0.14 m. Even at 0.06 the shaft's lower end stays 23 mm off the table, so the
turn is floor-free across the whole range.

    perturbation            peak    final    obj z   contacts   force
    baseline (mu 2.4)       0.901   0.901    0.113      3        4.2 N
    friction x0.5 (mu 1.2)  1.000   0.967    0.120      3       19.2
    friction x0.75          0.948   0.948    0.116      3        8.7
    friction x1.5           0.765   0.765    0.113      3        1.5
    friction x2.0 (mu 4.8)  ---- DROPS ----
    solimp dmax 0.999       0.729   0.729    0.119      3       53.6
    solimp dmax 0.9995      0.530  -0.986    0.102      2       50.3   (overshoots past vertical)
    object mass x0.5        0.846   0.846    0.118      3       24.1
    object mass x2.0        ---- DROPS ----

Two things worth carrying into hardware.

**Friction hurts in the direction nobody expects.** Lower friction is *better* — µ 1.2 gives the
cleanest turn in the table — and µ 4.8 drops the shaft. The carry needs the pads to slip a little
against the shaft as it comes round; too much grip and the fingers fight the object instead of
turning it. The scene's µ 2.4 is already high for TPU on steel, so the real hand sits on the
favourable side of this. That is the opposite sign from the inline hand's friction cliff
(`project_inline_sim2real_robustness`), and it is because this is a pivot, not a roll.

**Mass is the live risk.** The scene's cylinder is 24.5 g. At 2x it drops. A real screwdriver is
heavier than 24.5 g, so the object model needs the real tool's mass before any of this is a
hardware claim — and the anchor will need re-tuning (or the CEM grip re-optimising) when it does.

## 2026-08-28, trained results and the head-to-head

Four anchored runs finished (`results/rl/20260828-{1140,1215,1251,1327}-policyB_*`). Protocol:
`rl_demo_handoff_continuous.py`, one physical rollout each, deterministic actor, n repeats
because GPU contact solves are non-deterministic. `kept` = rollouts whose hold-phase min-z clears
0.05 m. The open-loop row is the SAME env with `--zero-b`, i.e. the scripted carry riding Policy
A's real delivery rather than a scripted grasp.

    design         method                          n  held-cos    sd     min-z  kept  force
    rv05_manual    RL, anchor (it 270)             3   +0.848   0.003    0.114  3/3   24.0 N
    rv05_manual    open loop, in harness           3   -0.915   0.000    0.096  3/3    5.8
    rv05_manual    baseline b_liveA (27 Aug)       1   -0.014     -      0.104  1/1    6.0
    rv03_narrowy   RL, anchor + gate (it 150)      3   +0.645   0.007    0.110  3/3   10.6
    rv03_narrowy   RL, anchor only  (it 150)       4   +0.629   0.031    0.075  3/4   10.4
    rv03_narrowy   RL, anchor only  (it 270)       4   +0.542   0.046    0.034  1/4   10.3
    rv03_narrowy   open loop, in harness           3   +0.343   0.464    0.022  0/3    0.0
    rv03_narrowy   baseline b_liveA (27 Aug)       1   +0.052     -      0.105  1/1    5.6
    rv00_wide      RL, anchor (it 270)             3   +0.222   0.017    0.126  3/3    8.2
    rv00_wide      open loop, in harness           3   +0.375   0.051    0.006  0/3    3.5
    rv00_wide      baseline b_liveA (27 Aug)       1   -0.093     -      0.116  1/1   17.4

**The open-loop schedule does not survive Policy A's delivery.** From a scripted grasp it reaches
0.909–0.996 on all three viable designs; from A's real delivery it keeps the shaft 0/3 on rv00 and
rv03, and on rv05 — where it does keep it 3/3 — it turns the shaft to the OPPOSITE POLE (−0.915).
The schedule is an open-loop joint trajectory and A's delivered object pose differs from the
scripted one; near vertical the carry sits at a bifurcation.

**What RL adds is not the trajectory. It is pole selection and grip regulation**: sd 0.003–0.031
against the open loop's 0.46, three fingers loaded at 8–24 N against 0–6 N, 3/3 kept against 0/3.
On rv05 both methods run the SAME anchor and end 1.76 apart in cos.

**Checkpoint choice is not monotone** — rv03's iteration 150 beats its own 270 on every column.
Judge on the handoff eval, never on the reward curve.

### Which finger slips (`contact_trace_rv05.json`)

Per-finger normal force, friction-cone utilisation |f_t|/(mu·f_n) and pad travel across the
shaft's own surface, through the open-loop carry on rv05_manual:

    cmd     cos     thumb fn/util/slip    index fn/util/slip    middle fn/util/slip
    -13d   0.081    4.2N / 0.11 / 0.4mm   2.2N / 0.49 / 0.7mm   2.0N / 0.74 / 3.4mm
    -22d   0.178    5.5  / 0.23 / 0.8     1.7  / 0.63 / 1.2     3.7  / 0.62 / 2.7
    -31d   0.315    6.2  / 0.25 / 1.2     1.2  / 0.67 / 1.6     5.0  / 0.46 / 2.1
    -49d   0.679    5.8  / 0.19 / 0.7     0.4  / 0.71 / 2.1     5.4  / 0.24 / 1.1
    -71d   0.817    6.8  / 0.13 / 0.7     5.3  / 0.30 / 1.1     1.5  / 0.44 / 0.4
    -88d   0.836   10.9  / 0.04 / 0.5     7.5  / 0.10 / 0.5     3.3  / 0.07 / 0.2

The slip is real, concentrated in the first third of the turn, and in the two fingers that supply
the couple. **The distribution is wrong, not the total**: at −13° the thumb has 2× the normal
force and a seventh of the utilisation — it pins the shaft and does no tangential work — while
the middle pad slides 3.4 mm per sample at utilisation 0.74 on 2.0 N. The intervention is to move
normal force from the thumb to the pair (per-finger squeeze in the CEM objective, or
`--finger-residual-scale-per-joint`), NOT to raise total grip force, which is already 24 N on the
best RL policy and flagged `over_clamp`. Past −70° every utilisation collapses to 0.04–0.10: the
shaft hangs into the grip and the fingers only hold it.

Full methods spec and head-to-head: https://claude.ai/code/artifact/3632efc4-8132-4f3c-bed9-4588705efc6d
