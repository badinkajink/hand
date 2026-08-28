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
