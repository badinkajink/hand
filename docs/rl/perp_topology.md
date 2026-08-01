# Perpendicular-finger topology ("perp") — opposed index/middle pair

Engineering log for the 2026-07-28 topology switch: **index and middle rotated 90° about the
palm Z so they FACE each other along Y**, thumb still along +X. Every finger is perpendicular
to its neighbour; index/middle are 180° apart. Requested as an exploratory "super underlying
morphology switch up" — can the existing stack grasp and reorient the screwdriver at all on a
hand this different?

Short answer: **yes, and better than the baseline hand does.** The headline is not the grasp,
it is that this topology makes the reorientation nearly free.

---

## The mechanism (the actual finding)

The opposed pair pinches the shaft **across** its axis. Two opposed point contacts on a
cylinder sit ON the axis of rotation about Y — they are a near-frictionless **pin joint**, so
they can carry the shaft but cannot control its pitch. On the baseline hand that would be a
defect. Here it is the whole point:

> Pinch the shaft **35 mm off its centre of mass**, lift, and **gravity does the
> reorientation**. The long side swings down and the shaft ends hanging vertically.

Because the object is respawned along world X, its body +Z (the reward's tracked axis) starts
along world +X, so "+X end up" is exactly `cos = +1` for the **signed** target-axis reward.
Pinching toward the +X end makes gravity rotate it the correct way. Nothing has to be learned.

Open-loop scripted probe on the frozen scene (`scripts/probe_perp_mechanism.py`), no policy:

| | result |
|---|---|
| final alignment cos | **+0.983** |
| object height | **0.133 m — airborne, no floor contact** |
| thumb contribution | **~0 N** |

Robustness to spawn jitter (`--object-x`, the RL env randomises the spawn):

| object x | −0.015 | −0.008 | 0.000 | +0.008 | +0.015 |
|---|---|---|---|---|---|
| peak cos | 1.000 | 0.998 | 0.993 | 0.974 | 0.879 |

It reorients across the whole range. It never fails flat.

**The thumb is not needed for the reorient.** The user's intuition was that the thumb would
come in on the way up to stabilise and reorient; it turns out gravity alone does the rotating
and the thumb registers ~0 N throughout. Its useful role is as a stabiliser/catcher, and CEM
does recruit it for the *grasp* (see below). Pressing it on the shaft's top does drive pitch
in the correct direction when the pinch is centred — it is simply dominated by the gravity
route once the pinch is off-centre.

---

## Geometry that had to be right (all found by rendering the scene and looking at it)

These are the failure modes; each one silently produced "this topology can't grasp".

1. **A facing finger must not reach far INWARD.** Ask for a tip 40 mm inboard and the target
   lands near the finger's *minimum* reach (0.0949 m, pip at its flexion limit). The 90 mm
   proximal chain then overshoots the midline and the distal link hooks back outward — the two
   fingers cross and self-collide. Mounts at |y| = 48 mm keep each arm nearly straight
   (|tip − mount| = 0.115 of a 0.12 reach) with its elbow outboard of its tip.

2. **The pinch must contact BELOW the shaft's equator.** A 10 mm pad cannot reach below the
   equator of a 12.5 mm shaft without burying itself in the floor, so it can only grip the
   converging upper flank. The shaft settles a few mm on lift-off, the contact rides up the
   shoulder, penetration and grip force decay together, and it drops. **Friction is never the
   limit** — measured friction-cone utilisation was 0.03–0.13. The *wedge sign* is. Narrow
   6 mm pads reach ~2 mm below the equator: the normal tilts upward and settling re-tightens
   the grip instead of releasing it.

3. **The distal phalange then becomes the binding contact.** These fingers meet the shaft
   almost vertically, so a 7.5 mm distal shaft grazes the upper flank just above the pad, with
   its normal tilted DOWN — it shoves the shaft into the floor and unloads the pad entirely.
   Slender 4.5 mm distal phalanges on the opposed pair fix it.

4. **The thumb opposes from ABOVE, not from below.** Its baseline pip range (−1.2) put the top
   of the shaft inside its minimum-reach shell; widened to −1.8.

5. **Palm height matters.** At the baseline 0.134 m the equator of a shaft on the floor is
   *outside* the fingers' reach (lowest reachable z = 0.014 > 0.0125). Palm lowered to 0.120.

---

## Grip force window

The open-loop grip has a **narrow squeeze window**, and the trade is grip-vs-rotation:

| pad set-point \|y\| | outcome |
|---|---|
| 0.0105 | too tight — holds firmly (z 0.164) but **blocks** the swing, cos 0.28 |
| **0.0140** | **cos 0.983, airborne at z 0.133** ← used |
| 0.0165–0.0180 | swings to vertical but touches down (cos ~1.0 at z 0.050, standing on floor) |
| 0.0195 | drops flat |

A fixed joint set-point also bleeds grip force under sustained load (9 N → 0.6 N over an
episode). A closed-loop policy that keeps squeezing should widen this window considerably —
this is the clearest thing for RL to add on top of the geometry.

---

## Pipeline status

The topology is expressed as a **base scene**, so the whole existing pipeline works unchanged:

- `assets/mjcf/perp/scenes/scene_screwdriver_medium_perp.xml` — base scene, morph joints intact.
  The 90° rotation lives on `<finger>_yaw_frame`, **not** `<finger>_mount`, so the morph x/y
  slide joints stay expressed in the palm frame and `create_rigid_morphology_xml` bakes the
  9-param design exactly as for every other scene. **A morphology sweep can be run over this
  topology as-is** — and since the pinch offset is baked into the hand (mounts at x = +0.035),
  the ±0.03 morph x range explores exactly the lever we now know dominates.
- `generate_morphology_xml.py` → rigid scene → `open_ik` keyframe → CEM → Policy A. All ran.

**CEM grasp** (`results/phase1/perp/perp_v1`, gitignored) — notable given this program's
history of degenerate 2-finger pinches and idle thumbs on the baseline hand:

| metric | value |
|---|---|
| cube_lift | 0.1026 (target 0.10) |
| cube_z_drop_from_peak | 0.0 |
| contact persistence, all three fingers | 1.0 |
| finger_persistence_imbalance | **0.0** |
| cube_axis_tilt | 0.036 rad |

A genuinely balanced 3-finger grasp — CEM *does* recruit the thumb for the grasp even though
the reorient does not need it.

---

## Open issue: the A recipe contradicts this design

`terminate_object_orientation_slip` uses the **full geodesic quaternion distance** from the
spawn orientation, threshold 0.5 rad — it fires on rotation about *any* axis. The `a_lift`
recipe assumes the object stays level during the lift, which is right for the baseline hand
and **exactly backwards here**: on perp, pitching during the lift is the desired behaviour.

### This is not theoretical — it killed the first run

`policyA_perp` was launched with the stock guard on the reasoning that it was self-consistent
(the CEM grasp holds the shaft level at 0.036 rad, so A *can* lift without tipping). The
collapse watchdog **aborted it at iteration 44**: `object_height` fell 0.09 → 0.0447.

The failure mode is instructive. Once airborne the shaft pitches — that is what this hand does
— and the episode terminates as `object_orientation_slip`. The one reliable way to avoid the
pitch is to **not lift at all**: on the floor the shaft cannot rotate. So the guard actively
selects against the design's own mechanism, and PPO found the degenerate optimum. Aggregate
reward never flagged it; the object-height watchdog did.

Relaunched as `policyA_perp_freeRot` with `--term-object-slip-yaw 10.0 --term-object-slip-xy
0.5` — the same relaxation `b_liveA` already uses for the rotation-permitting phase — keeping
drop and tip-loss terminations intact (gotcha #7: never strip those). Orientation-slip
terminations went to **0.0** and `object_height` held at 0.09–0.11.

### Two ways forward

1. **Relax the orientation guard for A on this design** (done above), or make the term
   pitch-aware, so A is allowed to deliver a partially reoriented shaft.
2. **Skip the A/B split entirely.** On this topology lift and reorient are the *same* motion.
   A single policy doing lift+reorient is a natural fit, and the A→B handoff seam — which has
   cost this program the most time — may simply not exist here.

---

## Reproduce

```bash
# look at the design
MUJOCO_GL=egl uv run python scripts/mj_snap.py \
  --scene assets/mjcf/perp/scenes/scene_screwdriver_medium_perp.xml --keyframe open \
  --views front,side,top,iso --out /tmp/perp.png

# the mechanism, open-loop, on the frozen scene
MUJOCO_GL=egl uv run python scripts/probe_perp_mechanism.py \
  --scene assets/mjcf/generated/scene_perp_tp0d0000p0d0000p0d0000_ip0d0000p0d0000p0d0000_mp0d0000p0d0000p0d0000.xml \
  --lift 0.16 --lift-steps 900 --press-steps 1400 --out /tmp/perp_probe.png

# re-author any keyframe from explicit fingertip world targets
MUJOCO_GL=egl uv run python scripts/pose_open_keyframe.py \
  --scene assets/mjcf/perp/scenes/scene_screwdriver_medium_perp.xml --keyframe open --out-keyframe closed \
  --index-tip "0.035 0.0140 0.0095" --middle-tip "0.035 -0.0140 0.0095" \
  --thumb-tip "-0.030 0 0.038" --write
```

`scripts/mj_snap.py` is the loop-closer: it renders a scene/keyframe from canonical viewpoints
into one tiled PNG *and* prints tip positions, object pose and per-contact forces, so the
picture and the numbers can be checked against each other. Every geometry bug above was found
by looking at the render, not by reading metrics.

---

## Training results (2026-07-28)

### Policy A — trains clean, but suppresses the reorient

`results/rl/20260728-2028-policyA_perp_freeRot` (from scratch, orientation guard relaxed).
Ran to completion, no collapse. Health gate WARN with every critical check passing:

| check | result |
|---|---|
| late_finger | PASS — thumb@2, index@1, middle@1 |
| idle_finger | PASS — all three at persistence 1.00 |
| drop | PASS — min hold-phase obj-z **0.111 m** |
| jitter | PASS — ang-jerk 12.7 |
| de_centering | WARN — path 5.6 cm vs net 0.0 cm |
| over_clamp | WARN — mean fingertip 9.2 N (thumb alone 22.3 N) |

**`peak_cos` under A is 0.014** — the shaft never rotates. A clamps hard and carries it level,
which is the same regime as the too-tight end of the squeeze sweep above (0.0105 → cos 0.28).
Relaxing the orientation *termination* was necessary but not sufficient: `a_lift` still *pays*
for holding the object still, so A trades away the free gravity reorient.

Note when reading `rl_eval_reorient_metrics.py` on an A run: its `min_z`/`drop` columns take the
minimum over the WHOLE rollout, which is dominated by the pre-lift floor at z≈0.012, so they
read `0.011 / 1.00` on a policy that never drops anything. `peak_cos` is the trustworthy column.

### Policy B — BOTH variants drop at the seam; the seam is the cause

| condition | drop check | lateral drift | reorient |
|---|---|---|---|
| **A alone, full 240 steps (control)** | **PASS — min hold-phase obj-z 0.111 m** | net 0.0 cm, path 0.0 cm | cos 0.014 |
| A→B, warmstart b33 | FAIL — 0.009 m | 15.1 cm | peak cos 0.543, obj_jerk 680 |
| A→B, **from scratch** | FAIL — 0.008 m | 20.2 cm | — |

`results/rl/20260728-2128-policyB_perp` (warmstart b33) and
`results/rl/20260728-2208-policyB_perp_scratch` (from scratch) fail the continuous handoff
**essentially identically** — the object is thrown at the switch step and the hand is empty for
the rest of the rollout (`handoff_perp.mp4`, `handoff_perp_scratch.mp4`).

The warmstart was first assumed to be the cause, by analogy with gotcha #5 (a warmstarted A
loads a grip-specific residual that ejects a re-CEM'd object). **That attribution was wrong.**
Training B from scratch — no b33 residual anywhere — reproduces the failure exactly, and the
control run settles it: with `--handoff-step 240` (A driving the whole rollout, no switch) the
same A holds perfectly for 240 steps — obj-z 0.111 m, net lateral drift 0.0 cm, ang-jerk 0.0,
only the over-clamp WARN. A's hold does not decay over the long horizon; the closed-loop policy
does not suffer the open-loop grip bleed measured earlier.

**The seam itself is the cause.** This topology's grip has a narrow stability window — the
squeeze sweep above spans hold-but-no-rotation (0.0105) to dropped (0.0195) over 9 mm of pad
set-point — and it is held by a two-point pinch that provides no pitch resistance at all. Any
discontinuity in finger targets at the policy switch lands outside that window and the shaft
escapes. A's hard clamp (thumb 22 N) is what keeps the object in the narrow window at all, and
B has no way to inherit it.

That makes the "collapse the A/B split" recommendation below a **measured result rather than an
aesthetic preference**: two independently trained B policies, and a no-seam control that passes
every health check the seam runs fail.

#### Detail: the b33-warmstart run

`results/rl/20260728-2128-policyB_perp` (live-A reset, warmstart b33). Trains without
collapsing, `peak_cos` 0.543 vs A's 0.014, but `obj_jerk` 680 vs A's 0.83. The continuous
handoff eval says why — **FAIL**, min hold-phase obj-z 0.009 m, net lateral drift 15.1 cm,
object flung at 3 m/s. The video is unambiguous: A lifts cleanly, then at the handoff step B
throws the shaft away and the hand is empty for the rest of the rollout
(`docs/rl/videos/20260728_reorient/2207_handoff_perp.mp4`).

`train_handoff_liveA_reset.sh` now accepts `B_CKPT=none` to train B from scratch, mirroring
`WARMSTART=none` in `train_A_on_morph.sh`. That is the right default here: this topology's
reorient is nearly free (the policy mostly has to STOP clamping), so it does not need the
hard-exploration prior that the b33 warmstart exists to supply.

---

## Axial load capacity — the screwdriver requirement

Holding a screwdriver is not the same as holding a mass. In use the driver is pushed into the
screw and the screw pushes back UP the shaft, so the functional requirement is an axial (world-Z)
load capacity in both directions, not just enough friction to carry 0.24 N of weight.

`scripts/probe_axial_load.py` measures it: reach the held state, then ramp an external force on
the object along +Z / -Z until it slips more than 10 mm **relative to the palm** (a world-frame
threshold would just measure the palm actuator).

| grip set-point | reorient cos | grip force | escape +Z | escape -Z |
|---|---|---|---|---|
| loose (pad \|y\| 0.0140) | **0.983** — vertical | 0.3 N (bled) | 1.24 N | 0.47 N |
| tight (pad \|y\| 0.0105) | 0.276 — barely rotates | 14.7 N | 14.62 N | 1.09 N |

Two things fall out.

**1. Capacity is just μN, so the lever is holding grip force.** In the loose/vertical case the
open-loop grip has bled to 0.3 N per pad; μ = 2.4 over two pads predicts 1.44 N, and 1.24 N was
measured. Axial capacity is therefore *not* a geometry limit — it is a direct readout of how much
normal force the grip is still carrying when the shaft reaches vertical. An open-loop set-point
cannot maintain that (the bleed is what it is); a closed-loop policy can, and 9 N of grip would
predict ~43 N of axial capacity. This is exactly what `perp_single`'s `grip_force_reduce: min`
plus the contact-min term are there to buy.

**2. The thumb currently contributes nothing to load capacity either.** With the press phase on
vs off: 1.24 vs 1.20 N up, 0.47 vs 0.58 N down — inside the noise, and its measured contact force
is 0.0 N. The thumb is geometrically stranded: it mounts at x = −0.065 and the shaft, once
vertical, hangs at x = +0.035, which is outside the thumb's reach shell. **Making the thumb
load-bearing on the reoriented shaft needs a geometry change, not a reward change** — either move
the thumb mount toward +x, or give the pinch a smaller x-offset so the vertical shaft lands
nearer the thumb. That is the concrete next design iteration.

The tight-grip row also should not be read as a real axial measurement: at cos 0.276 the shaft is
still nearly horizontal, so a Z force is transverse and levers about the pinch rather than
sliding along the shaft. That asymmetry (14.62 up vs 1.09 down) is the lever arm, not friction.

---

## Single-stage policy, attempt 1: still clamps — and the reward table says why

`results/rl/20260729-1225-perp_single` (single policy, no A/B seam, orientation TERMINATION
removed, target-axis reward on). Trains clean, no collapse, object held at 0.135 — and
**`peak_cos` 0.015**. It clamps, exactly like Policy A did.

The episode reward breakdown is unambiguous about the cause, and it is not the grip terms:

| term | value |
|---|---|
| contact_min | +9.30 |
| contact_mean | +6.23 |
| lift_height | +5.75 |
| **track_object_quat** | **+0.72** — pays for matching the SPAWN quaternion |
| grip_force / grip_force_spread | +1.19 / +1.34 |
| **target_axis_alignment** | **+0.115** — the term that wants the pitch |
| **target_axis_progress** | **−0.18** |
| **object_orientation_drift** | **−0.93** — penalty for rotating away from spawn |

**Removing the termination was necessary but nowhere near sufficient.** Two *base* reward terms
still pay for keeping the shaft at its spawn orientation — `object_orientation_drift` (−0.93) and
`track_object_quat` (+0.72) — roughly 1.65 of pressure against rotating versus 0.115 for it. The
policy is being paid an order of magnitude more to hold the shaft still than to turn it. This is
the same trap as the termination, one layer down, and it is invisible unless you read the
per-term table rather than the aggregate reward.

`object_orientation_drift_weight` was already a knob; `track_object_quat`'s weight was hardcoded
from `DEFAULT_REWARD_WEIGHTS`, so `track_object_quat_weight` was added (default `None` = the
built-in value, so every existing run is unchanged). Both are zeroed in `perp_single`, the
target-axis weights raised (250 / 600) so they are not dwarfed by contact_min + contact_mean, and
the contact/grip weights trimmed since they pull toward the rotation-blocking tight grip that the
axial probe measured. Relaunched as `perp_single_freeRot`; the live log confirms
`track_object_quat 0.0000` and `object_orientation_drift 0.0000`.

---

## Attempt 2 (`perp_single_freeRot`): the terms were silenced, and it still failed — but NOT the way the reward table reads

`results/rl/20260729-1316-perp_single_freeRot`, 339 iterations, 42 min, trained clean: object
held at 0.171, no collapse, both anti-rotation terms confirmed at `0.0000` all run. Final
reward table:

| term | value |
|---|---|
| track_object_quat | **0.0000** (silenced ✓) |
| object_orientation_drift | **0.0000** (silenced ✓) |
| target_axis_alignment | +0.028 |
| target_axis_progress | **−0.524** |
| contact_min / lift_height | +3.55 / +4.87 |

Read as a reward table this says "still won't rotate" — alignment flat at ~0.03 after the terms
that were blamed for suppressing it were removed. **That reading is wrong**, and the render is
what proved it.

### It rotates hard. It rotates the WRONG WAY.

Deterministic eval (N=32): **`held_cos` −0.449**, `peak_cos` 0.127, mean peak |cos| **0.655**,
**28/32 envs end negative**. Per-step trace:

| step | 0 | 20 | 30 | 40 | 50 | 65 | 100 | 249 |
|---|---|---|---|---|---|---|---|---|
| cos | −0.00 | +0.02 | −0.02 | −0.30 | −0.46 | −0.48 | −0.46 | −0.38 |
| obj z | 0.012 | 0.011 | 0.123 | 0.165 | 0.170 | 0.171 | 0.161 | 0.138 |

The gravity swing this topology was built around **is happening, at full magnitude** — it just
lands on the opposite pole. `target_axis_alignment` uses `exp(−α(1−cos))` with α=4, which is
≈0.003 for any cos < 0: flat, no gradient, indistinguishable from "never tried". The persistent
negative `target_axis_progress` was the only honest signal in the table and it was read as noise.

**Two things follow, and both are timing/sign facts that no reward-weight change addresses:**

1. **The rotation is over before the reward starts.** cos commits between step 20 and 50 (during
   the lift ramp); `reorient_start_step` is **65**. The reorient reward is a spectator to the
   only decision that matters. Raising its weight from 250 to 2500 would change nothing.
2. **A negative held-cos is invisible in every aggregate**, including `peak_cos` — which just
   picks up noise around zero on a wrong-way swing.

### Same scene, open-loop: +0.957. So it is not the morphology.

The decisive control: the scripted probe on the **exact frozen scene the RL run uses**
(`results/phase1/perp/perp_v1/frozen_scene.xml`) reaches **final cos +0.957, held**. Same hand,
same object, same lift height. The geometry is fine; a morphology sweep launched off the reward
table would have been days spent on the wrong variable.

### The real variable: the reorient is GRIP-FORCE-GATED, window ≈ 4–9 N per finger

Substituting grips into that open-loop probe (symmetry preserved, only clamp depth varied)
gives a clean monotonic curve — and the CEM grip the RL run inherits sits far outside the window:

| per-finger force (N) | 1.2 | 3.8 | 6.0 | **8.3** (authored) | 9.9 | 11.4 | 12.9 | 14.3 | **21.1** (CEM) |
|---|---|---|---|---|---|---|---|---|---|
| final cos | DROP | 0.965 | 0.977 | **0.957** | 0.656 | 0.401 | 0.320 | 0.292 | **0.298** |

Below ~3 N it drops; above ~10 N the pitch is choked off. The mechanism is exactly as documented
— two opposed contacts on the axis are a near-frictionless **pin joint** — but only while the
friction torque about that axis stays under gravity's moment. Clamp harder and the pin joint
becomes a rigid clamp. **Force is the dominant variable for reorient on this topology.**

This is a **topology-specific reversal of a durable program lesson**. On the baseline hand grip
force was measured to be decoupled from hold quality and jitter, and the standing guidance is
"stop re-chasing grip force". That lesson does **not** transfer to the perp hand, where force is
the thing that gates the task.

### The inherited CEM grip is the proximate cause

`finger_default_ctrl` in the run config is the CEM grasp. Rendered against the authored
keyframe on the same scene (`mj_snap.py --ctrl … --contacts`):

| | authored `closed` | CEM grip |
|---|---|---|
| index tip / middle tip | `[0.035, +0.0174, 0.0085]` / `[0.035, −0.0174, 0.0085]` — symmetric | `[0.0236, +0.0111, 0.0076]` / `[0.0266, −0.0201, 0.0077]` — skewed in x, y and z |
| tip forces | 3.74 / 3.74 N — balanced | 11.34 / 5.44 N — 2:1 |
| pip forces | 4.55 / 4.55 N | 9.91 / 15.34 N |
| total | ~16 N | **~44 N** |
| thumb | 0 N (free) | 2.71 N (engaged) |
| pinch offset from COM | 33 mm | 26 mm |
| open-loop result | **+0.957** | **+0.298** |

CEM optimises **grasp stability** — force closure — and on an opposed pair that means clamping
hard and breaking the symmetry to wedge the object. That objective is *directly opposed* to the
reorient mechanism, which needs a light, symmetric, off-COM pinch that lets the shaft pivot.
The grip alone costs 0.957 → 0.298; the learned residual on top of it takes 0.298 → −0.449.

### What this does and does not license

- Confirmed: the mechanism works on this exact morphology, open-loop, at +0.957. The thumb reads
  0 N throughout — still not needed for the rotation, consistent with the original probe.
- Confirmed: the failure is the **inherited grip + reward timing**, not the finger layout. The
  thumb-reach limitation (mounts at x −0.065, vertical shaft at x +0.035) remains real, but it
  is a *post-reorient stabilisation* question, not the reason the reorient fails.
- Not established: that any reward re-weighting alone fixes this. The two structural problems —
  a reward gated after the event, and a grip outside the force window — are not weights.

Tooling added for this: `scripts/policy_filmstrip.py` (phase-aligned frames from a run's own
eval video into one PNG) and camera/resolution overrides on `rl_render_reorient.py`
(`--width/--height/--distance/--elevation/--azimuth`; `viewer_*` fields on `MorphoHandEnvCfg`
default to the previous hardcoded values, so existing runs are bit-identical). Skill:
`policy-eyes`.

---

## Thumb morphology sweep (2026-07-30): the thumb CAN be recruited — but not on a shaft that reoriented

User's call after the grip-force finding: change the morphology and see whether thumb support can
be elicited. Swept with `scripts/sweep_perp_thumb.py` (new), which per design generates the rigid
scene, gates on self-collision, stows the thumb clear of the swing corridor, runs
settle → close → lift → hold, then IKs the thumb onto the settled shaft and ramps an axial load.
New helper `scripts/probe_thumb_reach.py` reports the reach geometry alone.
Table: `docs/experiments/perp_thumb_sweep.md`.

### Result 1 — `thumb_x` is unusable. The whole forward half of the range self-collides.

Every design with thumb_x ≥ +0.010 buries `thumb_mcp_frame` in `palm_pose`, 497–689 N:

| thumb_x | +0.000 | +0.010 | +0.020 | +0.030 |
|---|---|---|---|---|
| verdict | ok | INVALID 499 N | INVALID 687 N | INVALID 689 N |

So "move the thumb toward +x", the obvious fix for a thumb that cannot reach a shaft at x = +0.035,
is not in the design space at all — it needs a palm change, not a morphology parameter. Any sweep
that did not gate on self-collision would have reported these as ordinary results; the contact
forces are large enough to throw the object across the scene.

### Result 2 — `thumb_len` buys the reach for free

+35 mm of thumb closes the 30.6 mm reach deficit at **zero cost to the swing**:

| thumb_len | reach shell | d(mount→shaft) | verdict | swing cos |
|---|---|---|---|---|
| 0.000 | [0.0882, 0.1200] | 0.1506 | SHORT by 30.6 mm | +0.986 |
| 0.020 | [0.1072, 0.1400] | 0.1506 | SHORT by 10.6 mm | +0.986 |
| **0.035** | [0.1217, 0.1550] | 0.1506 | **INSIDE** | **+0.986** |

### Result 3 — reach is not support, and the two objectives fight over one parameter

Moving the pinch (index+middle x, applied symmetrically) trades the reorient against the thumb,
because the same offset is the swing's lever arm *and* the thumb's reach deficit:

| index/middle x | pinch x | swing cos | d(mount→shaft) | thumb N | axial +Z / −Z |
|---|---|---|---|---|---|
| **−0.030** | 0.005 | **+0.168** | 0.1194 INSIDE | **4.0** | **9.21 / 4.48 N** |
| −0.020 | 0.015 | +0.618 | 0.1367 | 0.0 | dropped at press |
| −0.010 | 0.025 | +0.953 | 0.1390 | 0.0 | dropped at press |
| 0.000 | 0.035 | +0.986 | 0.1506 | 0.0 | dropped at press |
| +0.010 | 0.045 | +0.993 | 0.1861 | — | SHORT by 30 mm |

**The thumb absolutely can be recruited: 4.0 N of thumb contact and axial capacity 9.21 N up /
4.48 N down, against the baseline's 1.24 N — a 7.4× gain.** It just requires the pinch essentially
on the COM, where cos stays at 0.168 and there is no reorient. Every design that reorients well
loses the object the moment the thumb touches it.

### Result 4 (the one that reframes the program) — the perp hand does not HOLD the reoriented shaft

"Dropped at press" is not the thumb's fault. Running the hold past the horizon the original probe
used shows the pinch unloading as the rotation completes:

| steps after lift | 0 | 400 | 800 | 1200 | 1600 | 2000 |
|---|---|---|---|---|---|---|
| cos | +0.622 | +0.807 | +0.967 | +0.989 | +0.998 | +1.000 |
| obj z | 0.1266 | 0.1210 | 0.1146 | 0.1137 | **0.0506** | 0.0500 |
| grip N (each) | 8.11 | 7.53 | 5.18 | 1.82 | **0.00** | 0.00 |

**Grip force and alignment are anti-correlated all the way to zero.** The shaft rotates to vertical
*by sliding out of the pinch*, and then falls. The documented headline (cos +0.983, "airborne,
HELD") is a **short-horizon artifact**: the original probe stops at step 2200, right in the window
after the rotation completes and before the shaft escapes. Extend it by 800 steps and the run ends
with the screwdriver **standing upright on the floor** — which reports `cos +1.000, z 0.050, HELD`,
the best possible numbers, for a drop. The filmstrip shows it plainly.

Two artifacts fixed as a result:

* `probe_perp_mechanism.py`'s verdict was `final_z > 0.03`. A 100 mm shaft standing on the floor
  sits at z = 0.050 and passed. It now asks the physics — total grip force and object↔floor
  contact — and reports `RELEASED-on-floor` for exactly that state.
* **`object_min_z: 0.05` in the `perp_single` recipe does not block it either.** The termination is
  `z < min_z`, and a standing shaft's centre is 0.0500, so it never fires while
  `target_axis_alignment` pays +1.000 forever. On this topology that is a reachable reward hack,
  and it is *specifically* a hazard for the recommended change of moving the reorient reward
  earlier and raising its weight. Raised to **0.09** (held-vertical sits at 0.113).

### Where this leaves the two options

The morphology route answers the question asked and returns a real number — 7.4× axial capacity
with the thumb engaged — but it cannot be combined with the gravity reorient inside the 9-param
space: `thumb_x` is blocked by the palm, and pinch offset cannot be small (thumb reaches) and
large (shaft swings) at once. Getting both needs geometry outside the current parameterisation
(a palm that lets the thumb sit forward), not another sample of it.

Separately, **the reorient itself now needs re-examining before more is built on it**: on the
current hand the vertical shaft is not held at all. "Reorient" here means "release into a vertical
pose", which is a different capability from in-hand reorientation and is not what the RL task is
scored on.

---

## 2026-07-31 — train the reorient instead of scripting it: the first HELD reorient (cos +0.947)

The open question the section above closes on — "on the current hand the vertical shaft is not
held at all" — now has an answer, and it is better than expected. **It is held.** The release was
never a property of the hand; it was a property of the scripted open-loop probe.

User's call: skip the scripted two-phase grip schedule and train a reorienter suited to the perp
hand. That turned out to be right, because three of the four things blocking the reorient were
reward-level defects and the fourth was a missing phase gate — none of them needed a script.

### First: the thumb reach was measured through the palm

The preceding work widened the `thumb_x` range to a T-shaped workspace and picked `+0.0325` from
an FK reach study, where the tip closes to within 1 mm of a held-vertical shaft. **That position
is inside the palm.** Settled contact force, `thumb_mcp_frame` vs `palm_pose`:

| thumb_x | +0.000 | +0.010 | +0.020 | **+0.0325** | +0.0525 |
|---|---|---|---|---|---|
| force | 0 N | 499 N | 687 N | **689 N** | 690 N |

This restates Result 1 of the thumb sweep, which had already flagged `thumb_x >= +0.010` as
INVALID — the FK study did not see it because forward kinematics does not model the palm. The tip
does reach the shaft; it reaches it *through* solid geometry. Two consequences:

* **A reach result is necessary, never sufficient.** Gate on `robot_self_collision` before
  treating any thumb design as real. Nothing in the generate/sample path does this for you.
* **The scripted press failed for an unrelated reason.** A press at `thumb_x +0.0325` reported
  "cos 0.000, z 0.012, all three fingers 0 N" — that is the 689 N palm contact ejecting the
  shaft, not a grip-sequencing failure. No conclusion about press sequencing survives from it.

Recorded in the scene XML beside the joint range, since the range itself still spans the invalid
region (the box is the mount rail, and the rail passes through the palm). Moving the thumb
forward remains a palm change, not a morphology parameter.

Everything below therefore runs at `thumb_x = 0`, which costs the reorient nothing: the thumb
reads 0 N throughout the successful swing anyway.

### Two infrastructure traps, both of which silently produce garbage

**1. `init_noise_std` 0.3 (the default) NaNs this scene.** The first launch died at iteration 0 —
and so did a faithful replay of run `20260729-1316`, which had trained 339 iterations clean. The
perp hand is dynamically fragile under exploratory actions in a way the baseline hand is not.
512 envs, 260 steps:

| actions | zero | randn×0.05 | randn×0.15 | randn×0.5 | randn×1.0 |
|---|---|---|---|---|---|
| result | stable, max\|qvel\| 0.05 | stable, max\|qvel\| ~15 | NaN @ step 227 | NaN @ step 14 | NaN @ step 12, \|qvel\| 2e13 |

One env in 512 going NaN aborts the entire run through `rsl_rl`'s `check_nan`. The run that
worked had passed `0.05` as a bare CLI flag, invisible to anyone reading the recipe, so the next
launch inherited the 0.3 default and died — exactly the flag-soup parity bug the recipe layer
exists to kill. Now pinned in `perp_single`.

**2. The 3-tip contact requirement is unsatisfiable on this hand.** The first launch that trained
looked healthy on reward, but its in-training metrics did not: **mean episode length 62 of 250**,
`tip_lost` 128.8, and `contact_min` / `grip_force` both flat at `0.0000` for the whole run. Every
episode was terminating three steps after the lift phase opened; nothing past the lift was ever
trained. `tip_lost` required all three tips, `contact_min` was min-over-3, and `grip_force` used
`reduce=min` — all three gated on a thumb that cannot touch the shaft at any pinch offset which
still permits the reorient. Two reward rows sat in the table looking active while being
structurally incapable of taking a non-zero value.

The CEM grip had masked this by happening to engage the thumb at 2.71 N; the lighter authored
grip exposed it. New `min_tips_in_contact` (default 3, so every prior run is bit-identical) is
set to 2 here, and `contact_min` becomes the worst of the k best tips. This adapts gotcha #8's
guard to the topology rather than stripping it — drop, slip and floor-proximity are untouched,
and losing either *opposed* finger still terminates. Effect, same config otherwise:

| | episode length | mean reward | target_axis_alignment | target_axis_progress |
|---|---|---|---|---|
| 3 tips required | 62 | 61 | 7.1 | +0.27 |
| 2 tips required | 149 | 335 | 49.1 | **+1.76** |

### The four fixes to the reward, against the −0.449 run

1. **`reorient_start_step` 65 → 25.** The pole is chosen between step 20 and 50, so the only
   signed term (`target_axis_progress`) was switched off during the very decision it grades.
2. **`target_axis_alpha` 4.0 → 2.0.** `exp(−α(1−cos)²)` is ~1e-7 for cos ≤ −0.5, so a
   full-magnitude rotation to the wrong pole scored identically to never moving.
3. **The grip set-point comes from the authored `closed` keyframe, not CEM** (new
   `--closed-ctrl-from-keyframe`, and the matching flag on `policy_healthcheck.py` — evaluating a
   keyframe-grip policy against the CEM grip is gotcha #13 in another coordinate, 21 N vs 8 N).
4. **The grip rewards are phase-gated** (new `grip_phase_start_step`). The reorient needs a loose
   pinch, but the shaft reaches vertical by sliding out of one, so the grip must firm up
   afterwards. One always-on weight cannot ask for both. This is the user's two-phase schedule
   expressed as something the policy *learns*, not a scripted ramp.

### The result: the release was an artifact of the scripted probe

Run `20260731-1146-perp_single_r2`, 339 iterations, 41 min, no collapse. Deterministic trace,
run to 800 steps — well past its own 250-step training horizon:

| step | 0 | 100 | 200 | 300 | 400 | **484** | 500 | 520 |
|---|---|---|---|---|---|---|---|---|
| cos | 0.000 | +0.296 | +0.441 | +0.615 | +0.848 | **+0.947** | +0.940 | 0.000 |
| obj z | 0.012 | 0.141 | 0.134 | 0.125 | 0.115 | 0.111 | 0.111 | 0.012 |
| grip (N) | 17.5 | 20.2 | 22.5 | 24.5 | 22.3 | ~10 | 6.2 | 0.00 |

**Grip force and alignment are no longer anti-correlated.** That was the headline finding of the
section above — the shaft rotates to vertical *by sliding out of the pinch*, grip decaying to
0.00 N, ending upright on the floor with a perfect-looking cos. This policy instead climbs
monotonically to **+0.947 with the grip maintained at 10–27 N and object z never leaving
0.111–0.145**. It is walking the shaft around between the two opposed pads. That is in-hand
reorientation, which is the capability the task is actually about, as opposed to "release into a
vertical pose".

Scorecard at the trained 250-step horizon: **held-cos +0.372**, peak 0.435, `drop` PASS at
min hold-phase z 0.134. (Previous published run: held-cos −0.449.) The `idle_finger` FAIL is the
thumb at 0 N and is structural on this topology, not a policy defect — the scorecard was written
for the baseline 3-finger hand.

### What it costs, and the defect that exposes

The held reorient is **~4× slower than the gravity swing** — ~480 steps against ~120. That is the
price of holding on: the fast version gets to vertical by letting go. And it exposes the next
gate-timing defect, the same error as the original step-65 reorient gate one layer out:

* the episode is 250 steps, so training ended at cos ~0.52 with the rotation still in progress;
* the `grip_phase_start_step` "catch" reward was gated at 150 — less than a third of the way
  through a manoeuvre that finishes at 484;
* the loss of the shaft at ~step 510 happens 260 steps into **untrained extrapolation**, so it is
  not evidence about what the policy would do if the completion were scored.

Revision 3 (running): `episode_length_s` 5 → 12 (600 steps), grip and ang-acc gates 150 → 450.

> **Gate timings are trajectory-relative, not absolute.** Both defects found here were a phase
> gate pinned to a step count that came from a differently-shaped trajectory. When the horizon or
> the manoeuvre speed changes, re-measure every `*_phase_start_step` against the actual trace.

---

## Revisions 3 and 4: gate the catch on ROTATION, and measure policies as distributions

### Revision 3 failed — and the way it failed is the useful part

600-step horizon, catch gate moved 150 → 450. It peaked at iteration 50 and then degraded
monotonically until it NaN'd at 188:

| iter | 18 | 50 | 80 | 110 | 140 | 170 | 187 |
|---|---|---|---|---|---|---|---|
| reward | 734 | **883** | 783 | 571 | 528 | 452 | 430 |
| target_axis_alignment | 45.8 | **59.5** | 54.2 | 41.0 | 35.1 | 30.7 | 28.8 |
| episode length | 409 | 362 | 313 | 239 | 203 | 170 | 165 |

The longer horizon exposed the drop at the top of the rotation as a terminal penalty, but the
catch reward that would have taught the policy to hold through it was gated at step 450 while
episodes had shortened to 165 — so it never fired. With no reachable way to hold on, the only
remaining way to stop losing the shaft was to stop rotating, and PPO took it.

**Both step-gate attempts were wrong, in opposite directions**: 150 covered less than a third of
a 480-step manoeuvre; 450 then missed entirely. A wall-clock gate must be guessed from the
*previous* run's trajectory and is invalidated by the very learning it exists to shape. New
`grip_phase_align_thresh` gates on task progress instead — "clamp firm once it has actually
rotated past cos 0.7, however long that took". (This was the user's original framing, "clamp-firm
on a rotation threshold"; the first implementation rendered it as a step count and got it wrong
twice.)

### Revision 4: the reorient is solved; the indefinite hold is not

`20260731-1300-perp_single_r4`, 339 iterations, 41 min, no NaN, and unlike r3 it improves
monotonically end to end (reward 234 → 1375, alignment 14 → 100, episode length 85 → 279).

### Single rollouts of this stack are not reproducible

Three deterministic rollouts of the same r4 checkpoint — stochastic sampling off,
`cube_spawn_*_jitter` at 0 — ended three different ways (held at step 400 / lost at 430 / lost
before 400). The actions are deterministic, so the spread is the simulator: parallel contact
solves do not reduce in a fixed order on GPU. **A single trace cannot support a claim about a
policy**, which invalidates the method used for every per-policy number in the sections above.
New `scripts/policy_eval_suite.py` runs N envs batched (N rollouts ≈ one rollout of wall-clock)
and reports rates with spreads. Skill: `policy-metrics`.

### N = 64, 500 steps

| metric | r2 (step-gated catch) | r4 (align-gated catch) |
|---|---|---|
| align_rate (ever cos ≥ 0.9) | 71.9% (46/64) | **100% (64/64)** |
| t_align (steps) | 409 ± 29 | **89 ± 46** |
| peak_cos | 0.905 ± 0.090 | **0.996 ± 0.006** |
| hold_steps (aligned AND held) | 49 ± 41 | **331 ± 55** |
| hold_rate @ step 500 | 45.3% | 0% |
| drop_step | 443 ± 30 (35/64 envs) | 433 ± 32 (64/64) |

**r4 reorients reliably and fast**: every rollout reaches vertical, at cos 0.996, in ~89 steps
(1.8 s), and stays vertical and held for ~331 steps (6.6 s). Against the original run's
held-cos of −0.449 that is the task, solved.

**Neither holds indefinitely, and they fail at the same absolute step.** r2's 45% `hold_rate`
is an artifact of the 500-step horizon truncating before the slower policy finishes failing —
`hold_rate` at a fixed step ranks a slower policy higher, which is the same short-horizon trap
documented earlier in this file, reappearing inside the new metric. Compare on `hold_steps`.

That both policies lose the shaft at ~435–443 steps *regardless of when they got vertical* is
the pointer to the remaining defect. The 4-panel eval plot shows it directly: object z slips
**~12 mm monotonically** across the whole hold (0.120 → 0.108) while grip decays to ~6 N; the
align-gated catch does fire (grip climbs back to ~16 N around step 300) but does not arrest the
slip, and at ~440 steps the shaft has simply run out of pad. The open problem is therefore no
longer "does it reorient" or "does it release" — it is a **slow steady slip**, and the lever is
something that penalises the slip rate or re-seats the grip, not more rotation reward.

Artefacts: `docs/rl/videos/20260731_perp/` (960×720 render, filmstrip, per-run and overlay
training curves, N=64 eval plots), `docs/experiments/perp_r{2,4}_eval.json`.

---

## 2026-07-31 (later) — the perp hand had no hand file, and the workspace was never sampled

User's observation, and it is correct: every perp result to date — the scripted probes, the
thumb sweep, r2/r3/r4 — ran on **one morphology, the all-zeros corner of the workspace**.
`PERP_T_WORKSPACE` was written down in `morphohand.sampling.morphology` when the T-shaped mount
layout was specified, and then **never referenced by anything**. It was dead code. The shipped
mount positions are not a chosen design; they are simply where the scene was authored.

Two mechanical reasons that persisted, both now fixed:

**1. The perp topology had no hand XML.** `assets/mjcf/perp/` contained only `scenes/`, and
`create_rigid_hand_and_scene_xmls` needs a hand/scene *pair*. Everything that generated a perp
design therefore passed the scene as **both** arguments — which is why
`assets/mjcf/generated/hand_perp_*.xml` is a byte-identical copy of `scene_perp_*.xml`,
screwdriver and all. Now added, mirroring the baseline convention:

* `assets/mjcf/perp/perp_hand.xml` — morph joints present, unactuated. **The generation source.**
* `assets/mjcf/perp/perp_hand_morphology_actuated.xml` — all 18 DoF drivable, for dragging a
  design in the viewer. Exploration only; never generate or measure from it.

Parity with the scene is exact (0.0 m worst body mismatch at all four keyframes) and pinned by
`tests/test_perp_hand_scene_parity.py`.

**2. Nothing gated designs on self-collision**, so the workspace looked bigger than it is.
New `scripts/morph_selfcollision_gate.py` bakes a design, settles it, and asks the contact
solver. Always run it with `--retarget`, which IKs each design onto the reference grasp's
fingertip world targets — without it a moved mount is judged in the pose authored for the old
one and reports self-collisions it would not actually have (gotcha #5, resurfacing).

### Two generator bugs this exposed

* **Only the `open` keyframe was stripped.** `_strip_scene_morph_qpos` rewrote just that one
  key, so any source with a second keyframe emitted a model MuJoCo refuses to load
  (`keyframe 'closed': invalid qpos size, expected 9, got 18`). Invisible until now because the
  baseline hand has exactly one keyframe and the perp hand has four. Now `_strip_morph_qpos`,
  layout-aware, applied to every key.
* **`--` inside an XML comment.** MuJoCo's parser accepts it; `ElementTree` — which the
  generator uses — does not. A file can therefore load fine everywhere except the one place
  that matters. Pinned by a test that parses each MJCF with *both*.
* Generating from a `*_morphology_actuated.xml` now **raises** instead of silently substituting
  the baseline hand's open angles and discarding the perp pose.

### What the workspace actually allows

`perp_compact_design(thumb_t, pair_x_t, pair_y_t)` interpolates from the shipped spread-out
mounts (0,0,0) to the most compact layout the boxes allow (1,1,1). Gate results, retargeted:

| direction | stock scene | palm plate raised 12 mm |
|---|---|---|
| **pair back in x** (`pair_x_t`) | **OK, full range** | OK, full range |
| **thumb forward** (`thumb_t`), thumb stowed | palm 686–690 N | **OK, full range** |
| **pair inward in y** (`pair_y_t`) | palm 462–687 N | index↔middle 22–81 N |

**`pair_x_t` is available right now, unmodified.** The pair moves from palm x +35.0 to +12.5 mm,
closing the thumb→pair tip gap 71.4 → 51.7 mm, with the pinch preserved (34.9 → 34.0 mm) and
zero self-collision. This design was reachable the entire time and was never tried.

**The palm plate is the sole blocker on the other two directions.** It is a 96 × 72 mm slab at
`z = 0` — the *same plane as the finger mounts* — so any mount sliding over its footprint buries
a 10 mm proximal capsule in it. That is what the `thumb_x` force table earlier in this file was
measuring; it is not specific to the thumb. Raising the plate 12 mm clears it, and **costs the
existing lineage nothing**: over 1200 settle steps from each of `open`/`closed`/`press`, the
palm plate makes **no contact with anything at all**. It is inert geometry in every perp result
so far. (Excluding palm↔proximal contacts instead gives bit-identical gate results.)

**Pair separation in y is a genuine constraint, not a modelling artifact.** With the palm out of
the way the pair still fails, now on index↔middle: the two facing fingers scissor into an X and
eject the shaft sideways — visible directly in the render, and the failing bodies move from the
middle phalanges (22–42 N) to the proximals (81 N) as the mounts close. Retargeting lowers the
force an order of magnitude versus the naive pose but does not remove it. Bringing the opposed
pair closer needs geometry outside the 9-param space (shorter proximals, or mounting the pair
at an angle), not another sample of it.

**The thumb-stow trick works.** A thumb moved forward cannot reach the authored `open` target —
that target sits behind it, inside its minimum-reach shell — so it reads UNREACHABLE (residual
3.4–6.1 mm). Stowing it with `thumb_yaw` (`--thumb-yaw 1.1`) exempts it from the grasp entirely
and the whole forward range opens with the pinch untouched at 34.9 mm. `thumb_yaw` rolls the
finger about its own proximal axis, which swings the tip laterally *because* the mcp is flexed
at every authored perp keyframe.

**Open decision (user's call, nothing launched):** whether to raise the palm plate. It unlocks
the thumb-forward half of the workspace at no measured cost, but it is a change to the scene
every published perp policy trained against.

### r5 queue launched (2026-07-31 17:07) — and the compact designs are MORE NaN-prone

User's call: palm made **visual-only** (`contype/conaffinity 0`) rather than raising it — the real
palm is a complex mechanism that has to actuate the XY morphology, and is deliberately not
modelled at all. This matches `assets/mjcf/baseline/hand.xml`, whose palm has always been
non-colliding, and is free: the plate made no contact with anything over 1200 settle steps from
every keyframe. Gate goes 5/21 → 9/21 (thumb full range + pair_x full range).

`scripts/sweep_perp_compact.py` ranks the workspace by the MECHANISM before spending GPU
(gotcha #7: per-design RL is seed-dominated, so training every design learns nothing). 50
designs, gate → retarget → settle/close/lift → **3.2 s hold**. 17/50 gate-valid, **12 HELD**.

The long hold earns its keep immediately: designs `t0.50_x0.00`, `t0.75_x0.00`, `t0.75_x0.25`
all report **peak cos +1.000** — and are **on the floor** at grip 0.0 N, z 0.050. That is the
release artifact this file documented at 2026-07-30, reproduced across the morphology axis.
`held` is asked of the physics (tip force + height + floor contact), never inferred from cos.

Top 5 by held-peak-cos: `t0.00_x0.25` (+0.995), `t0.00_x0.00` (shipped, +0.991), `t0.25_x0.25`
(+0.987), `t0.50_x0.50` (+0.879), `t0.00_x0.50` (+0.746). Table:
`docs/experiments/perp_compact_sweep.md`.

**The shipped hand is genuinely good on the open-loop mechanism** (+0.991, 2nd of 50) — the
morphology complaint was never that the mechanism was broken, it was that one design was the
only one ever tried. `t0.00_x0.25` edges it.

`scripts/train_perp_compact_queue.sh` runs r5 sequentially over the top 5 (single GPU),
resumable via per-design `.DONE`, own Warp cache per process, GPU-drain wait between runs.

**⚠ OPEN: `t0.00_x0.25` NaN'd at iteration 0.** Same failure the recipe pins `init_noise_std:
0.05` to prevent — but that value was tuned on the SHIPPED morphology, and a moved-mount design
is more fragile still. The queue skipped it and is training `t0.00_x0.00`. **This means the
top-ranked design is currently unmeasured, and the ranking may be systematically biased toward
designs that happen to survive the pinned noise.** Next: per-design `init_noise_std` (try 0.02
/ 0.01 on failure) and a retry loop in the queue rather than a skip. ETA is ~6 h per design, so
the queue will not finish in one sitting — it is resumable by design.

### 2026-07-31 20:55 — the non-colliding palm NaN'd every run; excludes are the right fix

**The whole r5 queue failed.** All five designs NaN'd at iteration 0–121, *including the shipped
morphology* — the same design r4 trained for 339 iterations. So the "moved-mount designs are
more fragile" hypothesis was wrong, and the ranking bias it predicted does not exist.

Diffing my baked scene against r4's `perp_v1/frozen_scene.xml`: **identical in every respect —
nq 22, nu 15, ngeom 24, same four keyframes, identical `open_ik` qpos and fingertip world
positions — except `palm_plate contype/conaffinity`, `[1]/[1]` vs `[0]/[0]`.** One attribute.

**The "it's free to disable the palm" claim was wrong, and the way it was wrong is the familiar
one.** The evidence for "free" was: no palm contact over 1200 settle steps from each keyframe.
That is a *scripted settle with zero actions* — precisely the too-narrow test this project keeps
getting burned by. Under RL exploration the palm↔object contact is load-bearing, and removing it
lets the shaft escape into a state the solver cannot integrate. A geom reading 0 N in a settle
is not an inert geom.

**Fix: keep the plate collidable, exclude only palm↔finger-link pairs** (12 excludes, all three
fingers' mcp/len/pip/tip). This is the user's stated intent read literally — *ignore
palm-proximal contacts* — rather than the broader "make the plate disappear" I substituted for
it. The palm remains a physical backstop for the object; it is simply not an obstacle to the
fingers it carries, which was always a modelling artifact of putting the mounts in its plane.

| | gate valid | shipped design trains |
|---|---|---|
| plate collidable (original) | 5/21 | yes (r4, 339 iters) |
| plate non-colliding | 9/21 | **no — NaN at 0–121** |
| plate collidable + 12 excludes | **9/21** | **yes — 204 iters clean** |

The workspace is open *and* the sim is stable. `t0.00_x0.25_y0.00`, the top-ranked design that
had never trained at all, is running now.

**Queue hardening:** `train_perp_compact_queue.sh` now retries a NaN'd design at
`init_noise_std` 0.02 then 0.01 instead of skipping it, and only retries on an actual NaN. The
comment there carries the caveat this episode earned: **if the whole queue fails, the cause is
not the noise — diff the scene against a run that trained.**

Still ~6 h per design, sequential, resumable. Nothing has finished yet, so there are no r5
policy numbers to compare against r4 (t_align 89±46, peak_cos 0.996±0.006, hold_steps 331±55).

---

## 2026-08-01 — the r5/r6 wipeouts were TWO launcher flags, not the palm

**Everything the previous section concluded about r5 is wrong.** The queue finished 0/5, was
relaunched, and finished 0/5 again. Neither wipeout had anything to do with the palm plate,
which I blamed twice (first as a non-colliding geom, then as 12 contact excludes) and which was
innocent both times.

### What actually broke

Two flags that r4 passed on the command line and that no config carried. `r4`'s dumped
`config.yaml` records both; `scripts/train_perp_compact_queue.sh` passed neither.

| flag | r4 | queue | consequence |
|---|---|---|---|
| `--num-envs` / `--total-timesteps` | 3072 / 25M → **339 iters** | omitted → 1024 / 200M → **8138 iters** | 5.7 h/design instead of 42 min |
| `--open-finger-from-keyframe` | **True** | omitted → **False** | the IK-retargeted keyframe is discarded |

`max_iterations` is not a flag — it is derived as
`total_timesteps // (num_envs * num_steps_per_env)` — so omitting the budget silently rescales
the run. And the recipe layer, which exists to stop exactly this, carries **no timestep budget
at all**.

The second flag is the one that produced the fake science. Without it the env falls back to
`env_cfg.open_finger_qpos`, which is the **baseline hand's** open pose:

```
baseline default : (0.0, 3.14, 0.0,   0, 0, 0,        0, 0, 0)
perp open_ik     : (0.0, 1.909, -1.8, 0, 1.138, 0.729, 0, 1.138, 0.729)
```

70° off at `thumb_mcp`, 103° at `thumb_pip`, index and middle left **straight out** instead of
curled. The hand closes on nothing. It is wrong even at zero morphology change, so it is a
property of the perp topology, not of the retarget. The queue therefore spent real time
IK-retargeting `open_ik` into every scene, logged sub-0.1 mm residuals, and threw the result
away.

Both failure modes are this one bug, and each impersonated a genuine finding:

* moved-mount designs → grip never forms → object never leaves the floor (`obj_z` 0.0123 =
  spawn, against r4's 0.060–0.097) → reads as **"this morphology is ungraspable"**;
* shipped design → fingers start in violent contact → NaN at iteration 0 → reads as **"the perp
  scene is intrinsically unstable"**.

### The decisive measurements

* r4's **own** frozen scene, 3072 envs, recipe noise, **without** the flag → NaN at iteration 0.
  With it → 339 clean iterations. Same file, same morphology, same everything else.
* The scripted open-loop grip lifts **all five** baked designs (obj z 0.138–0.151, 2–3 tips in
  contact) — i.e. none of them is ungraspable.
* Three CPU probes (400–500 seeds) could not separate the palm-exclude scenes at all: 0/500 vs
  0/500 under finger-only noise, 107/300 vs 104/300 under all-DOF noise. They could not,
  because the defect was never in the scene.

That second bullet is standing test #2 in `CLAUDE.md` — *good open-loop + bad learned is a
policy/grip problem, never morphology*. It was available an hour before I used it.

### Two traps worth naming

**Reading solver settings off the XML is misleading.** The scene says Newton
`iterations=100, ls_iterations=50`, pyramidal cone, impratio 1. mjlab **overrides all of it** at
sim build (`env_build.py` → `SimulationCfg(mujoco=MujocoCfg(...))`): `iterations=10`,
`ls_iterations=20`, `impratio=10`, `cone="elliptic"`. So (a) the "untuned solver" speedup is
illusory — it is already lean — and (b) any CPU probe built from the raw XML simulates physics
the trainer never runs. Apply the overrides to `m.opt` before concluding anything.

**A health gate calibrated on a bug is calibrated on nothing.** The collapse watchdog was first
set to `obj_z < 0.04`, fitted to r5/r6 curves where every dead run sat at 0.0123 — the
*signature of the keyframe bug*. With the bug fixed the failure distribution moved and 0.04
promptly false-killed a design that was recovering (obj_z 0.0268 → 0.0349 → 0.0390 with reward
climbing 62.9 → 76.7). Re-validated against every curve on record, 0.02 catches 11/11
genuinely-dead runs with zero false kills, at 1.6× above spawn and 1.34× below that run's true
minimum. The gate answers **one binary question — did the object ever leave the floor** — in
~7 min; "lifts poorly" is the measurement itself and costs the full 42 min. Raising it to
reclaim GPU time would make early kills correlate with design quality and bias the ranking.

### Guards added (the whole cost of both bugs was silence)

* `_assert_keyframe_not_silently_discarded` (`morphohand.rl.env_build`) — refuses to start when
  the keyframe is an IK-retarget output (`*_ik`), the flag is off, and the poses actually differ
  by >0.05 rad. Scoped by the name suffix so baseline-hand runs are unaffected.
* `open_finger_from_keyframe: true` pinned in `configs/recipes/perp_single.yaml`; budget flags
  passed explicitly by the queue launcher.
* Regression tests: guard (4) + recipe pin (1). 108 pass.

**Before any cluster sweep, generalise this**: assert at startup that what the launcher
*prepared* is what the env *used*. At 200 runs these bugs produce confident, plausible, wholly
wrong conclusions with no error anywhere.

Commits: `795535d` (budget), `3f6d032` (keyframe flag + guard), `02272ab` (gate recalibration).
