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
