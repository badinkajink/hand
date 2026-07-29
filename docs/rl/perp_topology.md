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

- `assets/mjcf/scene_screwdriver_medium_perp.xml` — base scene, morph joints intact.
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

The current Policy-A run keeps the strict guard, which is self-consistent (the CEM grasp holds
the shaft level, tilt 0.036 rad, so A *can* lift without tipping) and preserves the A→B split.
But it means A is being trained to suppress the free reorientation, and B then has to pay to
re-create it.

Two ways forward, both worth trying:

1. **Relax `--term-object-slip-yaw` for A on this design** (or make the term pitch-aware) so A
   is allowed to deliver a partially reoriented shaft.
2. **Skip the A/B split entirely.** On this topology lift and reorient are the *same* motion.
   A single policy doing lift+reorient is a natural fit, and the A→B handoff seam — which has
   cost this program the most time — may simply not exist here.

---

## Reproduce

```bash
# look at the design
MUJOCO_GL=egl uv run python scripts/mj_snap.py \
  --scene assets/mjcf/scene_screwdriver_medium_perp.xml --keyframe open \
  --views front,side,top,iso --out /tmp/perp.png

# the mechanism, open-loop, on the frozen scene
MUJOCO_GL=egl uv run python scripts/probe_perp_mechanism.py \
  --scene assets/mjcf/generated/scene_perp_tp0d0000p0d0000p0d0000_ip0d0000p0d0000p0d0000_mp0d0000p0d0000p0d0000.xml \
  --lift 0.16 --lift-steps 900 --press-steps 1400 --out /tmp/perp_probe.png

# re-author any keyframe from explicit fingertip world targets
MUJOCO_GL=egl uv run python scripts/pose_open_keyframe.py \
  --scene assets/mjcf/scene_screwdriver_medium_perp.xml --keyframe open --out-keyframe closed \
  --index-tip "0.035 0.0140 0.0095" --middle-tip "0.035 -0.0140 0.0095" \
  --thumb-tip "-0.030 0 0.038" --write
```

`scripts/mj_snap.py` is the loop-closer: it renders a scene/keyframe from canonical viewpoints
into one tiled PNG *and* prints tip positions, object pose and per-contact forces, so the
picture and the numbers can be checked against each other. Every geometry bug above was found
by looking at the render, not by reading metrics.
