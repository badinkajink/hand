# Perp topology → prototype: what has to be true before a sim2real attempt

Written 2026-08-17 alongside the sim2real review in `docs/rl/perp_topology.md` §2026-08-17.
Everything here is derived from the sim scene and `hand_paper/main.tex` Table I. **None of it
has been checked against the physical build or the CAD** — the point of the note is to turn the
open questions into ones someone can answer in an afternoon, not to claim they are settled.

## Why perp is the right candidate at all

Two measured results, both in the review:

1. **The reorient is compliance-insensitive.** r4 zero-shot at geom `solimp` dmax 0.995 / 0.997
   / 0.999: align_rate 100% at every level, peak cos 0.994–0.996. The baseline m05 lineage
   collapses under the same hardening because it reorients by *rolling*; perp *pivots* about two
   opposed contacts and does not spend compliance. Contact-stiffness mismatch is the usual
   sim2real killer for in-hand manipulation, and on this topology it is not in play.
2. **The policy fits the actuator envelope.** Every joint travels less than Table I allows and
   no joint's p99 rate exceeds 70 deg/s (`scripts/probe_joint_envelope.py`).

So the risk is concentrated in **geometry and mounting**, not in control or contact physics.

## Blocker 1 — the 90° opposition is a mount, not a joint

The opposed pair is created by a fixed rotation on the finger's frame, not by the yaw servo:

| body | pos (m) | quat | yaw |
|---|---|---|---|
| `thumb_mount` | −0.065, 0.000, 0 | identity | 0° |
| `index_mount` | +0.035, +0.048, 0 | identity | 0° |
| `index_yaw_frame` | 0, 0, 0 | (0.7071, 0, 0, −0.7071) | **−90°** |
| `middle_mount` | +0.035, −0.048, 0 | identity | 0° |
| `middle_yaw_frame` | 0, 0, 0 | (0.7071, 0, 0, +0.7071) | **+90°** |

The yaw *joint* then moves ±1.1 rad (±63°) about that rotated frame. Table I gives the yaw
servo **±45°**. So the opposition cannot be produced by actuation from a neutral mount — under
any yaw command the pair is at most 90° apart, never the 180° the mechanism needs.

**Question for the build:** can index and middle be physically remounted on their gantries
rotated 90° about the palm normal? If the finger module bolts to the gantry carriage on a
symmetric pattern this may be free; if the carriage interface is keyed, it is a machining job.

Note this is *not* the same as the `thumb_x` palm interference documented earlier in the perp
log — that one is about a mount sliding over the palm plate. This is about mount *orientation*.

## Blocker 2 — the pair's y offset may exceed the gantry travel

Table I gives base travel **X ±60 mm, Y ±40 mm**. The perp mounts sit at:

| finger | palm-frame mount | vs X ±60 | vs Y ±40 |
|---|---|---|---|
| thumb | −65.0, +0.0 mm | **outside by 5 mm** | ok |
| index | +35.0, +48.0 mm | ok | **outside by 8 mm** |
| middle | +35.0, −48.0 mm | ok | **outside by 8 mm** |

**This comparison is only valid if each gantry's zero coincides with the palm origin**, and
that is exactly what I cannot check from here — the travel figures are per-rail and the rails
may be offset from palm centre by design. If the rails *are* palm-centred, all three fingers sit
just outside their travel and the perp layout is not reachable without moving the rails.

Worth resolving first, because it is cheap to answer and it gates everything else. The `|y| =
48 mm` figure is not arbitrary either — the perp log records that closing the pair inward makes
the two fingers scissor and eject the shaft, and that constraint is geometric, not a modelling
artifact. Moving the pair *outward* is untested.

## Blocker 3 (minor) — `thumb_pip` sign convention

The policy drives `thumb_pip` over [−98.5, −85.6]°, against a Table I range of 0–100°. That is
**13° of travel inside a 100° servo**, sitting at a different zero: the thumb is mounted
mirrored, so its MJCF flexion runs negative. It needs a zero offset agreed between the MJCF and
the real thumb, not a larger actuator. `probe_joint_envelope.py` reports this as `OFFSET`
rather than a range failure for exactly this reason.

Check the sign of `thumb_mcp` too while doing this — it runs +84.7 to +109.2°, near the top of
its 0–110° range, so a zero error there *would* hit a real limit.

## What a bare-minimum transfer looks like if the blockers clear

r4 gives, on the current scene, N=64: the shaft picked off the floor, rotated to vertical (100%
of rollouts, peak cos 0.994) in ~121 steps, and held vertical **and physically gripped for
257 ± 83 steps ≈ 5.1 s** before the axial slip loses it. That is a demonstrable pick-and-stand,
and it does not depend on getting contact stiffness right.

The r6 runs in progress add a penalty on that slip; if they land, the hold extends. If they do
not, r4 is still the demo policy and the 5 s window is the thing to show.

## Things deliberately NOT proposed

* **No contact-stiffness curriculum / compliance DR.** Measured unnecessary for this topology.
* **No thumb recruitment.** The thumb reads ~0 N throughout the successful swing and every
  design that makes it load-bearing flattens the reorient (perp log, thumb sweep Result 3).
  For a bare-minimum transfer it is a passive stabiliser.
* **No morphology change.** The shipped design ranked 2nd of 50 on the open-loop mechanism and
  is the only one with a working policy. The re-queued sweep may change this; it has not yet.
