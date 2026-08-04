# S0 — sphere-packed contact geometry: the packing works, and it says pack the OBJECT

Branch `experiment/sphere-contact`. Scene: `assets/mjcf/perp/scenes/scene_screwdriver_medium_perp.xml`
(the perp scene is used for S0/S1 because it has `closed`/`press` keyframes; the m05 frozen scene of
the a10/b33 lineage has only open poses, so its grasp exists solely inside a rollout — that is S3).

## Headline

**Packing the hand is inert in the working regime. Packing the object is what creates a contact map.**

Under real grip load (`press`, 8.2 N total — the regime the policies actually operate in):

| scene | hand↔obj contacts | sum Fn | penetration | per-tip patch width |
|---|---|---|---|---|
| capsule baseline | **5** | 8.17 N | −0.444 mm | 0.0 mm (point) |
| arm A, hand packed ε=0.02 | **5** | 8.18 N | −0.444 mm | 0.0 mm |
| arm A, hand packed ε=0.005 | **5** | 8.18 N | −0.444 mm | 0.0 mm |
| arm B, object packed ε=0.02 | **18** | 7.64 N | −0.203 mm | 1.9–2.9 mm |
| arm B, object packed ε=0.01 | **29** | 7.67 N | −0.155 mm | 1.7–3.4 mm |

Arm A at ε=0.005 puts 257 spheres on the hand and produces **byte-identical contact behaviour** to
the single-capsule baseline. Not "similar" — identical count, identical force to 0.01 N, identical
penetration.

## Why (the geometric reason, which generalizes)

**Contact resolution is set by the flatter of the two surfaces.** The fingertip is the
high-curvature side; the screwdriver barrel is the low-curvature side. Spheres packed onto the
*curved* side all hide behind whichever one is nearest the object — only one can reach the surface,
so you get one contact no matter how many you add. Spheres packed onto the *flat* side spread out
along the patch and each generate their own contact.

This also explains the one case where arm A does help. At the light `closed` touch (1.8 N) the
*proximal phalanges* lie along the barrel — a low-curvature hand link against a low-curvature object
— and there arm A goes 6 → 13 contacts with a 1.9 mm patch on `*_pip_frame`. But at `press` those
links lift off and only the tips carry load, so the gain vanishes. The refined statement:

> Hand-side packing only pays where a low-curvature hand link cradles the object. In the actual
> fingertip grip it buys nothing.

## Throughput cost (CPU, single world, `mj_step`)

| scene | ngeom | steps/s | rel |
|---|---|---|---|
| capsule baseline | 24 | 81 489 | 1.00× |
| A hand ε=0.05 | 101 | 76 331 | 0.94× |
| A hand ε=0.02 | 143 | 68 458 | 0.84× |
| A hand ε=0.01 | 192 | 65 498 | 0.80× |
| A hand ε=0.005 | 263 | 58 726 | 0.72× |
| B obj ε=0.05 | 159 | 57 614 | 0.71× |
| B obj ε=0.02 | 357 | 46 725 | 0.57× |
| B obj ε=0.01 | 653 | 34 377 | 0.42× |

Much cheaper than feared: 11× the geoms costs 1.4× wall clock (arm A), and the useful arm-B
configuration costs ~1.8–2.4×. Cost tracks *contact* count, not sphere count, as expected — most
spheres never touch. **This is CPU MuJoCo at nworld=1 and does not settle the GPU question**;
mjwarp broadphase and `nconmax` scaling are the real S2.

## Fidelity

Renders at `render_capsule_baseline.png`, `render_armA_eps0.005.png`, `render_armB_eps0.010.png`
(iso/side/top, contact arrows on). Arm A at ε=0.005 is visually indistinguishable from the capsule
baseline — same silhouette, same pose, same object placement. Arm B shows the shell as surface
texture with the cylinder silhouette preserved, and the contact arrows become a visible *spray*
rather than three spikes.

Mass is preserved exactly (max relative drift 4e-9), because
`scripts/generate_sphere_packed_scene.py` compiles the unpacked scene and bakes
`body_mass`/`body_inertia`/`body_ipos`/`body_iquat` into an explicit `<inertial>` before
substituting geoms. Without that step the finger links would have gained ~18× mass from overlapping
sphere volume × density — the frozen scenes carry no explicit inertials.

## Confounds: status

- **N× stiffness — real but mild, and it does not distort force.** Penetration drops (−0.444 →
  −0.155 mm in arm B) so the packed shapes *are* stiffer, but the solver redistributes and total
  normal force is preserved to ~1% (arm A) / ~6% (arm B). Still needs the S1 renormalization before
  any rolling claim, since rolling is the compliance-sensitive part.
- **Sensor `reduce="none", num_slots=1`** — untouched so far, and it is now the top-priority fix:
  arm B puts 3–4 contacts on a single tip body, so the reported force becomes one arbitrary
  sphere's share. Everything downstream of `policy_eval_suite.py --held-min-n` would read a healthy
  grasp as a drop.
- **Floor contacts inflate under arm B.** A scalloped cylinder resting on a plane touches at many
  spheres (11–19) where a smooth one gives 2. This is the scalloping risk, now visible; it inflates
  the `nconmax` budget and slightly changes how the object rests. The free-roll fidelity probe is
  still required.
- **Self-collision: zero** intra-hand contacts in every packed scene at these poses. The
  anticipated intra-finger `<exclude>` work has not proved necessary yet; recheck at flexed poses.

## What this means for the plan

Arm B is the arm. It is also the one S6 needs anyway, since a contact *trajectory* has to be
expressed in object-surface coordinates. Arm A should be kept as the control that shows hand-side
resolution is inert — that is a result, not a failure — but it should not carry the program.
