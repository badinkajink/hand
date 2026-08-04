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

## S1 — the roll fidelity probe: arm B has an irreducible rolling cost

`scripts/probe_roll_fidelity.py`. Bare object on a plane, no hand, launched at 0.5 m/s with
matched spin so it rolls rather than skids. (Getting the spin wrong makes it *skid*, which
measures friction and not rolling — the freejoint's `qvel[3:6]` is body-local and this body is
rotated 90° about X, so the roll axis is `qvel[5]`.)

| object | distance | roll ratio | bob p-p | mean ncon | v_end |
|---|---|---|---|---|---|
| smooth cylinder | 1.498 m (1.00×) | 0.996 | 0.0000 mm | 2.0 | 0.499 |
| packed ε=0.02 | 0.806 m (0.54×) | 0.997 | 0.138 mm | 13.0 | 0.132 |
| packed ε=0.01 | 0.907 m (0.61×) | 0.998 | 0.109 mm | 19.4 | 0.152 |
| packed ε=0.005 | 1.023 m (**0.68×**) | 0.999 | 0.091 mm | 28.7 | 0.178 |
| packed ε=0.002 | 0.983 m (0.66×) | 0.999 | 0.059 mm | 52.4 | 0.151 |
| packed ε=0.001 | 0.857 m (0.57×) | 0.999 | 0.060 mm | 94.4 | 0.086 |

**Good news:** `roll_ratio` stays 0.996–0.999 everywhere. The rolling constraint is intact — the
packed cylinder rolls, it does not slip. Scallop bob falls monotonically with ε exactly as the
tolerance formula predicts.

**Bad news:** rolling *distance* does not converge to smooth. It peaks at ε=0.005 (0.68×) and gets
**worse** at finer ε, because contact count climbs (28 → 94) faster than the scallop shrinks. There
is no ε that recovers smooth rolling.

Cause, isolated:

| configuration | distance | rel |
|---|---|---|
| packed ε=0.005, default solver | 1.023 m | 0.68× |
| packed ε=0.005, Newton iter=200 tol=1e-10 | 1.023 m | 0.68× |
| packed ε=0.005, CG iter=500 tol=1e-12 | 1.035 m | 0.69× |
| packed ε=0.005, **timestep 0.002 → 0.0005** | 1.249 m | **0.83×** |

Solver iterations, tolerance and algorithm change nothing. **Quartering the timestep recovers half
the deficit.** So the loss is time-discretization of the micro-collisions at every facet crossing,
not solver convergence — and our production timestep is 0.002 (`env_build.py` `MujocoCfg`), where
quartering it would quarter throughput on top of arm B's own 1.8–2.4× cost.

### The tension this creates

- **Arm A** is faithful and free, and **inert**: 257 spheres reproduce the capsule baseline exactly
  at the fingertip grip.
- **Arm B** is the only one that produces a contact map, and it **costs ~32% of rolling distance**
  at production timestep, irreducibly.

The reorient we are chasing *is* a roll, so this is not a nuisance term — arm B perturbs the exact
behaviour it was brought in to make legible. It does not kill the arm (rolling still happens, and
happens without slip), but any arm-B policy result must be read against a hand that lives in a
measurably higher-rolling-resistance world, and a b33 failure under arm B cannot be attributed to
representation without controlling for this.

## S4 — the contact map: the trajectory signal was already there at 5 contacts

`scripts/contact_map_render.py` → `contact_map.png`. Scripted `open → closed → press`, every
hand↔object contact transformed into the object's own frame and mapped to cylindrical surface
coordinates (θ around the barrel, axial along it). Top panel: force-weighted occupancy on the
unrolled surface. Bottom panel: per-finger contact θ, unwrapped — the "contact trajectory" a
Pollard-style tracking controller would be handed.

Aggregating θ across fingers is meaningless and the first draft did it: the fingers sit on opposite
sides of the barrel, so a circular mean lands on the ±180 seam and flips every step. Per-finger and
unwrapped is the only readable form.

Two findings, and the second is the awkward one.

**1. The contact is PINNED.** Over the full 1000-step grasp, index holds θ ≈ +90°, middle ≈ −90°,
thumb dead flat at −180°. Total migration is ~20° for index and middle and ~0° for the thumb. The
fingers grab three fixed spots on the barrel and never migrate. That is what "the fingers barely
move" looks like in contact coordinates, and it is visible in one plot. (Caveat: this is a scripted
grip, not b33 — a policy that *tries* to reorient has not been run through this yet. That is S3.)

**2. The capsule baseline draws the SAME trajectory as the packed scene.** Both panels show the
same three flat lines at the same angles. The packing multiplies contact *samples* 7× (5 445 →
38 920) and widens the patch, but it does not change the trajectory readout — because a single
contact point already tells you where the finger is riding.

This substantially deflates the case for packing as a *control* enabler while validating the
contact-trajectory framing as a *diagnostic*. The signal S6 would track is already extractable from
the existing 5-contact representation, today, with no packing and no physics change. What the
packing adds is patch *shape* (how wide, how the force distributes within it), not patch *location*.
