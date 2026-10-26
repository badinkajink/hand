# Outlines for the fingertip documentation of 2026-10-26 (scratch for the hand-off note)

The owner (2026-10-26) wants to understand three things this week and will read the pages: (1) the UR5 calibration —
motions, geometry, the sensing result wanted (contact place, wrench: normal, shear, torsion), the protocol justified
mathematically and by experiment, the learning architecture, data processing and pooling across tips; (2) the FE, the
printed tip and the magnets/magnetometers as the physical model; (3) the contact overview as the link from the real tip
through the FE to the MuJoCo sphere pads, with the gaps between each component and what they mean for simulation,
training and control. Done: (2) as `docs/experiments/20261026-fingertip_physical_model/20261026-fingertip_physical_model.html`
(commit 8e425c2b; builder `scripts/fingertip_physical_model_page.py`, text `_text.py`, figures `_figures.py`, FE `_fe.py`).
Copy its builder pattern, figure style (serif, 10 in wide at 150 dpi, text >= 12 pt) and prose (prose_lint within
targets) for the two pages below. Writing rules: `~/.claude/CLAUDE.md` Writing and `~/.claude/writing_reference.md`.
The owner pasted a "Claudish to English" prompt on 10-26: collapse restatements, no contrast framing, no staged emphasis,
decode metaphors ("gated", "load-bearing", "surface", "lands"), lowest useful abstraction.

## A. Contact overview: open with the chain from the printed tip to the simulator

Builder `scripts/contact_overview_page.py` + `scripts/contact_overview_text.py` + template; output
`docs/overviews/contact_model.html` (stable name, artifact 4w96Kt4J3tunu1xuz4XbdN on personal). Keep the existing
derivations (steps 1-10), comparisons, cost and sprung skin, but put a new first part before them and shorten what
repeats. The lede states the chain and its gaps with numbers. New Part 1, "From the printed fingertip to the simulated
contact", prerequisites linked: the physical-model page above and the calibration page of B.

1. Three models of one fingertip: the print as measured on the UR5 (HEX-E force, registered indentation, board), the FE
   of the print, and the pressure law that Drake hydroelastic and the 1 mm sphere pads integrate. One figure: the chain
   with the number at each link (draw like `20261026-model_chain.png`).
2. Print against FE (UR5 ground-truth page Sections 3, 4, 8; checkerboard page): at the registered depth the FE pushes
   back with 5.1x the measured force (sphere2), so the measured contact is softer than the FE; but the board's force is
   0.78-1.31 of the HEX-E over four sessions, so the magnets' motion per newton, the pad's bulk, follows the FE within
   about 30 %. The extra indentation is 0.33 mm + 0.20 mm/N under the 5 mm sphere and 0.07 + 0.10 mm/N under the 12.5 mm
   rod: at most 0.10 mm/N in series (mount, arm, stand), at least 0.10 mm/N local to a small contact. Answer the owner's
   question plainly: the FE is stiffer than the measured tip at the contact and about right in bulk.
3. FE against the pressure law (`docs/experiments/20261018-fe_contact_calibration/`, v2 checks page): the FE approach grows
   as F^0.71-0.93, a local law as F^0.5-0.67, so one modulus misses by 20-28 % RMS (41-61 % largest). A stiff local
   modulus behind a series spring (9.7 N/mm on the 11/22 tip) fits within 5.2 % (Drake) and 5.6 % (0.5 mm pads), twist arm
   0.87-1.15; with the spring the two simulators end a turn 6.3 deg apart, without it 18.6 deg. The capsule cannot be fit
   (4.1x stiffer along the finger).
4. Pads against the UR5 presses (ground-truth page Section 7): pads on the 812 FE surface points; with a series spring
   pads and FE both follow the measured indentation within 133-135 um RMS; without a spring the best single modulus is
   0.90 MPa, 11x under the scenes' 10 MPa (`E_TPU = 1e7` in `scripts/reorient_backends.py`; the default builder v2 has no
   series spring: state this). Figure: force against indentation at the face under the 5 mm sphere, measured holds
   (sphere2 or cb2t holds.npz: f_n against registered indentation) with FE, pads at 10, 5.4 and 0.90 MPa and pads at
   153 MPa + 8.45 N/mm (pads.json of sphere2 has the fits; check its keys M3, by_level, delta_*).
5. Pads against Drake: the existing comparisons, in one short subsection (onset torque 0.94, turn within 3 deg on 22-23 of
   24, creep 9-160x Drake's).
6. What none of the models has: tangential compliance before slip (FE 48-177 um/N), presliding, rate dependence (board
   change 1.24x and 1.79x at 0.6 and 2 mm/s against 0.15 mm/s, cb2t checks.json rates), the measured friction coefficient
   (scenes use 1).
7. What the gaps mean for the hand: through the servos each fingertip is a 250 N/m spring; the pad contact (34.7 kN/m at
   1 N) in series makes 248 N/m, so the servo sets the grip force and the pad's normal stiffness error barely changes it;
   the friction torque (patch, arm) and creep decide twist, brake and reorientation; policies on pads and mesh hold alike
   but move differently, hence one contact model through training, planning and evaluation (owner rule 2026-10-15).
8. "Can the same calibration be done for MuJoCo": the pads are the MuJoCo model and were calibrated twice (to the FE: M3;
   to the UR5: spring or 0.90 MPa); the scenes still run 10 MPa without a spring. Name the step that would put the UR5 fit
   into the scenes (a series slide joint per tip, as the M3 study built) under Future work, with the check that decides it
   (turn and grip-force trajectories against Drake with seed spread).
9. Masterjohn, Guoy, Shepherd and Castro, RA-L 7(4) 2022 ("Velocity level approximation of pressure field contact
   patches", arXiv 2021): Drake's discrete solvers replace each polygon of the hydroelastic contact surface by one point
   contact at its centroid with stiffness g A (combined pressure gradient times area) and its own friction cone. The pads
   are the same approximation on a lattice fixed in the fingertip: each sphere a point contact of stiffness E a_i / h_i.
   Differences: Drake rebuilds the polygons from the overlap every step; the pads' resolution is the 1 mm lattice; both
   are rigid-body models with no deformation state. What the pads add (steps 7-9 and the cross-simulator use) is stated
   once, plainly, without praise or denial.

Then check deck 23: `python3 scripts/contact_overview_page.py --deck-check`; if a number the deck quotes changed, say so
under "Waiting on owner" (the owner also found the deck hard to read; revising it is the owner's call), else set
DECK_CHECKED in the builder.

## B. Calibration page

New folder `docs/experiments/20261026-fingertip_calibration/`, page `20261026-fingertip_calibration_on_the_ur5.html`,
builder `scripts/fingertip_calibration_page.py` (+ `_text.py`, `_figures.py`), topic fingertip. Title: "Calibration of
the magnetic fingertip on the UR5 against a six-axis force sensor". Data: `docs/experiments/20261022-ur5_tip_calibration/`
(protocol page, ground-truth page, checkerboard page, drift page, lab runs page, `data/*/summary.json`, `holds.npz`,
`checks.json`, `density_curve.json`, `pooled_checks.json`, `predict/*.json`; scripts `ur5_tip_calibration_predict.py`,
`_cb1r_checks.py`, `_cb2t_checks.py`, `_density_curve.py`, `_pooled_checks.py`, `_tip_transfer.py`; commits bcd8300e,
0a8a5d2c, 67c0d776, b00e2b15, 776b0f88, 23d4f867). Sections:

1. Terms and metrics.
2. What the calibration must deliver: contact place on the surface (2 numbers), force (normal + 2 shear), torsion about the
   normal (1); why each matters (grip force, slip margin = shear/normal against mu, twist); accuracy reached.
3. Cell and tool: frames B, E, S, M, P, I and the stack (protocol page Section 2, Tables 1 and 3); reuse
   `media/20261022-cell_stills.png`; add a frame-chain schematic.
4. Motions and what each identifies, one table + one new schematic figure (six panels on a pad section over the sphere:
   touch-off, offset touch-off 3 mm beside the top, press along the normal at force levels, slide = turn about the
   sphere's centre (constant indentation), roll = turn about the contact point, twist = turn about the normal). For each:
   the quantity it excites, the count on cb2t, the reason (e.g. touch-offs on the top have vertical normals and miss a
   horizontal shift, so the offset touch-offs fix the place along the finger: contact sd 0.348 -> 0.105 mm along it;
   twins mirror the grid about y = 0 because it had 30 targets at y < -1.5 and 19 at y > 1.5; pass 6's 38 targets by
   farthest-point sampling; rate presses at 0.045-2 mm/s; 87 slides for shear; twists at 1, 2.5 and 5 N for torsion;
   reference presses for drift). Pass table for cb2t: passes 1, 2a, 2b, 3, 5, 6 with 28, 31, 67, 55, 101, 73 actions.
5. Registration (protocol page Section 4: onset fit y = a (s - s0)_+^p + b; 15-parameter Levenberg-Marquardt with the
   widths of Table 7); results 0.045-0.057 mm RMS; simulated contact error 0.39 -> 0.07 mm.
6. Ground truth of a hold (protocol page Section 5 equations: HEX-E deflection, c_P, p as the surface point whose normal
   passes through c_P, indentation, force less the payload's weight change, shear, torsion as half the out-and-back
   difference); board change = hold mean less the rests before and after interpolated to the hold, each chip divided by
   its gain at its temperature. One real press trace (HEX-E normal force, board-change norm, holds shaded) if the lab
   streams are readable, else the protocol page's simulated Figure 7.
7. Sessions table: tip, mount, date, passes, press holds, turn holds (sphere2, rod1, sph3t, cb1t, cb1r, cb2t; cb3t ran on
   10-26 from 12:25).
8. Learning architecture, one diagram: inputs = the 15-component board change (temperature change and time since rest
   tried; tool orientation left out because it identifies the target); outputs = contact point (3, pad frame, projected
   to the surface) and force (3); GP with a squared-exponential kernel on standardized inputs, one length and noise by
   the marginal likelihood over a grid; MLP ensemble of 5 (two hidden layers of 64 SiLU units, AdamW, steps chosen on the
   latest fifth); ridge baseline; physical model + GP correction (input the physical estimate, output its correction).
   Splits by press, never by hold; five folds grouped by action; density curve groups points within 0.75 mm, 8 seeds.
   Pooling = training on more than one tip's holds.
9. Results, each with a figure: location by force level (physical vs GP vs MLP, all tips); density (error against the
   distance to the nearest training hold: 0.32 mm at 0.5-1 mm, 0.98 at 1.5-2, 6.1 beyond 4 mm, floor 0.22 mm; flanks
   74-79 % of FE points over 1.5 mm from any press); normal force 0.06-0.09 N; shear on cb2t's 87 slides 0.117 N and
   5.9 deg with slides in training, 0.179 N and 9.6 deg presses only, physical 1.30 N and 38.7 deg; torsion r 0.39;
   press speed; day repeat (0.80 mm); transfer and pooling (tip 1's GP on tip 2: 1.42 vs own 0.39 mm; physical model
   1.08 mm; tip 1's twins 1.64/2.09 -> 0.75/0.40 mm with tip 2's presses; shear direction 11.0 -> 7.6 deg).
10. Limitations (HEX-E Fz gain 0.78-0.87 of the other axes, dropouts, torsion under a ~230 N mm push moment, light
    presses) and Future work (ATI Mini45 reference, N52 magnets, flank coverage, random-order grid, torsion on the
    sensor's axis).

After B exists: set `L["cal"]` in `scripts/fingertip_physical_model_page.py` to the new page and rebuild it.

## C. Fingertip overview and publishing

`docs/overviews/printed_fingertips.html` (`scripts/topic_overview_page.py fingertip`, text `scripts/overview_fingertip_text.py`,
also edited by the lab window: edit only the opening): after the lede, a short reading path: physical model page, then the
calibration page, then the contact overview's Part 1. Publish from personal: the two new pages as new artifacts (URL in
an `artifact_url.txt` beside each, INDEX row via docs_index), republish the contact overview (4w96Kt4J3tunu1xuz4XbdN) and
the fingertip overview (XMCDZm1LyCpdrZuLufM37T) and docs/index.html, docs/pages.html; read each back.
