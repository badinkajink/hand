# Five-magnet N52 tip (v2 ID 1) and better contact sensing: work after the weekly reset

Owner, 2026-10-26 18:4x: until the 04:00 reset (10-27) the workstation and the lab session only run the calibration of
v2 ID 1 (R28 in `docs/experiments/20261022-ur5_tip_calibration/QUEUE.md`); planning or debugging that needs more goes to
an autopilot job on the personal account. After the reset: "full license to do all the analysis and further calibration
test development and deployment" within the envelope of forces, joint limits and speeds defined so far.

State: v1 ID 3 (cb3t = `lab/20261026-134522-cb3t`) stopped early for the swap. Passes 1 and 2a, the 24 parity
actions above joint 6 +45 deg, R26 (hard1 and hard2, clean; afterwards the pad reached 2 N up to 0.22 mm deeper near
the reference point, recovering) and R27's rod pass 1 with the rate set of rod pass 2 (`lab/20261026-180938-rod3t`,
builder `scripts/ur5_tip_calibration_rod_pass2.py`) ran; passes 2b to 6 did not. v2 ID 1 (five N52 magnets) runs R28 tonight
(session `v2id1`); the owner changed it from v3 ID 1 (four magnets) at 18:57.

Next, in order (owner, 18:57: "you can table the new question regarding 4 v 5 magnets and just focus on the improving
sensing components. seems like contact density is a big one. i will mount the taller indenter tmw too"):
1. Analyse v2id1 with the cb2t checks and the density curve (`scripts/ur5_tip_calibration_density_curve.py --session`),
   and pool it with ID 1 to 3 after scaling its board change by the strength fitted at rest (same layout, so the field
   scales by one factor): data from other tips filled ID 1's gaps on 10-26 (`scripts/ur5_tip_calibration_pooled_checks.py`).
2. Contact density: the face and edges level off near 1 mm spacing (0.36 mm against a 0.22 mm floor); the flanks are
   thin (74 to 79 % of their FE points over 1.5 mm from a press). The tall indenter mounts on 10-27: give it its
   cell-model kind and a reach map (spec `docs/experiments/20261026-tall_indenter/20261026-tall_indenter_spec.md`), then
   a pass of flank and edge presses and slides.
3. Then: contact pose, its smoothness in motion, variable forces and speeds (R26), the rod (R27, rod pass 2's builder
   `scripts/ur5_tip_calibration_rod_pass2.py`), the simulation models; a results page for cb1r, cb2t, cb3t and v2id1, and
   `docs/overviews/printed_fingertips.html`.

Waiting on owner: the four-magnet tip v3 ID 1 (tabled); the comparison of four and five magnets (tabled).
