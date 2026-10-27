# Four-magnet N52 tip (v3 ID 1) against the five-magnet tips: work after the weekly reset

Owner, 2026-10-26 18:4x: until the 04:00 reset (10-27) the workstation and the lab session only run the calibration of
v3 ID 1 (R28 in `docs/experiments/20261022-ur5_tip_calibration/QUEUE.md`); planning or debugging that needs more goes to
an autopilot job on the personal account. After the reset: "full license to do all the analysis and further calibration
test development and deployment" within the envelope of forces, joint limits and speeds defined so far.

State: v1 ID 3 (cb3t = `lab/20261026-134522-cb3t`) stopped early for the swap. Passes 1 and 2a, the 24 parity
actions above joint 6 +45 deg, R26 (hard1 and hard2, clean; afterwards the pad reached 2 N up to 0.22 mm deeper near
the reference point, recovering) and R27's rod pass 1 with the rate set of rod pass 2 (`lab/20261026-180938-rod3t`,
builder `scripts/ur5_tip_calibration_rod_pass2.py`) ran; passes 2b to 6 did not. v3 ID 1 runs R28 tonight (session
`v3id1`).

Next, in order (owner's priority):
1. Resolution and overall sensing of the four-magnet N52 tip against the five-magnet tips (ID 1 cb1t/cb1r, ID 2 cb2t,
   ID 3 cb3t): rest noise per chip, change per newton, GP location error by force band (light presses under 1.5 N
   first), shear and force direction with slides, torsion, the density curve
   (`scripts/ur5_tip_calibration_density_curve.py --session`), with the cb2t checks (`scripts/ur5_tip_calibration_cb2t_checks.py`
   generalised to a session argument). The owner moved the four-magnet tip first to settle crosstalk before resolution.
2. Physical model for four magnets: an FE row for the layout and a 16-pattern pole search in
   `scripts/fingertip_wrench_model.py` (it assumes five magnets, the centre third), then the viewer.
3. Then: contact pose, its smoothness in motion, variable forces and speeds (R26), the rod (R27, rod pass 2 not built),
   the simulation models; a results page for cb1r, cb2t, cb3t and v3 ID 1, and `docs/overviews/printed_fingertips.html`.

Waiting on owner: v2 (five N52 magnets) is printed and not yet calibrated; the tall indenter
(`docs/experiments/20261026-tall_indenter/20261026-tall_indenter_spec.md`) goes to another agent.
