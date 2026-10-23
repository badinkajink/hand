Status: waiting on the owner. The mount was redrawn to ATI's drawing and the calibration plan built and simulated on 2026-10-23 from 11:05 to 12:30; nothing is printed, connected or measured.
Task (owner, morning of 2026-10-23): "i think the only way to permanently fix the HEX-E robot is to calibrate one of our uncalibrated ATI sensors. see the attached notes from chatgpt. we do not have to do the desired calibration." Later: "we can probalby calibrate the ATI with just a 3dp mount to the HEX-E, with the current indenter setup". At 10:51 the owner moved it out of the tip calibration's channel. At 11:05: "the most we can do is design a mount for the ATI for the HEX-E and build up a calibration plan ... the ati will be stuck onto the F/T sensor with the Z-axis aligned. however, we now know that the Z-axis is the noisiest axis ... maybe we have to design an alternate mount that has it rotated? ... i can set up the ati when i get back".
Safety rule of the owner's ATI note, verbatim: "Never enable force-feedback control, contact motion, or safety thresholds based on an unvalidated borrowed matrix."
Read first:
- the plan page docs/experiments/20261023-ati_ft_calibration/20261023-ati_ft_calibration_plan.html (local, no artifact): the reference question, the drawing, the stack, the hook cross, the robot plan, the simulated sessions, the procedure;
- docs/experiments/20261023-ati_ft_calibration/QUEUE.md: requests A1-A3 to the lab computer;
- the owner's ATI note, docs/experiments/20261023-ati_ft_calibration/20261023-ati_mini40_mini45_calibration_handoff.md.
State (12:30, 10-23):
- Reference: the ATI stays coaxial. Simulated on 40 pairs: weighed mass 0.001-0.004 N, 0.1-0.2 N mm; HEX-E coaxial 0.08-0.21 N; HEX-E with the ATI turned 90 deg 0.08-0.22 N (its Fz gain 0.78 moves into the ATI's Fx). scripts/ati_reference_compare.py.
- Stack v2 (scripts/ati_stack_mount.py, femtip env): ATI drawing 9230-05-1094 rev 20, fetched from ati-ia.com on 10-23 (linked on the page; ATI forbids copies). Mounting side M3 5.0 deep on 39.24, pairs +-10 deg about 60/180/300, dowel at +X; tool side M3 3.0 deep on 28.2, pairs +-20 deg, dowels at 0 and 120 deg; cable radial at 110 deg. Base 70 mm + 2 mm pad, cable channel and tie tunnel; screws M3 x 12 (4.0 of 5.0 mm), M3 x 10 (2.5 of 3.0 mm); clock 114 deg (3.07 mm walls), cable toward 224 deg in M; tool 46.7 mm longer, ATI origin z_M 50.7. Hook cross: arms to 80 mm, post to 118 mm, eyes 1.2 mm holes in 1.6 mm walls with knot slots.
- Plate ati_stack (scripts/v3b_ur5v2_pla_plate.py ati_stack): 3 h 28 min, 129.7 g, no problems.
- Plan (scripts/ati_calibration_plan.py, about 2.5 min under resguard): plan/20261023-ati_plan_mass.json, P0 (-180,-420,430) mm, 39 stations (8 per eye + reference), 91 steps, 26.8 min, excitation 19.6..12.8, least clearances tool-arm 24.0, port 19.6, mass to tool/links/cell 41/77/92 mm; joint 5 up to 129 deg (past the tip calibration's 100 deg window: watch in the dry run).
- Session program scripts/ati_calibration_session.py (RDT reader tested on the fake box; --sim with scripts/ati_calibration_sim_session.py), analysis scripts/ati_calibration_analysis.py (pairs, drift from the reference station, full/rows6/rows12, leave-one-eye-out, HEX-E gains, truth score), film scripts/ati_calibration_film.py, driver scripts/ati_sim_sessions.py (5 seeds: truth 0.001-0.005 N, 0.1-0.4 N mm; leave-one-eye-out 0.012-0.035 N). Logger reads every record of a datagram.
- Status page rows updated (requests row 47, decision row 24).
Waiting on owner:
- the box: state saved as found, #16 FT00000 with lbf and lbf-in, RDT on, its IP; then A1 (box check);
- the print, four M3 inserts, the eyes drilled to 1.0-1.2 mm, pigtails, screws (4 x M3x20, 6 x M3x12, 6 x M3x10, 4 x M3x12), a 0.5 kg mass with S hook weighed to 0.1 g;
- the go for A2 (dry run, half speed) and A3 (the run) on the UR5 between tip calibration sessions.
Next (each step can run once its input exists):
1. After A1: read the box's netftapi2.xml and the box check's ati.bin (rate, status, rest noise per gage); if the rest noise differs much from 3 counts, rerun scripts/ati_sim_sessions.py with it.
2. After A3: scripts/ati_calibration_analysis.py on the session folder; add a results section to the page (rebuild with scripts/ati_ft_calibration_page.py) with C, the leave-one-eye-out errors and the HEX-E's gains for the tip calibration (R5).
3. Plan the presses for the ATI against HEX-E comparison with the tool 46.7 mm longer (the tip calibration's plan builder and checks, unchanged; plans into this folder's plan/).
Tip calibration interface:
- The stack changes the tool's length and payload (0.555 kg, 1.055 kg with the mass), so it goes on the robot only between tip calibration sessions, with the owner's go.
- Requests to the lab for this work go only in docs/experiments/20261023-ati_ft_calibration/QUEUE.md. The lab answers in docs/experiments/20261023-ati_ft_calibration/lab/.
Claimed: scripts/ati_*.py (stack, plan, session, sim_session, analysis, film, sim_sessions, reference_compare, fit, sim, logger, gravity_pose_check, page); docs/experiments/20261023-ati_ft_calibration/; the plate ati_stack in scripts/v3b_ur5v2_pla_plate.py; the status page's two ATI rows.
Do not:
- write in docs/experiments/20261022-ur5_tip_calibration/ (QUEUE.md, lab/, plan/) or send the lab session messages about this work;
- write the Net F/T box's configuration from a script, or send it a bias command;
- put the stack on the robot during a tip calibration session;
- start a printer or move the robot;
- publish artifacts from the colorado login;
- run a mesh process outside resguard.sh;
- run git pull, rebase or stash in this working tree, because other sessions' uncommitted edits are in it.
Pushing: the tip calibration window shares this working tree. It fast-forwards main from the lab's pushes and pushes every local commit. Commit with explicit paths (git commit -m ... -- <paths>). To push yourself, run git fetch origin first. If git merge-base --is-ancestor origin/main HEAD fails, the lab has pushed. Then list the local commits (git rev-list --reverse origin/main..HEAD), run git reset --keep origin/main, cherry-pick them in order, and run git push origin HEAD:main.
