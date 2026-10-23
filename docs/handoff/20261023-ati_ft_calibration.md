Status: waiting on the owner. The calibration was planned and simulated on 2026-10-23 from 08:18 to 08:27; nothing is printed, connected or measured.
Task (owner, morning of 2026-10-23): "i think the only way to permanently fix the HEX-E robot is to calibrate one of our uncalibrated ATI sensors. see the attached notes from chatgpt. we do not have to do the desired calibration." Later: "we can probalby calibrate the ATI with just a 3dp mount to the HEX-E, with the current indenter setup". This was R20 of the tip calibration's queue. At 10:51 the owner moved it out of that channel ("i dont want to pollute the calibration channels"). It continues in its own conversation from this note.
Safety rule of the owner's ATI note, verbatim: "Never enable force-feedback control, contact motion, or safety thresholds based on an unvalidated borrowed matrix."
Read first:
- the owner's ATI note, docs/experiments/20261023-ati_ft_calibration/20261023-ati_mini40_mini45_calibration_handoff.md (hardware, factory matrices of other serials, the raw-gage mode; copied unchanged from the upload);
- the plan page docs/experiments/20261023-ati_ft_calibration/20261023-ati_ft_calibration_plan.html (a local page with no artifact);
- R20 in docs/experiments/20261022-ur5_tip_calibration/QUEUE.md, read only.
Hardware: Mini45 FT11452 (US-120-160 as received), Mini40 FT16552 (SI-20-1) and the Net F/T box NETBA FT35796. No calibration file exists for either sensor.
State (10:51, 10-23):
- Simulation (scripts/ati_calibration_sim.py, 20261023-ati_calibration_sim.json). Presses on the sphere excite three of the six wrench directions: the scaled singular values are 50, 21 and 8, against 0.25, 0.22 and 0.10. A fit to 59 presses is off by 1.5 N in Fx and Fy on other wrenches. Five hook points at eight orientations each, with a weighed 0.5 kg mass, recover the simulated Mini45 to 0.043-0.047 N and 1.0-1.1 N mm (median absolute error on held-out wrenches). The simulation's stack (10 + 15.7 + 8 mm) is older than the parts' 18 + 15.7 + 11 mm.
- Fit (scripts/ati_calibration_fit.py; --selftest wrote 20261023-ati_fit_selftest.json). It fits three models: a full 6 x 6 matrix; "rows6", FT31565's row shapes with six output scales; and "rows12", which adds six gage gains. Self-test medians: full 0.041-0.046 N and 0.9-1.0 N mm; rows12 0.040-0.046 N and 1.3-1.5 N mm; rows6 0.055-0.091 N and 40-64 N mm.
- Logger (scripts/ati_netft_logger.py). It streams RDT over UDP 49152 and writes ati.bin and ati.json into a session folder. It reads the box's configuration and never writes it. It has run only against its own simulated box (--fake).
- Parts (scripts/ati_stack_mount.py; cad/*.stl and 20261023-ati_stack_parts.json). The base (18 mm) sits over v3's M6 heads, and the top (11 mm) takes v3's foot pattern. The hook cross has hooks at (+-60, 0, 30), (0, +-60, 30) and (0, 0, 100) mm. The tool grows 44.7 mm. The ATI clocks at 54 deg, with 3.22 mm of wall. The Mini45's interface is quoted from ATI drawing 9230-05-1094 through a web search. Its pair angle (20 deg assumed), thread depth (4 mm assumed), height (15.7 mm) and dowels are unmeasured.
- Gravity stations (scripts/ati_gravity_pose_check.py, 20261023-ati_gravity_pose_check.json). All 16 stations of the lab plan stay clear with the stack and cross on the tool: 63.7 mm from the robot's links and 67 mm from the cell.
- Plate (scripts/v3b_ur5v2_pla_plate.py ati_stack; print/20261023-ati_stack_p1s_pla.3mf). The three parts fit one P1S plate: 3 h 38 min, 142 g. It stays a draft until the Mini45 is measured.
- Commits: ba8f47e4, 5f2e35ce, ce04ba9d, b28ba705 and 4a56aec2, and the commit of this note.
Waiting on owner:
- which sensor comes first (the plan takes the Mini45; the Mini40's 1 N m range suits 0.2 kg on the same hooks);
- the Mini45 measured: height, the angle between the two holes of a pair, the dowels' places, thread depth;
- a weighed mass (0.5 kg for the Mini45) and an S hook;
- the box: its IP address and the sensor cable checked against ATI's pinout. Its present state is saved first (every web page, netftapi2.xml, the selected slot). Then raw-gage slot #16 (FT00000) is selected by hand;
- the plate's print after the measurement, and time on the UR5 between tip calibration sessions.
Next (each step can run once its input exists):
1. With the measurements, set ATI_H, ATI_PAIR_DEG and ATI_THREAD (and the dowels) in scripts/ati_stack_mount.py and rebuild the parts through resguard.sh run. Then rerun ati_gravity_pose_check.py, reslice with v3b_ur5v2_pla_plate.py ati_stack, rerun ati_calibration_sim.py on the final stack and rebuild the page (scripts/ati_ft_calibration_page.py).
2. Write the pairing step, which does not exist yet. It turns a session folder (ati.bin, robot.bin, the gravity stations in events.jsonl) into pairs.npz for ati_calibration_fit.py. G is the gage values averaged over each station's rest. W is the mass's wrench at the ATI origin, from the hook point and gravity in the sensor frame. The group is the station, and mass-off is subtracted from mass-on at each station. The sensor's turn about the tool axis (the 54 deg clocking and the ATI's own axes) is fitted or taken from the drawing. Test it on a simulated session first.
3. Plan the presses for the ATI against HEX-E comparison with the tool 44.7 mm longer. The plan builder and the checks are the tip calibration's scripts (scripts/ur5_tip_calibration_plan.py, ur5_tip_calibration_check.py --clearance); use them unchanged, and write the plans into this folder's plan/.
Tip calibration interface:
- The stack changes the tool's length and payload, so it goes on the robot only between tip calibration sessions, with the owner's go.
- Requests to the lab computer for this work go in their own file, docs/experiments/20261023-ati_ft_calibration/QUEUE.md, which only this conversation writes. The lab answers in docs/experiments/20261023-ati_ft_calibration/lab/. The owner points a lab session at that file.
- The same gravity poses give the HEX-E's gains (R5 of the tip calibration). Report them on the ATI page, where the tip calibration reads them.
Claimed: scripts/ati_calibration_sim.py, ati_calibration_fit.py, ati_netft_logger.py, ati_stack_mount.py, ati_gravity_pose_check.py and ati_ft_calibration_page.py; docs/experiments/20261023-ati_ft_calibration/; the plate ati_stack in scripts/v3b_ur5v2_pla_plate.py; the status page's two ATI rows.
Do not:
- write in docs/experiments/20261022-ur5_tip_calibration/ (QUEUE.md, lab/, plan/) or send the lab session messages about this work;
- write the Net F/T box's configuration from a script;
- put the stack on the robot during a tip calibration session;
- start a printer or move the robot;
- publish artifacts from the colorado login;
- run a mesh process outside resguard.sh;
- run git pull, rebase or stash in this working tree, because other sessions' uncommitted edits are in it.
Pushing: the tip calibration window shares this working tree. It fast-forwards main from the lab's pushes and pushes every local commit. Commit with explicit paths (git commit -m ... -- <paths>). To push yourself, run git fetch origin first. If git merge-base --is-ancestor origin/main HEAD fails, the lab has pushed. Then list the local commits (git rev-list --reverse origin/main..HEAD), run git reset --keep origin/main, cherry-pick them in order, and run git push origin HEAD:main.
