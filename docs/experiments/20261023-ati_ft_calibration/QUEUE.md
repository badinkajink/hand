# Requests to the lab computer for the calibration of the ATI Mini45 on the UR5

Only the workstation's ATI conversation writes this file. The lab session answers in `lab/NOTES.md` of this folder and in
its commit messages, and copies each session folder whole to `lab/<YYYYMMDD>-<time>-<name>/`. The owner points a lab
session at this file; nothing here runs before the owner says so. The design, the plan and the procedure are on the
page `20261023-ati_ft_calibration_plan.html` in this folder.

Fetch the plan, which this repository stores in LFS: `git -c lfs.fetchexclude= lfs pull
--include=docs/experiments/20261023-ati_ft_calibration/plan/20261023-ati_plan_mass.json`. The programs are
`scripts/ati_netft_logger.py` and `scripts/ati_calibration_session.py`; the session program imports
`src/morphohand/driver/reskin5x/host/ur5_tip_calibration.py` and needs numpy and ur_rtde, as the tip calibration does.
Run the session in its simulator first, on the lab computer: `$PY scripts/ati_calibration_session.py --plan <plan>
--mass-g 500 --sim --name sim` takes about 20 s.

- A1 (open, 2026-10-23 12:25; waits for the owner's setup of the box). Check the Net F/T box (NETBA FT35796) and the
  Mini45 (FT11452) before any print. The owner saves the box's state as found first: its DIP switches photographed, every
  web page and `http://<box>/netftapi2.xml` saved into `lab/<date>-ati_box_as_found/`. Then the owner selects #16
  (FT00000), lbf and lbf-in on the Configuration page and enables RDT on comm.htm. The lab session runs
  `$PY scripts/ati_netft_logger.py --ip <box> --out docs/experiments/20261023-ati_ft_calibration/lab/<date>-<time>-boxcheck --seconds 20`
  with the sensor at rest for 10 s, then pressed by hand on its tool side for 10 s. It passes with status 0
  throughout, the RDT rate as set, no gage at its limit and every gage changing under the press. Then record the gages'
  drift: the same command with `--seconds 600` and `<date>-<time>-boxdrift`, nothing touching the sensor (added 13:13:
  the calibration with the 100 g weight is limited by the drift between a station's two rests in simulation). Push both
  folders.

- A2 (open, 12:25; after A1 passes and the stack is assembled). Dry run of the plan with no mass, at half speed, the
  owner's hand on the emergency stop. Jog the robot to the plan's homing pose first (joints 48.3, -86.3, 109.5, -24.8,
  47.3, 6.4 deg, the lab's home of 22 October). Command:
  `$PY scripts/ati_calibration_session.py --plan docs/experiments/20261023-ati_ft_calibration/plan/20261023-ati_plan_mass.json --ati <box> --dry --speed 0.5 --name dry`.
  About 25 min: no mass changes, the moves at half speed. Joint 5 reaches 129 deg at the +x and +y eyes' stations, past the tip calibration's window of 100 deg;
  the owner watches the HEX-E's cable port and the ATI's cable there. Note in `lab/NOTES.md` any station where a cable
  pulls or where anything in the cell comes within 50 mm of the tool. Push the session folder.

- A3 (open, 12:25; after A2). The calibration run, about 27 min, with the operator at the robot for ten mass changes.
  The mass is the owner's 100 g weight in its printed holder (plate `print/20261023-ati_mass_holder_p1s_pla.3mf`, two
  M3 x 12; cap 2 of 15:34, `print/20261023-ati_mass_cap2_p1s_pla.3mf`, fits the measured knob and has a tie hole), or a
  heavier mass that fits the plan's envelope. A line is tied through the cap's tie hole: polyester thread, dental floss
  or braided fishing line, 0.2 to 0.5 mm across, never wire or monofilament (a stiff line moves the force off the eye).
  At each mass change its free end goes through the eye from outside and out of the slot, and a figure-eight stopper
  knot in the slot holds it with the tie hole 55 to 70 mm below the eye. Measure the line's diameter and weigh the
  weight, holder, screws and line together to 0.1 g first. Command: the one of A2 with `--mass-g <grams> --line-mm <mm>
  --hexe 192.168.0.6 --name run1` in place of `--dry --speed 0.5 --name dry`, and `--eye-hole-mm <mm>` if the eyes were
  drilled larger than 1.2 mm. The program stops on a gage saturation, on a gap of over
  0.5 s in the box's stream and on a protective stop; `--resume <folder>` continues. Push the whole session folder.

- A4 (open, 17:52; after A3; the owner's go). The Mini40 (FT16552) on the same box. With the box off, move the
  transducer cable from the Mini45 to the Mini40. Unscrew the cross, the top, the Mini45 and its base. Screw the Mini40
  to its own base (plate `print/20261023-ati_stack_mini40_p1s_pla.3mf`) with three M3 x 12 from below, the base onto the
  adapter with four M3 x 20, its top onto it with three M3 x 10 (never longer: its tool-side threads are 3.0 mm), and the
  cross onto the top with four M3 x 12. Its cable lies in the base's channel toward the Mini45's route. Note in
  `lab/NOTES.md` whether the Mini40 is a -R (short fitting), -E (17 mm strain relief) or -A (cable out of the mounting
  face's centre: the base does not fit it; stop). Then the box check of A1 for the Mini40, into `lab/<date>-<time>-boxcheck40`
  and `-boxdrift40`; the box stays on #16 (FT00000).
- A5 (open, 17:52; after A4). The Mini40's dry run: A2's command with
  `--plan docs/experiments/20261023-ati_ft_calibration/plan/20261023-ati_plan_mass_mini40.json --name dry40`. Its torque
  range is 1 N m: a push of about 12 N at an arm's end saturates it, and the program stops on a saturation.
- A6 (open, 17:52; after A5). The Mini40's run: A3's command with the Mini40's plan and `--name run40`, the same weight,
  holder and line.
