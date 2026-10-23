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
  throughout, the RDT rate as set, no gage at its limit and every gage changing under the press. Push the folder.

- A2 (open, 12:25; after A1 passes and the stack is assembled). Dry run of the plan with no mass, at half speed, the
  owner's hand on the emergency stop. Jog the robot to the plan's homing pose first (joints 48.3, -86.3, 109.5, -24.8,
  47.3, 6.4 deg, the lab's home of 22 October). Command:
  `$PY scripts/ati_calibration_session.py --plan docs/experiments/20261023-ati_ft_calibration/plan/20261023-ati_plan_mass.json --ati <box> --dry --speed 0.5 --name dry`.
  About 25 min: no mass changes, the moves at half speed. Joint 5 reaches 129 deg at the +x and +y eyes' stations, past the tip calibration's window of 100 deg;
  the owner watches the HEX-E's cable port and the ATI's cable there. Note in `lab/NOTES.md` any station where a cable
  pulls or where anything in the cell comes within 50 mm of the tool. Push the session folder.

- A3 (open, 12:25; after A2). The calibration run, about 27 min, with the operator at the robot for ten mass changes.
  Weigh the mass with its S hook to 0.1 g first. Command: the one of A2 with `--mass-g <grams> --hexe 192.168.0.6
  --name run1` in place of `--dry --speed 0.5 --name dry`. The program stops on a gage saturation, on a gap of over
  0.5 s in the box's stream and on a protective stop; `--resume <folder>` continues. Push the whole session folder.
