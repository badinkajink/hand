# HEX-E link losses against the UR5's joints (2026-10-30)

How to rebuild what QUEUE R31 reports at 11:2x on 10-30. The data run from 10-25 to 11:03 on 10-30.

- The logs: robot.bin (host time, q0..q5, 125 Hz) and ft.bin (one row per HEX-E sample, 100 Hz) of the 23 sessions since
  10-25, copied read-only from corvus (`/home/will/workspace/hand/logs/<date>-tip_calibration/<HHMMSS>-<name>/`) to the
  workstation's `logs/20261030-hexe_losses/<HHMMSS>-<name>/` (4.6 GB, gitignored), for example
  `scp "will@100.93.255.76:/home/will/workspace/hand/logs/20261029-tip_calibration/153034-tallv3id3/{robot.bin,ft.bin}" logs/20261030-hexe_losses/153034-tallv3id3/`.
- `20261030-hexe_losses_tol3.json`: `scripts/ur5_hexe_link_losses.py logs/20261030-hexe_losses --out ... --fig ...`,
  run under `~/.claude/bin/resguard.sh run --mem 3G` (6 s, 0.86 GB). A loss is a gap of more than 1 s in the HEX-E's
  samples while robot.bin runs. The file lists each loss with the joints at its last sample and the abort that
  followed it. It also lists the visits within 3 deg of each loss pose on all six joints, and every travel (a motion in
  which the wrists move more than 10 deg) with its class and whether a loss came during it or in the 10 s after it.
- `20261030-hexe_losses_joints.png`: wrist 1 against wrist 2 and wrist 3. A grey point marks each second with the link
  alive, and a red cross marks each loss.
