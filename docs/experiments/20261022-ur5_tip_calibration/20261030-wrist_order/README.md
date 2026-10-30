# Tall stage 2's travels reordered for the wrist (2026-10-30)

How to rebuild what QUEUE R31 reports at 07:4x on 10-30. The plan for ID 3's resume is
`../lab/20261030-plan_tall_lab_stage2_night_wrist_v3id3.json`; the next tip's suite gets the same reorder from
`scripts/ur5_tip_calibration_next_tip_suite.py`.

- `20261030-plan_tall_stage2_v3id3_reordered_full.json`: ID 3's 101 tall stage-2 actions reordered as a whole, a test
  of the reorder only (46 of them ran on 10-29): `scripts/ur5_tip_calibration_wrist_order.py
  lab/20261029-plan_tall_lab_stage2_night_v3id3.json --session lab/20261029-153034-tallv3id3 --out ...`.
- `20261030-paths_tall_stage2_v3id3*_on153034.json`: `scripts/ur5_tip_calibration_holder_paths.py PLAN --session
  lab/20261029-153034-tallv3id3 --other sphere:lab/20261029-134508-spherev3id3 --other rod:lab/20261029-230820-rodv3id3`
  for the night plan and the reordered one.
- `20261030-travels_*.json`: each travel's joints at its start and end (deg) and wrist 1's and wrist 2's extremes,
  replayed as the path checker replays them without clearances: the night plan, the reordered one, and R29's plan
  (`lab/superseded/20261027-plan_tall_lab_r29_v2id1.json` on `lab/20261027-143529-tallv2id1`).
