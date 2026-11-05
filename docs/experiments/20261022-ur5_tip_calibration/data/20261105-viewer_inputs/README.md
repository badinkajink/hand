# Contact viewer inputs for the SR2 hand (2026-11-05)

Copies of the v3 fingertips' viewer inputs from the workstation's `logs/`, which git does not track, so that corvus can
run `src/sr2hand/driver/reskin5x/host/wrench_viewer.py` on the hand's mux. The originals are unchanged.

- `calib/20261102-v3_id_{1,2,3}-calib.json`: each tip's magnets fitted at rest (from `logs/20261020-contact_estimation/calib/`).
- `correction/<session>/`: the UR5 sphere sessions whose press holds train each tip's correction (`holds.npz`,
  `calib_rest.json`, `summary.json`; from `logs/<session>/`): ID 1 `20261028-spherev3id1_twists`, ID 2
  `20261029-spherev3id2`, ID 3 `20261030-spherev3id3`.
- The board GP is `../20261105-board_gp_v3_id1_id3_rig_id3_id6.npz` (QUEUE R33).

On the hand the mux channels are ch0 middle finger (v3 ID 2), ch1 index (v3 ID 3) and ch3 thumb (v3 ID 1), so
`--names`, `--calib` and `--correction` go in that order: ID 2, ID 3, ID 1.
