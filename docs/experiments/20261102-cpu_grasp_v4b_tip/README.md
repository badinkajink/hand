# 2026-11-02: real_v3 with the v4b magnetic-sensing fingertip

`assets/mjcf/real_v3/real_hand.xml` now carries the v4b fingertip (the one on the UR5 calibration rig since 10-23;
in simulation on `main` since 72f382c3 as `assets/mjcf/experimental/20261030-cad_v2`). The TPU-block hand that
policy A was trained on is kept as `real_hand_block.xml`; `CPU_GRASP_HAND=block` loads it in `cpu_grasp`.

Built by `scripts/build_real_v3_v4b_tip.py` from `real_hand_block.xml` and the parts in `assets/mjcf/real_v3/v4b/`
(copied from `main`):

| per finger | block tip (old) | v4b (new) |
|---|---|---|
| distal parts | STEP distal link, 32 MorphIt spheres | DIP link b v4 (board mount) + 3T cassette + TW15H3 cap with M2 heads, 315 inscribed spheres + meshes |
| pad | flat TPU block, 238 spheres r 1 mm | 11/22 half-ellipsoid (7.4 across, 8.5 deep, 11 distal / 22 proximal of the apex), 435 spheres r 1 mm at 1.5 mm |
| PIP axis to pad contact point | 26.3 mm | 36.4 mm (apex) |
| PIP axis to lowest point | 37.5 mm | 48.5 mm |
| distal mass (link + pad) | 10.65 g | 12.92 g (board, cable and the four magnets not modelled) |

Frame check: the v4b parts are in the cad_v1 finger frame at the PIP axis; registered against real_v3's distal link
above the pad they sit 0.6 mm / 0.65 deg from it with no transform (the residual is the design change), so they go
in unchanged (thumb mirrored about z, as both models do). `v4b_finger_closeup.png`: block vs v4b, visuals and
collision spheres.

## Plan changes (`src/morphohand/cpu_grasp`)

The block plan's flat-face IK put the curved pad's contact only near its apex: only 40 mm was feasible. For the v4b
hand (detected by its pad mesh; the block path is unchanged and still gives 0.209 open loop):

- `kin.py`: the pad's distance to the cylinder axis is a soft minimum over its spheres; the contact normal must be
  within 35 deg of the pad's palmar normal (without it the IK pressed the pads' ends on top of the shaft, 60-70 deg
  above the equator); the link / cassette / cap spheres stay 1 mm farther from the axis than the pad.
- `plan.py`: the IK's palm seed puts the D1 pads level with the axis (the fingers are 10 mm longer), and the plan
  table walks outward from 25 mm, each radius seeded by its neighbour's touch pose (solved alone, 20 and 32.5 mm
  fell into a +-75 deg yaw pose). Every radius 17.5-42.5 mm is now feasible; contacts within 10 deg of the equator,
  2 mm table clearance.

## E1 trial set (same 1008 trials as E1 v2 and policy A)

| | block tip | v4b tip |
|---|---|---|
| plan alone: success / lifted | 0.209 / 0.799 | **0.671** / 0.739 |
| plan alone: trials with finger–table contact | 0.889 | 0.111 |
| policy A (trained on the block tip), unchanged | 0.975 | 0.712 |

`plan_alone/` and `policyA_block_trained/`: trials, grid and heatmaps. The 1008 open-loop trials take 55 s
(39 s with the block).

## Policy A retrained on the v4b hand (2026-11-03)

`runs/cpu_grasp/A_v4b_s0` (same settings as A_palm_s0_full; best evaluation checkpoint iteration 120, 0.996).
On the 1008 E1 trials: **success 0.997, lifted 1.000, no trial with finger–table contact**; the only cell below 1.0
is 20 mm × 150 g (0.81). Residuals smaller than on the block hand (mean |finger| 1.8 deg vs 2.7, peak |palm| 1.8 mm
vs 4.9, no longer at the 5 mm limit). `policyA_v4b/` (compare.png: plan alone vs policy on v4b),
`learning_curve_block_vs_v4b.png`; videos in `manta/sim_media/20261103-v4b_policyA/`.

Videos: `manta/sim_media/20261102-v4b_tip/` (top: plan alone, bottom: policy A as trained on the block tip).
