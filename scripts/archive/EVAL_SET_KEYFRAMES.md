# Eval-set scenes for run18 — standalone MJCFs for manual review

These are the `_short_proximal` scenes used by `run18_multi_object_sweep.py`.
Each is a self-contained MJCF you can open in the MuJoCo viewer
(`python -m mujoco.viewer --mjcf <path>`) and edit by hand. The Lightning-
authored keyframe is named `open_short` in each new scene; the original
`open` keyframe (authored for the long hand) is also preserved for
reference.

All scenes have **capsule fingertips** (geom = capsule, fromto = -0.006..0.006,
radius = 0.005) and **short proximal phalanges** (each `*_len_frame` body
at `pos="0.025 0 0"` instead of `0.05`).

## Scene → keyframe map

| Object | Scene path | Keyframe to use | Object body name | Notes |
|---|---|---|---|---|
| cube                       | [assets/mjcf/baseline/scenes/scene_cube_short_proximal.xml](../assets/mjcf/baseline/scenes/scene_cube_short_proximal.xml)                            | `open_short`         | `cube`              | Lightning-authored |
| prism                      | [assets/mjcf/baseline/scenes/scene_prism_short_proximal.xml](../assets/mjcf/baseline/scenes/scene_prism_short_proximal.xml)                          | `open_short`         | `prism`             | Lightning-authored |
| power_drill                | [assets/mjcf/baseline/scenes/scene_power_drill_short_proximal.xml](../assets/mjcf/baseline/scenes/scene_power_drill_short_proximal.xml)              | `open_flat_gripping` | `power_drill`       | Hand-authored from run17 |
| screwdriver_medium_flat    | [assets/mjcf/baseline/scenes/scene_screwdriver_medium_flat_short_proximal.xml](../assets/mjcf/baseline/scenes/scene_screwdriver_medium_flat_short_proximal.xml) | `open_short`         | `screwdriver_medium` | Lightning-authored |
| screwdriver_medium_vertical| [assets/mjcf/baseline/scenes/scene_screwdriver_medium_vertical_short_proximal.xml](../assets/mjcf/baseline/scenes/scene_screwdriver_medium_vertical_short_proximal.xml) | `open_short` | `screwdriver_medium` | Lightning-authored; original `open` keyframe palm was relocated to mirror `open_vertical` |
| screwdriver_medium_90vert  | [assets/mjcf/baseline/scenes/scene_screwdriver_medium_short_proximal.xml](../assets/mjcf/baseline/scenes/scene_screwdriver_medium_short_proximal.xml) | `open_90vertical`    | `cube` (legacy)     | Existing keyframe (no Lightning needed) |
| screwdriver_small_flat     | [assets/mjcf/baseline/scenes/scene_screwdriver_small_flat_short_proximal.xml](../assets/mjcf/baseline/scenes/scene_screwdriver_small_flat_short_proximal.xml) | `open_short`         | `screwdriver_small` | Lightning-authored |

## To validate a keyframe yourself

```bash
.venv/bin/python scripts/smoke_eval_set_short_proximal.py
```
reports `verdict / ncon / fingers / lift` per scene. `verdict=OK` means
≥2 fingers in contact and ≥2.5 cm lift on a 5 cm palm lift.

## To add or replace a keyframe

The standard MuJoCo `<keyframe><key name="..." qpos="..." ctrl="..."/></keyframe>`
block sits at the end of every scene file. You can:

1. Open in mujoco viewer, pose the hand, capture `qpos` from `data.qpos` and
   `ctrl` from `data.ctrl`, and paste into a new `<key>` element.
2. Or hand-edit by mirroring an existing key. qpos layout for these scenes
   (31 entries):
   ```
   obj_pos(3) obj_quat(4)              # freejoint
   palm_px palm_py palm_pz palm_rx palm_ry palm_rz   # palm 6-dof
   thumb_x_morph thumb_y_morph thumb_yaw thumb_mcp thumb_len_morph thumb_pip
   index_x_morph index_y_morph index_yaw index_mcp index_len_morph index_pip
   middle_x_morph middle_y_morph middle_yaw middle_mcp middle_len_morph middle_pip
   ```
   The 9 ctrl entries (after the 6 palm pose ctrls) skip the 6 morph dims:
   ```
   palm_px palm_py palm_pz palm_rx palm_ry palm_rz
   thumb_yaw thumb_mcp thumb_pip
   index_yaw index_mcp index_pip
   middle_yaw middle_mcp middle_pip
   ```

## To re-run Lightning for a single scene

```bash
.venv/bin/python scripts/build_short_proximal_keyframes.py \
  --objects cube --batch-outer 256 --batch-inner 128
```

Multiple comma-separated labels are accepted. The script auto-builds a
short-hand URDF from the scene's current `open` keyframe morphology, swaps
it into Lightning's assets, runs `lightning_grasp_runner.py`, scores via
`lightning_grasp_eval.py --mode init_pose`, picks the best by score, and
replaces the `open_short` `<key>` in the scene XML.
