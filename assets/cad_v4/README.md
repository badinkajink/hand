# CAD of the SR2 finger, version 4: two actuator layouts

cad_v4 starts as two layouts of the finger that take the PIP servo off the finger (2026-11-18). Both are envelopes
(boxes and capsules for servo cases, housings, links, cranks and rods) placed on the cad_v3 hand, not printable parts.
The analysis is on the result page
`docs/experiments/20261118-sr2_actuator_layout/20261118-sr2_finger_actuator_layouts.html`.

- **v4a, PIP servo on the yaw link.** The yaw servo, yaw link and MCP servo are cad_v3's. The PIP servo (an STS3032)
  sits in a housing on the yaw link behind the MCP servo. A parallel four-bar turns an idler on the MCP axis, and a rod
  parallel to the MCP link turns the distal link with the idler. Below the MCP joint the finger is as wide as the sensor
  cassette, 21 mm.
- **v4b, gimbal MCP.** The yaw axis passes through the MCP axis at G, on cad_v3's yaw axis. The MCP servo stands on the
  yoke with its shaft through G, the PIP servo sits on the yoke right behind it, and the yaw servo moves back along the yaw
  axis. The yaw link's 20.75 mm moves into the MCP link (51.55 mm). The distal drive is v4a's.

## Files

| File | What it is |
|---|---|
| `concepts/20261118-layout_params.json` | every dimension of both layouts (mm, the fingers' cad_v3 body frames) |
| `concepts/step/20261118-cad_v4_v4a_index_envelopes.step` | v4a's new parts as named solids (not in git; CAD files move by rsync) |
| `concepts/step/20261118-cad_v4_v4b_index_envelopes.step` | v4b's new parts as named solids (not in git) |
| `../mjcf/experimental/20261118-cad_v4/v4a.xml`, `v4b.xml` | MuJoCo scenes: the cad_v3 hand with index and middle rebuilt; under physics, equality constraints keep the drive on the MCP and PIP joints |

The STEP solids are in the index finger's mount frame at the zero pose, in mm: the origin is the point of the carriage's
yaw axis that cad_v3's model calls `index_mount`, with P's axes (x from the thumb toward the index and middle fingers, y
from the middle toward the index, z up). In cad_v3's P frame the index mount sits at (56.18, 58.40, 0) mm. The parts of
v4a that do not change (yaw link, yaw servo, MCP servo, cassette and cap) are the ones in the Onshape document.

## Frames and conventions

Each finger's body frames follow the cad_v3 MJCF: x dorsal (+) and palmar (-), y along the MCP and PIP axes, z up. The
servo bases of both fingers face +y as built. Joint angles are in the simulator's convention (`manta_hand.plan`
JOINT_SIGN maps the servo readings).

## Viewing the scenes

    .venv/bin/python scripts/cad/cad_v4_layouts.py play v4a                # under physics
    .venv/bin/python scripts/cad/cad_v4_layouts.py play v4b --kinematic    # joints take the sliders' values

The viewer starts at the key `lifted` and is posed from the Control panel's sliders. The actuators are cad_v3's, in joint
angles: `a_index_yaw`, `a_index_mcp`, `a_index_pip`, the same for the middle finger and the thumb, and six for the palm's
pose. The idler, rod, coupler and PIP crank follow the MCP and PIP joints; in both layouts the PIP servo's crank turns at
MCP + PIP. The least index-middle gap of the replay's collision sets (mm, with its pair of parts) is shown at the bottom
left. Under physics touching parts push each other apart; with `--kinematic` they may overlap and the gap goes negative.
The key `open` is the base scene's start pose, with the fingertips 3.9 mm into the floor.

The scenes have no morphology slides (mount x and y, finger length): contact forces dragged them under physics. Mount
positions and link lengths change in the params file, followed by `build`. The carriage's MorphIt spheres touch v4a's MCP
plate at yaw -21 to -12 deg and MCP 31 to 50 deg, where the carriage's mesh is 3.45 mm away (`check`, key `carriage`).
`python -m mujoco.viewer --mjcf=...` also opens a scene, at qpos0 with straight fingers; load a key from its panel.

## Rebuild

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python scripts/cad/cad_v4_layouts.py all
    env -u VIRTUAL_ENV -u PYTHONPATH PATH=/usr/bin:/bin CADV4_REPO=$PWD freecadcmd scripts/cad/cad_v4_step.py
    .venv/bin/python scripts/cad/cad_v4_page.py
