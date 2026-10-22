# UR5e calibration mount v3: files and how to rebuild them

Mount v3 of 2026-10-21 for the owner's M6 x 20 screws (owner, 17:54: "i only have 20mm long m6 screws ... make the holes
accessible from the top plate"), built by `scripts/ur5_calibration_mount.py` with `UR5M_REV=v3` (v2 and v1 build as before
with `UR5M_REV=v2` or unset). The adapter is 15 mm thick, so an M6 x 20 seated on it enters the HEX-E 5.0 mm (3.4 mm over a
DIN 125 washer); the heads sit captive in pockets of the standoff-fork's 13 mm foot, two under straight key holes and two under
channels for a ball-end key at 20 deg; everything above the foot is v2's, 11 mm higher. The page is
`20261021-ur5_calibration_mount_v3.html` in this folder.

- `cad/*.step`: exact solids of the adapter, the standoff-fork, the clamp plate (v2's, moved with the fork) and the two
  indenters (v2's, unchanged) in the mount frame M (origin at the adapter's face on the HEX-E, z along the tool axis toward
  the tip, x toward the palmar side and the dowel, y along the PIP axis); `cad/*.stl` the same as meshes; `cad/*_leg_molds.json`
  the walls around the legs (from DIP links b v2, v3 and v4). Inventor: File > Open, STEP, Convert Model.
- `print/20261021-ur5_calibration_mount_v3_p1s_pla.3mf`: the adapter (U1, six walls) and the standoff-fork (U2, four walls) at
  Bambu's 0.20 mm Standard with 15 % grid, 1 h 58 min, 75.9 g; `_p1s_pla.json` the slicer's check, `_map.png` the plate.
- `*_parts.json` (parameters, volumes), `*_step.json` (STEP read-back), `*_check.json` (the b links in the fork, homing and
  touching poses), `*_screws.json` (the M6 screws seated and lifted, the keys at each tilt, the heads captive), `*_payload.json`.

Rebuild (femtip = `~/miniconda3/envs/femtip/bin/python`; `unset VIRTUAL_ENV PYTHONPATH` before freecadcmd):

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- env UR5M_REV=v3 NUMPY_MADVISE_HUGEPAGE=0"
    $RG ~/miniconda3/envs/femtip/bin/python scripts/ur5_calibration_mount.py build
    $RG env UR5M_CMD=step UR5M_REPO=$PWD freecadcmd scripts/ur5_calibration_mount.py
    $RG ~/miniconda3/envs/femtip/bin/python scripts/ur5_calibration_mount.py check
    $RG ~/miniconda3/envs/femtip/bin/python scripts/ur5_calibration_mount.py screws
    $RG ~/miniconda3/envs/femtip/bin/python scripts/v3b_ur5v2_pla_plate.py mount_v3
    $RG ~/miniconda3/envs/femtip/bin/python scripts/ur5_calibration_mount.py payload
    $RG ~/miniconda3/envs/femtip/bin/python scripts/ur5_calibration_mount.py sections
    $RG env MUJOCO_GL=egl .venv/bin/python scripts/ur5_calibration_mount.py render
    .venv/bin/python scripts/ur5_mount_v3_page.py

Sizes of v3 are the `P.update(...)` under `if REV == "v3"` near the top of the script (on top of v2's block): `ad_t` sets the
thread engagement (20 - ad_t mm without a washer), `key_tilt` the lower screws' channels, `m6_pocket_top` how far a head lifts.
