# UR5e calibration mount v2: files and how to rebuild them

Mount v2 of 2026-10-20 for DIP link concept b (v2 as printed on 2026-10-19 and v3) on the OnRobot HEX-E, built by
`scripts/ur5_calibration_mount.py` with `UR5M_REV=v2` (without it the script builds v1 of 2026-10-18 into
`../20261018-ur5_calibration_mount/`). The page is `20261020-dip_link_b_v3_and_ur5_mount_v2.html` in this folder.

- `cad/*.step`: exact solids of the adapter, the standoff-fork and the two indenters (mount frame M: origin at the adapter's
  face on the HEX-E, z along the tool axis toward the tip, x toward the palmar side and the dowel, y along the PIP axis;
  indenters in the table frame, z up); `cad/*.stl` the same as meshes. Inventor: File > Open, STEP, Convert Model.
- `print/20261020-p1s_pla_fork_and_indenters.3mf` (19:10: the standoff-fork with walls around both legs of the DIP link and the
  two indenters with keyed homing posts, Bambu's 0.20 mm Standard, 2 walls, 15 % grid), `print/20261020-p1s_pla_link.3mf`
  (DIP link b v3, cover W00) and `print/20261020-p1s_pla_calibration.3mf` (15:40, printed: the ears fork, adapter, clamp
  plate, round-pin indenters, covers W00/W10/W20): Bambu Studio projects for the P1S, PLA in AMS slot 1, each part's settings
  inside. `cad/*_leg_molds.json`: the x-z polygons of the fork's walls per leg layer (the STEP build reads them).
- `*_parts.json` (volumes, bounds), `*_step.json` (read-back of the STEP files), `*_check.json` (the b links in the mount,
  insertion along the tool axis, gaps, turn to contact, foot-screw access, homing and touching poses), `*_payload.json` (set_payload values).

Rebuild (femtip = `~/miniconda3/envs/femtip/bin/python`; `unset VIRTUAL_ENV` before freecadcmd):

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- env UR5M_REV=v2 NUMPY_MADVISE_HUGEPAGE=0"
    $RG ~/miniconda3/envs/femtip/bin/python scripts/ur5_calibration_mount.py build
    $RG env UR5M_CMD=step UR5M_REPO=$PWD freecadcmd scripts/ur5_calibration_mount.py
    $RG ~/miniconda3/envs/femtip/bin/python scripts/ur5_calibration_mount.py check
    $RG env MUJOCO_GL=egl .venv/bin/python scripts/ur5_calibration_mount.py render
    $RG ~/miniconda3/envs/femtip/bin/python scripts/v3b_ur5v2_pla_plate.py        # fork_and_indenters and link; 'calibration' when named
    $RG ~/miniconda3/envs/femtip/bin/python scripts/ur5_calibration_mount.py payload
    .venv/bin/python scripts/dip_b_v3_ur5_v2_page.py

Sizes and clearances of v2 are the `P.update(...)` under `if REV == "v2"` near the top of the script; the DIP link's faces it
fits are `B_LINK`. DIP link b v3 rebuilds with `DLV2_PARAMS_B=docs/experiments/20261018-dip_link_board_mount/params/20261020-v3_concept_b.json
scripts/dip_link_v2.py b build step check sense render print docs` (see `../20261018-dip_link_board_mount/v2/README.md`).
