# UR5e calibration mount: files and how to rebuild them

Mount for a DIP link as printed (the CAD hand's, v2 a or v2 b of the DIP study: all keep the clevis arms) on the tool side of the HEX-E QC on the UR5e, built by `scripts/ur5_calibration_mount.py`. The page is `20261018-ur5_calibration_mount.html` in this folder.

## Files

- `cad/*.step`: exact solids per part, in the mount frame M (origin at the flange face's centre, z along the tool axis toward the tip, x toward the palmar side, y along the PIP axis toward the slotted arm); the indenters in a table frame (z up). `cad/*.stl`: the same as meshes.
- `print/*.3mf`: PrusaSlicer projects for the Prusa XL on the owner's v17 template, PLA on tool 3, sliced and checked (`print/*.json`).
- For the Bambu Lab P1S (PLA from 2026-10-19 on): the three mount parts and the two indenters are on one plate with DIP-link v2 a and b, `../20261019-v2_ur5_pla_plate/20261019-v2_ur5_pla_p1s.3mf` (built by `scripts/v2_ur5_pla_plate.py`, each part's settings in the project as object settings: 0.15 mm, the standoff-fork and the clamp plate solid, the adapter and the indenters 40 % gyroid). The M3 holes stay 2.75 mm. With a v2 link, the slotted arm's M3 washer (DIN 9021, 9 mm) covers the end of the cable's groove on the arm's outer face (its anchor lies 5.0 mm below the PIP axis): lead the cable off the arm before the anchor when the link sits in the mount.
- `20261018-ur5_calibration_mount_interfaces.json` (flange and DIP-link interfaces with sources), `_parts.json`, `_step.json`, `_check.json`, `_estimate.json` (deflection, payload, wrist field), `_assembly.json` (parts list, assembly).

## Opening the STEP files in Inventor

File > Open, file type STEP, Options > Convert Model: each part arrives as one editable solid body. The DIP study's STEP files (frame D of `fingertip_mount_DIP_link.stl`) go into this frame by a translation only: x_M = x_D, y_M = y_D - 17.40, z_M = z_D + 46.0.

## Rebuild

    E=~/miniconda3/envs/femtip/bin/python
    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 400 -- taskset -c 12-15 env NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1"
    $RG $E scripts/ur5_calibration_mount.py interfaces
    $RG $E scripts/ur5_calibration_mount.py build
    $RG env UR5M_CMD=step UR5M_REPO=$PWD freecadcmd scripts/ur5_calibration_mount.py
    $RG $E scripts/ur5_calibration_mount.py check
    $RG $E scripts/ur5_calibration_mount.py estimate
    $RG env MUJOCO_GL=egl .venv/bin/python scripts/ur5_calibration_mount.py render
    $RG $E scripts/ur5_calibration_mount.py sections
    $RG $E scripts/ur5_calibration_mount.py print
    $E scripts/ur5_calibration_mount.py docs

Sizes and clearances are the `P` dictionary at the top of the script; change one there and rerun from `build`.
