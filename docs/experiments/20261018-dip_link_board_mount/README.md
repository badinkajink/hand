# DIP link: parameter files, viewer, export to Inventor

Runbook for changing the DIP-link designs a (window plate and dorsal clamp plate) and b (the board in a cassette behind a
pad) by their parameters, looking at the result, and exporting it as STEP for Inventor. The printable v2 of both designs
and the order to print them are in `v2/README.md`; the design study is the
[DIP page](20261018-dip_link_magnetometer_board_mount.html).

## Parameter files

`params/*.json`, one per design: every size, clearance and spacing with its value, unit, allowed range, what it sets and
the checks it changes, and the design's discrete choices (`design`). The committed designs:

- `params/20261018-v2_concept_a.json`, `params/20261018-v2_concept_b.json`: the printable v2 (J1 kept, five-magnet tips,
  M2 self-forming screws); `v2/a/` and `v2/b/` reproduce from them (`scripts/dip_link_v2.py a|b build ...`).
- `params/20261018-concept_a.json`, `params/20261018-concept_b_final.json`, `params/20261018-concept_b.json`: the judge's
  final of a (`final/`), b's final and b as first committed; `scripts/dip_link_params.py verify` regenerates them and
  compares volumes with the committed STL files.

A saved set (`params/YYYYMMDD-HHMM-concept_<a|b>-<name>.json`) has no `outputs`, so building it never overwrites a
committed design; `changed_from_default` lists what differs.

## Viewer

From a VS Code terminal on the workstation (VS Code forwards the port; open the address it shows):

    cd ~/Programs/hand
    ~/.claude/bin/resguard.sh run --mem 3G --cpu 400 -- taskset -c 8-11 env NUMPY_MADVISE_HUGEPAGE=0 \
        ~/miniconda3/envs/femtip/bin/python scripts/dip_link_viewer.py          # http://127.0.0.1:8780/

`http://127.0.0.1:8780/?concept=b&file=20261018-v2_concept_b.json` opens a given file. The page shows the chosen file's
parameters as numeric fields with their ranges (a value outside its range is outlined red, a changed one yellow), the
assembly (link, clamp plate or cover, board with J1 and its plug, tip or pad with magnets, screws) with per-part
visibility and opacity, a section plane, and the MCP link and PIP servo on a PIP slider from -18 to 92 deg. A change
rebuilds the parts with manifold3d in 2-5 s and updates the readouts: least clearance over the PIP range (keep-out
estimate, about 0.13 mm under the exact check), board-to-tip gap, chip-to-magnet distance, least wall and mass.

- Save: writes the current values as a dated file under `params/`.
- Clearance check: runs `scripts/dip_link_params.py check` on the saved file under resguard (the exact sweep, 3-6 min) and
  shows the result; output under `logs/20261018-dip_link/params/checks/`.
- Export: runs `scripts/dip_link_params.py export` on the saved file (below).

Ctrl-C in the terminal stops the viewer.

## Export

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 400 -- taskset -c 8-11 env NUMPY_MADVISE_HUGEPAGE=0"
    $RG ~/miniconda3/envs/femtip/bin/python scripts/dip_link_params.py export --params params/FILE.json [--name NAME]

writes `exports/YYYYMMDD-HHMM-<name>/` in about a minute: `cad/` with an STL and a STEP per part (exact solids for the
link's new features or new body, the clamp plate or cover and the tip cutter; faceted solids for the tips and pads; STEP
files over 1.5 MB zipped), `<name>_assembly_D.step` with the board as supplied, J1, the plug, the screws and the printed
parts as named solids in the coordinates of `assets/cad_v1/fingertip_mount_DIP_link.stl`, the 11/22 tip or pad among them
as a faceted solid, `print/` with the PLA parts as print-oriented STLs for Bambu Studio (their settings in
`<name>_pla_bambu.json`) and each tip or pad sliced for the XL with its magnet pauses, and `export.json` with every STEP
read back in FreeCAD. The v2 designs with the routed cable are in `exports/20261019-1816-v2_concept_a_plug7_cable2p6/`
and `exports/20261019-1818-v2_concept_b_plug7_cable2p6/` (the 7.8 mm J1 slot for the 7.0 mm SHR-04V-S-B plug, the 3.0 mm
cable groove); the exports of 15:08 and 15:16 have the 6.6 mm slot and the 2.4 mm groove.

## Opening the STEP files in Inventor

1. Unzip a `.step.zip` first; the plain `.step` files open as they are.
2. File > Open, file type STEP (`*.stp, *.ste, *.step`), choose the file, then Options.
3. Under Import options choose Convert Model: the exact parts arrive as editable solid bodies (the assembly STEP as an
   assembly of named parts). Reference Model keeps them as non-editable references.
4. The fused DIP link (`*_dip_link.step.zip`) contains the clevis arms as facets of the original STL; edit the exact new
   body or new features instead and keep the arms as they are. STL files open only as reference meshes.

The parts lie in the DIP link STL's frame: x_D = -x_F, y_D = y_F + 21.25, z_D = -z_F - 51.55 (PIP axis along y, finger
along +z, palmar face toward +x).
