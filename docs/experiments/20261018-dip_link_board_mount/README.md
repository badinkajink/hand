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
read back in FreeCAD. The v2 designs with the routed cable are in `exports/20261019-1851-v2_concept_a_m2hole1p9/`
and `exports/20261019-1853-v2_concept_b_m2hole1p9/` (the 7.8 mm J1 slot for the 7.0 mm SHR-04V-S-B plug, the 3.0 mm
cable groove, 1.9 mm M2 holes); the exports of 15:08 and 15:16 have the 6.6 mm slot and the 2.4 mm groove, those of
18:16 and 18:18 the 1.8 mm holes.

## Caps over b v3's tip end

Two PLA caps over the tip end of b v3 (owner, 2026-10-20 19:48, the pad popping off its rails): an end plate on the tip
end, ears under the two distal board screws (M2 x 8 in place of M2 x 5), a 2 mm pin 2.7 mm into the pad's flat end; cap TW
adds side walls along the pad's distal flanks. The pad that takes the pin is B00P (B00 with a 2.0 mm hole, 3.0 mm deep).

    unset VIRTUAL_ENV PYTHONPATH
    RG="$HOME/.claude/bin/resguard.sh run --mem 6G --cpu 300 -- taskset -c 14,15 env NUMPY_MADVISE_HUGEPAGE=0"
    $RG ~/miniconda3/envs/femtip/bin/python -u scripts/dip_b_tip_cap.py build step render       # caps, checks, STEP, figure
    $RG env UR5M_REV=v2 ~/miniconda3/envs/femtip/bin/python -u scripts/v3b_ur5v2_pla_plate.py tip_caps   # P1S plate
    $RG ~/miniconda3/envs/femtip/bin/python -u scripts/v3b_pad_b00_magnets_xl.py --magnets five --pin-hole  # pad B00P, XL

Sizes are in the `CAP` dict at the top of `scripts/dip_b_tip_cap.py`. Outputs: `v3/b/cad/20261020-dip_link_v3b_tip_cap_{TP,TW}.step`
and `.stl` (frame F), `v3/b/print/20261020-dip_link_v3b_p1s_pla_tip_caps.3mf` (both caps, 17 min),
`v3/b/print/20261020-dip_link_v3b_print_pad_ell11_5_B00_pin_85a_t2_g30.3mf`, the checks in
`v3/b/20261020-dip_link_v3b_tip_caps.json`, the FE bounds as cases `bm_five_d75_capend` and `bm_five_d75_cap` of
`scripts/v2_tip_fe_checks.py`. The page is Section 9 of
[DIP link b v3 and UR5e mount v2](../20261020-ur5_calibration_mount_v2/20261020-dip_link_b_v3_and_ur5_mount_v2.html).

### Second revision (2026-10-21)

After the pad, pushed sideways, came off cover W00 and broke a wall's proximal end (a 0.66 mm fin beside the proximal
screw's counterbore): covers I05, I10, I15 with `cover_walls` "solid" in `scripts/dip_link_concept_b.py` (walls straight up
the 21 mm edge with a 1 mm chamfer, ending at the U3/U5 windows, 0.05/0.10/0.15 mm interference per side; the proximal
screws' heads on solid wall `seat_raise` 3 mm up, M2 x 8), caps TP2 (1.6 mm plate, chamfers, filleted pin) and TW2 (solid
5 mm side blocks, M2 x 10 for the distal screws), and pad B00R (proximal notches 4.5 mm deep, the pin hole chamfered):

    $RG ~/miniconda3/envs/femtip/bin/python -u scripts/dip_b_tip_cap.py covers2 step2 render2
    $RG env UR5M_REV=v2 ~/miniconda3/envs/femtip/bin/python -u scripts/v3b_ur5v2_pla_plate.py solid_covers_caps2
    $RG ~/miniconda3/envs/femtip/bin/python -u scripts/v3b_pad_b00_magnets_xl.py --magnets five --pin-hole --raised-seats

Outputs: `v3/b/cad/20261021-dip_link_v3b_cover_I{05,10,15}.step`, `*_tip_cap_TP2.step`, `*_tip_cap_TW2.step`,
`v3/b/print/20261021-dip_link_v3b_p1s_pla_solid_covers_caps2.3mf` (40 min),
`v3/b/print/20261021-dip_link_v3b_print_pad_ell11_5_B00_pin_raised_85a_t2_g30.3mf`, checks in
`v3/b/20261021-dip_link_v3b_solid_covers_and_caps2.json`; page Section 10.

### v4 and the wall-height sweep (2026-10-21 16:20)

DIP link b v4 (`params/20261021-v4_concept_b.json`: design `bridge` recessed, `col_margin` 3.0, `col_side_margin` 1.45, cover I15):
the bridge's palmar face from x_F -10.55 (flush with the pad's apex) to -6.46 with a 1 mm chamfer, the slotted arm's spur cut to the same
plane, the column under the pad 3.0 mm inside the pad's mount face on the palmar side. The sweep: covers I15 with walls 4, 3 and 2 mm
(I15H4-H2) and caps with side blocks 4, 3 and 2 mm at 0.15 mm (TW15H4-H2), the height engraved on each.

    DLV2_PARAMS_B=docs/experiments/20261018-dip_link_board_mount/params/20261021-v4_concept_b.json \
        $RG ~/miniconda3/envs/femtip/bin/python -u scripts/dip_link_v2.py b build check step
    $RG ~/miniconda3/envs/femtip/bin/python -u scripts/dip_b_v4.py link heights step figures
    $RG env UR5M_REV=v2 ~/miniconda3/envs/femtip/bin/python -u scripts/v3b_ur5v2_pla_plate.py heights link_v4

Outputs in `v4/b/`: `cad/20261021-dip_link_v4b_*` (link, cover I15, covers I15H4-H2, caps TW15H4-H2; STEP without the engraved
IDs), `print/20261021-dip_link_v4b_p1s_pla_heights.3mf` (42 min) and `_p1s_pla_link.3mf` (1 h 8 min), checks in
`20261021-dip_link_v4b_checks.json` and `_heights.json`; page
[DIP link b v4 and a sweep of retaining-wall heights](../20261021-dip_link_b_v4/20261021-dip_link_b_v4_and_wall_heights.html).

### Chosen heights and tighter rails (2026-10-22 11:30)

The owner chose the sweep's 3 mm pair (cover I15H3, cap TW15H3) and asked for tighter rails and a smaller pin hole. Cover 3T is
I15H3 with each rail 0.05 mm wider per flank and its top edge 0.05 mm more (`rail_grow`, `rail_top_grow` in `cover_part`; absent
keys build v1-v4 unchanged); pad B05R is B00R with its grooves 0.05 mm inside the nominal rail and a 1.6 mm pin hole. The rail
presses 0.10 mm into each groove wall at the cover and 0.16 mm at its top. Sizes are `FIT3` in `scripts/dip_b_v4.py`.

    $RG ~/miniconda3/envs/femtip/bin/python -u scripts/dip_b_v4.py fit3 fit3fig step3
    $RG ~/miniconda3/envs/femtip/bin/python -u scripts/v3b_pad_b00_magnets_xl.py --magnets five --pin-hole --raised-seats \
        --groove-fit 0.05 --hole-d 1.6 --pid B05R --copies 3 --out docs/experiments/20261018-dip_link_board_mount/v4/b/print --date 20261022
    $RG env UR5M_REV=v2 ~/miniconda3/envs/femtip/bin/python -u scripts/v3b_ur5v2_pla_plate.py fit3

Outputs in `v4/b/`: `cad/20261022-dip_link_v4b_cover_I15H3_3T.{stl,step}`, `print/20261022-dip_link_print_pad_ell11_5_B05R_x3_85a_t2_g30.3mf`
(three pads on the XL, pauses at 7.3, 12.2, 17.1 mm, 1 h 20 min), `print/20261022-dip_link_v4b_p1s_pla_fit3.3mf` (three 3T covers and
three TW15H3 caps on the P1S, 43 min), checks in `20261022-dip_link_v4b_fit3.json`; page section 8 of the v4 page.

### Lowest heights with the tighter rails (2026-10-22 16:17)

The owner asked for the sweep's lowest pair again, now that the rails fit tighter. Cover 2T is I15H2 with 3T's rails (`FIT2` in
`scripts/dip_b_v4.py`); the cap is the sweep's TW15H2 and the pads are the B05R already printed. Both screw pairs are M2 x 6, with
2.5 mm of thread in the tongue.

    $RG ~/miniconda3/envs/femtip/bin/python -u scripts/dip_b_v4.py fit2 step2
    $RG env UR5M_REV=v2 ~/miniconda3/envs/femtip/bin/python -u scripts/v3b_ur5v2_pla_plate.py fit2

Outputs in `v4/b/`: `cad/20261022-dip_link_v4b_cover_I15H2_2T.{stl,step}`, `print/20261022-dip_link_v4b_p1s_pla_fit2.3mf` (three 2T
covers and three TW15H2 caps on the P1S, 39 min), checks in `20261022-dip_link_v4b_fit2.json`; Table 9 of the v4 page. Two more
DIP links b v4 on one P1S plate: `print/20261022-dip_link_v4b_p1s_pla_link_x2.3mf` (1 h 52 min).

### Snug sets without the added interference (2026-10-22 16:37)

With the 3T set the pad no longer sat flush on the cassette and was locked in too tightly (owner: "now i want to try a setting
where it's a more very-snug fit without all the compression"). The snug sets remove the interference of 11:30 for the B05R pads
already printed: the rails of covers 3S and 2S are as wide as B05R's grooves along the whole flank (1.20 mm at the cover, 1.771 mm
at the top; `rail_grow` -0.05, `rail_top_grow` -0.0143), and caps TW15H3S and TW15H2S have a 1.6 mm pin for B05R's 1.6 mm hole.
The walls and the caps' blocks keep I15 (0.15 mm per side). `FIT3S`, `FIT2S` in `scripts/dip_b_v4.py`; screws M2 x 8 on the
3 mm parts, M2 x 6 on the 2 mm parts.

    $RG ~/miniconda3/envs/femtip/bin/python -u scripts/dip_b_v4.py fit3s fit2s step3s step2s
    $RG env UR5M_REV=v2 ~/miniconda3/envs/femtip/bin/python -u scripts/v3b_ur5v2_pla_plate.py snug

Outputs in `v4/b/`: `cad/20261022-dip_link_v4b_cover_I15H3_3S.{stl,step}`, `_cover_I15H2_2S`, `_tip_cap_TW15H3S`, `_tip_cap_TW15H2S`,
`print/20261022-dip_link_v4b_p1s_pla_snug.3mf` (three of each part on the P1S, 1 h 12 min), checks in `20261022-dip_link_v4b_fit3s.json`
and `_fit2s.json`; Table 10 of the v4 page.

### Design as of 2026-10-23

The owner kept the 3T set ("in the end, i thinkt eh interference fits are quite important. we will have to ilve with the squish. so
we are still using the 3mm side wall settings", 10-23 19:19): cover 3T, cap TW15H3 and pad B05R with M2 x 8 screws. The 2T and snug
sets are not used; their files stay for the record.

## Opening the STEP files in Inventor

1. Unzip a `.step.zip` first; the plain `.step` files open as they are.
2. File > Open, file type STEP (`*.stp, *.ste, *.step`), choose the file, then Options.
3. Under Import options choose Convert Model: the exact parts arrive as editable solid bodies (the assembly STEP as an
   assembly of named parts). Reference Model keeps them as non-editable references.
4. The fused DIP link (`*_dip_link.step.zip`) contains the clevis arms as facets of the original STL; edit the exact new
   body or new features instead and keep the arms as they are. STL files open only as reference meshes.

The parts lie in the DIP link STL's frame: x_D = -x_F, y_D = y_F + 21.25, z_D = -z_F - 51.55 (PIP axis along y, finger
along +z, palmar face toward +x).
