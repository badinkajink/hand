# DIP link v2: Qwiic cable route and print files

Both v2 designs keep the ReSkin 5X board's own Qwiic connector J1 and lead the Qwiic cable from J1's plug in a groove open
to the back, through the slotted arm's lower dorsal corner and up the arm's outer face to an anchor 5.0 mm below the PIP
axis. Both DIP links are 9.75 mm longer than the final, so the PIP axis lies 46.91 mm from the tip's end (37.16 mm on the
current hand). The Qwiic cable is a flat ribbon about 4 mm wide and 1 mm thick; massaged into a bundle, it is modelled as
2.6 mm round in a 3.0 mm groove, and it clears the MCP link by 1.44 mm in a and by 1.24 mm in b over PIP -18 to 92 deg
(R7 asks 1.0 mm), and the joint keeps its whole range. J1's plug, JST's SHR-04V-S-B, is 7.0 mm wide at its protrusions;
the J1 slot is 7.8 mm wide. Until the fit prints of 2026-10-19 the model had a 5.0 mm plug (SHR-04V-S, so a 6.6 mm slot
that the plug did not enter) and a 2.0 mm bundle in a 2.4 mm groove (which the ribbon had to be squished into). PLA parts print on the Bambu printer from print-oriented STLs, the
85A tips and pads on the Prusa XL 5T from PrusaSlicer projects.

Terms. F is the finger frame of `../spec.json` (mm; x toward the dorsal side, y along the PIP axis, z up; the PIP axis
through (0, y, -51.55)). Clearance is the least signed distance from a part fixed to the DIP link to the MCP link and the
PIP servo over PIP -18 to 92 deg in 0.5 deg steps, in mm (`dip_link_geometry.points_clearance_exact`); R7 asks 0.5 mm for
rigid parts and 1.0 mm for the cable. Lengthening is how much longer the DIP link is between its bridge and the tip's mount
face than the final. Lead is the straight run of the cable out of the plug before its first turn. Free span is the cable's
length between the anchor and a tie on the MCP link's side plate at (0, 17.40, -33.0) in F at PIP 0.

## Cable route

In a the plug's wires leave at x 5.90 mm. Right at the plug's rear face the cable turns to +y over the plug (2.0 mm bend
radius; a 0.5 mm straight lead came first until 2026-10-19, which with the 2.6 mm bundle would leave about 0.98 mm), runs along
the DIP link's dorsal face to the slotted arm, turns up into the arm's outer face at y 17.40 and runs half sunk in a
groove to the anchor at (4.0, 17.40, -56.5). It is 27.2 mm long; its least clearance is 1.44 mm, to the MCP link at
PIP 15.5 deg. Between the J1 slot and the arm the bridge's arc lies only 0.07-1.4 mm above the groove (the wall check
found 0.24 mm), so there the groove is open to the top as well: the cable lies on a ledge of the bridge's dorsal corner
and a strip of Kapton tape keeps it there.

In b the plug sits in the neck's slot palmar of the board, so its wires leave at x 2.40 mm, 0.7 mm closer to the PIP axis
at the height of the first turn, where the end of the MCP link sweeps a cylinder of radius 12.70 mm about the axis. a's
turn right over the plug would clear the MCP link by 0.28 mm there. b's cable therefore turns toward the back at the plug's
rear face, with no lead and a 1.5 mm bend radius (the ribbon still lies flat there and bends flatwise; a 2.0 mm radius
would leave about 0.86 mm with the 2.6 mm bundle), then to +y at x 8.0 with a 2.0 mm radius, and from the slotted arm on
follows a's path to the same anchor. It is 32.9 mm long; its least clearance is 1.24 mm, to the MCP link at PIP 6.5 deg,
at the first turn. b's bridge top keeps
0.8 mm from the MCP link's sweep, so over the plug and the short leg toward the back (x up to 6.0 mm) it lies less than
0.8 mm above the groove; there the groove is open to the top as well, and a strip of Kapton tape over it keeps the first
turn down. Along the run at x 8.0 the bridge leaves 1.13 mm or more of PLA over the groove. The variants computed for b
with the 2.0 mm bundle (`b/20261019-dip_link_v2b_cable_routes.json`; the run's x does not change the clearance, 6.4 and
7.4 mm both give 1.16 mm), and the two routes of today's design with the 2.6 mm bundle:

| Route | Lengthening (mm) | Lead (mm) | Clearance (mm) | At PIP (deg) |
|---|---|---|---|---|
| a: turn to +y over a's plug | 9.75 | 0.5 | 1.28 | 60 |
| b: turn toward the back first | 9.75 | 0 | 1.16 | 6 |
| b: turn toward the back first | 9.75 | 0.5 | 0.68 | 9 |
| b: turn toward the back first | 10.0 | 0.5 | 0.92 | 16 |
| a's turn to +y at b's plug | 9.75 | 0.5 | 0.28 | -4.5 |
| a's turn to +y at b's plug | 10.25 | 0 | 1.27 | 14.5 |
| a's turn to +y at b's plug | 10.5 | 0.5 | 1.02 | 56 |
| a, 2.6 mm bundle: turn to +y at the plug | 9.75 | 0 | 1.44 | 15.5 |
| b, 2.6 mm bundle: 1.5 mm first turn toward the back | 9.75 | 0 | 1.24 | 6.5 |

Outside y 17.40 no part of the MCP link or the PIP servo reaches at any PIP angle, and inside it the cable lies in the DIP
link's groove, so the cable takes nothing from the range of motion. The free span from the anchor is 22.66 mm at PIP -18
deg and 24.91 mm at its longest (PIP 39 deg); tied 26 mm from the anchor the cable never pulls on the joint. The parameter
files contain the route as `cable_*` keys and the design choice `cable = idler_side`; `cable_run_x` (b only) sets where b's
cable turns to +y.

## Print files

Plates of 2026-10-19 evening: `../../20261019-v2_ur5_pla_plate/20261019-v2_ur5_pla_p1s.3mf` prints a's link and clamp plate, b's link and cover and the UR5e calibration mount on the P1S (PLA, AMS slot 1, 6 h 42 min, 166 g; each part's settings in the project as object settings, so Bambu Studio keeps them when it slices again; the links at 0.1 mm with tree supports and 40 % gyroid in place of the 100 % shoulder modifiers). `print/20261019-dip_link_v2_tip_a_pad_b_ell11_5_85a_t2_g30.3mf` prints a's 11/22 tip and b's 11/22 pad on the XL, tool 2, with the settings of the owner's good print of 10-18 and three magnet pauses (7.3, 12.2, 17.1 mm; 55 min). `print/20261019-dip_link_v2_fit_sizes_tips_a_pads_b_85a_t2_g30.3mf` prints the fit sizes on the XL without magnets (3 h 43 min, 21.2 g, no pauses): a's 11/22 tip with its channels at 0.00, 0.05, 0.10 and 0.20 mm per flank (A00 to A20) and b's 11/22 pad with its rail grooves at 0.00, 0.05, 0.10 and 0.15 mm per flank (B00 to B15), each ID engraved in the mount face (`scripts/v2_tip_fit_sizes_xl.py`). The size that fits on the printed link or cover prints again with magnets; the design has 0.05 mm (fit_flank, groove_fit).

PLA, Bambu printer (Bambu Studio): STLs in print orientation, lying on z = 0, each one closed body. Their settings are in
`a/print/20261018-dip_link_v2a_pla_bambu.json` and `b/print/20261018-dip_link_v2b_pla_bambu.json`; for a's link that file
also lists the support regions and the two 100 % infill regions over the shoulders in the STL's own coordinates.

| File | Orientation | Layer (mm) | Walls | Infill | Supports, brim |
|---|---|---|---|---|---|
| `a/print/20261018-dip_link_v2a_dip_link_pla_bambu.stl` | tip end up, arms' ends on the plate | 0.1 | 3 | 15 % gyroid, 100 % over the shoulders | tree, under both arms' ends and the bridge's face toward the PIP axis; 5 mm brim |
| `a/print/20261018-dip_link_v2a_clamp_plate_pla_bambu.stl` | on its back face | 0.1 | 4 | 100 % | none |
| `a/print/20261018-dip_link_v2a_pin_fit_coupon_pla_bambu.stl` | pins up | 0.1 | 4 | 100 % | none |
| `b/print/20261018-dip_link_v2b_dip_link_pla_bambu.stl` | tip end up, arms' ends on the plate | 0.1 | 3 | 15 % gyroid | tree, under both arms' ends and the bridge's face toward the PIP axis; 5 mm brim |
| `b/print/20261018-dip_link_v2b_cover_pla_bambu.stl` | on its back face, rails up | 0.1 | 4 | 100 % | none |
| `b/print/20261018-dip_link_v2b_fit_coupon_rails_pla_bambu.stl` | on its back face, rails up | 0.1 | 3 | 100 % | none |

85A, Prusa XL 5T on tool 2 (0.2 mm layers, 30 % gyroid, M601 pauses at the 2.9 mm magnet seats): at each pause press the
magnets to the pocket floor, north pole toward the palmar face, and resume.

- `a/print/20261018-dip_link_v2a_print_tip_ell11_5_channels_85a_t2_g30.3mf` (the 11/22 tip with a's two key channels;
  pauses 7.3, 12.2, 17.1 mm) and `a/print/20261018-dip_link_v2a_fit_coupon_channels_85a_t2.3mf` (channel blocks).
- `b/print/20261018-dip_link_v2b_print_pad_ell11_5_85a_t2_g30.3mf` (the 11/22 pad; pauses 7.3, 12.2, 17.1 mm) and
  `b/print/20261018-dip_link_v2b_fit_coupon_grooves_85a_t2.3mf` (groove blocks at 0, 0.05, 0.10 mm per flank).
- The five-magnet block versions: `a/print/*_tip_block5_channels_85a_t2_g30.3mf` (pause 12.4 mm) and
  `b/print/*_pad_block5_85a_t2_g30.3mf` (printed on its cut back face, pause 8.8 mm).

The fit tests print as runs of one plate per printer, every piece with its ID printed on it: concept a's in
`../../20261019-dip_link_fit_prints/` and concept b's in `../../20261019-dip_link_fit_prints_b/` (print sheet,
projects, STLs and records.csv in each; `scripts/dip_fit_print_plates.py [--run 20261019b] pieces bambu xl`). The
earlier coupons of a, as XL projects, are in `coupons/`.

Hardware: a takes two M2 x 4 countersunk and two M2 x 5 cheese-head screws, b four M2 x 5 cheese-head screws, all brass
(R10: no ferromagnetic fastener within 15 mm of a chip); the board as supplied (J1 kept, tab not cut); a Qwiic cable; a
drop of CA glue or a strip of Kapton tape at the anchor; a cable tie on the MCP link's side plate 26 mm from the anchor.
The assembly sequences and parts lists are in `a/evaluation.json` and `b/evaluation.json` (`assembly`, `parts_list`).

## STEP files for Inventor

`scripts/dip_link_params.py export` wrote both designs with the 7.8 mm J1 slot, the 3.0 mm cable groove and 1.9 mm M2 holes,
every STEP file read back valid in FreeCAD (the exports of 15:08 and 15:16, `../exports/20261019-1508-v2_concept_b_cable/` and
`../exports/20261019-1516-v2_concept_a_cable/`, have the 6.6 mm slot and the 2.4 mm groove):

- `../exports/20261019-1851-v2_concept_a_m2hole1p9/`: the DIP link's new features and the clamp plate as exact
  solids, the fused link (zipped), the tip cutter, the cable-groove cutter, the five-magnet block tip, and
  `20261019-1851-v2_concept_a_m2hole1p9_assembly_D.step.zip` with the board, J1 on its tab, the 7.0 mm plug, the cable
  as a 2.6 mm pipe to its anchor, the four screws and the 11/22 tip in the DIP link STL's coordinates.
- `../exports/20261019-1853-v2_concept_b_m2hole1p9/`: the new body and the cover as exact solids, the fused link
  (zipped), the cable-groove cutter, both pads (faceted), and the assembly with the board, J1, the plug, the cable, the four
  screws and the 11/22 pad.

Each folder's `print/` has the PLA parts as Bambu STLs with their settings and the tips or pads sliced for the XL;
`export.json` lists every file and its read-back. Opening them in Inventor: `../README.md`, section Opening the STEP files.

## Concept c with J1 kept

Concept c (`../concept_c/`) places the board into a PLA carrier at an M601 pause and prints the PLA cap and the 85A tip
over it in the same job. With J1 kept at c's board placement (component face x 7.65 mm, parts toward the tip), J1 and its
plug clear the MCP link and the PIP servo by 0.5 mm from a lengthening of 8.5 mm (0.67 mm there), and a's cable route
from c's plug (wires at x 6.25 mm) clears the MCP link by 0.30, 0.75, 1.20 and 1.42 mm at 8.5, 9.0, 9.5 and 9.75 mm. c
would need 9.5 mm, or 9.75 mm for one finger length with a and b (`20261019-v2_concept_c_j1.json`).

The print is the obstacle. c's print plan relies on nothing standing above the last layer before the pause (z 4.60 mm in
the cartridge's print frame), so the cap and the tip print with no Z-hop or travel rule. J1 is 2.9 mm tall: at the pause
its top stands at z 6.5 mm, 1.9 mm above that layer, and its body begins 2.3 mm from the tip's mount face, past which
tool 1 prints the cap and tool 2 the 85A tip. The carrier's tab region would also have to end before J1. c with J1 kept is
not built; the choice waits on the owner (a test print of the pause with a dummy board on the XL would decide it).

## Regenerate

From the parameter files `../params/20261018-v2_concept_a.json` and `../params/20261018-v2_concept_b.json`
(`unset VIRTUAL_ENV` first: freecadcmd otherwise loads the .venv's numpy):

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 400 -- taskset -c 8-11 env NUMPY_MADVISE_HUGEPAGE=0"
    E=~/miniconda3/envs/femtip/bin/python
    $RG $E scripts/dip_link_v2.py a build step check sense print docs
    $RG .venv/bin/python scripts/dip_link_v2.py a render
    $RG $E scripts/dip_link_v2.py b build step check sense render print docs
    $E scripts/dip_link_v2.py summary
