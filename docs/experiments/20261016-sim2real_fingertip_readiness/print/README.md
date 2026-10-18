# Siraya Tech TPU 85A fingertip prints for 3 x 2 mm magnets (2026-10-16)

Every file here is an 85A print: PrusaSlicer projects on the owner's v17 template (Original Prusa XL 5T Input Shaper, 0.4 mm nozzles) with Siraya Tech's Prusa XL 0.4 mm TPU 85A filament profile in slot 2 and the part on extruder 2. The 95A projects of the same geometry in docs/experiments/20261016-metric_magnet_prints/print/ use Generic FLEX at 230 C and are not 85A prints. `scripts/sim2real_fingertip_prints.py` (template, build, coupon, readme) writes them; each project has a sheet (JSON) with the geometry and the G-code checks.

## Seat test first

The owner's seat test of 2026-10-18 chose 2.9 mm ('the 2.9mm holes are what we should use'): print the seat2p9 projects, listed first in the table of tips. Two tips change at that seat to keep the 1.8 mm pocket-wall rule: tcd_ell7.4x8.5x11x22, magnets 7.5 mm deep (1.76 mm to the surface at 7.0 mm); five_z7p4, corners 7.4 mm apart along the finger (1.71 mm web at 7.0 mm). The 2.7 mm projects below them are the provisional set printed before the test. `scripts/sim2real_fingertip_prints.py build --seat 2.9 --depth tcd_ell7.4x8.5x11x22=7.5 --five five_z7p4` rewrites the 2.9 mm set.

| Coupon | Walls at each seat | Print time | Project |
|---|---|---|---|
| Seats 2.7, 2.8, 2.9 mm, round and slot | 2 | 8m 37s | 20261016-siraya85a_seat_coupon_d3x2_seats2p7_2p8_2p9_g30.3mf |
| Seats 2.7, 2.8, 2.9 mm, round and slot | 1 | 8m 18s | 20261016-siraya85a_seat_coupon_d3x2_seats2p7_2p8_2p9_pocketwalls1_g30.3mf |

## Profile

Starting point: "Siraya Tech TPU 85A @XL 0.4mm Nozzle" and ST TPU 85A 0.20mm QUALITY @XL 0.4 from Siraya Tech TPU 85A XL 0.4mm 20260610.3mf (Siraya Tech, for the single-tool Original Prusa XL). Taken over in slot 2: nozzle 240 C (first layer 240 C), bed 35 C, extrusion multiplier 0.96, maximum volumetric speed 12 mm3/s, retraction 0.6 mm, fan 100 %. Taken over from Siraya's print profile: perimeters at 60 mm/s, external perimeters 40 mm/s, infill 60 mm/s, top layers 40 mm/s, bridges 25 mm/s, a 3-loop skirt. Kept from the template, with the reason in template_changes.json: the 5T printer and tool 2, two 0.45 mm perimeters and four top and four bottom layers (the finite-element recipe and the pocket-wall rule), thin bridges (a thick bridge's 0.4 mm strand would hang onto the magnet under the 0.1-0.2 mm air gap), and the Arachne perimeter generator. Infill is gyroid at 30 % or 50 %, the stiffness choice; the 2026-10-12 models put 85A at 30 % near half the pad modulus of 50 % (6.9 against 16.5 MPa with a 4.76 mm magnet), with a solid modulus of 12 MPa taken from NinjaFlex 85A, so Siraya's printed stiffness is still to be measured.

## Tips

| Print | Infill | Orientation and layers | Magnets, top under the apex | Least TPU around a pocket (mm) | Pauses | Print time | Project |
|---|---|---|---|---|---|---|---|
| Block | 30 % | back face on the bed, 0.2 mm | 4, 5 mm | 2.95 | 12.4 mm (4) | 17m 3s | 20261016-siraya85a_block_four_d3x2_depth5_seat2p9_g30.3mf |
| Block | 50 % | back face on the bed, 0.2 mm | 4, 5 mm | 2.95 | 12.4 mm (4) | 21m 54s | 20261016-siraya85a_block_four_d3x2_depth5_seat2p9_g50.3mf |
| Block | 30 % | back face on the bed, 0.2 mm | 5, 5 mm | 2.95 | 12.4 mm (5) | 17m 27s | 20261016-siraya85a_block_five_z7p4_d3x2_depth5_seat2p9_g30.3mf |
| Block | 50 % | back face on the bed, 0.2 mm | 5, 5 mm | 2.95 | 12.4 mm (5) | 22m 14s | 20261016-siraya85a_block_five_z7p4_d3x2_depth5_seat2p9_g50.3mf |
| Block, one wall at the pockets | 30 % | back face on the bed, 0.2 mm | 4, 5 mm | 2.95 | 12.4 mm (4) | 16m 47s | 20261016-siraya85a_block_four_d3x2_depth5_seat2p9_pocketwalls1_g30.3mf |
| Block, one wall at the pockets | 50 % | back face on the bed, 0.2 mm | 4, 5 mm | 2.95 | 12.4 mm (4) | 21m 42s | 20261016-siraya85a_block_four_d3x2_depth5_seat2p9_pocketwalls1_g50.3mf |
| Capsule r 8.5 mm | 30 % | upright on the mount face, 0.1 mm | 4, 5 mm | 2.01 | 9.2 mm (2); 16.2 mm (2) | 33m 36s | 20261016-siraya85a_tcd_cap8p5_four_d3x2_depth5_seat2p9_g30.3mf |
| Capsule r 8.5 mm | 50 % | upright on the mount face, 0.1 mm | 4, 5 mm | 2.01 | 9.2 mm (2); 16.2 mm (2) | 39m 37s | 20261016-siraya85a_tcd_cap8p5_four_d3x2_depth5_seat2p9_g50.3mf |
| Half-ellipsoid 7.4 x 8.5 x 16/32 mm | 30 % | upright on the mount face, 0.1 mm | 4, 5.5 mm | 1.86 | 9.2 mm (2); 16.2 mm (2) | 33m 33s | 20261016-siraya85a_tcd_ell7p4x8p5x16x32_four_d3x2_depth5p5_seat2p9_g30.3mf |
| Half-ellipsoid 7.4 x 8.5 x 16/32 mm | 50 % | upright on the mount face, 0.1 mm | 4, 5.5 mm | 1.86 | 9.2 mm (2); 16.2 mm (2) | 38m 41s | 20261016-siraya85a_tcd_ell7p4x8p5x16x32_four_d3x2_depth5p5_seat2p9_g50.3mf |
| Half-ellipsoid 7.4 x 8.5 x 11/22 mm | 30 % | upright on the mount face, 0.1 mm | 4, 7.5 mm | 1.85 | 9.2 mm (2); 16.2 mm (2) | 33m 2s | 20261016-siraya85a_tcd_ell7p4x8p5x11x22_four_d3x2_depth7p5_seat2p9_g30.3mf |
| Half-ellipsoid 7.4 x 8.5 x 11/22 mm | 50 % | upright on the mount face, 0.1 mm | 4, 7.5 mm | 1.85 | 9.2 mm (2); 16.2 mm (2) | 37m 32s | 20261016-siraya85a_tcd_ell7p4x8p5x11x22_four_d3x2_depth7p5_seat2p9_g50.3mf |
| Block | 30 % | back face on the bed, 0.2 mm | 4, 5 mm | 3.05 | 12.4 mm (4) | 16m 58s | 20261016-siraya85a_block_four_d3x2_depth5_seat2p7_g30.3mf |
| Block | 50 % | back face on the bed, 0.2 mm | 4, 5 mm | 3.05 | 12.4 mm (4) | 21m 53s | 20261016-siraya85a_block_four_d3x2_depth5_seat2p7_g50.3mf |
| Block, five magnets | 30 % | back face on the bed, 0.2 mm | 5, 5 mm | 3.05 | 12.4 mm (5) | 17m 20s | 20261016-siraya85a_block_five_d3x2_depth5_seat2p7_g30.3mf |
| Block, five magnets | 50 % | back face on the bed, 0.2 mm | 5, 5 mm | 3.05 | 12.4 mm (5) | 22m 9s | 20261016-siraya85a_block_five_d3x2_depth5_seat2p7_g50.3mf |
| Block, one wall at the pockets | 30 % | back face on the bed, 0.2 mm | 4, 5 mm | 3.05 | 12.4 mm (4) | 16m 42s | 20261016-siraya85a_block_four_d3x2_depth5_seat2p7_pocketwalls1_g30.3mf |
| Block, one wall at the pockets | 50 % | back face on the bed, 0.2 mm | 4, 5 mm | 3.05 | 12.4 mm (4) | 21m 40s | 20261016-siraya85a_block_four_d3x2_depth5_seat2p7_pocketwalls1_g50.3mf |
| Capsule r 8.5 mm | 30 % | upright on the mount face, 0.1 mm | 4, 5 mm | 2.03 | 9.3 mm (2); 16.3 mm (2) | 33m 39s | 20261016-siraya85a_tcd_cap8p5_four_d3x2_depth5_seat2p7_g30.3mf |
| Capsule r 8.5 mm | 50 % | upright on the mount face, 0.1 mm | 4, 5 mm | 2.03 | 9.3 mm (2); 16.3 mm (2) | 39m 37s | 20261016-siraya85a_tcd_cap8p5_four_d3x2_depth5_seat2p7_g50.3mf |
| Half-ellipsoid 7.4 x 8.5 x 16/32 mm | 30 % | upright on the mount face, 0.1 mm | 4, 5.5 mm | 1.94 | 9.3 mm (2); 16.3 mm (2) | 33m 34s | 20261016-siraya85a_tcd_ell7p4x8p5x16x32_four_d3x2_depth5p5_seat2p7_g30.3mf |
| Half-ellipsoid 7.4 x 8.5 x 16/32 mm | 50 % | upright on the mount face, 0.1 mm | 4, 5.5 mm | 1.94 | 9.3 mm (2); 16.3 mm (2) | 38m 45s | 20261016-siraya85a_tcd_ell7p4x8p5x16x32_four_d3x2_depth5p5_seat2p7_g50.3mf |
| Half-ellipsoid 7.4 x 8.5 x 11/22 mm | 30 % | upright on the mount face, 0.1 mm | 4, 7 mm | 1.81 | 9.3 mm (2); 16.3 mm (2) | 32m 56s | 20261016-siraya85a_tcd_ell7p4x8p5x11x22_four_d3x2_depth7_seat2p7_g30.3mf |
| Half-ellipsoid 7.4 x 8.5 x 11/22 mm | 50 % | upright on the mount face, 0.1 mm | 4, 7 mm | 1.81 | 9.3 mm (2); 16.3 mm (2) | 37m 33s | 20261016-siraya85a_tcd_ell7p4x8p5x11x22_four_d3x2_depth7_seat2p7_g50.3mf |

## Plate of the candidate tips

20261018-siraya85a_plate_block_cap8p5_ell16x32_ell11x22_seat2p9_g30.3mf prints block, capsule, ellipsoid 16/32, ellipsoid 11/22 (30 % gyroid, 2.9 mm seats) on one bed, each from its tip's STL with its tip's recipe and layer height (0.2 mm: block; 0.1 mm: capsule, ellipsoid 16/32, ellipsoid 11/22); front row, left to right: block, capsule; back row, left to right: ellipsoid 16/32, ellipsoid 11/22 (figure 20261018-siraya85a_plate_block_cap8p5_ell16x32_ell11x22_seat2p9_g30_bed.png). The recipe is written on each object and, where the tips agree, in the print config: PrusaSlicer's GUI save of the 11/22 project on 2026-10-18 dropped the per-volume recipe of the tip projects and kept the object's layer height. The plate stops 3 times for its 16 magnets, where the tips' own projects stop 7 times; the wipe tower is off, since PrusaSlicer refuses one over objects of unequal layer heights and one tool prints every tip. At each pause only the tips named in the printer's message take magnets. Print time 1h 40m 54s, filament 12.3 g. The G-code labels each tip as an object (M486), so a failed tip can be cancelled at the printer without stopping the others.

| Pause | Height (mm) | Time from start (min) | Magnets | Tips |
|---|---|---|---|---|
| 1 | 9.2 | 45.9 | 6 | capsule 2 proximal, ellipsoid 16/32 2 proximal, ellipsoid 11/22 2 proximal |
| 2 | 12.4 | 61.9 | 4 | block 4 |
| 3 | 16.2 | 82.9 | 6 | capsule 2 distal, ellipsoid 16/32 2 distal, ellipsoid 11/22 2 distal |

Checks of the plate's G-code: only T1 selected; 3 pauses (M601) at 9.2, 12.4, 16.2 mm; every pocket of every tip empty on the layers it spans (largest share under a bead 0.000), the layer under it covering 100 % or more of it, and bridged right after the pause assigned to it (97-100 %); each tip's extent in the G-code within 0.23 mm of its place on the bed.

## Checks of the sliced G-code

PrusaSlicer 2.9.2 sliced every project (G-code in logs/20261016-sim2real_fingertip_readiness/print/, untracked). In every file the only tool selected is T1 (extruder 2), the part prints at 240 C on that tool and the bed at 35 C, and the configuration block lists Siraya's filament in slot 2 with extrusion multiplier 0.96. The largest volumetric flow of any extruding move on a part is 7.66 mm3/s against the profile's 12. Each pocket is empty on every layer it spans (largest share of a pocket's section under a bead: 0.000), the layer under it covers its whole footprint, and the first layer over it is printed right after its pause and covers 96-100 % of the footprint. Islands per layer: at most 1. At mid-pocket height every pocket has 2 perimeter loops in the two-wall prints and 1 in the one-wall prints, and the outer surface 2. Filament on the part: 2.9-4.3 g per tip (skirt included).

## Pauses

The printer stops (M601) before the first layer over a pocket. Press each magnet to its floor with a flat, non-magnetic tool, north pole toward the palmar face, and resume. The block takes all its magnets at one pause; the curved tips take the two proximal magnets at the first pause and the two distal ones at the second. Check each pole with a marked magnet: a flipped magnet reverses its field at the chips.

## Not yet measured

Siraya 85A's printed modulus and friction, the seat fit and retention in 85A, and the tips' indentation at 0.5, 1 and 3 N. The study page (docs/experiments/20261016-sim2real_fingertip_readiness/) gives the model values and the bench procedure.
