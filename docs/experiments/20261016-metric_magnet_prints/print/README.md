# Fingertips for 3 x 2 mm magnets (2026-10-16)

Four fingertips with four magnets each are ready to print: the block of the 2026-10-13 prints and the three curved finalists of 2026-10-15, each with four press-fit pockets for the owner's 3 x 2 mm magnets. A coupon of 12 min tests the 2.7 mm seat against 2.8 and 2.9 mm before the tips. Every print has a PrusaSlicer project (3MF on the owner's v17 template, TPU on tool 2, pauses included), its STL and a sheet (JSON) with the geometry and G-code checks, written by `scripts/metric_magnet_prints.py build` and `coupon`.

Recipe, as in the earlier prints: TPU 95A (the template's Generic FLEX @XL at 230 C), gyroid infill 20 %, two perimeters of 0.45 mm, four top and four bottom solid layers, no supports. The block prints on its back face in 0.2 mm layers. The curved tips print upright on their mount face in 0.1 mm layers after a 0.2 mm first layer, as on 2026-10-15; PrusaSlicer raises their solid layers to the template's minimum shells, 0.7 mm on top and 0.5 mm at the bottom.

Magnets: AplysiaTech N52 disc 3 x 2 mm, axially magnetized, nickel coated, +-0.1 mm (amazon.com/dp/B0G52HW8SY). Four per tip on the grid of the readout model, 6 mm across and 7 mm along the finger and centred on the contact point, each axis along the palmar normal, every north pole toward the palmar face. Seat: 2.7 mm, 90 % of the diameter, the owner's press fit. Magnet depth: from the highest point of the palmar surface, over the contact point, to the magnets' top faces.

| Print | Orientation and layers | Magnet top under the apex (mm) | Least TPU around a pocket (mm) | Pauses | Print time | Files |
|---|---|---|---|---|---|---|
| Block | back face on the bed, 0.2 mm | 5 | 3.05 | 12.4 mm (layer 62, 17 min in), 4 magnets | 22m 35s | 20261016-block_four_d3x2_depth5_seat2p7_w95_20.3mf, 20261016-block_four_d3x2_depth5_seat2p7.stl, 20261016-block_four_d3x2_depth5_seat2p7_w95_20.json |
| Half-ellipsoid 7.4 x 8.5 x 16/32 mm | upright on the mount face, 0.1 mm | 5.5 | 1.94 | 9.3 mm (layer 92, 16 min in), 2 magnets; 16.3 mm (layer 162, 28 min in), 2 magnets | 37m 18s | 20261016-tcd_ell7p4x8p5x16x32_four_d3x2_depth5p5_seat2p7_w95_20.3mf, 20261016-tcd_ell7p4x8p5x16x32_four_d3x2_depth5p5_seat2p7.stl, 20261016-tcd_ell7p4x8p5x16x32_four_d3x2_depth5p5_seat2p7_w95_20.json |
| Capsule r 8.5 mm | upright on the mount face, 0.1 mm | 5 | 2.03 | 9.3 mm (layer 92, 17 min in), 2 magnets; 16.3 mm (layer 162, 29 min in), 2 magnets | 37m 7s | 20261016-tcd_cap8p5_four_d3x2_depth5_seat2p7_w95_20.3mf, 20261016-tcd_cap8p5_four_d3x2_depth5_seat2p7.stl, 20261016-tcd_cap8p5_four_d3x2_depth5_seat2p7_w95_20.json |
| Half-ellipsoid 7.4 x 8.5 x 11/22 mm | upright on the mount face, 0.1 mm | 7 | 1.81 | 9.3 mm (layer 92, 17 min in), 2 magnets; 16.3 mm (layer 162, 29 min in), 2 magnets | 36m 47s | 20261016-tcd_ell7p4x8p5x11x22_four_d3x2_depth7_seat2p7_w95_20.3mf, 20261016-tcd_ell7p4x8p5x11x22_four_d3x2_depth7_seat2p7.stl, 20261016-tcd_ell7p4x8p5x11x22_four_d3x2_depth7_seat2p7_w95_20.json |

## Pauses

The printer stops (M601) before the first layer over a pocket. Press each magnet to the pocket's floor with a flat, non-magnetic tool, north pole toward the palmar face, and resume. On the block all four magnets go in at one pause, lying flat, north pole up. On the curved tips the magnets stand on their edge in slots: the two proximal magnets go in at the first pause and the two distal ones at the second, north pole toward the curved palmar side. The layer after each pause bridges 0.20 mm (block) or 0.19 mm (curved tips) of air over the magnets, so the nozzle lays no plastic on them.

## Magnet depth and TPU around the pockets

Least TPU around a pocket: the smallest distance from the complete pocket (for a slot, its part above the seated magnet included) to the outer surface or to another pocket, from 400,000 samples of the outer surface and 20,000 of each pocket. Each tip uses the shallowest magnet depth, in 0.5 mm steps from 5 mm, that leaves 1.8 mm, four 0.45 mm perimeters, around every pocket. The block and the capsule keep 5 mm. On the half-ellipsoids the top corners of the distal slots approach the distal surface: at 5 mm the 16/32 tip leaves 1.72 mm, and the 11/22 tip leaves 1.13, 1.35, 1.54, 1.69 mm at 5, 5.5, 6, 6.5 mm, so they use 5.5 and 7 mm.

A deeper magnet lies closer to the magnetometer. With the five-chip board against the back face, the four magnets put at most 5.5, 6.1, 5.5, 8.0 mT on one axis of a chip (block, 16/32, capsule, 11/22). The MLX90393 reads that at gain setting GAIN_SEL 6, 6, 6, 4; its datasheet guarantees 4.8 mT at GAIN_SEL 7, and the range grows with the step size at lower settings. Field: point dipoles, N52 remanence 1.45 T.

## Checks of the sliced G-code

Every part is watertight with one outer shell and four void shells. PrusaSlicer 2.9.2 sliced each project with its own settings (G-code in logs/20261016-metric_magnet_prints/print/, untracked). In the G-code each pocket is empty on every layer it spans (no bead inside its section shrunk by 0.1 mm), the layer under it covers its footprint, and the first layer over it is bridge infill over 97-100 % of the footprint, printed right after the pause. Each layer is one island, and 87-93 % of the beads rest on the layer below. The template fits arcs (G2/G3); `scripts/gcode_audit.py` reads only straight moves, so the builder reads the arcs itself.

## Seat coupon

20261016-pocket_fit_coupon_d3x2_seats2p7_2p8_2p9_w95_20.3mf (20261016-pocket_fit_coupon_d3x2_seats2p7_2p8_2p9.stl, sheet 20261016-pocket_fit_coupon_d3x2_seats2p7_2p8_2p9_w95_20.json): 22 x 16 x 5 mm, printed flat with the same recipe in 0.2 mm layers, no pause, 12 min. One row has round pockets 2.2 mm deep, the block's seat with the magnet lying flat; the other has slots 2.1 mm across, the curved tips' seat with the magnet on its edge. Seats are 2.7, 2.8 and 2.9 mm wide from the chamfered end. Press a magnet into each seat with a flat, non-magnetic tool. The seat to use is the narrowest that takes the magnet to its floor without splitting the wall and keeps it in when the coupon is turned over and tapped. If a wider seat does better, `scripts/metric_magnet_prints.py build --seat 2.8` (or 2.9) writes the tips with it; the seat is in the file names.

## Print risks

The curved tips need someone at the printer for two pauses, about 17 and 29 minutes into the print. Magnets 6 mm apart pull toward each other: seat each one before picking up the next, and check each pole with a marked magnet, because a flipped magnet reverses its field at the chips. The slots are 2.1 mm across the magnet's thickness, 0.1 mm over the nominal 2 mm, and the magnets' tolerance is 0.1 mm: a thick magnet may stick in a slot printed undersize, which the coupon's slots show before a tip is printed.

The 2026-10-13 four-pocket block for 4.76 x 1.59 mm magnets and the one-magnet prints are in docs/experiments/20261009-tpu_tip_fem/print/; the curved single-magnet prints are in docs/experiments/20261015-curved_tips/print/.

## Five magnets

20261016-block_five_d3x2_depth5_seat2p7_w95_20.3mf (20261016-block_five_d3x2_depth5_seat2p7.stl, sheet 20261016-block_five_d3x2_depth5_seat2p7_w95_20.json) is the block with a fifth magnet at the contact point. Magnet centres (y across the finger, z from the mount face, mm): (-3, 7.36); (3, 7.36); (0, 10.86); (-3, 14.36); (3, 14.36), every top 5 mm under the palmar face. The board's chips lie under U1 = (0, 10.86) and 7 mm from it on both axes; magnets 7 mm across would reach past the 7.4 mm half-width, so only the centre magnet sits over a chip. The pockets keep 1.91 mm between the centre and the corner pockets (the 1.8 mm rule) and 3.05 mm to the surface; all five go in at the one pause at 12.4 mm, and the print takes 23m 8s. The five magnets put 7.7 mT on a chip, which needs GAIN_SEL 5.

Finite elements of the block (`scripts/metric_magnet_prints.py fe`, the model and grid of the 2026-10-13 layout study, whose four-magnet row they reproduce: 16.70 MPa) and the board's Cramer-Rao bound (`sense`; five chips against the back face, one 141 Hz sample at the noise of GAIN_SEL 7, contact position and the Earth's field unknown) compare the two:

| Magnets | Pad modulus E* (MPa) | Face compliance lost at the contact point (%) | Force noise at the contact point (N) | Force noise over 15 points, median and range (N) | Field on a chip (mT) |
|---|---|---|---|---|---|
| Four | 16.7 | 7 | 0.329 | 0.34 (0.31-0.36) | 5.5 |
| Five | 17.9 | 12 | 0.097 | 0.17 (0.09-0.28) | 7.7 |

E*: the sphere pads' modulus that reproduces the tip's approach under the screwdriver at 0.5, 1 and 3 N; face compliance lost: the drop in the palmar face's deflection under a 1 N patch (Gaussian, sigma 0.6 mm) against the tip without magnets. The bound at the contact point is 3.4 times lower with five magnets: the four corner magnets lie 4.6 mm from the contact point, the fifth under it. Below GAIN_SEL 7 the datasheet gives no noise, so the five-magnet bound, taken at GAIN_SEL 7's noise for a field that needs GAIN_SEL 5, is a lower bound.

On the upright curved tips the same layout fails the rule: on the capsule r 8.5 mm each slot, open upward to its own pause, comes within 0.99 mm of the next seat up, and the five magnets need three pauses (checks/20261016-tcd_cap8p5_five_d3x2_depth5_seat2p7.json). No curved five-magnet print is written.

## One wall at the pockets

20261016-block_four_d3x2_depth5_seat2p7_pocketwalls1_w95_20.3mf (sheet 20261016-block_four_d3x2_depth5_seat2p7_pocketwalls1_w95_20.json; part 20261016-block_four_d3x2_depth5_seat2p7.stl, modifier 20261016-block_four_d3x2_depth5_seat2p7_pocketwalls1_modifier.stl) is the four-magnet block with one perimeter around each pocket and the two of the recipe at the outer surface. PrusaSlicer has no setting for the walls of a hole alone, so the project adds a modifier volume with one perimeter: on each of the 11 layers that cross the pockets (10-12.2 mm), the block's section inset by 0.407 mm, the centre-to-centre spacing of the recipe's 0.45 mm perimeters at 0.2 mm layers. The band outside the modifier keeps the recipe's two perimeters and has room for one bead; the modifier adds its own outer loop beside it. In the G-code at 11 mm every pocket has one loop and the outer surface has 2 loops, their centrelines 0.20 and 0.63 mm in from the surface against the two-wall print's 0.23 and 0.63 mm. The pause, the bridges and the floors check as in the two-wall print; the one-wall print takes 22m 2s against 22m 35s.

In the finite elements one 0.45 mm wall at the pocket lowers the pad modulus from 16.70 to 16.40 MPa (-1.8 %), the face compliance lost at the contact point from 7.0 to 5.7 %, and the force noise at the contact point from 0.329 to 0.296 N, as each magnet moves 3 % farther under 1 N. The press fit then rests on one bead backed by 20 % gyroid instead of two; the bridge 0.2 mm over the magnet closes the pocket after the pause, so the fit has to keep the magnet in place only until that layer.

20261016-pocket_fit_coupon_d3x2_seats2p7_2p8_2p9_pocketwalls1_w95_20.3mf (sheet 20261016-pocket_fit_coupon_d3x2_seats2p7_2p8_2p9_pocketwalls1_w95_20.json, modifier 20261016-pocket_fit_coupon_d3x2_seats2p7_2p8_2p9_pocketwalls1_modifier.stl) is the seat coupon with the same modifier, 11 min: at 4 mm every seat has one loop and the outer surface 2. Pressing magnets into it next to the two-wall coupon shows whether one wall keeps them in.
