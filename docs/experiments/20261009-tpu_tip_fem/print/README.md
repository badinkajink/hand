# SR2 fingertips to print (2026-10-12)

Written by `scripts/tpu_tip_print_plan.py stl`; the page that explains them is
`docs/experiments/20261009-tpu_tip_fem/20261009-printed_tpu_fingertip_finite_elements.html`.

Every tip is the whole block (17 x 14.8 x 21.4 mm, palmar edges rounded r 2.7 mm, end edge r 3.0 mm, back square),
printed with its back face on the bed and the palmar face up: 2 walls of 0.45 mm, 4 top and 4 bottom layers of
0.2 mm, gyroid infill, no supports.

| Print | Material | Infill | STL | Sheet | Pad modulus E* (MPa) |
|---|---|---|---|---|---|
| 95A, gyroid 20 % | TPU 95A | 20 % | 20261012-sr2_tip_d4p76x1p59_depth5.stl | 20261012-sr2_tip_w95_20_d4p76x1p59_depth5.json | 14.6 |
| 85A, gyroid 50 % | TPU 85A | 50 % | 20261012-sr2_tip_d4p76x1p59_depth5.stl | 20261012-sr2_tip_w85_50_d4p76x1p59_depth5.json | 16.5 |
| 85A, gyroid 30 % | TPU 85A | 30 % | 20261012-sr2_tip_d4p76x1p59_depth5.stl | 20261012-sr2_tip_w85_30_d4p76x1p59_depth5.json | 6.9 |
| 95A, gyroid 10 % | TPU 95A | 10 % | 20261012-sr2_tip_d4p76x1p59_depth5.stl | 20261012-sr2_tip_w95_10_d4p76x1p59_depth5.json | 7.2 |
| 95A, gyroid 20 %, no magnet | TPU 95A | 20 % | 20261012-sr2_tip_whole.stl | 20261012-sr2_tip_w95_20_nomagnet.json | 12.7 |

Magnet: N52 disc 4.76 x 1.59 mm, axially magnetized, north toward the palmar face, its top 5 mm under the contact point. Pocket 4.96 mm across and 1.69 mm deep. Pause before the layer at 12.2 mm (layer 61 at 0.2 mm layers), press the magnet in, fix it with a drop of cyanoacrylate, resume.

Other pocket depths (1.5 and 3 mm) and a 3.18 x 1.59 mm magnet are in the other STLs; their pause heights come from
`scripts/tpu_tip_design.py sheet --magnet <name> --depth <mm>`.

After printing: `scripts/tpu_tip_calibrate.py fit` turns an indentation and a pull test into E_s, E* and the friction
coefficient, and prints the command that writes the pad scene at those values.
