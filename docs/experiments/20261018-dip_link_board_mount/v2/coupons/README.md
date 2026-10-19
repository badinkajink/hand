# Fit coupons of DIP-link concept a, v2 (2026-10-19)

Superseded for printing by the runs of one plate per printer with printed IDs: concept a's in
`../../../20261019-dip_link_fit_prints/` and concept b's in `../../../20261019-dip_link_fit_prints_b/`
(`scripts/dip_fit_print_plates.py`). The projects below date from 13:08: their PLA is on the XL's tool 3, and their board
carrier stands on 5.4 mm2 under the bridge's curved face and keeps the PLA roof over the cable groove that v2 a lost
at 15:35. `scripts/dip_link_v2.py coupons` now fills that face down to the bed (199.5 mm3 under the carrier, 7.1 mm3
under each window gauge, the J1 slot and the cable groove left open).

The owner calibrates the magnetometer board's fit and the tip's snap fit before printing a whole DIP link (owner,
2026-10-19: "calibrate the magnetometer board fit and the snap fits, dont need to print the finger"). Every project is a
sliced PrusaSlicer 3mf for the Prusa XL 5T on the v17 template; its JSON beside it records the settings and the slice
checks. PLA on tool 3 at 0.1 mm layers; Siraya Tech 85A on tool 2 at 0.2 mm layers, 30 % gyroid. The board carrier and
the window gauges stand as the DIP link prints (the bridge's flat face on the bed, the window plate vertical).
`scripts/dip_link_v2.py coupons` writes them from `../../params/20261018-v2_concept_a.json` (the routed-cable design,
DIP link 9.75 mm longer).

| Order | Project | Tool | Time | Filament | What it holds |
|---|---|---|---|---|---|
| 1 | 20261018-dip_link_v2a_coupon_keys_detents_pla_t3.3mf | 3, PLA | 15 min | 1.0 g | three PLA keys of the DIP link, detents 0.15, 0.25 and 0.35 mm (1, 2, 3 notches on the flange) |
| 2 | 20261018-dip_link_v2a_coupon_channels_5_flanks_85a_t2.3mf | 2, 85A | 9 min | 1.6 g | five 85A channel blocks of the tip, 0, 0.05, 0.10, 0.15, 0.20 mm narrower than the key per flank (1 to 5 notches) |
| 3 | 20261018-dip_link_v2a_coupon_board_carrier_clamp_pins_pla_t3.3mf | 3, PLA | 1 h 4 min | 5.2 g | the DIP link's board region at the design (chip windows 0.20 mm, keys, shoulders, J1 slot, cable groove), the clamp plate, and four pins of 2.25, 2.30, 2.35, 2.40 mm |
| 4 | 20261018-dip_link_v2a_coupon_window_gauges_pla_t3.3mf | 3, PLA | 43 min | 2.0 g | the window plate alone at chip-window clearances 0.15, 0.20 and 0.30 mm |

## What to check

Snap fit (1 with 2): push each key into each channel block along the channel by hand. Note whether it slides on, whether
the detent clicks, whether it stays when pulled along the channel and when twisted. The channel interference to use is the
largest that still slides on by hand; the detent height is the smallest that clicks and holds.

Board fit (3 and 4): lay the ReSkin 5X board component side down on the carrier's window plate. All five chips should
drop into their windows with the board flat on the plate (no rocking), the chip tops 0.05 mm proud of the palmar face
(feel with a straightedge). J1 with a Qwiic plug mated sits in the slot without touching its walls, and the cable lies in
the groove, around the corner of the cut-off arm. Put on the clamp plate: its two pins pass the board's proximal holes into
the window plate; two M2 x 4 countersunk screws from the palmar side form their thread in the clamp plate's 1.8 mm holes;
two M2 x 5 screws through the ears into the shoulders' 1.8 mm holes. The pin coupon shows which pin diameter goes through
the board's 2.40 mm holes without play. The gauges show which window clearance takes the chips without force.

Write the results into `records.csv` (one row per pairing or check).
