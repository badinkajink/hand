# Seat block for the inverted calibration rig: print, assembly and first session

The seat block is the only thing on the table for the inverted rig. It holds three fingertips pad up for the probe on
the HEX-E, with plate v2e's seats at v2e's place, and two homing keys between its columns. The owner asked for it on
2026-10-30 ("a smaller ... print out block that screws into the same 2x4 ... can you space it out farther from the
fingertip holders so a drill can more easily drive in?"; then "add the homing indexers to the sensor blocks. i want to
replace everything on the table with just three sensor mounts. i also would like to have all the indenters + the plate on
one bambu p1s plate"). The same day the tools' fit on the probe base changed ("can we actually create a fastening point on
the homing block for different size indenters?"), and a sixth tool came to carry a fingertip face out ("a fingertip mount
for this new indenter that doesnt use the dip link, it just uses the cassette ... the crosstalk should be way easier to
measure with the palmar face in line with the z axis"). The rig's page,
`../20261029-inverted_calibration_rig/20261029-inverted_rig.html`, section 8, gives the design, the homing check and the
replays; the rig's runbook, `../20261029-inverted_calibration_rig/README.md`, gives the probe, the tools and the program
changes. This file is the seat block's runbook.

## Files

| what | where |
|---|---|
| CAD in the rig's frames (M for the probe, A for the block) | `cad/20261030-seat_block_*.stl`, `cad/20261030-seat_block_parts.json` |
| print-oriented STLs (on z = 0, centred) | `print/20261030-seat_block_*_print.stl` |
| Bambu Studio project for the P1S, each part's settings inside | `print/20261030-seat_block_p1s_one.3mf` |
| plate map and slicing record | `print/20261030-seat_block_p1s_one_map.png`, `print/20261030-seat_block_p1s_one.json` |
| probe plans replayed on each seat, and their path checks | `data/20261030-seat_block_plan_up_xm_seat{1,2,3}.json`, `data/20261030-seat_block_paths_up_xm_seat{1,2,3}.json` |
| homing with each tool on both keys | `data/20261030-seat_block_homing.json` |

## Print (Bambu Lab P1S, PLA, textured PEI at 55 C): one plate, 18 h 26 min, 576 g

| part | ID | settings |
|---|---|---|
| seat block, 135 x 220 x 145 mm | SB | 0.16 mm, 5 walls, 6 top and 5 bottom layers, 25 % gyroid, no brim |
| probe base, with the captive-nut pockets | PB | 0.15 mm, 4 walls, 100 % |
| tools: 5 mm sphere (S5), 2 mm sphere (S2), half rod (RD), flat (FL), fingertip (FT) | | 0.1 mm, 4 walls, 100 % |
| tool: flat at 30 deg | F30 | 0.08 mm, 4 walls, 100 % |

Every part prints without supports in its CAD orientation, each at its own layer height; the block starts 24 mm from
the bed's left edge, clear of the wiper's corner. Plate v2e's two plates took 21.8 h and 702 g.

## Hardware

- Four M3 hex nuts (5.5 mm across flats), one in each pocket of the probe column, slid in from the column's +y or -y
  face 3.6 mm under its top. Each tool goes on with two M3 x 10 socket head screws on a diagonal, which turn into the
  nuts. Snug them by hand: the PLA over each pocket is 3.6 mm thick.
- The probe base onto mount v7's adapter disc with the four M3 x 12, once. It stays there while tools change.
- Each held tip onto its seat with four M2 x 8 button heads into the block's 1.9 mm holes, as on v2e.
- The fingertip tool's tip with the same four M2 x 8 onto the tool's station block. Put the tool on the probe base first:
  its station block covers the flange's M3, which a hex key then reaches at about 5 deg from the side.
- The block onto the 2x4s with seven wood screws through 4.6 mm holes, heads in 9 mm counterbores, 3 mm of PLA under each
  head: three at x_A 15 mm into the front 2x4 and four at x_A 130 mm into the back one. Each screw stands 35.5 to 44.5 mm
  from the nearest column, so a drill comes down on it from above. The columns stand 145 mm high, so a bit or extension
  of 75 mm or more keeps the drill's chuck above them.

## Assembly

1. Clear the table: the low sphere, the rod, the tall indenter and the holder come off. The holder's two held tips go
   onto the block's seats.
2. Set the block across both 2x4s, the keys toward the robot. Its frame is the old low sphere's frame A: the middle
   seat's apex 70 mm behind where the low sphere's centre stood, away from the robot. A ruler to 1 to 2 mm is enough,
   since homing on a key and pass 1's touch-offs measure where it is.
3. The tall indenter's base stood 1.4 mm higher than the low sphere's (z_B 71.6 against 70.2 mm), so the back 2x4 may
   stand higher. Shim under the front screws until the block sits flat, then drive the seven screws.
4. Slide a nut into each of the column's four pockets, put a tool on the boss and pull it down with its two M3 x 10. The
   socket is 0.11 mm smaller a face than the boss's taper, so the tapered faces seat and the flange stops about 0.3 mm
   over the column's top. A flange that sits on the top means the taper printed loose.
5. Screw each tip pad up onto its seat, its end toward the robot. Its cable leaves away from the robot and runs down
   behind its column.

## First session

As the rig's runbook, with two changes:
- The probe homes on a key of the block (key 1 at y_A -40 mm for seats 1 and 2, key 2 at +40 mm for seat 3), with the
  palmar side down and the tool reaching between the columns, as the fork homes on a key.
- After each tool change, touch off one pad point. A shift over 0.02 mm means the tool's centre needs pass 1's
  touch-offs again.

## Regenerate

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 100 --"
    $RG ~/miniconda3/envs/femtip/bin/python scripts/ur5_inverted_rig.py --plate block build           # cad/
    $RG .venv/bin/python scripts/ur5_inverted_rig.py --plate block replay --seat 1                   # seat 2's plan and check
    $RG .venv/bin/python scripts/ur5_inverted_rig.py --plate block home-check                        # data/*_homing.json
    env -u VIRTUAL_ENV ~/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- taskset -c 14-18 \
        ~/miniconda3/envs/femtip/bin/python -u scripts/inverted_rig_print.py --block                  # print/, one plate
    .venv/bin/python scripts/inverted_rig_page.py                                                    # the rig's page
