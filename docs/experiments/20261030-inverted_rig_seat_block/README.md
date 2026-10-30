# Seat block for the inverted calibration rig: print, assembly and first session

The seat block holds three fingertips pad up for the probe on the HEX-E. It replaces plate v2e of 2026-10-29 at the
same place, with v2e's seats and without its indenters and keys (owner, 2026-10-30: "a smaller ... print out block that
screws into the same 2x4 ... can you space it out farther from the fingertip holders so a drill can more easily drive
in?"; the owner chose v2e's place, with the tall indenter and the holder off). The same day the tools' fit on the probe
base changed (owner: "can we actually create a fastening point on the homing block for different size indenters? ...
if we constrain it enough there shouldn't be too much slop/error"). The rig's page,
`../20261029-inverted_calibration_rig/20261029-inverted_rig.html`, section 8, gives the design and the replays; the
rig's runbook, `../20261029-inverted_calibration_rig/README.md`, gives the probe, the tools and the program changes.
This file is the seat block's runbook.

## Files

| what | where |
|---|---|
| CAD in the rig's frames (M for the probe, A for the block) | `cad/20261030-seat_block_*.stl`, `cad/20261030-seat_block_parts.json` |
| print-oriented STLs (on z = 0, centred) | `print/20261030-seat_block_*_print.stl` |
| Bambu Studio projects for the P1S, each part's settings inside | `print/20261030-seat_block_p1s_a.3mf`, `print/20261030-seat_block_p1s_b.3mf` |
| plate maps and slicing records | `print/20261030-seat_block_p1s_{a,b}_map.png`, `print/20261030-seat_block_p1s_{a,b}.json` |
| probe plans replayed on each seat, and their path checks | `data/20261030-seat_block_plan_up_xm_seat{1,2,3}.json`, `data/20261030-seat_block_paths_up_xm_seat{1,2,3}.json` |

## Print (Bambu Lab P1S, PLA, textured PEI at 55 C)

| plate | parts | settings | time, filament (command-line slice) |
|---|---|---|---|
| a | probe base (PB) with the captive-nut pockets | 0.15 mm, 4 walls, 100 % | 6 h 11 min, 177 g for the plate |
| a | tools: 5 mm sphere (S5), 2 mm sphere (S2), half rod (RD), flat (FL) with the taper-seated socket | 0.1 mm, 4 walls, 100 % | |
| a | tool: flat at 30 deg (F30) | 0.08 mm, 4 walls, 100 % | |
| b | seat block (SB), 130 x 220 x 145 mm | 0.16 mm, 5 walls, 6 top and 5 bottom layers, 25 % gyroid, no brim | 10 h 44 min, 348 g |

Plate v2e took 15 h 37 min and 524 g. Every part prints without supports in its CAD orientation; the nut pockets'
roofs bridge 5.8 mm.

## Hardware

- Four M3 hex nuts (5.5 mm across flats), one in each pocket of the probe column, slid in from the column's +y or -y
  face 3.6 mm under its top. Each tool goes on with two M3 x 10 socket head screws on a diagonal, which turn into the
  nuts. Snug them by hand with a hex key: the PLA over each pocket is 3.6 mm thick.
- The probe base onto mount v7's adapter disc with the four M3 x 12, once. It stays there while tools change.
- Each held tip onto its seat with four M2 x 8 button heads into the block's 1.9 mm holes, as on v2e.
- The block onto the 2x4s with eight wood screws through 4.6 mm holes, heads in 9 mm counterbores, 3 mm of PLA under
  each head: four at x_A 22 mm into the low sphere's 2x4 and four at x_A 130 mm into the tall indenter's. Each screw
  stands 35.5 to 39.8 mm from the nearest column, so a drill comes down on it from above. The columns stand 145 mm
  high, so a bit or extension of 75 mm or more keeps the drill's chuck above them.

## Assembly

1. Take the tall indenter and the holder off their 2x4. Their two held tips go onto the block's seats.
2. Set the block across both 2x4s, its notch round the low sphere's base and the row of columns behind it. The block's
   frame is the low sphere's frame A: the middle seat's apex 70 mm behind the low sphere's centre (away from the robot)
   and level with it across. A ruler to 1 to 2 mm is enough: pass 1's touch-offs measure each seat's pose.
3. The tall indenter's base stood 1.4 mm higher than the low sphere's (z_B 71.6 against 70.2 mm), so the back 2x4 may
   stand higher. Shim under the front screws until the block sits flat, then drive the eight screws.
4. Slide a nut into each of the column's four pockets, put a tool on the boss and pull it down with its two M3 x 10. The
   socket is 0.11 mm smaller a face than the boss's taper, so the tapered faces seat and the flange stops about 0.3 mm
   over the column's top. A flange that sits on the top means the taper printed loose.
5. Screw each tip pad up onto its seat, its end toward the robot. Its cable leaves away from the robot and runs down
   behind its column.

## First session

As the rig's runbook, with two changes:
- The probe homes on the low sphere's key, as the fork does now, since the probe base's socket is the fork's
  (registration `../20261022-ur5_tip_calibration/lab/20261027-225213-spherev3id1`).
- After each tool change, touch off one pad point. A shift over 0.02 mm means the tool's centre needs pass 1's
  touch-offs again.

## Regenerate

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 100 --"
    $RG ~/miniconda3/envs/femtip/bin/python scripts/ur5_inverted_rig.py --plate block build           # cad/
    $RG .venv/bin/python scripts/ur5_inverted_rig.py --plate block replay --seat 1                   # seat 2's plan and check
    env -u VIRTUAL_ENV ~/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- taskset -c 14-18 \
        ~/miniconda3/envs/femtip/bin/python -u scripts/inverted_rig_print.py --block                  # print/
    .venv/bin/python scripts/inverted_rig_page.py                                                    # the rig's page
