# Seat strip for the inverted calibration rig: print, assembly and first session

The seat strip is the only thing on the table for the inverted rig. It holds three fingertips pad up for the probe on the
HEX-E, at plate v2e's places in x and y, with two homing keys between the seats. The owner asked for it on 2026-10-30:
first a smaller block that screws into the 2x4s ("can you space it out farther from the fingertip holders so a drill can
more easily drive in?"), then the table cleared for it ("add the homing indexers to the sensor blocks. i want to replace
everything on the table with just three sensor mounts. i also would like to have all the indenters + the plate on one
bambu p1s plate"). At 12:3x, with about 300 g of PLA left on the spool, the owner asked for less: "can we drastically
reduce the height and material usage of these parts? ... there's no need for these fingertip holders to come so high up
off the ground anymore". The tall block of 12:0x (commit 1690e3df) became this strip: the pads' apexes stand 40 mm over
the 2x4 instead of 160 mm, and the plate takes 189 g instead of 576 g. The same day the tools' fit on the probe base
changed ("can we actually create a fastening point on the homing block for different size indenters?"), and a sixth tool
came to take a fingertip face out ("a fingertip mount for this new indenter that doesnt use the dip link, it just uses
the cassette"). The rig's page, `../20261029-inverted_calibration_rig/20261029-inverted_rig.html`, section 8, gives the
design, the homing check and the replays; the rig's runbook, `../20261029-inverted_calibration_rig/README.md`, gives the
probe, the tools and the program changes. This file is the strip's runbook. Its files keep the name `seat_block`.

## Files

| what | where |
|---|---|
| CAD in the rig's frames (M for the probe, A for the strip) | `cad/20261030-seat_block_*.stl`, `cad/20261030-seat_block_parts.json` |
| print-oriented STLs (on z = 0, centred) | `print/20261030-seat_block_*_print.stl` |
| Bambu Studio project for the P1S, each part's settings inside | `print/20261030-seat_block_p1s_one.3mf` |
| plate map and slicing record | `print/20261030-seat_block_p1s_one_map.png`, `print/20261030-seat_block_p1s_one.json` |
| probe plans replayed on each seat, and their path checks | `data/20261030-seat_block_plan_up_xm_seat{1,2,3}.json`, `data/20261030-seat_block_paths_up_xm_seat{1,2,3}.json` |
| homing with each tool on both keys | `data/20261030-seat_block_homing.json` |

## Print (Bambu Lab P1S, PLA, textured PEI at 55 C): one plate, 9 h 56 min, 189 g

| part | ID | settings | PLA |
|---|---|---|---|
| seat strip, 35 x 216 mm, 34.5 mm tall at the keys | SB | 0.16 mm, 3 walls, 5 top and 4 bottom layers, 15 % gyroid | 44 g |
| probe base, with the captive-nut pockets | PB | 0.15 mm, 4 walls, 6 top and bottom layers, 50 % gyroid | 57 g |
| tools: 5 mm sphere (S5), 2 mm sphere (S2), half rod (RD), flat (FL), fingertip (FT) | | 0.1 mm, 4 walls, 6 top and bottom layers, 50 % gyroid | 72 g |
| tool: flat at 30 deg | F30 | 0.08 mm, as the other tools | 12 g |

Every part prints without supports in its CAD orientation, each at its own layer height; the strip starts 24 mm from the
bed's left edge, clear of the wiper's corner. Until 12:3x the probe base and the tools printed solid and the plate took
18 h 26 min and 576 g.

## Hardware

- Four M3 hex nuts (5.5 mm across flats), one in each pocket of the probe column, slid in from the column's +y or -y
  face 3.6 mm under its top. Each tool goes on with two M3 x 10 socket head screws on a diagonal, which turn into the
  nuts. Snug them by hand: the PLA over each pocket is 3.6 mm thick.
- The probe base onto mount v7's adapter disc with the four M3 x 12, once. It stays there while tools change.
- Each held tip onto its seat with four M2 x 8 button heads into the station block's 1.9 mm holes, as on v2e.
- The fingertip tool's tip with the same four M2 x 8 onto the tool's station block. Put the tool on the probe base first:
  its station block covers the flange's M3, which a hex key then reaches at about 5 deg from the side.
- The strip onto the tall's 2x4 with four wood screws through 4.6 mm holes, heads in 9 mm counterbores, 3 mm of PLA under
  each head: at x_A 75 mm and y_A -100, -21.5, 21.5 and 100 mm. The keys stand 34.5 mm tall, so a bit of 50 mm or more
  keeps the drill's chuck above them.

## Assembly

1. Clear the table: the low sphere, the rod, the tall indenter and the holder come off. The holder's two held tips go
   onto the strip's seats.
2. Lay the strip on the tall's 2x4, the keys toward the robot's side of the seats. Its frame is the old low sphere's
   frame A: the middle seat's apex 70 mm behind where the low sphere's centre stood, away from the robot. The strip's
   front edge is then at x_A 50 mm, about 5 mm over the joint with the low sphere's 2x4; the screws go 20 mm behind that
   joint. A ruler to 1 to 2 mm is enough, since homing on a key and pass 1's touch-offs measure where it is.
3. The tall's 2x4 stands 1.4 to 1.6 mm higher than the low sphere's, so the strip's front edge stays clear of the lower
   one. Drive the four screws.
4. Slide a nut into each of the column's four pockets, put a tool on the boss and pull it down with its two M3 x 10. The
   socket is 0.11 mm smaller a face than the boss's taper, so the tapered faces seat and the flange stops about 0.3 mm
   over the column's top. A flange that sits on the top means the taper printed loose.
5. Screw each tip pad up onto its seat, its end toward the robot. Its cable leaves away from the robot and drops behind
   the strip onto the 2x4. Each tip keeps its side of the bench and goes on the held tips' mux board
   (`b4:3a:45:b0:d9:94`, `--held-board`; owner, 10-30 14:3x):

   | seat | y_A (mm) | as the owner faces the bench | tip | mux channel |
   |---|---|---|---|---|
   | 1 | -80 | right | v3 ID 1 | 3 |
   | 2 | 0 | middle | v3 ID 3 | 1 |
   | 3 | 80 | left | v3 ID 2 | 0 |

## First session

As the rig's runbook, with three changes:
- The probe homes on a key of the strip (key 1 at y_A -40 mm for seats 1 and 2, key 2 at +40 mm for seat 3), with the
  palmar side down and the tool pointing away from the robot. The probe's column then stands between two held tips,
  16 mm from each; homing before the tips go on is easier.
- Home on both keys once. The keys stand 80 mm apart in the print, so the two homings should place them 80 mm apart
  within 0.1 mm.
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
