# Inverted calibration rig: print, assembly and first session

A probe on the HEX-E presses fingertips held still on indenter plate v2e. The owner approved building it on 2026-10-29
("yea, we can build the inverted rod ... let's shore up the plan and get it ready to print tomorrow"). The result page is
`20261029-inverted_rig.html` in this folder: reach, plans, the choice of hold and the program changes. This file is the
runbook.

## Files

| what | where |
|---|---|
| CAD in the rig's frames (M for the probe, A for the plate) | `cad/20261029-inverted_rig_*.stl`, `cad/20261029-inverted_rig_parts.json` |
| print-oriented STLs (on z = 0, centred) | `print/20261029-inverted_rig_*_print.stl` |
| Bambu Studio projects for the P1S, each part's settings inside | `print/20261029-inverted_rig_p1s_a.3mf`, `print/20261029-inverted_rig_p1s_b.3mf` |
| plate maps and slicing records | `print/20261029-inverted_rig_p1s_{a,b}_map.png`, `print/20261029-inverted_rig_p1s_{a,b}.json` |
| probe plans replayed on each seat, and their path checks | `data/20261029-inverted_rig_plan_up_xm_seat{1,2,3}.json`, `data/20261029-inverted_rig_paths_up_xm_seat{1,2,3}.json` |

## Print (Bambu Lab P1S, PLA, textured PEI at 55 C)

| plate | parts | settings | time, filament (command-line slice) |
|---|---|---|---|
| a | probe base (PB) | 0.15 mm, 4 walls, 100 % (solid, as v7's standoff-fork) | 6 h 13 min, 178 g for the plate |
| a | tools: 5 mm sphere (S5), 2 mm sphere (S2), half rod (RD), flat (FL) | 0.1 mm, 4 walls, 100 % | |
| a | tool: flat at 30 deg (F30) | 0.08 mm (its face slopes 30 deg), 4 walls, 100 % | |
| b | indenter plate v2e (V2E), 220 x 236 x 145 mm | 0.16 mm, 5 walls, 6 top and 5 bottom layers, 25 % gyroid, no brim | 15 h 37 min, 524 g |

- Every part prints without supports in its CAD orientation: the probe base on its foot, each tool on its flange with
  the sphere, rod or flat face up, the plate on its base.
- The plate fills 220 x 236 mm of the 256 mm bed, clear of the wiper's corner (18 x 28 mm at the front left). It has
  no brim; add mouse ears in Bambu Studio if its corners lift. At 0.2 mm layers it prints faster, with coarser steps on
  the 5 mm sphere and the keys.
- The seats' columns bend as printed shells: five walls (2 mm) and 25 % gyroid give about half of solid PLA's
  stiffness across a column. Fewer walls or less infill make the seats softer.

## Hardware

- Probe base onto mount v7's adapter disc: the four M3 x 12 that hold the standoff-fork now, into the disc's brass
  inserts. The adapter disc stays on the HEX-E with its four M6 x 20.
- Each tool onto the probe base: two M3 x 10 socket head screws on a diagonal, threaded into the column's 2.75 mm holes.
  The four holes let a tool go on at 0, 90, 180 or 270 deg.
- Each held tip onto its seat: four M2 x 8 button heads through the cap and the board into the block's 1.9 mm holes,
  as on the fingertip holder's up station.
- The plate onto the 2x4s: seven wood screws through 4.6 mm holes, heads in 9 mm counterbores, 3 mm of PLA under each
  head (three at the front edge, four on the rail behind the seats).

## Assembly

1. Take the standoff-fork (with the finger) off the adapter disc and put the probe base on with the same four M3 x 12.
   Its foot goes on one way only (the screw circle is not symmetric).
2. Seat a tool on the column's boss and pull it down with its two M3 x 10 until the flange's face sits on the
   column's top. The socket's sides are 0.05 mm off the boss's at 20 deg draft, so the screws pull it home.
3. Screw each tip pad up onto its seat's block, its end toward the robot, from above as on the holder's up station.
   The board's cable leaves on the far side (away from the robot); run it down behind the column to the QT Py mux.

## The plate in the cell

- Plate v2e replaces the low sphere, the rod, the tall indenter and the holder. Its frame A is the low sphere's frame
  of v3 ID 1's registration (`lab/20261027-225213-spherev3id1`): x away from the robot, z up from the plate's
  underside, which lies at the 2x4's top (z_B about 70.8 mm). Every plan and check here assumes that place; the
  homing on the plate's keys measures where it is.
- The plate needs flat support under its whole footprint (220 x 236 mm). The low sphere's 2x4 lies under about x_A
  -35 to 55 mm (the sphere's key overhangs its front edge), so the front 75 mm with the three keys overhangs it and the
  front screws at x_A -100 meet no wood. The seats stand at x_A 55 to 110 mm, about where the tall's 2x4 is; the tall's
  base stands 1.6 mm higher than the low sphere's, so a shim may be needed for the plate to lie flat.

## First session

1. Payload: weigh the probe base with its tool and screws (about 0.11 kg; the adapter disc as before) and set the
   arm's payload. Each press is tared at its rest.
2. Home on key 2 with the probe base's socket, as with the fork. The socket stands where the fork's does
   (z_M 49 mm, its face at x_M 17 mm), so the homing pose should match the fork's within 0.1 mm.
3. Seat compliance: a dial indicator on seat 2's block while the probe presses the face at 1 to 5 N. The beam
   estimate is 0.2 um/N along the press (pad up, the column loaded along its axis) and 8 to 11 um/N across it.
4. Probe compliance: a dial indicator on the tool's sphere at 1 to 5 N across the stem. The estimate is 77 um/N,
   nearly all of it the HEX-E's (data sheet).
5. Field: hover the probe at rest poses around a held tip and log the held board. The moving wrist and HEX-E should
   change the held chips' field by less than 1 uT between a press's rest and its depth.
6. Pass 1 on each seat: touch-offs of the 5 mm sphere on the held pad (0.15 mm/s to 0.5 N, about 18 directions over
   the face, edges and flanks). They fit the held pad's pose in B (6 numbers) and the sphere's centre in M (3).
7. The plan of each seat (about 275 actions, 2 mm apart on the pad; normal presses, pushes 20 and 35 deg off the
   normal, twists of 15 deg), rebuilt on that registration.

## What the lab program needs

The program drives the finger against fixed indenters. The probe mode needs a probe action, the held board as the
measured board, the registration with the roles swapped (step 6), the force on the pad as the opposite of the HEX-E's
reading, and a gravity model of the payload for the twists, fitted from the rests of the plan's own presses (no
sweep). The page's section "What the lab program needs for the probe" lists them. Every robot run needs the owner's go
in the lab's session.

## Regenerate

    RG="$HOME/.claude/bin/resguard.sh run --mem 3G --cpu 100 --"
    $RG ~/miniconda3/envs/femtip/bin/python scripts/ur5_inverted_rig.py build                 # cad/
    $RG .venv/bin/python scripts/ur5_inverted_rig.py replay --seat 1                         # seat 2's plan and check
    env -u VIRTUAL_ENV ~/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- taskset -c 14-18 \
        ~/miniconda3/envs/femtip/bin/python -u scripts/inverted_rig_print.py                 # print/
    .venv/bin/python scripts/inverted_rig_page.py                                            # the page
