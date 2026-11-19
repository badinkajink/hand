# Modular tag blocks and stems, 20261117

The tag block and its stem as two parts (owner, 2026-11-17: "it's quite costly printing these, in terms of filament and time, and im trying to anticipate breaks, which would make these blocks useless. i am not sure how to design joinery here since we cant interfere with the side faces with the blocks on them"; "some kind of identification for these modular components would also be helpful"). A broken stem is reprinted in white and the block kept. Written by `scripts/hardware/tracking_objects_print.py modular build|check|bambu`.

| Part | Files | Print |
|---|---|---|
| 25 mm cylinder: tag block, socket in its top face, counterbore in its bottom face, ids 10-15, label C25 | `20261117-block_C25_white.stl`, `20261117-block_C25_black.stl` | upright, bottom face on the bed |
| 25 mm cylinder: stem, square-taper foot to the cotter joint's taper and post, C25, 1 dimple | `20261117-stem_C25_white.stl` | upright, foot on the bed, 5 mm brim |
| 40 mm sphere: tag block, socket in its top face, counterbore in its bottom face, ids 16-21, label S40 | `20261117-block_S40_white.stl`, `20261117-block_S40_black.stl` | upright, bottom face on the bed |
| 40 mm sphere: stem, square-taper foot to the cotter joint's taper and post, S40, 2 dimples | `20261117-stem_S40_white.stl` | upright, foot on the bed, 5 mm brim |
| 45 mm cylinder: tag block, socket in its top face, counterbore in its bottom face, ids 32-37, label C45 | `20261117-block_C45_white.stl`, `20261117-block_C45_black.stl` | upright, bottom face on the bed |
| 45 mm cylinder: stem, square-taper foot to the cotter joint's taper and post, C45, 3 dimples | `20261117-stem_C45_white.stl` | upright, foot on the bed, 5 mm brim |
| 67 mm sphere: tag block, socket in its top face, counterbore in its bottom face, ids 38-43, label S67 | `20261117-block_S67_white.stl`, `20261117-block_S67_black.stl` | upright, bottom face on the bed |
| 67 mm sphere: stem, square-taper foot to the cotter joint's taper and post, S67, 4 dimples | `20261117-stem_S67_white.stl` | upright, foot on the bed, 5 mm brim |
| fit test: the C45 block in white without tags | `20261117-test_block_white.stl` | upright, bottom face on the bed |
| fit test: the stem's foot, shank and root | `20261117-test_stem_white.stl` | upright, foot on the bed, 5 mm brim |

Joint. The block's six tag faces are those of the earlier blocks. Its top face has a square socket 10 mm across at the face, whose faces lie at 20 deg to the axis for 3 mm. Below the taper the socket runs straight to 12.6 mm, 4.4 mm above the bottom face. The stem's foot is the matching square taper, from 0.5 mm above the top face to 2.7 mm into the socket, and below it a square shank 7.4 mm across. The shank ends 5.4 mm above the bottom face, 1 mm above the socket's floor. Its end has a 3.2 mm hole 5.5 mm deep for the insert (M2 brass heat-set insert, 3.6 mm outside, 4 mm long). One M2 x 8 screw goes up from a counterbore 4.4 mm across and 2.4 mm deep in the bottom face, through 2 mm of floor, and threads 4 mm into the insert. It draws the taper into the socket. The four taper faces then fix the stem's position, tilt and turn on the block. The head sits below the bottom face, which stays flat on the seat. The thread is brass, so the stem comes off and goes on again without wearing the PLA, and the insert is in the stem, so a broken stem costs a stem and an insert. The socket's (-x_B, -y_B) corner is filled and the foot's cut, so the stem goes in one way only: turned by 90, 180 or 270 deg, its shank meets the filled corner at the socket's mouth and the stem stands 11.75 mm proud.

Stem. Above the foot a cone 8 mm across narrows to the 4 mm stem over 10 mm. The stem's other end is the cotter joint's taper and post of 2026-11-16 at the object's flat seat. Each object's stem has its own length, and its taper and post are turned as that object's socket requires.

Identification. Each object has a code. It is cut into the block's bottom face with the block's ids, and both the block's bottom face and the stem's root bear as many dimples as the code's number. The labels read from below.

| Code | Object | Block ids | Stem: block's top to the object's surface (mm) | Dimples | Stem as printed (mm) |
|---|---|---|---|---|---|
| C25 | 25 mm cylinder | 10-15 | 35.6 | 1 | 56.9 |
| S40 | 40 mm sphere | 16-21 | 25.7 | 2 | 45.7 |
| C45 | 45 mm cylinder | 32-37 | 31.0 | 3 | 51.8 |
| S67 | 67 mm sphere | 38-43 | 16.7 | 4 | 36.5 |

C45 and S67 are the objects of `assets/tracking/20261117-large_objects/`. C25 and S40 fit the cotter-joint objects of `assets/tracking/20261116-cotter_joint/`; the 25 mm cylinder and 40 mm sphere printed on 2026-11-13 have their blocks glued on.

Assembly. Heat-set the insert into the stem's foot, flush with its end. Push the foot into the socket, its cut corner to the socket's filled corner. Turn the M2 x 8 screw in from below until the stem no longer rocks. Then fit the stem's other end to the object with the cotter. To replace a broken stem, take the screw out and pull the stem out; push the foot's end through the hole with the screwdriver if the taper sticks.

Checks. In the CAD the seated foot and the block share 0 mm3 for all four objects. Every tag face of the four blocks' black bodies, drawn at 20 px/mm from the mesh, decodes as its own id, 0.022 mm RMS from the tracker's map (ids 32-43 included, `src/sr2hand/tracking/tag_bits.json` now has tag36h11 ids 0-63). Over the larger objects' 400 rollouts on the two seats no finger came within 4.0 mm of the root. Over the series' layouts of the rollouts of 2026-11-05 no finger came within 5.7 mm of the root (6.8 mm for the 6 mm root of 2026-11-16).

Mass. The stems are 1.64 to 1.96 g of solid PLA. The M2 x 8 steel screw is about 0.37 g and the insert about 0.28 g. Sliced alone in white, the test block is 5.94 g and the C45 stem 1.61 g; with the shallow foot and an M3 x 16 screw (1.45 g) of the first version they were 5.83 and 1.18 g. An object's block, stem and fastener are 0.26 g lighter than in that version: the shank's PLA weighs about twice as much per mm as the M2 screw it shortens. Weigh each assembled object; the simulation takes the measured mass.

Fit test. Print `20261117-modular_test_p1s.3mf` first: the C45 block in white without tags and a foot with its shank and root (0.38 h, 6.9 g white on filament 1). Set the insert and screw them together. The joint works if the foot seats without rocking when the screw is firm, and comes out by hand once the screw is out. If the insert's hole is too loose or too tight for the inserts on hand, INSERT in `tracking_objects_print.py` changes; only the stems carry the hole.

Bambu Studio project `20261117-modular_blocks_p1s.3mf`: the C45 and S67 blocks. Sliced again by the command line: 3.06 h, 42.1 g white on filament 2 and 10.2 g black on filament 1, 67 filament loads. Not in git; `tracking_objects_print.py modular bambu` rebuilds it.

Bambu Studio project `20261117-modular_stems_p1s.3mf`: the C45 and S67 stems, two each. Sliced again by the command line: 0.56 h, 5.7 g white on filament 1, 5 mm brim. Not in git; `tracking_objects_print.py modular bambu` rebuilds it.

The C45 and S67 stems with a glued spigot in place of the cotter joint's taper and post (owner, 2026-11-19) are in `assets/tracking/20261119-glued_objects/`; their feet and the blocks are these.
