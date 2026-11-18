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
| fit test: the stem's foot and root | `20261117-test_stem_white.stl` | upright, foot on the bed, 5 mm brim |

Joint. The block's six tag faces are those of the earlier blocks. Its top face has a square socket 8 mm across at the face. The socket's faces lie at 20 deg to the axis for 3 mm and run straight to 4 mm. The stem's foot is the matching square taper, from 0.5 mm above the top face to 2.7 mm into the socket, so it seats on the four taper faces with 1.3 mm left below it. One M3 x 16 screw goes up through a 3.4 mm hole from a counterbore 6 mm across and 3.4 mm deep in the bottom face. It threads 5.1 mm into a 2.75 mm pilot in the foot, the owner's size for self-forming M3 threads, and draws the taper into the socket. The head sits below the bottom face, which stays flat on the seat. The screw threads into the stem, so a stripped thread costs a stem, not a block. The socket's (-x_B, -y_B) corner is filled and the foot's cut, so the stem goes in one way only: turned by 90, 180 or 270 deg it stands 2.7 mm proud of its seat.

Stem. Above the foot a cone 8 mm across narrows to the 4 mm stem over 10 mm. The stem's other end is the cotter joint's taper and post of 2026-11-16 at the object's flat seat. Each object's stem has its own length, and its taper and post are turned as that object's socket requires.

Identification. Each object has a code. It is cut into the block's bottom face with the block's ids, and both the block's bottom face and the stem's root bear as many dimples as the code's number. The labels read from below.

| Code | Object | Block ids | Stem: block's top to the object's surface (mm) | Dimples | Stem as printed (mm) |
|---|---|---|---|---|---|
| C25 | 25 mm cylinder | 10-15 | 35.6 | 1 | 48.0 |
| S40 | 40 mm sphere | 16-21 | 25.7 | 2 | 36.8 |
| C45 | 45 mm cylinder | 32-37 | 31.0 | 3 | 42.9 |
| S67 | 67 mm sphere | 38-43 | 16.7 | 4 | 27.6 |

C45 and S67 are the objects of `assets/tracking/20261117-large_objects/`. C25 and S40 fit the cotter-joint objects of `assets/tracking/20261116-cotter_joint/`; the 25 mm cylinder and 40 mm sphere printed on 2026-11-13 have their blocks glued on.

Assembly. Push the stem's foot into the socket, its cut corner to the socket's filled corner. Turn the M3 x 16 screw in from below until the stem no longer rocks, and no further: self-forming threads in PLA strip after repeated screwing (owner, 2026-11-04). Then fit the stem's other end to the object with the cotter. To replace a broken stem, take the screw out and push the foot out of the socket with the screwdriver through the hole.

Checks. In the CAD the seated foot and the block share 0 mm3 for all four objects. Every tag face of the four blocks' black bodies, drawn at 20 px/mm from the mesh, decodes as its own id, 0.022 mm RMS from the tracker's map (ids 32-43 included, `src/sr2hand/tracking/tag_bits.json` now has tag36h11 ids 0-63). Over the larger objects' 400 rollouts on the two seats no finger came within 4.3 mm of the root. Over the series' layouts of the rollouts of 2026-11-05 no finger came within 5.7 mm of the root (6.8 mm for the 6 mm root of 2026-11-16).

Mass. The stems are 0.90 to 1.22 g of solid PLA, and an M3 x 16 steel screw is about 1.4 g. With the screw the 67 mm sphere comes to about 31 g with its block, stem and cotter, against the simulations' 25 g. Weigh each assembled object; the simulation takes the measured mass.

Fit test. Print `20261117-modular_test_p1s.3mf` first: the C45 block in white without tags and a foot with its root (0.35 h, 6.3 g white on filament 1). Screw them together. The joint works if the foot seats without rocking when the screw is firm, and comes out by hand once the screw is out. If it rocks with the screw firm, the foot bottoms out: report it, and FOOT_U or FOOT_DEPTH in `tracking_objects_print.py` changes.

Bambu Studio project `20261117-modular_blocks_p1s.3mf`: the C45 and S67 blocks. Sliced again by the command line: 3.05 h, 41.9 g white on filament 2 and 10.2 g black on filament 1, 67 filament loads. Not in git; `tracking_objects_print.py modular bambu` rebuilds it.

Bambu Studio project `20261117-modular_stems_p1s.3mf`: the C45 and S67 stems, two each. Sliced again by the command line: 0.47 h, 4.0 g white on filament 1, 5 mm brim. Not in git; `tracking_objects_print.py modular bambu` rebuilds it.
