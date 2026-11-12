# Tracked cylinder and sphere, 20261112

Print files of the objects of `docs/notes/20261105-object_seating_and_tracking/`, written by `scripts/hardware/tracking_objects_print.py`; the tracker's maps of the assembled objects are `object_tracking_mjcf.object_map(shape)`. Two-colour parts are a white body and its black cells in one frame: load both files at once as one object with two parts and give the black part the black filament.

| Part | Files | Print |
|---|---|---|
| cylinder block and stem, ids 10-15 | `20261112-cylinder_block_white.stl`, `20261112-cylinder_block_black.stl` | upright, block on the bed; 65 layers in both colours |
| sphere block and stem, ids 16-21 | `20261112-sphere_block_white.stl`, `20261112-sphere_block_black.stl` | upright, block on the bed; 65 layers in both colours |
| cylinder body, end tags 22 and 23 | `20261112-cylinder_white.stl`, `20261112-cylinder_black.stl` | standing on its -z end (tag 23 on the bed, tag 22 on top) |
| sphere, lower hemisphere with the seat | `20261112-sphere_lower_white.stl` | flat side down, the seat on top |
| sphere, upper hemisphere | `20261112-sphere_upper_white.stl` | flat side down |
| dowel pin of the sphere (print three) | `20261112-sphere_pin_white.stl` | standing; print three |

Geometry. The cylinder prints at 23.4 mm, 100 mm long, so 0.8 mm grip tape brings it to 25 mm; the end faces stay bare for their tags. The sphere is 40 mm and bare. The block's top lies 35.6 mm (cylinder) and 25.7 mm (sphere) below the object's taped or bare surface, 15 mm below the lowest finger point of the plan's five layouts and the measured servo's five best per task. The flat seat is 10 mm across and lies 1.12 mm (cylinder) and 0.64 mm (sphere) below the printed surface; the key is 2.4 x 5.0 x 4.0 mm with 0.1 mm clearance per side in its slot, its long side along the cylinder's axis.

Assembly. Glue the key into its slot with the seat flat on the object's flat (thin CA glue or epoxy), the block's tag faces then upright. Glue the hemispheres with the three dowels. Wrap the cylinder's curved surface in the tape, leaving a 10 mm hole at the seat. Weigh: each object with its block and tape is 25 g in the simulations. The plan expects 3 g for a block with 0.8 mm walls, and the stem with its cone and key is 0.75 g (cylinder) and 0.57 g (sphere) of solid PLA, so the cylinder body with its tape has about 21.2 g and the sphere about 21.4 g. Weigh the printed blocks and the tape first, then set the bodies' infill to make up the rest.

Calibration. A bundle adjustment of the assembled cylinder over its eight tags (`pose.bundle_adjust` with the block's three across-flats as lengths) gives its map, and with it where the cylinder sits on its block. The sphere's map is its block's, placed by the seat and key. In the CAD the assembled parts share 0 mm3 (cylinder) and 0 mm3 (sphere): the seats touch the flats and the keys clear their slots.

Check: every tag face of the black bodies, drawn at 20 px/mm from the mesh, decodes as its own id with corners 0.022 mm RMS from the map.

Bambu Studio project: `20261112-objects_p1s.3mf`, all parts on one plate, the white on filament 2 and the black on filament 3 (AMS slots 2 and 3). Sliced by the command line: 4.4 h, 81.3 g white and 10.1 g black, 73 filament loads. Not in git; `tracking_objects_print.py bambu` rebuilds it.
