# Tag test plate and tag block, 20261107

Print files of steps 1 and 3 of `docs/notes/20261105-object_seating_and_tracking/20261105-object_seating_and_tracking_plan.md`, written by `scripts/hardware/object_tracking_print.py`. Each part is a white body and its black cells in one frame, lying on z = 0. Load both files of a part at once as one object with two parts, and give the black part the black filament.

| Part | Files | Print |
|---|---|---|
| Tag test plate, 170 x 70 x 3 mm: tag36h11 ids 0-4 (upper row) and tag25h9 ids 0-4 at 10, 12, 13, 14 and 16 mm; two M3 clearance holes | `20261107-tag_test_plate_white.stl`, `20261107-tag_test_plate_black.stl` | face up; black cells 0.6 mm deep, 3 layers in both colours |
| Tag block, 30 mm across flats, 17 mm tall: tag36h11 ids 10-15 at 13 mm; 4.2 mm bore along the axis | `20261107-tag_block_ids10-15_white.stl`, `20261107-tag_block_ids10-15_black.stl` | upright; black cells 1.0 mm deep, 65 layers in both colours |

Settings for both: matte white and matte black PLA on the P1S with the AMS, 0.4 mm nozzle, 0.2 mm layers with a 0.2 mm first layer, textured PEI plate, prime tower on. The plate: 15 % infill and at least 3 top layers. The block: 2 walls (the plan's 0.8 mm), 15 % gyroid infill. On the block the black squares' top and bottom edges, 2.0 and 15.0 mm above the bed, fall on layer boundaries.

Frames: the plate's tag map (`object_tracking_mjcf.plate_maps`) is the file's frame moved down by 3 mm; the block frame B (`sr2hand.tracking.tags.HexBlock`) is the file's frame moved down by 8.5 mm, z_B up, face k's normal at 60k deg from x_B.

Check: each tag face of the black body drawn at 20 px/mm from the mesh decodes as its own id, with corners 0.015 to 0.027 mm RMS from the map; in each face's plane the white and black areas sum to the face's area. Numbers in `20261107-tag_prints.json`.
