# ChArUco board, 20261112

Lens and hand-eye calibration target for the D435 colour cameras of `docs/notes/20261105-object_seating_and_tracking/`, written by `scripts/hardware/charuco_board_print.py`.

OpenCV board: `cv2.aruco.CharucoBoard((7, 5), 21.0, 14.0, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))`, lengths in mm, marker ids 0-16. Both square counts are odd, so OpenCV 4.6 (`cv2.aruco.CharucoBoard_create(7, 5, 21.0, 14.0, d)` on corvus) and 4.10 draw the same pattern.

Files: `20261112-charuco_board_7x5_white.stl` and `20261112-charuco_board_7x5_black.stl`, one frame, face up on z = 0: the board in an 8 mm white margin, 163 x 121 x 4 mm, the black 0.6 mm deep, M3 clearance holes 4 mm in from the corners and a pair 12 mm apart at the middle of the top and bottom edges for the hand mount. Load both at once as one object with two parts and give the black part the black filament. Matte white and matte black PLA, 0.2 mm layers, at least 3 top layers, textured PEI plate.

Check: drawn from the black body at 10 px/mm, OpenCV's ChArUco detector finds 24 of 24 corners, 0.0010 mm RMS and at most 0.0013 mm from those found on OpenCV's own drawing of the board; on the top face the white and black areas sum to the face's area.

Bambu Studio project: `20261112-charuco_board_p1s.3mf`, the white on filament 2 and the black on filament 3 (AMS slots 2 and 3), 0.20mm Standard @BBL X1C and Bambu PLA Basic @BBL P1S 0.4 nozzle. Sliced by the command line: 1.6 h, 44.7 g white and 8.0 g black, 4 filament loads. It is not in git; `charuco_board_print.py bambu` rebuilds it.

Hand mount: `20261112-charuco_board_mount_white.stl`, white PLA, printed flange down (no AMS). A 20 x 20 mm post drops into the notch at the end of the hand's camera arm, and its flange rests on the arm and is bolted through the arm's M5 hole with an M5 bolt and nut. The board hangs from the post's -x_P face or its -y_P face by two M3 self-tapping screws through the pair at the middle of its top edge. Against the 121 solids of the hand's CAD assembly in its modelled pose nothing touches but the arm: the nearest solid is 21.4 mm from the mount, 34.5 mm from the board facing -x_P and 16.0 mm from the board facing -y_P (`charuco_board_mount_print.py build`).
