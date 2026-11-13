# Paper calibration boards, 20261113

Boards for the owner's rigid back of about 140 x 130 mm (owner, 2026-11-13), written by
`scripts/hardware/paper_board_print.py`.

`20261113-paper_boards_letter.pdf` is US Letter, two pages at actual size. Page 1 is a ChArUco board,
`cv2.aruco.CharucoBoard((7, 7), 15.0, 10.5, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))` with lengths in
mm: 36 inner corners and markers 0 to 23. Page 2 is a plain checkerboard of 9 x 8 squares of 12 mm, with 8 x 7 inner
corners. Each pattern is centred in a 140 x 130 mm trim outline with crop marks, and its white border is 12.5 mm or
more.

Print at 100 % (actual size). Measure seven squares of page 1 (105.0 mm) along both edges with calipers, and use the
measured square size in the calibration. Glue the sheet over the whole back. Use page 1: each corner is named by its
markers, so views with part of the board hidden still count.

Check: the boards' region rasterized at 300 dpi gives all 36 ChArUco corners within 0.020 mm of the board model, and
the checkerboard's 56 corners 12.000 and 11.996 mm apart (`paper_board_print.py check`, `20261113-paper_boards.json`).

With the board at an unmeasured place on the hand, the hand-eye poses need turns of the flange about two or more axes.
Translations alone leave each camera's position undetermined
(`docs/experiments/20261113-tracking_prints/20261113-tracking_prints_and_calibration_board.html`, section 7).
