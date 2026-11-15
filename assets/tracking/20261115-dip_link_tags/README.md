# AprilTag stickers for the back of the DIP link, 20261115

Stickers for the DIP link b v4, to track each finger's gantry with the two D435s of the HOM trials (owner, 2026-11-15:
"we can use probably the entire back", "although we will need some edge perimeter space for the glue"), written by
`scripts/hardware/dip_tag_print.py`.

`20261115-dip_link_tags_letter.pdf` is US Letter at actual size. It has four stickers per finger: tag36h11 with a 14.0 mm
black square, thumb 24, index 25 and middle 26. Each sticker's cut outline is 20.50 x 20.91 mm, 0.25 mm inside the face it
goes on. Outside the white border (one cell, 1.75 mm) a perimeter of 1.50 mm across the finger and 1.70 mm along it takes
the glue. One corner is cut off at 45 deg (1.2 mm) to mark the orientation, and the labels are printed outside the cut
outline. A drawing at 3:1 shows the placement.

The face is the tongue's back behind the fingertip board, 21.0 mm across the finger by 21.41 mm along it. A 1.45 mm step
bounds it at the PIP end, and the top cover's end plate is flush with it at the tip end. The four M2 pilot holes (1.9 mm)
run from the board's seat to the back at 8.0 mm from the centre lines; the screws end 0.1 mm inside the tongue, and the
paper bridges the holes, 0.46 mm outside the black square's corners. The cut-off corner goes at the PIP end, on the left
as seen from the back with the PIP joint up; the tag then reads upright.

Print at 100 % on matte paper, check the 100 mm bar and one black square (14.00 mm) with calipers, cut along the grey
outline and glue only the perimeter. `20261115-dip_link_tags.json` gives each tag's corners in the link's frame L
(`maps_L`; x_L dorsal, y_L along the PIP axis, z_L toward the PIP axis). The tag centre is at x_L 8.10, y_L 0, z_L -36.205
mm; the 0.10 mm is the paper and glue. Frame L is the `<finger>_pip_frame` body of
`assets/mjcf/experimental/20261104-cad_v3/ur5_full_k0.xml`, with the thumb's mesh turned 180 deg about z there.

Check: the sticker grid rasterized at 600 dpi gives 12 detections with the right ids, black-square edges of 13.99 to
14.03 mm and centres within 0.01 mm of the layout (`dip_tag_print.py check`, `check` in the JSON).
