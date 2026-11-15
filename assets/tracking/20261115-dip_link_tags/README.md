# AprilTag stickers for the back of the DIP link, 20261115

Stickers for the DIP link b v4, to track each finger's gantry with the two D435s of the HOM trials (owner, 2026-11-15),
written by `scripts/hardware/dip_tag_print.py`.

`20261115-dip_link_tags_letter.pdf` is US Letter at actual size. It has four stickers per finger: tag36h11 with a 13.0 mm
black square, thumb 24, index 25 and middle 26. Each sticker's cut outline is 20.50 x 20.91 mm, 0.25 mm inside the face it
goes on. A drawing at 3:1 shows the placement.

The face is the tongue's back behind the fingertip board, 21.0 mm across the finger by 21.41 mm along it. A 1.45 mm step
bounds it at the PIP end, and the top cover's end plate is flush with it at the tip end. Four M2 pilot holes of 1.9 mm
pass through it at 8.0 mm from the centre lines; they lie under the corners of the white border, 1.17 mm from the
black square's corners. The tag reads upright when the link is seen from the back with the PIP joint up.

Print at 100 % on matte paper, check the 100 mm bar and one black square (13.00 mm) with calipers, cut along the grey
outline and glue the sticker flat. `20261115-dip_link_tags.json` gives each tag's corners in the link's frame L
(`maps_L`; x_L dorsal, y_L along the PIP axis, z_L toward the PIP axis). The tag centre is at x_L 8.10, y_L 0, z_L -36.205
mm; the 0.10 mm is the paper and glue. Frame L is the `<finger>_pip_frame` body of
`assets/mjcf/experimental/20261104-cad_v3/ur5_full_k0.xml`, with the thumb's mesh turned 180 deg about z there.

Check: the sticker grid rasterized at 600 dpi gives 12 detections with the right ids, black-square edges of 12.99 to
13.02 mm and centres within 0.01 mm of the layout (`dip_tag_print.py check`, `check` in the JSON).
