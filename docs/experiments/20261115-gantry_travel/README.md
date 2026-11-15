# Gantry travel of the SR2 hand from tags on the DIP links: runbook

How to measure each finger's gantry travel with the two D435s of the HOM trials (owner, 2026-11-15: "for the middle/index,
if we actuate both the mcp and pip flex to ~90 degrees, and make sure the thumb is out of the way, we theoretically should
be able to track travel"). The measurement gives, per stage, the scale (measured mm per commanded mm), the axis direction
in the palm frame P, the straightness, the backlash between the two passes and whether the gantry reaches the ends of the
station's 60 mm travel. The pin-push layout (75, 56.8) puts the index and middle at that end.

## Tags

`assets/tracking/20261115-dip_link_tags/` (`scripts/hardware/dip_tag_print.py`): tag36h11, 14.0 mm black square, thumb 24,
index 25, middle 26, one sticker on the back of each DIP link (the tongue's back, 21.0 x 21.41 mm), the cut-off corner at
the PIP end. Its JSON gives each tag's corners in the link's frame L.

## A session (corvus, the owner present)

The arm does not move. Sync corvus first (its clone cannot fetch from GitHub): on the workstation
`git push --no-verify corvus:workspace/hand main:refs/remotes/origin/main`, then on corvus
`GIT_LFS_SKIP_SMUDGE=1 git merge --ff-only origin/main`.

1. The station homed (its web page or `/api/v1/home`) with torque on and a plan's morphology applied, so that it knows
   every gantry's position.
2. The viewing pose (sim degrees, checked against the station's limits; the owner judges the thumb's clearance):

       /usr/bin/python3 lab/gantry_travel.py pose --index 85,85 --middle 85,85 --thumb 0,0,0

3. The tags in both cameras, with their view angles (the tracker drops a tag beyond 65 deg) and distances:

       /usr/bin/python3 lab/gantry_travel.py look                          # window on DISPLAY :1, q quits
       /usr/bin/python3 lab/gantry_travel.py look --headless --seconds 10  # printed every 2 s

4. One finger's sweep: x from its low end to its high end and back in 5 mm steps, then y, then back to the start. Each
   point is a burst of 10 framesets per camera after 0.5 s of settling, about 3 s a point and 4 min a finger:

       /usr/bin/python3 lab/gantry_travel.py sweep --finger index --out ~/workspace/hand/logs/20261115-gantry_travel

5. The fit (camera stack; on corvus or on the workstation after an rsync of the sweep file):

       /usr/bin/python3 ../../../scripts/hardware/gantry_travel_fit.py SWEEP.jsonl --json FIT.json

## Checked without hardware

On a synthetic sweep of tag 25 projected through the calibrated cameras of 2026-11-13 (370 and 331 mm away) with 0.1 px of
corner noise, the fit recovered a scale of 1.0118 (true 1.012), an axis turn of 1.50 deg (1.5) and a backlash of
0.124 mm (0.12) on x, and 0.9939, 0.79 deg and 0.054 mm (0.994, 0.8, 0.05) on y; linearity sd 0.03 mm.
