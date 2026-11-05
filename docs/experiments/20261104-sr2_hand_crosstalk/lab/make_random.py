"""Random poses of all three fingers at the home layout (yaw within 40 deg, mcp and pip within each servo's range), each
at least MIN mm from finger to finger in the CAD hand and reached by a move that keeps MIN - 1 mm, with a zero-pose dwell
every 12 poses: the test set of the crosstalk subtraction.

    uv run ... python make_random.py OUT.json SEED N [MIN]
"""
import json
import sys

import numpy as np

from handgeo import F, HOME_LOCAL, Hand, ORIGIN

out, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
rng = np.random.default_rng(seed)
h = Hand()
LIM = {"thumb": {"yaw": (-40, 40), "mcp": (-10, 88), "pip": (-10, 85)},
       "index": {"yaw": (-40, 40), "mcp": (-10, 64), "pip": (0, 90)},
       "middle": {"yaw": (-40, 40), "mcp": (-10, 69), "pip": (-10, 76)}}
ZERO = {f: {"yaw": 0.0, "mcp": 0.0, "pip": 0.0} for f in F}


def clearance(p):
    h.set(HOME_LOCAL, p)
    return min(h.clearances().values())


def path(a, b, k=10):
    return min(clearance({f: {j: a[f][j] + (b[f][j] - a[f][j]) * s for j in a[f]} for f in F})
               for s in np.linspace(0, 1, k + 1))


steps, last, since, tries = [], ZERO, 0, 0
steps.append({"op": "joints", "family": "zero", "joints": ZERO, "dwell_s": 8.0, "speed": 80})
while sum(s["family"] == "random" for s in steps) < n:
    tries += 1
    p = {f: {j: round(float(rng.uniform(*LIM[f][j])), 1) for j in ("yaw", "mcp", "pip")} for f in F}
    c = clearance(p)
    if c < MIN:
        continue
    pc = path(last, p)
    if pc < MIN - 1.0:
        continue
    steps.append({"op": "joints", "family": "random", "joints": p, "dwell_s": 3.0, "speed": 80,
                  "clearance_mm": round(c, 1), "path_clearance_mm": round(pc, 1)})
    last, since = p, since + 1
    if since >= 12:
        zc = path(last, ZERO)
        assert zc >= MIN - 1.0
        steps.append({"op": "joints", "family": "zero", "joints": ZERO, "dwell_s": 8.0, "speed": 80,
                      "path_clearance_mm": round(zc, 1)})
        last, since = ZERO, 0
zc = path(last, ZERO)
steps.append({"op": "joints", "family": "zero", "joints": ZERO, "dwell_s": 30.0, "speed": 80, "path_clearance_mm": round(zc, 1)})
assert zc >= MIN - 1.0
cl = [s["clearance_mm"] for s in steps if s["family"] == "random"]
json.dump({"layout": "home", "seed": seed, "min_pose_mm": MIN, "steps": steps,
           "scene": "assets/mjcf/experimental/20261030-cad_v2/scene_screwdriver_medium.xml"}, open(out, "w"), indent=1)
dur = sum(s["dwell_s"] + 1.3 for s in steps)
print(f"{len(steps)} steps ({n} random from {tries} draws), about {dur / 60:.0f} min; clearance min {min(cl):.1f}, "
      f"median {np.median(cl):.1f} mm; under 20 mm: {sum(c < 20 for c in cl)}")
