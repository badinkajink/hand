"""make_random.py at another gantry layout, inside tonight's safe joint ranges (2026-11-17, the identification at 7.5 V,
docs/experiments/20261117-sr2_system_identification/QUEUE.md; owner, 19:1x: "good idea to get the magnetic fields
response recalibrated too with this new sysid"). Random poses of all three fingers, each at least MIN mm finger to finger
in the CAD hand at MOUNTS (palm mm) and reached by a move that keeps MIN - 1 mm, a zero-pose dwell every 12 poses. The
ranges keep every MCP at 55 deg or less (the MCPs' potentiometers jump past about 62 deg, sr2-mcp-pot-dead-zone), clear
of the MCPs' mechanical stops (-11.4 and -10.0 deg on the index and middle), and inside servo_limits' safe ranges.

    uv run ... python make_random_at.py OUT.json SEED N MIN --mounts=tx,ty,ix,iy,mx,my
"""
import json
import sys

import numpy as np

import handgeo
from handgeo import F, Hand, ORIGIN

arg = [x for x in sys.argv if x.startswith("--mounts=")]
sys.argv = [x for x in sys.argv if not x.startswith("--mounts=")]
v = [float(x) for x in arg[0].split("=", 1)[1].split(",")]
MOUNTS = {"thumb": (v[0], v[1]), "index": (v[2], v[3]), "middle": (v[4], v[5])}
HOME_LOCAL = {f: (MOUNTS[f][0] - ORIGIN[f][0], MOUNTS[f][1] - ORIGIN[f][1]) for f in F}

out, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
MIN = float(sys.argv[4]) if len(sys.argv) > 4 else 12.0
rng = np.random.default_rng(seed)
h = Hand()
LIM = {"thumb": {"yaw": (-35, 35), "mcp": (-8, 55), "pip": (-10, 80)},
       "index": {"yaw": (-35, 35), "mcp": (-8, 50), "pip": (3, 85)},
       "middle": {"yaw": (-35, 35), "mcp": (-7, 55), "pip": (-10, 70)}}
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
    if any(p[f]["mcp"] + p[f]["pip"] > 120.0 for f in ("index", "middle")):   # the curl limit of 19:4x on 11-04
        continue
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
json.dump({"layout": "mounts", "mounts_palm_mm": MOUNTS, "lim": LIM, "seed": seed, "min_pose_mm": MIN, "steps": steps,
           "scene": "assets/mjcf/experimental/20261030-cad_v2/scene_screwdriver_medium.xml"}, open(out, "w"), indent=1)
dur = sum(s["dwell_s"] + 1.3 for s in steps)
print(f"{len(steps)} steps ({n} random from {tries} draws), about {dur / 60:.0f} min; clearance min {min(cl):.1f}, "
      f"median {np.median(cl):.1f} mm; under 20 mm: {sum(c < 20 for c in cl)}")
