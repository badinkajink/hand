"""The night's crosstalk sweep on the SR2 hand at the gantries' home layout, as a list of steps for run_plan.py, each
pose and each move between poses checked for finger-to-finger clearance in the CAD hand (cad_v2 collision spheres).

Families: a baseline at the zero pose; servo torque off and on; each stepper axis 5 mm out and back; each finger alone
through yaw, mcp and pip and two grids (the other two straight: its own board's change with its own angles, and the
others' boards' change from a moving neighbour); the thumb and the middle finger toward each other (the pair that
meets at the home layout). A zero-pose dwell follows every 12 poses, to follow the boards' drift.
"""
import itertools
import json
import sys

import numpy as np

from handgeo import F, HOME_LOCAL, Hand, ORIGIN

MIN_POSE_MM = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0     # CAD clearance at every pose
MIN_PATH_MM = MIN_POSE_MM - 1.0                                     # and along each move (joint-space line)
DWELL = 3.0
SPEED = 80
h = Hand()
ZERO = {f: {"yaw": 0.0, "mcp": 0.0, "pip": 0.0} for f in F}
HOME_PALM = {f: (ORIGIN[f][0] + HOME_LOCAL[f][0], ORIGIN[f][1] + HOME_LOCAL[f][1]) for f in F}
# the station's servo ranges in sim degrees (sim = sign x servo; flexion signs +1, yaw thumb +1, index and middle -1)
RANGE = {"thumb": {"mcp": (-12.3, 89.06), "pip": (-18.0, 86.72)},
         "index": {"mcp": (-15.0, 64.75), "pip": (-2.93, 92.0)},
         "middle": {"mcp": (-15.0, 69.73), "pip": (-16.11, 77.05)}}


def full(partial):
    out = {f: dict(ZERO[f]) for f in F}
    for f, v in partial.items():
        out[f].update(v)
    return out


def ok_range(pose):
    for f in F:
        for j in ("mcp", "pip"):
            lo, hi = RANGE[f][j]
            if not lo + 0.5 <= pose[f][j] <= hi - 0.5:
                return False
        if abs(pose[f]["yaw"]) > 45:
            return False
    return True


def clearance(pose):
    h.set(HOME_LOCAL, pose)
    return min(h.clearances().values())


def path_clearance(a, b, n=12):
    return min(clearance({f: {j: a[f][j] + (b[f][j] - a[f][j]) * s for j in a[f]} for f in F})
               for s in np.linspace(0, 1, n + 1))


steps, poses_since_zero, last = [], 0, full({})
dropped = []


def pose_step(fam, partial, dwell=DWELL):
    global poses_since_zero, last
    pose = full(partial)
    if not ok_range(pose):
        dropped.append((fam, partial, "range"))
        return
    c = clearance(pose)
    if c < MIN_POSE_MM:
        dropped.append((fam, partial, f"pose {c:.1f} mm"))
        return
    pc = path_clearance(last, pose)
    if pc < MIN_PATH_MM:
        # go through the zero pose instead
        zc = path_clearance(last, full({}))
        pc2 = path_clearance(full({}), pose)
        if min(zc, pc2) < MIN_PATH_MM:
            dropped.append((fam, partial, f"path {pc:.1f} mm"))
            return
        steps.append({"op": "joints", "family": "via_zero", "joints": full({}), "dwell_s": 0.5, "speed": SPEED,
                      "clearance_mm": round(clearance(full({})), 1)})
        pc = min(zc, pc2)
    steps.append({"op": "joints", "family": fam, "joints": pose, "dwell_s": dwell, "speed": SPEED,
                  "clearance_mm": round(c, 1), "path_clearance_mm": round(pc, 1)})
    last = pose
    poses_since_zero += 1
    if poses_since_zero >= 12:
        zero_step()


def zero_step(dwell=8.0, fam="zero"):
    global poses_since_zero, last
    pc = path_clearance(last, full({}))
    assert pc >= MIN_PATH_MM, pc
    steps.append({"op": "joints", "family": fam, "joints": full({}), "dwell_s": dwell, "speed": SPEED,
                  "clearance_mm": round(clearance(full({})), 1), "path_clearance_mm": round(pc, 1)})
    last = full({})
    poses_since_zero = 0


# F0 baseline
zero_step(120.0, "baseline")
# F1 servo torque
for _ in range(2):
    steps.append({"op": "torque", "family": "torque", "state": "off", "dwell_s": 30.0})
    steps.append({"op": "torque", "family": "torque", "state": "on", "dwell_s": 30.0})
zero_step(10.0)
# F2 each stepper axis 5 mm toward the palm's centre and back, fingers straight
for f in F:
    x0, y0 = HOME_PALM[f]
    for axis in ("x", "y"):
        dx = 5.0 if axis == "x" else 0.0
        dy = 5.0 if axis == "y" else 0.0
        sx = -1.0 if x0 > 0 else 1.0
        sy = -1.0 if y0 > 0 else 1.0
        steps.append({"op": "mounts", "family": "stepper", "mounts": {f: {"x": x0 + sx * dx, "y": y0 + sy * dy}},
                      "dwell_s": 20.0})
        steps.append({"op": "mounts", "family": "stepper", "mounts": {f: {"x": x0, "y": y0}}, "dwell_s": 20.0})
zero_step(10.0)
# F3 each finger alone, inside each servo's range
MCP_MAX = {"thumb": 88, "index": 64, "middle": 69}
PIP_MIN = {"thumb": -10, "index": 0, "middle": -10}
PIP_MAX = {"thumb": 85, "index": 90, "middle": 76}


def span(lo, hi, n):
    return [round(float(v), 1) for v in np.linspace(lo, hi, n)]


for f in F:
    for yaw in range(-40, 41, 10):
        pose_step(f"{f}_yaw", {f: {"yaw": yaw}})
    zero_step()
    for mcp in span(-10, MCP_MAX[f], 10):
        pose_step(f"{f}_mcp", {f: {"mcp": mcp}})
    zero_step()
    for pip in span(PIP_MIN[f], PIP_MAX[f], 10):
        pose_step(f"{f}_pip", {f: {"pip": pip}})
    zero_step()
    for i, mcp in enumerate(span(0, MCP_MAX[f], 5)):
        pips = span(0, PIP_MAX[f], 5)
        for pip in (pips if i % 2 == 0 else pips[::-1]):
            pose_step(f"{f}_mcp_pip", {f: {"mcp": mcp, "pip": pip}})
    zero_step()
    for i, yaw in enumerate((-30, -15, 0, 15, 30)):
        mcps = span(0, MCP_MAX[f], 5)
        for mcp in (mcps if i % 2 == 0 else mcps[::-1]):
            pose_step(f"{f}_yaw_mcp", {f: {"yaw": yaw, "mcp": mcp}})
    zero_step()
# F4 thumb and middle toward each other: one parked, the other moving, then both
PARK = {"thumb": (60, 75, 88), "middle": (45, 60, 69)}
MOVE = {"thumb": (30, 45, 60, 70, 80, 88, 80, 60, 30), "middle": (20, 35, 50, 60, 69, 60, 45, 20)}
for parked, moving in (("middle", "thumb"), ("thumb", "middle")):
    for p in PARK[parked]:
        for mv in MOVE[moving]:
            pose_step(f"tm_{moving}_to_{parked}{p}", {parked: {"mcp": p}, moving: {"mcp": mv}})
        zero_step()
for a, b in itertools.product((60, 70, 80, 88), (45, 55, 63, 69)):
    pose_step("tm_grid", {"thumb": {"mcp": a}, "middle": {"mcp": b}})
zero_step()
for a, b in itertools.product((0, 20, 40), repeat=2):
    pose_step("tm_pip", {"thumb": {"mcp": 80, "pip": a}, "middle": {"mcp": 65, "pip": b}})
zero_step()
for ty, my in itertools.product((-20, 0, 20), repeat=2):
    pose_step("tm_yaw", {"thumb": {"mcp": 80, "yaw": ty}, "middle": {"mcp": 65, "yaw": my}})
zero_step(60.0, "baseline_end")

dur = sum(s.get("dwell_s", 0) for s in steps) + sum(1.2 for s in steps if s["op"] == "joints") + \
    sum(8.0 for s in steps if s["op"] == "mounts")
fam = {}
for s in steps:
    fam[s["family"]] = fam.get(s["family"], 0) + 1
plan = {"layout": "home", "mounts_palm_mm": HOME_PALM, "min_pose_mm": MIN_POSE_MM, "min_path_mm": MIN_PATH_MM,
        "scene": "assets/mjcf/experimental/20261030-cad_v2/scene_screwdriver_medium.xml", "steps": steps,
        "dropped": [[a, b, c] for a, b, c in dropped]}
json.dump(plan, open(sys.argv[1], "w"), indent=1)
print(f"{len(steps)} steps, about {dur / 60:.0f} min; families {fam}")
print(f"closest pose {min(s['clearance_mm'] for s in steps if 'clearance_mm' in s)} mm; "
      f"closest move {min(s.get('path_clearance_mm', 99) for s in steps)} mm; dropped {len(dropped)}: {dropped[:8]}")
