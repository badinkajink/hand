"""A plan that goes on with another after a stop. It moves the gantries from the home layout (where homing leaves them)
to the layout in force at the step it resumes from, takes a baseline there, then runs the plan from that step.
--middle-yaw-min leaves out the random poses that send the middle finger's yaw below it. On 11-05 the middle's yaw servo
pushed at -210 to -345 (load) past about -30 deg, short by 4 to 6 deg, and plan_layouts_s1 stopped at step 32 with it
6.2 deg short of -34 deg.

Each move between two poses that a removed pose used to separate is checked again along the straight joint-space line
in the CAD hand (make_layouts.move_clearance, MIN_MOVE). A move that does not clear goes through the zero pose: the plan
already checks every pose's way to zero.

    uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python resume_plan.py PLAN OUT FROM_STEP \\
        [--middle-yaw-min -25]
"""
import argparse
import json

from make_layouts import HOME_PALM, MIN_MOVE, ZERO, gantry_path, move_clearance

ap = argparse.ArgumentParser()
ap.add_argument("plan")
ap.add_argument("out")
ap.add_argument("from_step", type=int)
ap.add_argument("--middle-yaw-min", type=float, default=None)
a = ap.parse_args()
plan = json.load(open(a.plan))
steps = plan["steps"]
mounts = [i for i in range(a.from_step + 1) if steps[i]["op"] == "mounts"]
lay = steps[mounts[-1]]
palm = {f: (m["x"], m["y"]) for f, m in lay["mounts"].items()}
g = gantry_path(HOME_PALM, palm)
assert g >= 12.0, g
name = lay["family"].removeprefix("layout_")
out = [{"op": "joints", "family": "zero", "joints": ZERO, "dwell_s": 5.0, "speed": 60},
       dict(lay, gantry_clearance_mm=round(g, 1)),
       {"op": "joints", "family": f"baseline_{name}", "joints": ZERO, "dwell_s": 30.0, "speed": 80}]
dropped, checked, via_zero = [], [], []
prev, prev_orig = ZERO, None          # the pose before, as run, and its index in PLAN
for i in range(a.from_step, len(steps)):
    s = steps[i]
    if s["op"] == "mounts":
        palm = {f: (m["x"], m["y"]) for f, m in s["mounts"].items()}
        out.append(s)
        prev, prev_orig = ZERO, i
        continue
    if (a.middle_yaw_min is not None and s["family"].startswith("random")
            and s["joints"]["middle"]["yaw"] < a.middle_yaw_min):
        dropped.append(i)
        continue
    # the move from the pose before was checked in PLAN only if that pose came just before this one there
    if prev_orig != i - 1 or i == a.from_step:
        mc = move_clearance(palm, prev, s["joints"])
        checked.append((i, round(mc, 1)))
        if mc < MIN_MOVE:
            z1, z2 = move_clearance(palm, prev, ZERO), move_clearance(palm, ZERO, s["joints"])
            assert min(z1, z2) >= MIN_MOVE, (i, z1, z2)
            out.append({"op": "joints", "family": "zero", "joints": ZERO, "dwell_s": 3.0, "speed": 80})
            via_zero.append(i)
            mc = z2
        s = dict(s, move_clearance_mm=round(mc, 1))
    out.append(s)
    prev, prev_orig = s["joints"], i
report = {"from_plan": a.plan, "from_step": a.from_step, "layout": name, "gantry_from_home_mm": round(g, 1),
          "middle_yaw_min": a.middle_yaw_min, "dropped": len(dropped), "moves_checked": len(checked),
          "least_checked_mm": min((c for _, c in checked), default=None), "via_zero": via_zero}
json.dump({"resumes": report, "layouts": plan.get("layouts"), "home_palm_mm": plan.get("home_palm_mm"),
           "min_pose_mm": plan.get("min_pose_mm"), "min_move_mm": plan.get("min_move_mm"), "scene": plan.get("scene"),
           "steps": out}, open(a.out, "w"), indent=1)
dur = sum(s["dwell_s"] + (1.0 if s["op"] == "joints" else 60.0) for s in out)
print(json.dumps(report))
print(f"{len(out)} steps, {sum(s['family'].startswith('random') for s in out)} random poses, about {dur / 60:.0f} min -> {a.out}")
