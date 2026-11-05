"""Random poses of the SR2 hand at several gantry layouts, for the next crosstalk sweep. The owner, 22:10 on 11-04,
relayed: "we could try more thumb-middle poses yea, we can use the steppers to move them to more realistic poses too".
The layouts are the grasp layouts of the transfer study's deploy plans. At each layout the hand goes straight. The
gantries then move one axis at a time in the station's order (thumb, index, middle; each finger's palm y before its
palm x), checked with straight fingers every 2 mm. A baseline at the zero pose follows, then N random poses, about half
of them with the thumb and the middle finger within CLOSE mm of each other.

Every pose clears MIN mm finger to finger in the CAD hand (cad_v2 collision spheres). Every move, and every pose's way
back to the zero pose (where a safety stop sends the hand), clears MIN_MOVE mm along the straight joint-space line.
run_plan.py walks each move along that line in waypoints 3 deg apart, so servos at different rates (the workstation's
replay of 11-04 took 0.6 to 1.4 times) stay near it. The index and the middle keep the night's limits: mcp at most 50
and 62 deg, mcp + pip at most 120 deg. A zero-pose dwell follows every 12 poses. The plan ends back at the home layout.

    uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python make_layouts.py OUT.json SEED N \\
        [LAYOUT ...]    (default: every layout below that clears MIN straight)
"""
import json
import sys

import numpy as np

from handgeo import F, HOME_LOCAL, Hand, ORIGIN

MIN, MIN_MOVE, CLOSE = 12.0, 12.0, 30.0
CURL_MAX = 120.0
LIM = {"thumb": {"yaw": (-40, 40), "mcp": (-10, 88), "pip": (-10, 85)},
       "index": {"yaw": (-40, 40), "mcp": (-10, 50), "pip": (0, 90)},
       "middle": {"yaw": (-40, 40), "mcp": (-10, 62), "pip": (-10, 76)}}
# palm-frame mounts (mm) of the transfer study's deploy plans (docs/experiments/2026083x and 20260902-residual-bench)
LAYOUTS = {
    "g12": {"thumb": (-42.5, 0.0), "index": (42.5, 40.0), "middle": (42.5, -40.0)},
    "rv04_mid": {"thumb": (-35.0, 0.0), "index": (35.0, 40.0), "middle": (35.0, -40.0)},
    "rv05_manual": {"thumb": (-39.0, 0.0), "index": (35.0, 25.0), "middle": (48.5, -35.0)},
    "sv1_w6689": {"thumb": (-54.3, -33.7), "index": (37.9, 42.1), "middle": (39.0, -43.0)},
    "sv1_w0099": {"thumb": (-43.8, -2.3), "index": (41.1, 35.7), "middle": (55.2, -44.2)},
    "sv1_u0308": {"thumb": (-37.8, 4.3), "index": (38.2, 39.1), "middle": (46.9, -76.0)},
    # The HOM task-conflict study's layouts (SCS0009 plant, friction 1.0: docs/experiments/20261103-hom_task_conflict/
    # 20261104-conflict_sv25.json and _refined_sv25.json), symmetric tripods (x, y) placed as fp_sr2_toy.layout2 does:
    # thumb (-x/2, 0), index (x/2, y/2), middle (x/2, -y/2). The owner, 13:59 on 11-05: "why dont we try adding layouts
    # determined from our very extensive recent experiments on hom control?". The pin-push specialist (75, 55) is left
    # out: its straight index and middle clear 5.1 mm, under MIN. plan_layouts_s1 was built with the five layouts above.
    "hom_roll_85_170": {"thumb": (-42.5, 0.0), "index": (42.5, 85.0), "middle": (42.5, -85.0)},
    "hom_press_70_150": {"thumb": (-35.0, 0.0), "index": (35.0, 75.0), "middle": (35.0, -75.0)},
    "hom_generalist_70_110": {"thumb": (-35.0, 0.0), "index": (35.0, 55.0), "middle": (35.0, -55.0)},
    "hom_twist_100_70": {"thumb": (-50.0, 0.0), "index": (50.0, 35.0), "middle": (50.0, -35.0)},
    "hom_cover_50_75": {"thumb": (-25.0, 0.0), "index": (25.0, 37.5), "middle": (25.0, -37.5)},
}
HOME_PALM = {f: (ORIGIN[f][0] + HOME_LOCAL[f][0], ORIGIN[f][1] + HOME_LOCAL[f][1]) for f in F}
ZERO = {f: {"yaw": 0.0, "mcp": 0.0, "pip": 0.0} for f in F}
J = ("yaw", "mcp", "pip")
h = Hand()


def local(palm):
    return {f: (palm[f][0] - ORIGIN[f][0], palm[f][1] - ORIGIN[f][1]) for f in F}


def clearances(palm, pose):
    h.set(local(palm), pose)
    return h.clearances()


def gantry_path(a, b, step=2.0):
    """Least clearance with straight fingers while the station moves a's mounts to b's, axis by axis."""
    cur = {f: list(a[f]) for f in F}
    worst = min(clearances(cur, ZERO).values())
    for f in F:
        for axis in (1, 0):                       # palm y, then palm x
            x0, x1 = cur[f][axis], b[f][axis]
            n = max(1, int(abs(x1 - x0) / step))
            for s in np.linspace(0, 1, n + 1)[1:]:
                cur[f][axis] = x0 + (x1 - x0) * s
                worst = min(worst, min(clearances(cur, ZERO).values()))
    return worst


def move_clearance(palm, a, b, n=16):
    """Least clearance along the straight joint-space line from pose a to pose b. run_plan.py walks every move along
    that line in waypoints 3 deg apart, so servos at different rates leave it by about 1.7 deg at most (0.6 against
    1.4 times), some 2.6 mm at the tip; MIN_MOVE keeps that margin over the line."""
    worst = 1e9
    for s in np.linspace(0, 1, n + 1)[1:]:
        pose = {f: {j: a[f][j] + (b[f][j] - a[f][j]) * s for j in J} for f in F}
        worst = min(worst, min(clearances(palm, pose).values()))
    return worst


def ok_limits(p):
    return all(p[f]["mcp"] + p[f]["pip"] <= CURL_MAX for f in ("index", "middle"))


def main():
    out, seed, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    names = sys.argv[4:] or list(LAYOUTS)
    rng = np.random.default_rng(seed)
    steps, report = [], {}
    here = HOME_PALM
    zero_step = lambda dwell, fam: {"op": "joints", "family": fam, "joints": ZERO, "dwell_s": dwell, "speed": 80}
    for name in names:
        palm = LAYOUTS[name]
        straight = clearances(palm, ZERO)
        if min(straight.values()) < MIN:
            report[name] = {"skipped": f"straight fingers clear {min(straight.values()):.1f} mm"}
            continue
        g = gantry_path(here, palm)
        if g < MIN:
            report[name] = {"skipped": f"gantry move clears {g:.1f} mm"}
            continue
        steps.append(zero_step(5.0, "zero"))
        steps.append({"op": "mounts", "family": f"layout_{name}", "mounts": {f: {"x": palm[f][0], "y": palm[f][1]}
                                                                           for f in F},
                      "dwell_s": 5.0, "gantry_clearance_mm": round(g, 1)})
        steps.append(zero_step(30.0, f"baseline_{name}"))
        last, since, close, far, tries, cl = ZERO, 0, 0, 0, 0, []
        while close + far < n and tries < 200 * n:
            tries += 1
            p = {f: {j: round(float(rng.uniform(*LIM[f][j])), 1) for j in J} for f in F}
            if not ok_limits(p):
                continue
            c = clearances(palm, p)
            if min(c.values()) < MIN:
                continue
            is_close = c["thumb-middle"] <= CLOSE
            if (is_close and close >= n // 2) or (not is_close and far >= n - n // 2):
                continue
            mc = move_clearance(palm, last, p)
            if mc < MIN_MOVE:
                continue
            # the runner's safety stop sends the hand to zero from any pose, so every pose needs a clear way there
            zc = move_clearance(palm, p, ZERO)
            if zc < MIN_MOVE:
                continue
            steps.append({"op": "joints", "family": f"random_{name}", "joints": p, "dwell_s": 3.0, "speed": 80,
                          "clearance_mm": round(min(c.values()), 1), "thumb_middle_mm": round(c["thumb-middle"], 1),
                          "move_clearance_mm": round(mc, 1), "to_zero_mm": round(zc, 1)})
            cl.append(min(c.values()))
            close, far, last, since = close + is_close, far + (not is_close), p, since + 1
            if since >= 12:
                zc = move_clearance(palm, last, ZERO)
                assert zc >= MIN_MOVE, zc
                steps.append(zero_step(8.0, "zero"))
                last, since = ZERO, 0
        zc = move_clearance(palm, last, ZERO)
        assert zc >= MIN_MOVE, zc
        steps.append(zero_step(8.0, "zero"))
        report[name] = {"straight_mm": round(min(straight.values()), 1), "gantry_mm": round(g, 1), "close": close,
                        "far": far, "draws": tries, "min_pose_mm": round(min(cl), 1) if cl else None}
        here = palm
        print(name, report[name], flush=True)
    g = gantry_path(here, HOME_PALM)
    assert g >= MIN, g
    steps.append({"op": "mounts", "family": "layout_home", "mounts": {f: {"x": HOME_PALM[f][0], "y": HOME_PALM[f][1]}
                                                                    for f in F}, "dwell_s": 5.0,
                  "gantry_clearance_mm": round(g, 1)})
    steps.append(zero_step(30.0, "baseline_end"))
    json.dump({"layouts": {k: LAYOUTS[k] for k in report if "skipped" not in report[k]}, "home_palm_mm": HOME_PALM,
               "min_pose_mm": MIN, "min_move_mm": MIN_MOVE, "close_mm": CLOSE, "curl_max_deg": CURL_MAX, "seed": seed,
               "report": report, "steps": steps,
               "scene": "assets/mjcf/experimental/20261030-cad_v2/scene_screwdriver_medium.xml"}, open(out, "w"), indent=1)
    dur = sum(s["dwell_s"] + (1.0 if s["op"] == "joints" else 60.0) for s in steps)
    print(f"{len(steps)} steps, about {dur / 60:.0f} min -> {out}")


if __name__ == "__main__":
    main()
