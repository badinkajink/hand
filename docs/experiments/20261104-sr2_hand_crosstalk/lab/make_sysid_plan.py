"""The free-space identification of 11-06 on the SR2 hand at the gantries' home layout, as plans for run_plan.py, with
their clearance check in the CAD hand (handgeo's cad_v2 spheres) at four positions of the middle finger's gantry. The
owner, leaving at 12:2x on 11-06: "maybe we can double-check our sysid? the hands are only in free space right now, so we
should be able to reproduce the previous free-space only sysid". The coordinator's review of 12:43 set the blocks, their
order and the checks.

One plan a block. Every joint not named stays at zero; one joint moves at a time, at speed 80:
  a1, a2  09-02's staircase (scripts/archive/servo_hysteresis_sweep.py) on each finger's mcp and pip: 5 to 61 deg and
          back in 4 deg steps, the top commanded twice, the index's mcp to 49; dwells 0.45 s (a1, as on 09-02) and 2 s
          (a2, the steady droop)
  b1, b2  the same staircase on each yaw, -36 to +36 deg and back; dwells 0.45 and 2 s
  c       yaw stiction: each yaw 0 -> +A -> 0 -> -A -> 0 for A = 20, 30, 40, each pose held 10 s
  c40     c with the tested finger's mcp at 40 deg
They run in that order with --ramp-deg 0 (one command a step, as on 09-02), --min-clearance 5, --max-blocked 1 and
--keep-home; --max-err-yaw 10 for a and b, 15 for c and c40.

The check takes every step's pose and the move into it, at the middle's home mounts (80, -85) and cover's (25, -37.5),
and at the two corners between them, (80, -37.5) and (25, -85): one of its axes may not have followed 11-05's last move
(joint 4 skipped one at 13:34 on 11-05, and nothing on the hand sees a skip). Thumb and index stand at home. Each joint
lies in a box: a moving joint anywhere along its move, from its runner limit short of the command to 3 deg (yaw) or
1.5 deg (mcp, pip) past it; a held joint within 3 or 1.5 deg of its command. A moving joint is sampled every degree and
the held joints at the corners of their boxes, for each pair of fingers. c40's poses also get the box between them and
the zero pose, where a safety stop sends the hand with yaw and mcp moving at once. run_plan.py's measured-pose gate
computes clearance at the station's mounts (home) only, so the off-home corners rest on this check alone.

    uv run --no-project --python 3.10 --with mujoco --with numpy --with scipy python make_sysid_plan.py OUT_DIR
    python3 make_sysid_plan.py OUT_DIR --tsv-only      (the steps table again, from the plans in OUT_DIR)

OUT_DIR gets one plan a block (20261106-sysid_<block>.json), the check (20261106-sysid_clearance.json and .txt) and
every step in one table, 20261106-sysid_plan_steps.tsv: the plans' JSON are LFS files under docs/, the table is not.
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

if "--tsv-only" not in sys.argv:
    from handgeo import F, ORIGIN, Hand
else:
    F = ("thumb", "index", "middle")

J = ("yaw", "mcp", "pip")
SPEED = 80
STEP = 4.0
TOP = {"mcp": 61.0, "pip": 61.0}
CAP = {("index", "mcp"): 49.0}         # the index's mcp reaches the servo's protective torque near 55 deg (11-04)
AMPS = (20.0, 30.0, 40.0)
HOLD_C, MCP_C40 = 10.0, 40.0
HOME = {"thumb": (-80.0, -55.0), "index": (80.0, 85.0), "middle": (80.0, -85.0)}
MIDDLE = {"home (80, -85)": (80.0, -85.0), "cover (25, -37.5)": (25.0, -37.5),
          "x home, y cover (80, -37.5)": (80.0, -37.5), "x cover, y home (25, -85)": (25.0, -85.0)}
OVER = {"yaw": 3.0, "mcp": 1.5, "pip": 1.5}
HELD = {"yaw": 3.0, "mcp": 1.5, "pip": 1.5}
LIMIT = {"a": {"yaw": 10.0, "mcp": 6.0, "pip": 6.0}, "b": {"yaw": 10.0, "mcp": 6.0, "pip": 6.0},
         "c": {"yaw": 15.0, "mcp": 6.0, "pip": 6.0}}
MIN_MM = 12.0                          # the crosstalk plans' margin at every pose (make_layouts.py)
ZERO = {f: {j: 0.0 for j in J} for f in F}


def pose(**set_):
    """ZERO with finger_joint=value pairs set, e.g. pose(index_mcp=40)."""
    out = {f: dict(ZERO[f]) for f in F}
    for k, v in set_.items():
        f, j = k.split("_")
        out[f][j] = float(v)
    return out


def step(block, finger, joint, direction, cmd, dwell, joints):
    return {"op": "joints", "family": f"{block}_{finger}_{joint}_{direction}", "block": block, "finger": finger,
            "joint": joint, "dir": direction, "cmd": cmd, "joints": joints, "dwell_s": dwell, "speed": SPEED}


def staircase(block, joints_of, dwell):
    steps = []
    for f, j in joints_of:
        lo, hi = (-36.0, 36.0) if j == "yaw" else (5.0, CAP.get((f, j), TOP[j]))
        ups = [lo + i * STEP for i in range(int((hi - lo) / STEP) + 1)]
        steps.append(step(block, f, j, "start", lo, 1.0, pose(**{f"{f}_{j}": lo})))
        steps += [step(block, f, j, "up", v, dwell, pose(**{f"{f}_{j}": v})) for v in ups]
        steps += [step(block, f, j, "down", v, dwell, pose(**{f"{f}_{j}": v})) for v in ups[::-1]]
        steps.append(step(block, f, j, "park", 0.0, 1.0, pose()))
    return steps


def stiction(block, mcp):
    steps = []
    for f in F:
        if mcp:
            steps.append(step(block, f, "mcp", "on", mcp, 3.0, pose(**{f"{f}_mcp": mcp})))
        for amp in AMPS:
            for direction, v in (("out", amp), ("in", 0.0), ("out", -amp), ("in", 0.0)):
                steps.append(step(block, f, "yaw", direction, v, HOLD_C, pose(**{f"{f}_yaw": v, f"{f}_mcp": mcp})))
        if mcp:
            steps.append(step(block, f, "mcp", "off", 0.0, 3.0, pose()))
    return steps


FLEX = [(f, j) for f in F for j in ("mcp", "pip")]
YAWS = [(f, "yaw") for f in F]
BLOCKS = {"a1": staircase("a1", FLEX, 0.45), "b1": staircase("b1", YAWS, 0.45),
          "a2": staircase("a2", FLEX, 2.0), "b2": staircase("b2", YAWS, 2.0),
          "c": stiction("c", 0.0), "c40": stiction("c40", MCP_C40)}


def in_range(p):
    """The crosstalk plans' limits (make_layouts.py LIM, the night's caps of 11-04)."""
    lim = {"thumb": {"mcp": (-10, 88), "pip": (-10, 85)}, "index": {"mcp": (-10, 50), "pip": (0, 90)},
           "middle": {"mcp": (-10, 62), "pip": (-10, 76)}}
    return all(abs(p[f]["yaw"]) <= 40 and all(lim[f][j][0] <= p[f][j] <= lim[f][j][1] for j in ("mcp", "pip"))
               and (f == "thumb" or p[f]["mcp"] + p[f]["pip"] <= 120) for f in F)


def box(prev, cur, limit):
    """Each joint's interval: a moving joint over its move, from its limit short of the command (toward where it came
    from) to OVER past it; a held joint within HELD of its command."""
    iv = {}
    for f in F:
        for j in J:
            p, c = prev[f][j], cur[f][j]
            if p == c:
                iv[f, j] = (c - HELD[j], c + HELD[j], False)
            else:
                s = 1.0 if c > p else -1.0
                pts = (p, c, c - s * limit[j], c + s * OVER[j])
                iv[f, j] = (min(pts), max(pts), True)
    return iv


def values(lo, hi, moving, res):
    return list(np.arange(lo, hi, res)) + [hi] if moving else [lo, hi]


def local(middle_palm):
    palm = dict(HOME, middle=middle_palm)
    return {f: (palm[f][0] - ORIGIN[f][0], palm[f][1] - ORIGIN[f][1]) for f in F}


class Check:
    """handgeo's clearance between two fingers (centre distance less both radii, least over their spheres, mm), the
    squared distances from one matrix product: 0.3 ms against handgeo's 6 ms for 447 x 447 spheres."""

    def __init__(self):
        self.h = Hand()

    def pair(self, a, b):
        A, B = self.h.d.geom_xpos[self.h.geoms[a]] * 1000, self.h.d.geom_xpos[self.h.geoms[b]] * 1000
        d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
        return float((np.sqrt(np.maximum(d2, 0.0)) - self.h.r[a][:, None] - self.h.r[b][None, :]).min())

    def ev(self, loc, q, a, b):
        self.h.set(loc, q)
        return self.pair(a, b)

    def least(self, loc, iv, base, res):
        """The least clearance over the box, per pair of fingers, and the joints that give it. Each held joint goes to
        the end of its interval that brings the pair closer at the commanded pose; the moving joints are sampled every
        res deg; then every corner of the held joints is tried at the closest sample."""
        out = {}
        for a, b in itertools.combinations(F, 2):
            keys = [(f, j) for f in (a, b) for j in J]
            moving = [k for k in keys if iv[k][2]]
            held = [k for k in keys if not iv[k][2]]

            def q_of(setting):
                q = {f: dict(base[f]) for f in F}
                for (f, j), v in setting.items():
                    q[f][j] = float(v)
                return q

            corner = {k: min((iv[k][0], iv[k][1]), key=lambda v: self.ev(loc, q_of({k: v}), a, b)) for k in held}
            best = (np.inf, None)
            for combo in itertools.product(*(values(*iv[k], res) for k in moving)):
                setting = {**corner, **dict(zip(moving, combo))}
                c = self.ev(loc, q_of(setting), a, b)
                if c < best[0]:
                    best = (c, setting)
            at = {k: v for k, v in best[1].items() if k in moving}
            for ends in itertools.product(*((iv[k][0], iv[k][1]) for k in held)):
                setting = {**at, **dict(zip(held, ends))}
                c = self.ev(loc, q_of(setting), a, b)
                if c < best[0] - 1e-9:
                    best = (c, setting)
            out[f"{a}-{b}"] = (best[0], {f"{f}_{j}": round(float(v), 1) for (f, j), v in best[1].items()})
        return out


def moves_of(name, steps):
    """Each step's boxes: the move into it, and for c40's yaw-out poses the box down to the zero pose."""
    limit, prev, out = LIMIT[name[0]], ZERO, []
    for s in steps:
        mv = [(box(prev, s["joints"], limit), 1.0)]
        if name == "c40" and s["dir"] == "out":
            mv.append((box(ZERO, s["joints"], limit), 4.0))
        out.append(mv)
        prev = s["joints"]
    return out


def check_middle(m):
    """Every block's least clearance with the middle at MIDDLE[m]: per step, at the pose and in its boxes."""
    ck, loc, cache, res = Check(), local(MIDDLE[m]), {}, {}
    for name, steps in BLOCKS.items():
        rows = []
        for s, mv in zip(steps, moves_of(name, steps)):
            ck.h.set(loc, s["joints"])
            at = {f"{a}-{b}": ck.pair(a, b) for a, b in itertools.combinations(F, 2)}
            boxed = {}
            for iv, r in mv:
                key = (json.dumps(s["joints"], sort_keys=True), json.dumps(sorted(iv.items())), r)
                if key not in cache:
                    cache[key] = ck.least(loc, iv, s["joints"], r)
                for pair, (c, q) in cache[key].items():
                    if pair not in boxed or c < boxed[pair][0]:
                        boxed[pair] = (c, q)
            rows.append({"at": at, "box": boxed})
        res[name] = rows
    return m, res


def steps_tsv(out):
    """Every step of every plan in out, one row each, with its clearances."""
    rows = ["block\tstep\tfamily\tfinger\tjoint\tdir\tcmd_deg\tdwell_s\tspeed\tclear_home_pose_mm\t" +
            "\t".join(f"box_mm_middle_{m.split('(')[1].rstrip(')').replace(', ', '_')}" for m in MIDDLE)]
    for name in BLOCKS:
        for i, s in enumerate(json.load(open(out / f"20261106-sysid_{name}.json"))["steps"]):
            rows.append("\t".join(str(v) for v in (name, i, s["family"], s["finger"], s["joint"], s["dir"], s["cmd"],
                                                     s["dwell_s"], s["speed"], s["clearance_mm"],
                                                     *(s["clearance_box_mm"][m] for m in MIDDLE))))
    (out / "20261106-sysid_plan_steps.tsv").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    if "--tsv-only" in sys.argv:
        steps_tsv(out)
        raise SystemExit(0)
    from multiprocessing import Pool
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for name, steps in BLOCKS.items():
        assert all(in_range(s["joints"]) for s in steps), name
    with Pool(len(MIDDLE)) as pool:
        per = dict(pool.map(check_middle, list(MIDDLE)))
    print(f"checked in {time.time() - t0:.0f} s", flush=True)
    report = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "made_by": "lab/make_sysid_plan.py",
              "middle_mounts_palm_mm": MIDDLE, "home_palm_mm": HOME,
              "box_deg": {"over": OVER, "held": HELD, "runner_limit": LIMIT}, "margin_mm": MIN_MM, "blocks": {}}
    lines = []
    for name, steps in BLOCKS.items():
        for i, s in enumerate(steps):
            s["clearance_mm"] = round(min(per["home (80, -85)"][name][i]["at"].values()), 1)
            s["clearance_box_mm"] = {m: round(min(c for c, _ in per[m][name][i]["box"].values()), 1) for m in MIDDLE}
        dur = sum(s["dwell_s"] + (1.3 if s["dwell_s"] >= 10 else 0.5) for s in steps)
        limit = LIMIT[name[0]]
        json.dump({"block": name, "made_by": "lab/make_sysid_plan.py", "layout": "home", "mounts_palm_mm": HOME,
                   "scene": "assets/mjcf/experimental/20261030-cad_v2/scene_screwdriver_medium.xml",
                   "runner": {"ramp_deg": 0, "min_clearance": 5, "max_blocked": 1, "keep_home": True,
                              "max_err_yaw": limit["yaw"], "max_err": limit["mcp"]},
                   "steps": steps}, open(out / f"20261106-sysid_{name}.json", "w"), indent=1)
        rep = {"steps": len(steps), "minutes": round(dur / 60, 1), "least": {}}
        lines.append(f"{name}: {len(steps)} steps, about {dur / 60:.1f} min")
        for m in MIDDLE:
            rows = per[m][name]
            ia = min(range(len(rows)), key=lambda i: min(rows[i]["at"].values()))
            pa = min(rows[ia]["at"], key=rows[ia]["at"].get)
            ib = min(range(len(rows)), key=lambda i: min(c for c, _ in rows[i]["box"].values()))
            pb = min(rows[ib]["box"], key=lambda p: rows[ib]["box"][p][0])
            cb, qb = rows[ib]["box"][pb]
            rep["least"][m] = {"at_pose_mm": round(rows[ia]["at"][pa], 1), "pose_pair": pa, "pose_step": ia,
                               "pose_family": steps[ia]["family"], "pose_cmd": steps[ia]["cmd"],
                               "in_box_mm": round(cb, 1), "box_pair": pb, "box_step": ib,
                               "box_family": steps[ib]["family"], "box_cmd": steps[ib]["cmd"], "box_joints": qb}
            lines.append(f"   middle at {m:28s} at the poses {rows[ia]['at'][pa]:5.1f} mm ({pa}, step {ia} "
                         f"{steps[ia]['family']} {steps[ia]['cmd']:+.0f}); in the boxes {cb:5.1f} mm ({pb}, step {ib} "
                         f"{steps[ib]['family']} {steps[ib]['cmd']:+.0f}: {qb})")
        report["blocks"][name] = rep
    allmin = min(b["least"][m]["in_box_mm"] for b in report["blocks"].values() for m in MIDDLE)
    report["least_in_box_mm"], report["passes_margin"] = allmin, bool(allmin >= MIN_MM)
    lines.append(f"least anywhere in the boxes {allmin:.1f} mm; margin {MIN_MM} mm: "
                 f"{'pass' if allmin >= MIN_MM else 'FAIL'}")
    json.dump(report, open(out / "20261106-sysid_clearance.json", "w"), indent=1)
    (out / "20261106-sysid_clearance.txt").write_text("\n".join(lines) + "\n")
    steps_tsv(out)
    print("\n".join(lines))
