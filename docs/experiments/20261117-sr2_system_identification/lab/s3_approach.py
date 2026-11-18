"""The arm's approach to the owner's S3 block (2026-11-18 15:3x-15:4x), so that one finger's pad rests a few mm from a
plate of the block before s3_press.py presses it. The block, as the owner described it: a 20 x 20 mm 80/20 post about
128 mm tall standing on the table, two 45 x 45 x 1 mm plates glued on opposite faces of it, their top edges about 128 mm
above the table, "approximately centered but not perfectly"; tag36h11 id 0 (40 mm) on one plate, so that the other
plate's outer face is 22 mm behind the tag's face, and grip tape on that plate; a second 80/20 piece joined to the
post's foot as an L runs along the table and is clamped to it. The owner: "you should be able to model from this a safe
approach, at least on the approach axis, with expected noise laterally"; "the apriltag is not quite perfectly parallel
to the ground"; "the thumb will have to slide against the apriltag end".

The model sits on the tag's pose (s3_press.locate_block: both cameras' corners triangulated, or the tracker's solve with
the 40 mm edge): s along the tag's outward normal made level (0 at the tag's face), l along the plates' level width (0
at the tag's centre), z_B. Boxes: the tag plate s [-1, 0], the post s [-21, -1] and l [-10, 10] from the table to the
top, the pad plate s [-22 - tape, -21], the plates l [-22.5, 22.5] and z from the top less 45 mm to the top. Each box
grows by its noise: --noise-l along the width (the plates' and the post's centring), --noise-z up and down, --noise-n
along the normal. The base and its clamp are not modelled: every finger part stays --base-clear above the table
after the rise (the plates' bottom edge is 83 mm up; the clamp must stand lower than that).

The finger presses from the side its plate faces: index or middle at the pad plate (flexion moves their pads along
-x_P), the thumb at the tag plate (flexion along +x_P), so the hand's x_P should run against the tag's normal; the plan
prints the angle. The pressing finger's pad apex goes to --gap mm off its plate, at the tag centre's height and width
(--dl, --dz shift it). The path keeps the hand's orientation (no wrist turn): straight up until every finger part is
--over mm above the block's top, level to above the target, straight down. Every 2 mm of it, every finger part
(hom_setup.finger_pts: the 11 mm tip pad, the tip's end, the PIP joint, at the station's mounts and the fingers' present
joints) is checked against the boxes and the table.

    /usr/bin/python3 s3_approach.py plan --finger index [--gap 4] [--tape 1] [--out OUT]
        no motion: locates the block, prints the model, the target, the three legs and their least clearances, and
        writes OUT/<stamp>_approach_<finger>.json
    /usr/bin/python3 s3_approach.py go --plan PLAN [--go]
        the three legs with the HEX-E guarding each (hom_setup.Arm.move); refuses if the arm, the mounts, the joints or
        the block moved since the plan
    /usr/bin/python3 s3_approach.py back --plan PLAN [--go]
        the same legs in reverse, from the plan's target to its start
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s3_press as S3  # noqa: E402

ROOT = S3.ROOT
FINGERS, JOINTS = S3.FINGERS, S3.JOINTS
TOP = 128.0                      # mm above the table: the plates' top edges and the post's top (owner, 15:4x)
PLATE, PLATE_T = 45.0, 1.0       # mm: the plates' side and thickness
POST = 20.0                      # mm: the 80/20 post
TABLE_Z = 3.0


def model(loc, tape, noise_l, noise_z, noise_n, table_z=TABLE_Z):
    """(p, nh, w, boxes): the tag's centre p (B), its outward normal made level nh, the level width w = up x nh, and
    the boxes {name: ((s0, s1), (l0, l1), (z0, z1))} grown by their noise."""
    p, n, r, d = S3.block_frame_B(loc)
    up = np.array([0.0, 0.0, 1.0])
    nh = n - (n @ up) * up
    nh /= np.linalg.norm(nh)
    w = np.cross(up, nh)
    top, hw = table_z + TOP, PLATE / 2
    zp = (top - PLATE - noise_z, top + noise_z)
    boxes = {"tag plate": ((-PLATE_T - noise_n, noise_n), (-hw - noise_l, hw + noise_l), zp),
             "post": ((-S3.BEHIND + PLATE_T - noise_n, -PLATE_T + noise_n), (-POST / 2 - noise_l, POST / 2 + noise_l),
                      (table_z, top + noise_z)),
             "pad plate": ((-S3.BEHIND - tape - noise_n, -S3.BEHIND + PLATE_T + noise_n), (-hw - noise_l, hw + noise_l),
                           zp)}
    return p, nh, w, boxes


def box_gap(x, p, nh, w, box):
    """The distance (mm) from point x (B) to the box; 0 inside."""
    v = x - p
    c = np.array([v @ nh, v @ w, x[2]])
    lo, hi = np.array([b[0] for b in box]), np.array([b[1] for b in box])
    return float(np.linalg.norm(c - np.clip(c, lo, hi)))


def hand_spheres(HS, M, joints, T_BP):
    """[(finger, part, centre in B, radius)]: hom_setup.finger_pts of each finger at its joints on the mounts M."""
    plan = {"meta": {"hom": {"d6_P_mm": [*M["thumb"], *M["index"], *M["middle"]]}}}
    out = []
    for f in FINGERS:
        for part, (pt, rad) in zip(HS.PARTS, HS.finger_pts(plan, f, joints[f])):
            out.append((f, part, T_BP[:3, :3] @ pt + T_BP[:3, 3], rad))
    return out


def shifted(T, dv):
    T2 = T.copy()
    T2[:3, 3] = T[:3, 3] + dv
    return T2


def leg_clearance(HS, M, joints, T_BP0, d0, d1, mdl, table_z, base_clear, press, step=2.0):
    """The least clearances along the hand's straight translation from T_BP0 + d0 to T_BP0 + d1: to each box (mm and
    the finger part with the leg's fraction), to the table, and above table_z + base_clear; the pressing finger's tip
    pad against its own plate and the post behind it (press = (finger, plate)) apart, as "press pad"."""
    p, nh, w, boxes = mdl
    n = max(1, int(np.linalg.norm(d1 - d0) / step))
    least = {k: (math.inf, None) for k in (*boxes, "press pad", "table", "base layer")}
    for i in range(n + 1):
        u = i / n
        T = shifted(T_BP0, d0 + u * (d1 - d0))
        for f, part, c, rad in hand_spheres(HS, M, joints, T):
            for name, box in boxes.items():
                g = box_gap(c, p, nh, w, box) - rad
                key = "press pad" if f == press[0] and part == "tip pad" and name in (press[1], "post") else name
                if g < least[key][0]:
                    least[key] = (g, f"{f} {part} at {u:.2f}")
            g = float(c[2] - rad - table_z)
            if g < least["table"][0]:
                least["table"] = (g, f"{f} {part} at {u:.2f}")
            g = float(c[2] - rad - table_z - base_clear)
            if g < least["base layer"][0]:
                least["base layer"] = (g, f"{f} {part} at {u:.2f}")
    return {k: {"mm": round(v[0], 2), "where": v[1]} for k, v in least.items()}


def state(HS, a):
    import s1b_tags as S1
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    M = S1.model_mounts(HS.load_plan(str(S3.PLAN)), s["mount_positions"])
    return st, s, M, HS.sim_joints(s)


def cmd_plan(a):
    FT, HS, SL = S3.imports()
    import s1b_tags as S1
    T_BP, mode = S1.arm_T_BP(FT, HS)
    st, s, M, joints = state(HS, a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{time.strftime('%Y%m%d-%H%M%S')}_approach_{a.finger}"
    loc = S3.locate_block(FT, a.edge, a.seconds, frames_to=str(stem))
    if S3.block_frame_B(loc) is None:
        raise SystemExit(f"the block's tag is not in view (detections {loc['dets']}, camera errors "
                         f"{loc['camera_errors']}); frames {list(loc['frames'].values())}")
    src = "triangulated" if loc["tri"] else f"one camera's solve with the {a.edge:g} mm edge"
    noise_n = a.noise_n if a.noise_n is not None else (0.5 if loc["tri"] else 2.0)
    mdl = model(loc, a.tape, a.noise_l, a.noise_z, noise_n)
    p, nh, w, boxes = mdl
    top = TABLE_Z + TOP
    xP = T_BP[:3, 0] - (T_BP[:3, 0] @ np.array([0, 0, 1.0])) * np.array([0, 0, 1.0])
    xP /= np.linalg.norm(xP)
    off = math.degrees(math.acos(float(np.clip(-xP @ nh, -1, 1))))
    print(f"arm mode {mode}; palm origin B {np.round(T_BP[:3, 3], 1).tolist()}; tag ({src}): centre B "
          f"{np.round(p, 1).tolist()}, {p[2] - TABLE_Z:.1f} mm above the table (a centred tag on plates topped at "
          f"{TOP:g} mm: {TOP - PLATE / 2:.1f}); level normal B {np.round(nh, 3).tolist()}, tilt "
          f"{math.degrees(math.asin(float(np.clip(S3.block_frame_B(loc)[1][2], -1, 1)))):+.1f} deg")
    print(f"the hand's x_P runs {off:.1f} deg off the tag's inward normal (0: the flexion presses head on)")
    q = [float(joints[a.finger][k]) for k in JOINTS]
    _, _, tip, con = FT.fk(a.finger, M[a.finger], q)
    apex_B = T_BP[:3, :3] @ con + T_BP[:3, 3]
    s_t = a.gap if a.finger == "thumb" else -S3.BEHIND - a.tape - a.gap
    target = p + s_t * nh + a.dl * w + (a.dz + 0.0) * np.array([0, 0, 1.0])
    delta = target - apex_B
    T_end = shifted(T_BP, delta)
    lows = [c[2] - rad for _, _, c, rad in hand_spheres(HS, M, joints, T_BP)]
    rise = max(0.0, top + a.noise_z + a.over - min(lows))
    rise = max(rise, delta[2] + 0.0)                                 # never a leg that climbs on the level move
    legs = [("rise", np.zeros(3), np.array([0.0, 0.0, rise])),
            ("level", np.array([0.0, 0.0, rise]), np.array([delta[0], delta[1], rise])),
            ("descend", np.array([delta[0], delta[1], rise]), delta)]
    rec = {"what": "S3 approach plan", "t": time.time(), "finger": a.finger, "gap_mm": a.gap, "tape_mm": a.tape,
           "noise_mm": {"l": a.noise_l, "z": a.noise_z, "n": noise_n}, "dl": a.dl, "dz": a.dz, "over": a.over,
           "base_clear": a.base_clear, "edge_mm": a.edge, "locate": loc, "arm_mode": mode, "T_BP0": T_BP.tolist(),
           "T_BE0": (T_BP @ np.linalg.inv(HS.T_EP)).tolist(), "mounts_station": s.get("mount_positions"), "mounts_P": M,
           "joints": joints, "tag_B": p.tolist(), "nh_B": nh.tolist(), "w_B": w.tolist(),
           "boxes": {k: v for k, v in boxes.items()}, "x_P_off_deg": off, "apex_B": apex_B.tolist(),
           "target_B": target.tolist(), "delta_mm": delta.tolist(), "legs": []}
    ok = True
    print(f"target: the {a.finger}'s pad apex at s {s_t:+.1f}, l {a.dl:+.1f} mm, z_B {target[2]:.1f} "
          f"({target[2] - TABLE_Z:.1f} above the table); the hand moves {np.round(delta, 1).tolist()} mm in B, rising "
          f"{rise:.1f} mm first so that every finger part clears the block's top ({top - TABLE_Z:g} mm) by "
          f"{a.over:g} mm; noise l {a.noise_l:g}, z {a.noise_z:g}, n {noise_n:g} mm")
    press = (a.finger, "tag plate" if a.finger == "thumb" else "pad plate")
    for name, d0, d1 in legs:
        cl = leg_clearance(HS, M, joints, T_BP, d0, d1, mdl, TABLE_Z, a.base_clear, press)
        if name == "rise":
            cl["base layer"]["mm"] = None                            # starts at the present height
        # the pressing pad ends --gap off its plate (its 11 mm ball 0.45 mm past the apex, less the normal's noise)
        floor = {"base layer": 0.0, "press pad": 0.5}
        bad = [k for k, v in cl.items() if v["mm"] is not None and v["mm"] < floor.get(k, a.min_clear)]
        ok &= not bad
        rec["legs"].append({"name": name, "from_mm": d0.tolist(), "to_mm": d1.tolist(), "clear": cl, "bad": bad})
        print(f"  {name:7s} {np.round(d0, 1).tolist()} -> {np.round(d1, 1).tolist()} mm: "
              + "; ".join(f"{k} {num(v['mm'])} ({v['where']})" for k, v in cl.items())
              + (f"  -- TOO CLOSE: {bad}" if bad else ""))
    rec["T_BE_end"] = (T_end @ np.linalg.inv(HS.T_EP)).tolist()
    rec["ok"] = ok
    stem.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=float))
    print(("plan clear" if ok else "plan NOT clear") + f" -> {stem}.json")
    return 0 if ok else 1


def num(x):
    return "-" if x is None else f"{x:+.1f}"


def check_same(a, rec, FT, HS, at):
    """Refuses unless the arm is at `at` of the plan (T_BP0 or the target), the station's mounts and the joints are the
    plan's, and the block's tag is within 2 mm of where the plan saw it."""
    import s1b_tags as S1
    T_BP, mode = S1.arm_T_BP(FT, HS)
    if mode != 7:
        raise SystemExit(f"the arm is not running (robot mode {mode}): release the brakes on the pendant (ON, START)")
    want = np.array(rec["T_BP0"]) if at == "start" else shifted(np.array(rec["T_BP0"]), np.array(rec["delta_mm"]))
    dp = float(np.linalg.norm(T_BP[:3, 3] - want[:3, 3]))
    dr = math.degrees(math.acos(float(np.clip((np.trace(T_BP[:3, :3].T @ want[:3, :3]) - 1) / 2, -1, 1))))
    if dp > 2.0 or dr > 0.5:
        raise SystemExit(f"the hand is {dp:.1f} mm and {dr:.2f} deg from the plan's {at}")
    st, s, M, joints = state(HS, a)
    for f in FINGERS:
        if any(abs(joints[f][k] - rec["joints"][f][k]) > 1.0 for k in JOINTS):
            raise SystemExit(f"the {f}'s joints moved since the plan: {joints[f]} against {rec['joints'][f]}")
        if np.linalg.norm(np.subtract(M[f], rec["mounts_P"][f])) > 0.5:
            raise SystemExit(f"the {f}'s mount moved since the plan")
    loc = S3.locate_block(FT, rec["edge_mm"], 2.0)
    fr = S3.block_frame_B(loc)
    if fr is None:
        raise SystemExit("the block's tag is not in view")
    moved = float(np.linalg.norm(fr[0] - np.array(rec["tag_B"])))
    if moved > 2.0:
        raise SystemExit(f"the block's tag moved {moved:.1f} mm since the plan: plan again")
    print(f"checks: hand {dp:.2f} mm / {dr:.2f} deg from the plan's {at}, mounts and joints as planned, tag {moved:.2f} mm "
          f"from the plan's ({loc['dets']})")


def run_legs(a, back):
    FT, HS, SL = S3.imports()
    rec = json.loads(Path(a.plan).read_text())
    if not rec.get("ok"):
        raise SystemExit("the plan is not clear")
    T0 = np.array(rec["T_BP0"])
    legs = [(lg["name"], np.array(lg["from_mm"]), np.array(lg["to_mm"])) for lg in rec["legs"]]
    if back:
        legs = [(f"back {n}", d1, d0) for n, d0, d1 in reversed(legs)]
    v = {"rise": a.v_up, "level": a.v_level, "descend": a.v_down}
    print(f"{rec['finger']}: " + "; ".join(f"{n} {np.round(d1 - d0, 1).tolist()} mm" for n, d0, d1 in legs))
    check_same(a, rec, FT, HS, "target" if back else "start")
    if not a.go:
        return 0
    arm = HS.Arm(a.ur_ft, a.blank_s)
    try:
        arm.connect(True)
        for name, d0, d1 in legs:
            T_goal = shifted(T0, d1) @ np.linalg.inv(HS.T_EP)
            dist = float(np.linalg.norm(d1 - d0))
            vv = v[name.split()[-1]]
            stop_n = a.stop_down if name.endswith("descend") or name == "back descend" else a.stop_n
            why, moved = arm.move(T_goal, vv, a.acc, stop_n, dist / (1e3 * vv) + 6.0, f"approach {name}")
            if why:
                print(f"stopped in {name}: {why}; the arm holds there", flush=True)
                return 1
    finally:
        arm.close()
    print("done", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "go", "back"):
        p = sub.add_parser(name)
        p.add_argument("--station", default="http://10.99.99.2:8765")
        p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
        if name == "plan":
            p.add_argument("--finger", required=True, choices=FINGERS)
            p.add_argument("--gap", type=float, default=4.0, help="mm from the pad's apex to its plate")
            p.add_argument("--tape", type=float, default=1.0, help="mm of grip tape on the pad plate")
            p.add_argument("--dl", type=float, default=0.0, help="mm along the plates' width from the tag's centre")
            p.add_argument("--dz", type=float, default=0.0, help="mm up from the tag's centre")
            p.add_argument("--noise-l", type=float, default=4.0)
            p.add_argument("--noise-z", type=float, default=3.0)
            p.add_argument("--noise-n", type=float, help="default 0.5 mm triangulated, 2 mm from one camera")
            p.add_argument("--over", type=float, default=25.0, help="mm above the block's top on the level leg")
            p.add_argument("--base-clear", type=float, default=70.0, help="mm above the table after the rise")
            p.add_argument("--min-clear", type=float, default=5.0)
            p.add_argument("--edge", type=float, default=S3.BLOCK_EDGE)
            p.add_argument("--seconds", type=float, default=3.0)
            p.add_argument("--out", default=str(ROOT / "logs" / time.strftime("%Y%m%d-sr2_s3_press")))
        else:
            p.add_argument("--plan", required=True)
            p.add_argument("--v-up", type=float, default=0.015, help="m/s")
            p.add_argument("--v-level", type=float, default=0.015, help="m/s")
            p.add_argument("--v-down", type=float, default=0.008, help="m/s")
            p.add_argument("--acc", type=float, default=0.05, help="m/s^2")
            p.add_argument("--stop-n", type=float, default=3.0, help="N of force change that stops the rise and level legs")
            p.add_argument("--stop-down", type=float, default=2.0, help="N that stops the descent")
            p.add_argument("--ur-ft", action="store_true")
            p.add_argument("--blank-s", type=float, default=0.4)
            p.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if a.cmd == "plan":
        return cmd_plan(a)
    return run_legs(a, back=a.cmd == "back")


if __name__ == "__main__":
    sys.exit(main())
