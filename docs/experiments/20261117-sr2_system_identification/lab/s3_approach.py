"""The arm's approach to the owner's S3 block (2026-11-18 15:3x-15:4x), so that one finger rests a few mm from a plate of
the block before s3_press.py presses it. The block, as the owner described it: a 20 x 20 mm 80/20 post about 128 mm tall
standing on the table, two 45 x 45 x 1 mm plates glued on opposite faces of it, their top edges about 128 mm above the
table, "approximately centered but not perfectly"; tag36h11 id 0 (40 mm) on one plate, so that the other plate's outer
face is 22 mm behind the tag's face, and grip tape on that plate; a second 80/20 piece joined to the post's foot as an L
runs along the table and is clamped to it. The owner: "you should be able to model from this a safe approach, at least
on the approach axis, with expected noise laterally"; "the apriltag is not quite perfectly parallel to the ground";
"the thumb will have to slide against the apriltag end"; and at 16:0x "we can also rotate the wrist 90 degrees, if
that's necessary to properly evaluate the yaw".

The model sits on the tag's pose (s3_press.locate_block: both cameras' corners triangulated, or the tracker's solve with
the 40 mm edge): s along the tag's outward normal made level (0 at the tag's face), l along the plates' level width (0
at the tag's centre), z_B. Boxes: the tag plate s [-1, 0], the post s [-21, -1] and l [-10, 10] from the table to the
top, the pad plate s [-22 - tape, -21], the plates l [-22.5, 22.5] and z from the top less 45 mm to the top. Each box
grows by its noise: --noise-l along the width (the plates' and the post's centring), --noise-z up and down, --noise-n
along the normal. The base and its clamp are not modelled: every finger part stays --base-clear above the table
after the rise (the plates' bottom edge is 83 mm up; the clamp must stand lower than that).

Two presses. --press flex (default): the finger's flexion presses its pad's apex into the plate it faces, the index
or middle into the taped plate (their flexion moves the pad along -x_P), the thumb into the tag plate (+x_P), with the
hand's orientation kept; the plan prints how far x_P runs off the plates' normal. --press yaw --sign S: the yaw (sign
S moves the pad along S y_P) presses the tip's side into the plate head on (W3, 15:4x: a yaw through the tape's
friction slipped at about 1 N); the hand first turns about its own vertical axis z_P through the palm origin
(pinch_pickup's turn) until S y_P points into the plate. The finger's leading point goes to --gap mm off its plate (the
apex, or for the yaw the 11 mm pad ball's side, which covers the DIP link's 10.5 mm), at the tag centre's height and
width (--dl, --dz shift it). The legs: straight up until every finger part is --over mm above the block's top, the turn
(yaw only), level to above the target, straight down. Every 2 mm or 1 deg of them, every finger part
(hom_setup.finger_pts: the 11 mm tip pad, the tip's end, the PIP joint, at the station's mounts and the fingers'
present joints) is checked against the boxes and the table.

    /usr/bin/python3 s3_approach.py plan --finger index [--press yaw --sign -1] [--gap 4] [--tape 1] [--out OUT]
        no motion: locates the block, prints the model, the target, the legs and their least clearances, and writes
        OUT/<stamp>_approach_<finger>[_yaw].json
    /usr/bin/python3 s3_approach.py go --plan PLAN [--go]
        the legs with the HEX-E guarding each (hom_setup.Arm.move); refuses if the arm, the mounts, the joints or the
        block moved since the plan
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
PAD_R = 11.0                     # mm: hom_setup's tip pad ball (the v4b pad; the DIP link's distal end is 10.5 wide)
LIP = 3.0                        # mm: the lip on the block's top (owner, 16:25: "the dip link collided with the ~3mm
                                 # thick lip on top of the 80-20/top edge of the flat pad")
CAD_XML = ROOT / "assets/mjcf/experimental/20261104-cad_v3/ur5_k1.xml"
CAD_BODIES = ("mount", "yaw_frame", "mcp_frame", "pip_frame")


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
                           zp),
             "lip": ((-S3.BEHIND - tape - noise_n, noise_n), (-hw - noise_l, hw + noise_l), (top, top + LIP + noise_z))}
    return p, nh, w, boxes


def box_gaps(X, R, p, nh, w, box):
    """(clearances, in front of the face): each sphere's distance (mm) to the box less its radius, and whether its
    centre lies within the box's width and height (its nearest point on the box is on a face square to s)."""
    V = X - p
    C = np.stack([V @ nh, V @ w, X[:, 2]], axis=1)
    lo, hi = np.array([b[0] for b in box]), np.array([b[1] for b in box])
    Q = np.clip(C, lo, hi)
    front = (C[:, 1] >= lo[1]) & (C[:, 1] <= hi[1]) & (C[:, 2] >= lo[2]) & (C[:, 2] <= hi[2])
    return np.linalg.norm(C - Q, axis=1) - R, front


_CAD = {}


def cad_spheres(finger):
    """{body: (N, 4) array (x, y, z, r mm) in the body's frame}: the cad_v3 scene's collision spheres on the finger's
    mount, yaw, MCP and PIP (DIP link) bodies (on the CAD hand the DIP link reaches 23 mm to the side by the PIP)."""
    if finger not in _CAD:
        import xml.etree.ElementTree as ET
        names = {f"{finger}_{b}" for b in CAD_BODIES}
        out = {}
        for b in ET.parse(CAD_XML).getroot().iter("body"):
            if b.get("name") in names:
                out[b.get("name")] = np.array([[*(1e3 * float(v) for v in g.get("pos", "0 0 0").split()),
                                                1e3 * float(g.get("size").split()[0])]
                                               for g in b.findall("geom") if g.get("class") == "cad_collision"])
        _CAD[finger] = out
    return _CAD[finger]


def hand_points(FT, HS, M, joints):
    """(X_P (N, 3), R (N,), labels [(finger, part)]): every finger's CAD collision spheres on the chain at its mounts
    and joints, and hom_setup.finger_pts' tip pad, tip end and PIP joint (the TPU pad is not among the CAD's rigid
    spheres)."""
    from sr2hand.robot.tool_chain import ToolChain
    X, R, lab = [], [], []
    plan = {"meta": {"hom": {"d6_P_mm": [*M["thumb"], *M["index"], *M["middle"]]}}}
    for f in FINGERS:
        c, _ = FT.H.box(f)
        q = joints[f]
        links = FT.H.finger_links(f, M[f][0] - c[0], M[f][1] - c[1], *(math.radians(q[k]) for k in JOINTS))
        ch = ToolChain(f, links)
        for body, S in cad_spheres(f).items():
            T = ch.transform(upto=body)
            X.append(S[:, :3] @ T[:3, :3].T + T[:3, 3])
            R.append(S[:, 3])
            lab += [(f, body.split("_", 1)[1])] * len(S)
        for part, (pt, rad) in zip(HS.PARTS, HS.finger_pts(plan, f, q)):
            X.append(np.array([pt]))
            R.append(np.array([rad]))
            lab.append((f, part))
    return np.concatenate(X), np.concatenate(R), lab


def shifted(T, dv):
    T2 = np.array(T, float).copy()
    T2[:3, 3] = T2[:3, 3] + dv
    return T2


def turned(T_BP, deg):
    """The hand turned by deg about its own vertical axis z_P through the palm origin (pinch_pickup's turn)."""
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    Rz = np.eye(4)
    Rz[:2, :2] = [[c, -s], [s, c]]
    return np.array(T_BP, float) @ Rz


def leg_poses(leg, step=2.0):
    """The hand's poses T_BP along a leg: a straight translation every step mm, or a turn every deg."""
    T0, T1 = np.array(leg["T0"]), np.array(leg["T1"])
    if leg["kind"] == "turn":
        n = max(1, int(abs(leg["deg"])))
        return [(i / n, turned(T0, leg["deg"] * i / n)) for i in range(n + 1)]
    d = T1[:3, 3] - T0[:3, 3]
    n = max(1, int(np.linalg.norm(d) / step))
    return [(i / n, shifted(T0, d * i / n)) for i in range(n + 1)]


def leg_clearance(pts, leg, mdl, table_z, base_clear, press):
    """The least clearances along a leg (mm, and the finger part with the leg's fraction): to each box, to the table,
    and above table_z + base_clear. The pressing finger's parts in front of its own plate's face or the post's
    (press = (finger, plate)) go apart as "press face": they may end --gap off it; over an edge they count as usual."""
    p, nh, w, boxes = mdl
    X_P, R, lab = pts
    fing = np.array([lb[0] for lb in lab])
    least = {k: (math.inf, None) for k in (*boxes, "press face", "table", "base layer")}

    def keep(key, g, i, u):
        if g < least[key][0]:
            least[key] = (float(g), f"{lab[i][0]} {lab[i][1]} at {u:.2f}")

    for u, T in leg_poses(leg):
        X = X_P @ T[:3, :3].T + T[:3, 3]
        for name, box in boxes.items():
            g, front = box_gaps(X, R, p, nh, w, box)
            pf = (fing == press[0]) & front & (name in (press[1], "post"))
            if pf.any():
                i = int(np.argmin(np.where(pf, g, np.inf)))
                keep("press face", g[i], i, u)
            g2 = np.where(pf, np.inf, g)
            i = int(np.argmin(g2))
            keep(name, g2[i], i, u)
        gz = X[:, 2] - R - table_z
        i = int(np.argmin(gz))
        keep("table", gz[i], i, u)
        keep("base layer", gz[i] - base_clear, i, u)
    return {k: {"mm": None if v[1] is None else round(v[0], 2), "where": v[1]} for k, v in least.items()}


def state(HS, a, n=5):
    """(station, state, mounts in P, joints): the joints the median of n readings 0.2 s apart (a potentiometer reading
    jumps for one sample now and then: 1.5-2 deg on the thumb's yaw and PIP on 2026-11-18)."""
    import s1b_tags as S1
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    M = S1.model_mounts(HS.load_plan(str(S3.PLAN)), s["mount_positions"])
    samples = [HS.sim_joints(s)]
    for _ in range(n - 1):
        time.sleep(0.2)
        samples.append(HS.sim_joints(st.state()))
    joints = {f: {k: float(np.median([x[f][k] for x in samples])) for k in JOINTS} for f in FINGERS}
    return st, s, M, joints


def num(x):
    return "-" if x is None else f"{x:+.1f}"


def cmd_plan(a):
    FT, HS, SL = S3.imports()
    import s1b_tags as S1
    T0, mode = S1.arm_T_BP(FT, HS)
    st, s, M, joints = state(HS, a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / (f"{time.strftime('%Y%m%d-%H%M%S')}_approach_{a.finger}" + ("_yaw" if a.press == "yaw" else "")
                  + (f"_{a.plate}" if a.plate else ""))
    loc = S3.locate_block(FT, a.edge, a.seconds, frames_to=str(stem))
    if S3.block_frame_B(loc) is None:
        raise SystemExit(f"the block's tag is not in view (detections {loc['dets']}, camera errors "
                         f"{loc['camera_errors']}); frames {list(loc['frames'].values())}")
    src = "triangulated" if loc["tri"] else f"one camera's solve with the {a.edge:g} mm edge"
    noise_n = a.noise_n if a.noise_n is not None else (0.5 if loc["tri"] else 2.0)
    mdl = model(loc, a.tape, a.noise_l, a.noise_z, noise_n)
    p, nh, w, boxes = mdl
    top = TABLE_Z + TOP
    up = np.array([0.0, 0.0, 1.0])
    side = a.plate or ("tag" if a.finger == "thumb" else "pad")
    plate = f"{side} plate"
    d_in = -nh if side == "tag" else nh                            # into the chosen plate
    q = [float(joints[a.finger][k]) for k in JOINTS]
    _, _, tip, con = FT.fk(a.finger, M[a.finger], q)
    if a.press == "yaw":
        sign = a.sign if a.sign is not None else (-1 if a.finger == "index" else 1)
        lead_P = tip + PAD_R * np.array([0.0, float(sign), 0.0])
        dir_P = np.array([0.0, float(sign), 0.0])
    else:
        sign = None
        lead_P, dir_P = con, (con - tip) / np.linalg.norm(con - tip)
        if a.lead == "ball":                  # a pre-bent PIP turns the apex away from the plate: aim the pad's ball
            R0 = T0[:3, :3]
            dir_P = R0.T @ d_in
            dir_P = dir_P / np.linalg.norm(dir_P)
            lead_P = tip + PAD_R * dir_P

    def horiz(v):
        v = v - (v @ up) * up
        return v / np.linalg.norm(v)

    degs = np.arange(-180.0, 180.0, 0.05)
    deg = 0.0
    if a.press == "yaw":
        deg = float(degs[int(np.argmax([horiz(turned(T0, x)[:3, :3] @ dir_P) @ d_in for x in degs]))])
        if abs(deg) > a.max_turn:
            raise SystemExit(f"the yaw press needs a {deg:+.1f} deg turn of the hand, over --max-turn {a.max_turn:g}")
    off = math.degrees(math.acos(float(np.clip(horiz(turned(T0, deg)[:3, :3] @ dir_P) @ d_in, -1, 1))))
    print(f"arm mode {mode}; palm origin B {np.round(T0[:3, 3], 1).tolist()}; tag ({src}): centre B "
          f"{np.round(p, 1).tolist()}, {p[2] - TABLE_Z:.1f} mm above the table; level normal B "
          f"{np.round(nh, 3).tolist()}")
    print(f"{a.finger} {a.press}" + (f" sign {sign:+d}" if sign else "") + f": the hand turns {deg:+.1f} deg about z_P; "
          f"the press then runs {off:.1f} deg off the plate's normal")
    pts = hand_points(FT, HS, M, joints)

    def lowest(T):
        return float(np.min((pts[0] @ T[:3, :3].T + T[:3, 3])[:, 2] - pts[1]))

    top = top + LIP
    rise = max(0.0, top + a.noise_z + a.over - lowest(T0))
    T_r = shifted(T0, [0.0, 0.0, rise])
    T_t = turned(T_r, deg)
    lowest_t = min(lowest(turned(T_r, deg * u)) for u in np.linspace(0, 1, 13))
    if lowest_t < top + a.noise_z + a.over - 0.5:                 # a tilted z_P lowers a finger in the turn
        extra = top + a.noise_z + a.over - lowest_t
        rise += extra
        T_r = shifted(T0, [0.0, 0.0, rise])
        T_t = turned(T_r, deg)
    s_t = a.gap if side == "tag" else -S3.BEHIND - a.tape - a.gap
    target = p + s_t * nh + a.dl * w + a.dz * up
    lead_t = T_t[:3, :3] @ lead_P + T_t[:3, 3]
    delta = target - lead_t
    if delta[2] > 0:
        raise SystemExit(f"the target lies {delta[2]:.1f} mm above the turned hand's leading point: plan from lower")
    T_l = shifted(T_t, [delta[0], delta[1], 0.0])
    T_d = shifted(T_t, delta)
    legs = [{"name": "rise", "kind": "lin", "T0": T0, "T1": T_r}]
    if abs(deg) > 0.05:
        legs.append({"name": "turn", "kind": "turn", "deg": deg, "T0": T_r, "T1": T_t})
    legs += [{"name": "level", "kind": "lin", "T0": T_t, "T1": T_l},
             {"name": "descend", "kind": "lin", "T0": T_l, "T1": T_d}]
    print(f"target: the {a.finger}'s leading point at s {s_t:+.1f}, l {a.dl:+.1f} mm, z_B {target[2]:.1f} "
          f"({target[2] - TABLE_Z:.1f} above the table); up {rise:.1f} mm so that every finger part clears the block's "
          f"top ({TOP:g} mm) by {a.over:g} mm; noise l {a.noise_l:g}, z {a.noise_z:g}, n {noise_n:g} mm")
    ok, rec_legs = True, []
    for leg in legs:
        cl = leg_clearance(pts, leg, mdl, TABLE_Z, a.base_clear, (a.finger, plate))
        if leg["name"] == "rise":
            cl["base layer"]["mm"] = None                            # starts at the present height
        floor = {"base layer": 0.0, "press face": 0.5}
        bad = [k for k, v in cl.items() if v["mm"] is not None and v["mm"] < floor.get(k, a.min_clear)]
        ok &= not bad
        move = (f"{leg['deg']:+.1f} deg about z_P" if leg["kind"] == "turn"
                else f"{np.round(leg['T1'][:3, 3] - leg['T0'][:3, 3], 1).tolist()} mm")
        rec_legs.append({**{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in leg.items()}, "clear": cl,
                         "bad": bad})
        print(f"  {leg['name']:7s} {move}: " + "; ".join(f"{k} {num(v['mm'])} ({v['where']})" for k, v in cl.items())
              + (f"  -- TOO CLOSE: {bad}" if bad else ""))
    rec = {"what": "S3 approach plan", "t": time.time(), "finger": a.finger, "press": a.press, "sign": sign, "plate": side,
           "turn_deg": deg, "press_off_deg": off, "gap_mm": a.gap, "lead": getattr(a, "lead", "apex"), "tape_mm": a.tape,
           "noise_mm": {"l": a.noise_l, "z": a.noise_z, "n": noise_n}, "dl": a.dl, "dz": a.dz, "over": a.over,
           "base_clear": a.base_clear, "edge_mm": a.edge, "locate": loc, "arm_mode": mode, "T_BP0": T0.tolist(),
           "mounts_station": s.get("mount_positions"), "mounts_P": M, "joints": joints, "tag_B": p.tolist(),
           "nh_B": nh.tolist(), "w_B": w.tolist(), "boxes": boxes, "lead_P": lead_P.tolist(), "target_B": target.tolist(),
           "T_BP_end": T_d.tolist(), "legs": rec_legs, "ok": ok}
    stem.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=float))
    print(("plan clear" if ok else "plan NOT clear") + f" -> {stem}.json")
    return 0 if ok else 1


def check_same(a, rec, FT, HS, want):
    """Refuses unless the arm runs at the pose `want` (T_BP), the station's mounts and the joints are the plan's, and
    the block's tag is within 2 mm of where the plan saw it."""
    import s1b_tags as S1
    T_BP, mode = S1.arm_T_BP(FT, HS)
    if mode != 7:
        raise SystemExit(f"the arm is not running (robot mode {mode}): release the brakes on the pendant (ON, START)")
    dp = float(np.linalg.norm(T_BP[:3, 3] - want[:3, 3]))
    dr = math.degrees(math.acos(float(np.clip((np.trace(T_BP[:3, :3].T @ want[:3, :3]) - 1) / 2, -1, 1))))
    if dp > 2.0 or dr > 0.5:
        raise SystemExit(f"the hand is {dp:.1f} mm and {dr:.2f} deg from where this leg starts")
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
    print(f"checks: hand {dp:.2f} mm / {dr:.2f} deg from the leg's start, mounts and joints as planned, tag {moved:.2f} "
          f"mm from the plan's ({loc['dets']})")


def run_legs(a, back):
    FT, HS, SL = S3.imports()
    rec = json.loads(Path(a.plan).read_text())
    if not rec.get("ok"):
        raise SystemExit("the plan is not clear")
    T0 = np.array(rec["T_BP0"])
    legs = [dict(lg) if "kind" in lg else                           # plans of 15:4x: translations from T_BP0
            {"name": lg["name"], "kind": "lin", "T0": shifted(T0, lg["from_mm"]).tolist(),
             "T1": shifted(T0, lg["to_mm"]).tolist()} for lg in rec["legs"]]
    if back:
        legs = [{**lg, "name": f"back {lg['name']}", "T0": lg["T1"], "T1": lg["T0"], "deg": -lg.get("deg", 0.0)}
                for lg in reversed(legs)]
    v = {"rise": a.v_up, "turn": a.v_level, "level": a.v_level, "descend": a.v_down}
    print(f"{rec['finger']} {rec.get('press', 'flex')}: " + "; ".join(
        f"{lg['name']} " + (f"{lg['deg']:+.1f} deg" if lg["kind"] == "turn"
                            else f"{np.round(np.array(lg['T1'])[:3, 3] - np.array(lg['T0'])[:3, 3], 1).tolist()} mm")
        for lg in legs))
    check_same(a, rec, FT, HS, np.array(legs[0]["T0"]))
    if not a.go:
        return 0
    arm = HS.Arm(a.ur_ft, a.blank_s)
    try:
        arm.connect(True)
        for lg in legs:
            T_goal = np.array(lg["T1"]) @ np.linalg.inv(HS.T_EP)
            base = lg["name"].split()[-1]
            vv = v[base]
            if lg["kind"] == "turn":
                max_s, stop_n = 6.0 + 0.6 * abs(lg["deg"]), a.stop_n
            else:
                dist = float(np.linalg.norm(np.array(lg["T1"])[:3, 3] - np.array(lg["T0"])[:3, 3]))
                max_s = dist / (1e3 * vv) + 6.0
                stop_n = a.stop_down if base == "descend" else a.stop_n
            why, moved = arm.move(T_goal, vv, a.acc, stop_n, max_s, f"approach {lg['name']}")
            if why:
                print(f"stopped in {lg['name']}: {why}; the arm holds there", flush=True)
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
            p.add_argument("--press", choices=("flex", "yaw"), default="flex")
            p.add_argument("--sign", type=int, choices=(-1, 1), help="the yaw press's direction (default: index -1, "
                                                                     "middle +1, as in the night's squeezes)")
            p.add_argument("--max-turn", type=float, default=95.0, help="deg: the largest turn of the hand (owner: 90)")
            p.add_argument("--plate", choices=("pad", "tag"), help="the plate to press: the taped one (default for "
                                                                   "index and middle) or the tag's (default for the thumb)")
            p.add_argument("--gap", type=float, default=4.0, help="mm from the leading point to its plate")
            p.add_argument("--lead", choices=("apex", "ball"), default="apex",
                           help="flex presses: the pad's apex, or its 11 mm ball's side facing the plate (pre-bent PIP)")
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
            p.add_argument("--v-level", type=float, default=0.015, help="m/s (and the turn's moveL speed)")
            p.add_argument("--v-down", type=float, default=0.008, help="m/s")
            p.add_argument("--acc", type=float, default=0.05, help="m/s^2")
            p.add_argument("--stop-n", type=float, default=3.0, help="N of force change that stops the rise, turn and "
                                                                     "level legs (a turn adds hom_setup's allowance)")
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
