"""S3 of the identification queue (QUEUE.md; the owner's notes of 2026-11-18 09:0x, 09:55 and 15:0x): the torque scale
at 7.5 V. One joint of one finger steps its target past its reading, --step deg every --dwell s, toward a rigid face that
the finger rests a few mm from. The face is the flat pad of the owner's 80/20 block, 22 mm behind the face of its
tag36h11 id 0 and about the tag's size, 40 mm square (owner, 15:3x: "22mm on the otherside of it is an approximately equal
size flat pad that we can use to press against. the 8020 does flex moderately but should be fine"). The owner or the arm
places the block; this tool never moves the arm. It records the HEX-E (100 Hz), the station's joints and loads (20 Hz),
each step's servo flags, torque and temperatures (the station's diagnostics), the fingertip boards in the splitter's
file and, with --track-block, the block's tag in both cameras, so that the post's flex can be taken out of the joint's
advance (or, with --tag, the finger's DIP tag instead: the cameras serve one tracker at a time). The steps go on past
contact until the HEX-E's force has changed --max-n from its mean before the first step, a servo flags or loses torque,
the pressing servo's load passes --max-load, a servo reaches --max-temp, or the target is --max-offset past the reading.
Then the target steps back the same way to where it started (W3, the owner's alternative: with the clutch plant's gear
the force stays flat and the reading stays put until the target passes the reading; a gear the load can backdrive lets
both fall before that).

    /usr/bin/python3 s3_press.py check
        the HEX-E's mean and noise over 2 s, the station's torque, joints, loads and temperatures; no motion
    /usr/bin/python3 s3_press.py block [--edge 40] [--behind 22] [--pad-size 40]
        the block's tag (no motion): its pose from the tracker's solve with the printed edge and, when both cameras see
        it, from its corners triangulated in the two cameras, which assumes no size (39.8 mm at 15:28: the 40 mm print);
        its tilt; the press pad --behind mm behind the tag's face; and per finger at its present joints, for each joint
        and sign alone, the degrees until the finger (an --pad-r ball round the pad's centre, or the pad's apex) meets
        the pad within its square. One frame per camera is saved with the tag outlined.
    /usr/bin/python3 s3_press.py run --finger index --joint yaw --sign -1 --track-block --out OUT [--go]
        without --go, the plan and the same checks; --sign is the direction (simulator degrees) that moves the finger
        toward the pad

Directions (W2, 11-18 15:4x): the yaw presses go the way the yaws pushed in the night's squeezes, toward the sphere and
the hand's centre line: in the 11 tips-off g0 twists the index's yaw target ran 8.5-16.6 deg more negative than its
reading and the middle's 12.1-16.6 deg more positive, so index --sign -1 and middle --sign +1. With the finger
straight, +1 deg of yaw moves the tip 1.53 mm along +y_P (88 mm lever; the twist's grasp 1.44 mm). On the CAD hand the
DIP link's distal end stands 10.5 mm either side of the finger's axis and the pad only 7.4 mm, so a straight finger
pressed sideways meets the face with the link's side: no pad compression to subtract.

OUT/<stamp>_<finger>_<joint>.jsonl: a header, one line per step (target, reading, the pressing servo's load, the
HEX-E's force change from before the first step, mean over the dwell's last second; with --track-block the block's
centre in B and its displacement along the tag's outward normal from before the first step, positive away from the
finger for a press on the pad; with --tag the DIP tag's position and outward normal in P), an end line with the reason.
OUT/<stamp>_<finger>_<joint>_raw.jsonl holds the HEX-E's and the station's samples and every tracker sample, press and
step back alike. The workstation fits the gain, the give and the dead band (W2, W3). Prediction for the index's and
middle's yaws (hand-4a, QUEUE 1ad5a35ad): at 14-15 deg of target offset with the finger straight, about 0.6 N if the
gear passes 0.4 of the servo's torque (the clutch plant, kp 0.525 N m/rad) and 1.5 N if it passes all of it; the slope
needs neither gain: about 6.6 deg of reading per newton with the yaw chain's give of 0.76 N m/rad, under 1 deg per
newton for a rigid chain.
"""
import argparse
import json
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LAB = ROOT / "docs/experiments/20261115-hom_bench_trials/lab"
FINGERS = ("thumb", "index", "middle")
JOINTS = ("yaw", "mcp", "pip")
BLOCK_TAG = 0                    # tag36h11 id 0 on the back of the 80/20 (owner, 2026-11-18 15:0x); no map uses it
BLOCK_EDGE = 40.0                # mm: the print; the two cameras triangulated 39.8 mm (sides 39.9-40.3) at 15:28
BEHIND = 22.0                    # mm: the press pad behind the tag's face (owner, 15:3x)
PAD_SIZE = 40.0                  # mm: the press pad, "approximately equal size" to the tag
PLAN = LAB.parent / "plans/20261115-roll_press_plan.json"   # only the offset of the model's mounts from the station's


def imports():
    sys.path[:0] = [str(ROOT / "src"), str(LAB), str(ROOT / "src/sr2hand/driver/reskin5x/host")]
    import fingertip_tag_check as FT
    import hom_setup as HS
    from sr2hand.control import servo_limits as SL
    return FT, HS, SL


class Sampler(threading.Thread):
    """The station's joints (simulator degrees) and loads at about hz, kept as [(t, joints, loads)]."""

    def __init__(self, HS, st, hz=20.0):
        super().__init__(daemon=True)
        self.HS, self.st, self.dt = HS, st, 1.0 / hz
        self.rows, self.lock, self.halt, self.error = [], threading.Lock(), False, None

    def run(self):
        while not self.halt:
            try:
                s = self.st.state()
                row = (time.time(), self.HS.sim_joints(s), (s.get("telemetry") or {}).get("servo_load") or {})
                with self.lock:
                    self.rows.append(row)
            except Exception as e:                       # noqa: BLE001
                self.error = repr(e)
            time.sleep(self.dt)

    def since(self, t0):
        with self.lock:
            return [r for r in self.rows if r[0] >= t0]


def num(x, spec):
    return "-" if x is None else format(x, spec)


def num_list(v):
    return "-" if v is None else str(np.round(v, 1).tolist())


def hexe_mean(ft, seconds):
    t, v = ft.ring.since(time.time() - seconds)
    return (None, None, len(t)) if len(t) < 5 else (v[:, :3].mean(axis=0), v[:, :3].std(axis=0), len(t))


def diag(st):
    d = st.call("POST", "/api/v1/servos/diagnostics")
    return {k: d.get(k) for k in ("status", "temperature", "torque_enable", "voltage")}


def stop_tracker(arm, tracker):
    tracker.stop()
    arm.stop()
    arm.join(timeout=3)


def block_map(edge):
    """(map file, TagMap) of the block's tag alone in its own frame O. plate_map puts the tag's outward normal along +z_O
    (the DIP tags' maps of fingertip_tag_check put it along -z_O)."""
    import tempfile
    from sr2hand.tracking import tags as TG
    tm = TG.plate_map("tag36h11", edge, [BLOCK_TAG], [(0.0, 0.0, 0.0)])
    f = Path(tempfile.gettempdir()) / f"s3_block_tag{BLOCK_TAG}_{edge:g}mm_map.json"
    f.write_text(json.dumps(tm.to_json()))
    return str(f), tm


def locate_block(FT, edge, seconds=3.0, frames_to=None):
    """The block's tag over `seconds`, in B (mm): "joint", the tracker's solve with the printed edge `edge` (centre p_B,
    outward normal n_B, right r_B, down d_B, rms px, samples, cameras); "tri", when both cameras see it, its corners
    triangulated from paired frames of the two cameras (gantry_follow_fit.triangulate), which assumes no size (centre,
    outward normal, right, down, edge and its sd, pairs). frames_to: a path stem; one frame per camera is saved there,
    the tag outlined."""
    import cv2
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/hardware")]
    import gantry_follow_fit as GF
    from sr2hand.geometry.vectors import unit
    from sr2hand.tracking import live
    from sr2hand.tracking import tags as TG
    fused, dets, frames, lock = [], [], {}, threading.Lock()
    path, tm = block_map(edge)
    _, r_O, d_O, n_O = tm.frames[BLOCK_TAG]

    def on(smp):
        with lock:
            for sn, d in smp.dets.items():
                if d.corners and BLOCK_TAG in d.corners:
                    dets.append((sn, d.t_frame_ms, np.asarray(d.corners[BLOCK_TAG], float)))
            if smp.T_BO is not None and smp.msg.n_tags > 0:
                fused.append((np.array(smp.T_BO, float), float(smp.msg.rms_px), int(smp.msg.n_cams)))

    def on_frame(sn, y, det):
        seen = bool(det is not None and det.corners and BLOCK_TAG in det.corners)
        with lock:
            if frames_to is not None and (sn not in frames or (seen and not frames[sn][1])):
                frames[sn] = (np.array(y, copy=True), seen, np.asarray(det.corners[BLOCK_TAG]) if seen else None)

    class _A:
        map = path
    arm, tracker = FT.HT.start_tracker(_A, on, on_frame)
    try:
        time.sleep(1.5 + seconds)
    finally:
        errors = {w.sn: w.error for w in tracker.workers if w.error}
        stop_tracker(arm, tracker)
    out = {"tag": BLOCK_TAG, "edge_mm": edge, "seconds": seconds, "dets": dict(Counter(d[0] for d in dets)),
           "camera_errors": errors, "joint": None, "tri": None, "frames": {}}
    if fused:
        T = np.array([f[0] for f in fused])
        R = T[len(T) // 2, :3, :3]
        out["joint"] = {"p_B": np.median(T[:, :3, 3], axis=0).tolist(),
                        "n_B": unit(np.median(T[:, :3, :3] @ n_O, axis=0)).tolist(),
                        "r_B": (R @ r_O).tolist(), "d_B": (R @ d_O).tolist(),
                        "rms_px": float(np.median([f[1] for f in fused])), "n": len(fused),
                        "cams": dict(Counter(f[2] for f in fused))}
    by = {}
    for sn, tms, C in dets:
        by.setdefault(sn, []).append((tms, C))
    if len(by) == 2:
        cams = live.load_cameras(FT.HT.CAL / "20261113-lens_board_k3_arm.json",
                                 FT.HT.CAL / "20261113-handeye_tcp_fitted_lens_moved836.json")
        s1, s2 = sorted(by)
        tb = np.array([x[0] for x in by[s2]])
        X, E = [], []
        for tms, C in by[s1]:
            k = int(np.argmin(np.abs(tb - tms)))
            Xk, e = GF.triangulate(cams, [(s1, C), (s2, by[s2][k][1])])
            X.append(Xk)
            E.append(e)
        c, r, d, n = TG.TagMap.from_corners("tag36h11", float(np.median(E)),
                                            {BLOCK_TAG: np.median(np.array(X), axis=0)}).frames[BLOCK_TAG]
        out["tri"] = {"p_B": c.tolist(), "n_B": n.tolist(), "r_B": r.tolist(), "d_B": d.tolist(),
                      "edge_mm": float(np.median(E)), "edge_sd_mm": float(np.std(E)), "n": len(E)}
    for sn, (y, seen, C) in frames.items():
        img = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
        if C is not None:
            cv2.polylines(img, [C.astype(np.int32).reshape(-1, 1, 2)], True, (0, 0, 255), 3)
        f = f"{frames_to}_{sn}.jpg"
        cv2.imwrite(f, img)
        out["frames"][sn] = f
    return out


def block_frame_B(loc):
    """(centre, outward normal, right, down) of the block's tag in B: the triangulated one when both cameras saw it."""
    src = loc["tri"] or loc["joint"]
    return None if src is None else tuple(np.array(src[k]) for k in ("p_B", "n_B", "r_B", "d_B"))


def faces_P(loc, T_BP, behind, pad_size, edge):
    """The block's faces in P, [(name, point, outward normal, inside(x_P) -> bool)]: the tag's face and the press pad
    `behind` mm behind it, facing the other way; inside() is the face's square (the tag's edge, the pad's size) about
    the tag's centre and its projection."""
    from sr2hand.geometry.rotations import tf_inv
    T_PB = tf_inv(T_BP)
    pB, nB, rB, dB = block_frame_B(loc)
    R, t = T_PB[:3, :3], T_PB[:3, 3]
    p, n, r, d = R @ pB + t, R @ nB, R @ rB, R @ dB

    def square(c, half):
        return lambda x: abs(float((x - c) @ r)) <= half and abs(float((x - c) @ d)) <= half

    return [("tag face", p, n, square(p, edge / 2)), ("press pad", p - behind * n, -n, square(p - behind * n, pad_size / 2))]


def approach(FT, f, mount, q, face, pad_r, limits, max_deg=30.0, step=0.25):
    """(the finger's distance from the face's plane now, {(joint, sign): degrees of that joint alone until the finger
    (a pad_r ball round the pad's centre, or the pad's apex) meets the face, or None within max_deg and the station's
    range}). A meeting outside the face's square (inside() false) does not count, nor does a finger already behind
    the plane (distance <= 0: on the block's far side, or in it)."""
    _, p0, n, inside = face

    def gap(qq):
        _, _, tip, con = FT.fk(f, mount, qq)
        a, b = float((tip - p0) @ n) - pad_r, float((con - p0) @ n)
        return (a, tip - pad_r * n) if a < b else (b, con)

    d0 = gap(q)[0]
    res = {}
    if d0 <= 0.0:
        return d0, res
    for j, jn in enumerate(JOINTS):
        lo, hi = limits[jn]
        for s in (-1, 1):
            res[(jn, s)] = None
            for k in range(1, int(max_deg / step) + 1):
                qq = list(q)
                qq[j] += s * step * k
                if not lo <= qq[j] <= hi:
                    break
                g, x = gap(qq)
                if g <= 0.0:
                    if inside(x):
                        res[(jn, s)] = step * k
                    break
    return d0, res


def cmd_check(a):
    FT, HS, SL = imports()
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    ft = FT.HT.open_ft(True)
    time.sleep(2.5)
    F, sd, n = hexe_mean(ft, 2.0)
    ft.stop = True
    print("HEX-E: " + ("no samples: is it on?" if F is None else
                       f"{n} samples in 2 s, mean {np.round(F, 2).tolist()} N, sd {np.round(sd, 3).tolist()} N"))
    tel = s.get("telemetry") or {}
    j = HS.sim_joints(s)
    print(f"station: torque {s.get('servo_torque')}, mounts {s.get('mount_positions')}")
    print("joints " + "  ".join(f"{f} " + " ".join(f"{k} {j[f][k]:+.1f}" for k in JOINTS) for f in FINGERS))
    print(f"loads {tel.get('servo_load')}; temperatures {tel.get('servo_temperature')}")
    print(f"diagnostics {diag(st)}")
    return 0


def cmd_block(a):
    FT, HS, SL = imports()
    import s1b_tags as S1
    from sr2hand.geometry.rotations import tf_inv
    T_BP, mode = S1.arm_T_BP(FT, HS)
    T_PB = tf_inv(T_BP)
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{time.strftime('%Y%m%d-%H%M%S')}_block"
    loc = locate_block(FT, a.edge, a.seconds, frames_to=str(stem))
    P = lambda v: (T_PB[:3, :3] @ np.asarray(v) + T_PB[:3, 3])         # noqa: E731
    R = lambda v: (T_PB[:3, :3] @ np.asarray(v))                        # noqa: E731
    print(f"arm mode {mode}, palm origin B {np.round(T_BP[:3, 3], 1).tolist()}")
    print(f"tag {BLOCK_TAG}: detections {loc['dets'] or 'none'} in {a.seconds:g} s; frames {list(loc['frames'].values())}"
          + (f"; camera errors {loc['camera_errors']}" if loc["camera_errors"] else ""))
    rec = {"what": "S3 block", "t": time.time(), "arm_mode": mode, "T_BP": T_BP.tolist(), "edge_mm": a.edge,
           "behind_mm": a.behind, "pad_size_mm": a.pad_size, "locate": loc, "mounts": s.get("mount_positions"),
           "joints": HS.sim_joints(s)}
    for k in ("joint", "tri"):
        v = loc[k]
        if v is None:
            continue
        tilt = float(np.degrees(np.arcsin(np.clip(v["n_B"][2], -1, 1))))
        roll = float(np.degrees(np.arcsin(np.clip(v["r_B"][2], -1, 1))))
        extra = (f", edge {v['edge_mm']:.2f} mm (sd {v['edge_sd_mm']:.2f}, {v['n']} pairs)" if k == "tri" else
                 f" with edge {a.edge:g} mm, rms {v['rms_px']:.2f} px, {v['n']} samples, cameras {v['cams']}")
        print(f"{'triangulated' if k == 'tri' else 'solve':12s}: centre P {np.round(P(v['p_B']), 1).tolist()} "
              f"(B {np.round(v['p_B'], 1).tolist()}, {v['p_B'][2] - 3.0:.1f} mm above the table), outward normal P "
              f"{np.round(R(v['n_B']), 3).tolist()}: {tilt:+.1f} deg from horizontal, its right edge {roll:+.1f} deg "
              f"from level{extra}")
    if loc["joint"] and loc["tri"]:
        print(f"solve - triangulated: {np.round(np.subtract(loc['joint']['p_B'], loc['tri']['p_B']), 1).tolist()} mm in B")
    if block_frame_B(loc) is None:
        print("the block's tag is not in view")
        (stem.with_suffix(".json")).write_text(json.dumps(rec, indent=1))
        return 1
    faces = faces_P(loc, T_BP, a.behind, a.pad_size, a.edge)
    M = S1.model_mounts(HS.load_plan(str(PLAN)), s["mount_positions"])
    j = HS.sim_joints(s)
    limits = st.call("GET", "/api/v1/manual/limits")["joint_deg"]
    rec["faces"] = []
    for face in faces:
        name, p0, n, _ = face
        print(f"{name}: centre P {np.round(p0, 1).tolist()}, outward normal P {np.round(n, 3).tolist()}")
        fr = {"name": name, "p_P": p0.tolist(), "n_P": n.tolist(), "fingers": {}}
        for f in FINGERS:
            q = [float(j[f][k]) for k in JOINTS]
            d0, res = approach(FT, f, M[f], q, face, a.pad_r, limits[f], a.max_deg)
            hits = {f"{jn}{sg:+d}": v for (jn, sg), v in res.items() if v is not None}
            fr["fingers"][f] = {"q": q, "gap_mm": d0, "deg_to_face": hits}
            print(f"  {f:6s} q {np.round(q, 1).tolist()}: {d0:+6.1f} mm from the plane; "
                  + ("behind it (the block's far side, or in it)" if d0 <= 0 else
                     ", ".join(f"{k2} {v:.2f} deg" for k2, v in hits.items())
                     or f"no joint alone meets it within {a.max_deg:g} deg"))
        rec["faces"].append(fr)
    (stem.with_suffix(".json")).write_text(json.dumps(rec, indent=1))
    print(f"-> {stem}.json")
    return 0


def cmd_run(a):
    FT, HS, SL = imports()
    from sr2hand.geometry.rotations import quat_to_mat
    if a.tag and a.track_block:
        raise SystemExit("--tag or --track-block: the cameras serve one tracker at a time")
    P, SV = SL._driver()
    f, jn = a.finger, a.joint
    sids = {k: str(SV.FINGER_JOINTS[P.FINGER_ID[f]][P.SIM_JOINT_TO_SERVO[k]][0]) for k in JOINTS}
    sid = sids[jn]
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    if s.get("servo_torque") != 1:
        raise SystemExit("servo torque is not on")
    start = dict(HS.sim_joints(s)[f])
    q0 = float(start[jn])
    lo, hi = st.call("GET", "/api/v1/manual/limits")["joint_deg"][f][jn]
    n_max = int(a.max_travel / a.step)
    while n_max > 0 and not lo + 1.0 <= q0 + a.sign * a.step * n_max <= hi - 1.0:
        n_max -= 1
    if n_max < 4:
        raise SystemExit(f"{f} {jn} at {q0:+.1f} deg has under 2 deg of the station's range [{lo:g}, {hi:g}] toward "
                         f"sign {a.sign:+d}")
    ft = FT.HT.open_ft(True)
    time.sleep(1.5)
    F0, sd0, n0 = hexe_mean(ft, 1.0)
    if F0 is None:
        ft.stop = True
        raise SystemExit(f"the HEX-E gave {n0} samples in 1 s: is it on?")
    dg = diag(st)
    print(f"{f} {jn} (servo {sid}) from {q0:+.2f} deg, sign {a.sign:+d}: up to {n_max} steps of {a.step:g} deg every "
          f"{a.dwell:g} s (target to {q0 + a.sign * a.step * n_max:+.1f}, the station's range [{lo:g}, {hi:g}]); "
          f"the finger's other joints held at " + ", ".join(f"{k} {start[k]:+.1f}" for k in JOINTS if k != jn))
    print(f"stops: |dF| {a.max_n:g} N on the HEX-E, the target {a.max_offset:g} deg past the reading, load "
          f"{a.max_load:g}, a flag or torque off on the finger's servos, {a.max_temp:g} C; then back the same way")
    print(f"HEX-E before: {np.round(F0, 2).tolist()} N (sd {np.round(sd0, 3).tolist()}, {n0} samples in 1 s); "
          f"diagnostics {dg}")
    if not a.go:
        ft.stop = True
        return 0
    import tip_contacts as TC
    try:
        pids, how = TC.ensure_splitter(start=True)
    except RuntimeError as e:
        ft.stop = True
        raise SystemExit(f"fingertip boards: {e}")
    mux = sorted((ROOT / "logs").glob("*-three_finger_mux/*-mux.bin"), key=lambda p: p.stat().st_mtime)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{f}_{jn}"
    samp = Sampler(HS, st)
    samp.start()
    trk, tlock, arm, tracker, base = [], threading.Lock(), None, None, None
    if a.tag:
        def on(smp):
            m = smp.msg
            if m is not None and m.n_tags > 0 and np.all(np.isfinite(m.p_PO)):
                with tlock:
                    trk.append((time.time(), np.array(m.p_PO, float), -quat_to_mat(np.array(m.q_PO, float))[:, 2],
                                int(m.n_cams)))

        class _A:
            map = FT.tag_map_file(FT.TAG[f])
        arm, tracker = FT.HT.start_tracker(_A, on)
    elif a.track_block:
        path, tm = block_map(a.block_edge)
        n_O = tm.frames[BLOCK_TAG][3]

        def on(smp):
            if smp.T_BO is not None and smp.msg.n_tags > 0:
                with tlock:
                    trk.append((time.time(), np.array(smp.T_BO[:3, 3], float), smp.T_BO[:3, :3] @ n_O,
                                int(smp.msg.n_cams)))

        class _A:
            map = path
        arm, tracker = FT.HT.start_tracker(_A, on)
        time.sleep(3.0)
        with tlock:
            got = [x for x in trk if x[0] >= time.time() - 1.0]
        if len(got) < 5:
            stop_tracker(arm, tracker)
            samp.halt = True
            ft.stop = True
            raise SystemExit(f"the block's tag was seen {len(got)} times in 1 s: is it in view of the cameras "
                             f"(camera errors {[(w.sn, w.error) for w in tracker.workers if w.error]})?")
        base = (np.median([x[1] for x in got], axis=0), np.mean([x[2] for x in got], axis=0))
        base = (base[0], base[1] / np.linalg.norm(base[1]))
        print(f"block: centre B {num_list(base[0])}, outward normal B {np.round(base[1], 3).tolist()} "
              f"({len(got)} samples in 1 s)", flush=True)
    hdr = {"what": "S3 press", "finger": f, "joint": jn, "servo": sid, "sign": a.sign, "step_deg": a.step,
           "dwell_s": a.dwell, "speed": a.speed, "start_joints": start, "range": [lo, hi], "F0_N": F0.tolist(),
           "F0_sd_N": sd0.tolist(), "stops": {"max_n": a.max_n, "max_offset": a.max_offset, "max_load": a.max_load,
                                              "max_temp": a.max_temp}, "diag": dg, "splitter": how,
           "mux": str(mux[-1].relative_to(ROOT)) if mux else None, "mounts": s.get("mount_positions"),
           "tracker": "dip_tag" if a.tag else "block" if a.track_block else None,
           "block_base": None if base is None else {"p_B": base[0].tolist(), "n_B": base[1].tolist(),
                                                    "edge_mm": a.block_edge},
           "t": time.time(), "note": a.note}
    fh, raw = open(f"{stem}.jsonl", "w"), open(f"{stem}_raw.jsonl", "w")
    fh.write(json.dumps(hdr) + "\n")
    t_raw = [time.time() - 1.0]

    def dump():
        t1 = time.time()
        tt, vv = ft.ring.since(t_raw[0], t1)
        with tlock:
            tg = [[round(x[0], 4), *np.round(x[1], 3).tolist(), *np.round(x[2], 5).tolist(), x[3]] for x in trk
                  if t_raw[0] <= x[0] < t1]
        raw.write(json.dumps({"hexe_t": tt.tolist(), "hexe": np.round(vv, 4).tolist(),
                              "station": [(r[0], r[1][f], {k: r[2].get(sids[k]) for k in JOINTS})
                                          for r in samp.since(t_raw[0]) if r[1] and r[0] <= t1],
                              "tracker": tg}) + "\n")
        raw.flush()
        t_raw[0] = t1

    def send(target):
        cmd = dict(start)
        cmd[jn] = float(target)
        st.call("POST", "/api/v1/manual/joints", {"joints": {f: cmd}, "servo_speed": a.speed})

    def dwell_and_row(k, target, phase):
        """Waits --dwell s watching the HEX-E and the load; returns (row, stop reason or None)."""
        t0, why = time.time(), None
        while time.time() - t0 < a.dwell:
            time.sleep(0.1)
            t_l, _ = ft.ring.latest()
            if t_l is None or time.time() - t_l > 0.5:
                why = "the HEX-E stopped streaming"
                break
            Fn, _, _ = hexe_mean(ft, 0.2)
            if Fn is not None and np.linalg.norm(Fn - F0) >= a.max_n and phase == "press":
                why = f"|dF| {np.linalg.norm(Fn - F0):.2f} N reached {a.max_n:g} N"
                break
            rs = samp.since(time.time() - 0.2)
            if rs and abs(float(rs[-1][2].get(sid) or 0.0)) >= a.max_load and phase == "press":
                why = f"load {rs[-1][2].get(sid)} reached {a.max_load:g}"
                break
        tw = time.time() - min(1.0, a.dwell / 2)
        Fm, Fsd, nF = hexe_mean(ft, min(1.0, a.dwell / 2))
        rs = samp.since(tw)
        reading = float(np.median([r[1][f][jn] for r in rs if r[1]])) if rs else None
        load = float(np.median([float(r[2].get(sid) or 0.0) for r in rs])) if rs else None
        d = diag(st)
        row = {"k": k, "phase": phase, "t": time.time(), "target": round(target, 3), "reading": reading,
               "offset": None if reading is None else round(a.sign * (target - reading), 3), "load": load,
               "dF_N": None if Fm is None else np.round(Fm - F0, 4).tolist(),
               "dF_abs_N": None if Fm is None else round(float(np.linalg.norm(Fm - F0)), 4),
               "F_sd_N": None if Fsd is None else np.round(Fsd, 4).tolist(), "hexe_n": nF,
               "temp": {k2: (d.get("temperature") or {}).get(sids[k2]) for k2 in JOINTS},
               "status": {k2: (d.get("status") or {}).get(sids[k2]) for k2 in JOINTS},
               "torque": {k2: (d.get("torque_enable") or {}).get(sids[k2]) for k2 in JOINTS}}
        if tracker is not None:
            with tlock:
                tg = [x for x in trk if x[0] >= tw]
            ok = len(tg) >= 3
            pm = np.median([x[1] for x in tg], axis=0) if ok else None
            nm = np.mean([x[2] for x in tg], axis=0) if ok else None
            if a.tag:
                row["tag_P"] = np.round(pm, 2).tolist() if ok else None
                row["tag_n_P"] = np.round(nm / np.linalg.norm(nm), 4).tolist() if ok else None
                row["tag_n"] = len(tg)
            else:
                row["block_B"] = np.round(pm, 3).tolist() if ok else None
                row["block_d_mm"] = round(float((pm - base[0]) @ base[1]), 3) if ok else None   # +: toward the tag's side
                row["block_n"] = len(tg)
        if why is None and phase == "press":
            if any(v not in (None, 0, 0.0) for v in row["status"].values()):
                why = f"a servo flag {row['status']}"
            elif any(v not in (None, 1, 1.0) for v in row["torque"].values()):
                why = f"torque off {row['torque']}"
            elif any(v is not None and float(v) >= a.max_temp for v in row["temp"].values()):
                why = f"temperature {row['temp']}"
            elif row["offset"] is not None and row["offset"] >= a.max_offset:
                why = f"the target {row['offset']:.1f} deg past the reading"
        return row, why

    def track_txt(row):
        if a.tag:
            return f", tag {row.get('tag_P')}"
        if a.track_block:
            return f", block {num(row.get('block_d_mm'), '+.3f')} mm"
        return ""

    why, k, rows = None, 0, []
    try:
        for k in range(1, n_max + 1):
            target = q0 + a.sign * a.step * k
            send(target)
            row, why = dwell_and_row(k, target, "press")
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            dump()
            print(f"{time.strftime('%H:%M:%S')} step {k:2d}: target {target:+6.2f}, reading {num(row['reading'], '+6.2f')} "
                  f"(offset {num(row['offset'], '+5.2f')}), load {num(row['load'], '5.0f')}, |dF| "
                  f"{num(row['dF_abs_N'], '.3f')} N" + track_txt(row) + (f" -- STOP: {why}" if why else ""), flush=True)
            if why:
                break
        else:
            why = f"the target's travel ({n_max} steps) ran out"
        torque_on = all(v in (None, 1, 1.0) for v in rows[-1]["torque"].values()) if rows else True
        if torque_on:
            for kb in range(k - 1, -1, -1):
                target = q0 + a.sign * a.step * kb
                send(target)
                row, _ = dwell_and_row(kb, target, "back")
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                dump()
                print(f"{time.strftime('%H:%M:%S')} back {kb:2d}: target {target:+6.2f}, reading "
                      f"{num(row['reading'], '+6.2f')}, load {num(row['load'], '5.0f')}, |dF| "
                      f"{num(row['dF_abs_N'], '.3f')} N" + track_txt(row), flush=True)
        else:
            print("a servo of the finger lost torque: no step back. Torque on only through hom_setup.py torque-on "
                  "--go, with every MCP under 60 deg", flush=True)
    finally:
        dump()
        raw.close()
        samp.halt = True
        if tracker is not None:
            stop_tracker(arm, tracker)
        fh.write(json.dumps({"end": time.time(), "reason": why, "steps": k}) + "\n")
        fh.close()
        ft.stop = True
    print(f"S3 {f} {jn}: {why}; -> {stem}.jsonl")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "block", "run"):
        p = sub.add_parser(name)
        p.add_argument("--station", default="http://10.99.99.2:8765")
        p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
        if name in ("block", "run"):
            p.add_argument("--out", default=str(ROOT / "logs" / time.strftime("%Y%m%d-sr2_s3_press")))
        if name == "block":
            p.add_argument("--edge", type=float, default=BLOCK_EDGE, help="the printed black square of tag 0, mm")
            p.add_argument("--behind", type=float, default=BEHIND, help="the press pad behind the tag's face, mm")
            p.add_argument("--pad-size", type=float, default=PAD_SIZE, help="the press pad's square, mm")
            p.add_argument("--seconds", type=float, default=3.0)
            p.add_argument("--pad-r", type=float, default=11.0, help="the finger's ball round the pad's centre, mm")
            p.add_argument("--max-deg", type=float, default=30.0)
        if name == "run":
            p.add_argument("--finger", required=True, choices=FINGERS)
            p.add_argument("--joint", required=True, choices=JOINTS)
            p.add_argument("--sign", type=int, required=True, choices=(-1, 1))
            p.add_argument("--step", type=float, default=0.5)
            p.add_argument("--dwell", type=float, default=2.0)
            p.add_argument("--speed", type=int, default=20)
            p.add_argument("--max-travel", type=float, default=25.0, help="deg of target travel at most")
            p.add_argument("--max-n", type=float, default=3.0)
            p.add_argument("--max-offset", type=float, default=18.0)
            p.add_argument("--max-load", type=float, default=900.0)
            p.add_argument("--max-temp", type=float, default=55.0)
            p.add_argument("--track-block", action="store_true", help="track the block's tag through the press")
            p.add_argument("--block-edge", type=float, default=BLOCK_EDGE)
            p.add_argument("--tag", action="store_true", help="track the finger's DIP tag instead")
            p.add_argument("--note", default="")
            p.add_argument("--go", action="store_true")
    a = ap.parse_args()
    return {"check": cmd_check, "block": cmd_block, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
