"""S3 of the identification queue (QUEUE.md; the owner's notes of 2026-11-18 09:0x, 09:55 and 15:0x): the torque scale
at 7.5 V. One joint of one finger steps its target past its reading, --step deg every --dwell s, toward a rigid face that
the finger's pad rests 1-2 mm from (a flat land of the owner's 80/20 block, never across a T-slot). The owner or the arm
places the face; this tool never moves the arm. It records the HEX-E (100 Hz), the station's joints and loads (20 Hz),
each step's servo flags, torque and temperatures (the station's diagnostics) and, with --tag, the finger's DIP tag in
both cameras; the fingertip boards record in the splitter's file. The steps go on past contact until the HEX-E's force
has changed --max-n from its mean before the first step, a servo flags or loses torque, the pressing servo's load
passes --max-load, a servo reaches --max-temp, or the target is --max-offset past the reading. Then the target steps
back the same way to where it started, the tag still recording (W3, the owner's alternative: in the clutch plant the
force stays flat and the reading stays put until the target passes the reading; a gear the load can backdrive lets
both fall before that).

    /usr/bin/python3 s3_press.py check
        the HEX-E's mean and noise over 2 s, the station's torque, joints, loads and temperatures; no motion
    /usr/bin/python3 s3_press.py block [--edge MM] [--profile W[xH]]
        the block's tag36h11 id 0 (owner, 15:0x: "on the back of the 8020 for localization"), no motion: its pose from
        the tracker's solve with the printed edge --edge (calipers) and, when both cameras see it, from its corners
        triangulated in the two cameras, which assumes no size and gives the edge; the faces in P (the tag's face, with
        --profile the face W behind it and the top H above the table); and per finger at its present joints, for each
        joint and sign alone, the degrees until the pad (an --pad-r ball round its centre, or its apex) meets each
        face. One frame per camera is saved with the tag outlined.
    /usr/bin/python3 s3_press.py run --finger index --joint yaw --sign -1 --out OUT [--tag] [--block-edge MM] [--go]
        without --go, the plan and the same checks; --sign is the direction (simulator degrees) that moves the pad
        toward the face; --block-edge locates the block for 2 s before the press and after the step back (the
        cameras serve one tracker at a time) and records how far it moved

OUT/<stamp>_<finger>_<joint>.jsonl: a header, one line per step (target, reading, the pressing servo's load, the
HEX-E's force change from before the first step, mean over the dwell's last second, and with --tag the tag's position
and outward normal in P), an end line with the reason. OUT/<stamp>_<finger>_<joint>_raw.jsonl holds the HEX-E's and the
station's samples and every tag sample (t, p_PO, q_PO wxyz, cameras), press and step back alike. The output is force
against target offset; the workstation fits the gain, the give and the dead band (W2, W3). Prediction for the index's
and middle's yaws (W2, 11-18): the reading advances about 4 deg past first contact by the time the target is 14-15 deg
past it, at about 1.0 N if the gear passes 0.4 of the servo's torque and 2.4 N if it passes all.

The index's and middle's DIP tags face +x_P at zero, away from both cameras; on 2026-11-18 with the palm at z_B 123 mm
they were seen only with MCP 47-58 and PIP 65-80 deg (S1(b)), so --tag records them only in such a pose.
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
NOMINAL_EDGE = 30.0              # mm: the tracker's solve before the edge is measured (its distance scales with it)
TABLE_Z = 3.0                    # mm: the table in B (hom_setup's --table-z)
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
    import tempfile
    from sr2hand.tracking import tags as TG
    f = Path(tempfile.gettempdir()) / f"s3_block_tag{BLOCK_TAG}_{edge:g}mm_map.json"
    f.write_text(json.dumps(TG.plate_map("tag36h11", edge, [BLOCK_TAG], [(0.0, 0.0, 0.0)]).to_json()))
    return str(f)


def locate_block(FT, edge, seconds=3.0, frames_to=None):
    """The block's tag over `seconds`, in B (mm): "joint", the tracker's solve with the printed edge `edge` (centre p_B,
    outward normal n_B, rms px, samples, cameras); "tri", when both cameras see it, its corners triangulated from paired
    frames of the two cameras (gantry_follow_fit.triangulate), which assumes no size (centre, outward normal, right,
    down, edge and its sd, pairs). frames_to: a path stem; one frame per camera is saved there, the tag outlined."""
    import cv2
    sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "scripts/hardware")]
    import gantry_follow_fit as GF
    from sr2hand.geometry.vectors import unit
    from sr2hand.tracking import live
    from sr2hand.tracking import tags as TG
    fused, dets, frames, lock = [], [], {}, threading.Lock()

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
        map = block_map(edge)
    arm, tracker = FT.HT.start_tracker(_A, on, on_frame)
    try:
        time.sleep(1.5 + seconds)
    finally:
        stop_tracker(arm, tracker)
    out = {"tag": BLOCK_TAG, "edge_mm": edge, "seconds": seconds, "dets": dict(Counter(d[0] for d in dets)),
           "joint": None, "tri": None, "frames": {}}
    if fused:
        T = np.array([f[0] for f in fused])
        out["joint"] = {"p_B": np.median(T[:, :3, 3], axis=0).tolist(),
                        "n_B": unit(np.median(-T[:, :3, 2], axis=0)).tolist(),
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


def block_pose_B(loc):
    """(centre, outward normal) of the block's tag in B: the triangulated one when both cameras saw it."""
    src = loc["tri"] or loc["joint"]
    return (None, None) if src is None else (np.array(src["p_B"]), np.array(src["n_B"]))


def faces_P(loc, T_BP, profile, table_z):
    """The block's faces in P, [(name, point, outward normal, inside(x_P) -> bool)]: the tag's face; with a profile
    width W the face W behind it; with a height H the top at table_z + H. inside() checks only what the profile gives:
    the side faces between the table and the top, the top between the two side faces."""
    from sr2hand.geometry.rotations import tf_inv
    T_PB = tf_inv(T_BP)
    pB, nB = block_pose_B(loc)
    R, t = T_PB[:3, :3], T_PB[:3, 3]
    p, n = R @ pB + t, R @ nB
    W = profile[0] if profile else None
    H = profile[1] if profile and len(profile) > 1 else None
    zB = lambda x: float((T_BP[:3, :3] @ x + T_BP[:3, 3])[2])          # noqa: E731
    side = (lambda x: H is None or table_z - 1.0 <= zB(x) <= table_z + H + 1.0)
    out = [("tag face", p, n, side)]
    if W:
        out.append(("opposite face", p - W * n, -n, side))
    if H:
        top = R @ np.array([pB[0], pB[1], table_z + H]) + t
        out.append(("top", top, R @ np.array([0.0, 0.0, 1.0]),
                    lambda x: W is None or -W - 1.0 <= float((x - p) @ n) <= 1.0))
    return out


def approach(FT, f, mount, q, face, pad_r, limits, max_deg=30.0, step=0.25):
    """(the pad's distance from the face's plane now, {(joint, sign): degrees of that joint alone until the pad (a pad_r
    ball round its centre, or its apex) meets the face, or None within max_deg and the station's range}). A meeting
    outside the face's extent (inside() false) does not count, nor does a pad already behind the plane (distance <= 0:
    on the far side of the block, or in it)."""
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
    loc = locate_block(FT, a.edge or NOMINAL_EDGE, a.seconds, frames_to=str(stem))
    P = lambda v: (T_PB[:3, :3] @ np.asarray(v) + T_PB[:3, 3])         # noqa: E731
    R = lambda v: (T_PB[:3, :3] @ np.asarray(v))                        # noqa: E731
    print(f"arm mode {mode}, palm origin B {np.round(T_BP[:3, 3], 1).tolist()}")
    print(f"tag {BLOCK_TAG}: detections {loc['dets'] or 'none'} in {a.seconds:g} s; frames {list(loc['frames'].values())}")
    rec = {"what": "S3 block", "t": time.time(), "arm_mode": mode, "T_BP": T_BP.tolist(), "edge_given_mm": a.edge,
           "profile_mm": a.profile, "table_z": a.table_z, "locate": loc, "mounts": s.get("mount_positions"),
           "joints": HS.sim_joints(s)}
    for k in ("joint", "tri"):
        v = loc[k]
        if v is None:
            continue
        tilt = float(np.degrees(np.arcsin(np.clip(v["n_B"][2], -1, 1))))
        extra = (f", edge {v['edge_mm']:.2f} mm (sd {v['edge_sd_mm']:.2f}, {v['n']} pairs)" if k == "tri" else
                 f" with edge {loc['edge_mm']:g} mm{'' if a.edge else ' (nominal: give --edge)'}, rms "
                 f"{v['rms_px']:.2f} px, {v['n']} samples, cameras {v['cams']}")
        print(f"{'triangulated' if k == 'tri' else 'solve':12s}: centre P {np.round(P(v['p_B']), 1).tolist()} "
              f"(B {np.round(v['p_B'], 1).tolist()}), normal P {np.round(R(v['n_B']), 3).tolist()} ({tilt:+.1f} deg "
              f"from horizontal){extra}")
    if loc["joint"] and loc["tri"]:
        print(f"solve - triangulated: {np.round(np.subtract(loc['joint']['p_B'], loc['tri']['p_B']), 1).tolist()} mm "
              f"in B (a wrong edge scales the solve's distance from the camera)")
    if not (loc["tri"] or (loc["joint"] and a.edge)):
        print("no pose for the faces: both cameras must see the tag, or give --edge (the printed black square, calipers)")
        (stem.with_suffix(".json")).write_text(json.dumps(rec, indent=1))
        return 1
    faces = faces_P(loc, T_BP, a.profile, a.table_z)
    M = S1.model_mounts(HS.load_plan(str(PLAN)), s["mount_positions"])
    j = HS.sim_joints(s)
    limits = st.call("GET", "/api/v1/manual/limits")["joint_deg"]
    rec["faces"] = []
    for face in faces:
        name, p0, n, _ = face
        print(f"{name}: point P {np.round(p0, 1).tolist()}, outward normal P {np.round(n, 3).tolist()}")
        fr = {"name": name, "p_P": p0.tolist(), "n_P": n.tolist(), "fingers": {}}
        for f in FINGERS:
            q = [float(j[f][k]) for k in JOINTS]
            d0, res = approach(FT, f, M[f], q, face, a.pad_r, limits[f], a.max_deg)
            hits = {f"{jn}{sg:+d}": v for (jn, sg), v in res.items() if v is not None}
            fr["fingers"][f] = {"q": q, "gap_mm": d0, "deg_to_face": hits}
            print(f"  {f:6s} q {np.round(q, 1).tolist()}: pad {d0:+6.1f} mm from the plane; "
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
    block0 = None
    if a.block_edge:
        block0 = locate_block(FT, a.block_edge, 2.0, frames_to=f"{stem}_block_before")
        print(f"block before: {block0['dets'] or 'not seen'}; centre B {num_list(block_pose_B(block0)[0])}", flush=True)
    samp = Sampler(HS, st)
    samp.start()
    tags, tlock, arm, tracker = [], threading.Lock(), None, None
    if a.tag:
        def on(smp):
            m = smp.msg
            if m is not None and m.n_tags > 0 and np.all(np.isfinite(m.p_PO)):
                with tlock:
                    tags.append((time.time(), np.array(m.p_PO, float), np.array(m.q_PO, float), int(m.n_cams)))

        class _A:
            map = FT.tag_map_file(FT.TAG[f])
        arm, tracker = FT.HT.start_tracker(_A, on)
    hdr = {"what": "S3 press", "finger": f, "joint": jn, "servo": sid, "sign": a.sign, "step_deg": a.step,
           "dwell_s": a.dwell, "speed": a.speed, "start_joints": start, "range": [lo, hi], "F0_N": F0.tolist(),
           "F0_sd_N": sd0.tolist(), "stops": {"max_n": a.max_n, "max_offset": a.max_offset, "max_load": a.max_load,
                                              "max_temp": a.max_temp}, "diag": dg, "splitter": how,
           "mux": str(mux[-1].relative_to(ROOT)) if mux else None, "mounts": s.get("mount_positions"),
           "tag": a.tag, "block_before": block0, "t": time.time(), "note": a.note}
    fh, raw = open(f"{stem}.jsonl", "w"), open(f"{stem}_raw.jsonl", "w")
    fh.write(json.dumps(hdr) + "\n")
    t_raw = [time.time() - 1.0]

    def dump():
        t1 = time.time()
        tt, vv = ft.ring.since(t_raw[0], t1)
        with tlock:
            tg = [[round(x[0], 4), *np.round(x[1], 3).tolist(), *np.round(x[2], 5).tolist(), x[3]] for x in tags
                  if t_raw[0] <= x[0] < t1]
        raw.write(json.dumps({"hexe_t": tt.tolist(), "hexe": np.round(vv, 4).tolist(),
                              "station": [(r[0], r[1][f], {k: r[2].get(sids[k]) for k in JOINTS})
                                          for r in samp.since(t_raw[0]) if r[1] and r[0] <= t1],
                              "tag": tg}) + "\n")
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
        if a.tag:
            with tlock:
                tg = [x for x in tags if x[0] >= tw]
            ok = len(tg) >= 3
            row["tag_P"] = np.round(np.median([x[1] for x in tg], axis=0), 2).tolist() if ok else None
            nz = -np.mean([quat_to_mat(x[2])[:, 2] for x in tg], axis=0) if ok else None
            row["tag_n_P"] = np.round(nz / np.linalg.norm(nz), 4).tolist() if ok else None
            row["tag_n"] = len(tg)
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

    why, k, rows, block1, moved = None, 0, [], None, None
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
                  f"{num(row['dF_abs_N'], '.3f')} N"
                  + (f", tag {row.get('tag_P')}" if a.tag else "") + (f" -- STOP: {why}" if why else ""), flush=True)
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
                      f"{num(row['dF_abs_N'], '.3f')} N" + (f", tag {row.get('tag_P')}" if a.tag else ""), flush=True)
        else:
            print("a servo of the finger lost torque: no step back. Torque on only through hom_setup.py torque-on "
                  "--go, with every MCP under 60 deg", flush=True)
    finally:
        dump()
        raw.close()
        samp.halt = True
        if tracker is not None:
            stop_tracker(arm, tracker)
        if a.block_edge:
            try:
                block1 = locate_block(FT, a.block_edge, 2.0, frames_to=f"{stem}_block_after")
            except Exception as e:                       # noqa: BLE001 - the press's record comes first
                block1 = {"error": repr(e)}
        if block0 and block1 and "error" not in block1 and block_pose_B(block0)[0] is not None \
                and block_pose_B(block1)[0] is not None:
            moved = round(float(np.linalg.norm(block_pose_B(block1)[0] - block_pose_B(block0)[0])), 2)
        fh.write(json.dumps({"end": time.time(), "reason": why, "steps": k, "block_after": block1,
                             "block_moved_mm": moved}) + "\n")
        fh.close()
        ft.stop = True
    print(f"S3 {f} {jn}: {why}" + (f"; the block moved {moved} mm" if moved is not None else "") + f"; -> {stem}.jsonl")
    return 0


def profile_arg(s):
    v = [float(x) for x in s.lower().split("x")]
    if not 1 <= len(v) <= 2:
        raise argparse.ArgumentTypeError("W or WxH in mm")
    return v


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
            p.add_argument("--edge", type=float, help="the printed black square of tag 0, mm (calipers)")
            p.add_argument("--profile", type=profile_arg, help="W or WxH, mm: the block's depth behind the tag's face "
                                                               "and its height above the table")
            p.add_argument("--seconds", type=float, default=3.0)
            p.add_argument("--table-z", type=float, default=TABLE_Z)
            p.add_argument("--pad-r", type=float, default=11.0, help="the pad's ball round its centre, mm")
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
            p.add_argument("--tag", action="store_true", help="record the finger's DIP tag with the tracker")
            p.add_argument("--block-edge", type=float, help="locate the block (tag 0, this edge in mm) before and after")
            p.add_argument("--note", default="")
            p.add_argument("--go", action="store_true")
    a = ap.parse_args()
    return {"check": cmd_check, "block": cmd_block, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
