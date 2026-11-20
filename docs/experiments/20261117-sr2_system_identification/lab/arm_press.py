"""Arm presses on one fingertip pad (2026-11-19 evening). The owner, 16:3x: "the contact area vs force sensing is really
disconcerting. is there any way to estimate the size of the contact patch and thus get better performance. can we try
to learn that from our existing calibrations?" and "you have my go ahead to execute plans to further our understanding
of the existing models"; 16:4x: "you can calibrate more on the post, you can also run experiments on the sphere (which
you can pick up in the event of a knockover)".

The arm, not a finger joint, pushes one pad into a rigid indenter along the indenter's normal while the finger's servos
hold their targets (the gears do not backdrive). The indenters: the S3 block's taped plate (flat), the plate's free
vertical edge (a 1 mm plate under about 1 mm of tape: a line along the finger) and the sphere's top (R 20.75 mm with its
tape), the finger curled so that its pad faces down. The same place on the pad (near the apex) meets each indenter, so
the boards' field per newton of the HEX-E shows how the contact's size changes it. The MCP presses of 11-18/19 read
0.26-0.36 of the HEX-E with the pad flat on the plate (PIP 0) and 0.51-0.60 with the PIP bent 15-30 deg.

    /usr/bin/python3 arm_press.py sphere-plan --finger index [--gap 6]
        NOT USABLE on this hand (17:17): a curled finger's pad faces the palm, up, so no pad faces down onto the sphere's
        top; the plan refuses on the pad's facing (it found 173 deg off straight down at MCP 30, PIP 57).
        no motion: the sphere from the tracker and the S3 block's tag; the target (the finger's apex --gap above the
        ball's top, the finger at its present joints, curled beforehand with s3_press.py pose) and the legs (level at
        the present height, then straight down), every 2 mm of them each finger part's clearance (the cad_v3 collision
        spheres and hom_setup's pad, tip-end and PIP balls) to the ball, the stem, the block, the table and the S3
        block; writes OUT/<stamp>_sphere_<finger>.json
    /usr/bin/python3 arm_press.py go|back --plan PLAN [--go]
        s3_approach.py's legs and checks, and the sphere within 1.5 mm of where the plan saw it
    /usr/bin/python3 arm_press.py press --plan PLAN [--target 2.5] [--reps 3] [--indenter TEXT] [--go]
        PLAN an s3_approach.py plan (the press runs into its plate along the plate's level normal) or a sphere plan
        (straight down). From the plan's end pose: --v-fast (1 mm/s) to --gap-air mm short of the plan's gap, then --v
        (0.2 mm/s) until the force along the press reaches --target (guards: |dF| --max-n 3 N, a finger servo's load
        --max-load, any flag or torque loss, --travel mm past the plan's gap), --hold s still, back at --v to 0.5 mm
        short of where the force first passed 0.1 N, then --v-fast to the start; --reps times with --rest s between.
        The splitter records the boards; OUT/<stamp>_<finger>_arm_press.jsonl holds a header and one line per press,
        _raw.jsonl the HEX-E (100 Hz), the TCP (50 Hz) and the station (20 Hz) of each press.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s3_press as S3  # noqa: E402
import s3_approach as S3A  # noqa: E402
import s4b_drag as S4  # noqa: E402

ROOT = S3.ROOT
FINGERS, JOINTS = S3.FINGERS, S3.JOINTS
SPHERE_MAP = ROOT / "docs/experiments/20261115-hom_bench_trials/maps/20261116-sphere_fitted_map_O.json"
R_BALL = 20.75                   # mm: the 40 mm sphere under ~0.75 mm of tape (pinch_pickup.R_TAPED)


class TcpForceSampler(S4.TcpSampler):
    """The TCP pose and the UR's own TCP force estimate (getActualTCPForce, from the joints' currents: an independent,
    coarse check of the HEX-E's scale) at about hz: rows (t, pose 6, force 6)."""

    def run(self):
        while not self.halt:
            try:
                p, F = self.r.getActualTCPPose(), self.r.getActualTCPForce()
                with self.lock:
                    self.rows.append((time.time(), [round(v, 6) for v in p], [round(v, 3) for v in F]))
            except Exception:                             # noqa: BLE001
                pass
            time.sleep(self.dt)
        self.r.disconnect()


def sphere_caps(c, up, h, HS):
    """The sphere as capsules [(name, a, b, r)] in B (mm): ball, stem, block (pinch_pickup.object_caps)."""
    up = np.asarray(up, float) / np.linalg.norm(up)
    c = np.asarray(c, float)
    return [("ball", c, c, R_BALL), ("stem", c - R_BALL * up, c - (h - HS.BLOCK_H) * up, HS.STEM_R),
            ("block", c - (h - HS.BLOCK_H) * up, c - h * up, HS.BLOCK_R)]


def seg_dists(X, a, b):
    ab = b - a
    L = float(ab @ ab)
    t = np.zeros(len(X)) if L < 1e-12 else np.clip((X - a) @ ab / L, 0.0, 1.0)
    return np.linalg.norm(X - (a + t[:, None] * ab), axis=1)


def locate_sphere(HS, map_file, seconds=1.5):
    class _A:
        map = str(map_file)
    meas = HS.Measure(_A)
    try:
        from sr2hand.geometry.rotations import quat_to_mat
        p, q, info = meas.pose(seconds, settle=0.0)
        R_B = quat_to_mat(np.asarray(info["q_B"], float))
    finally:
        meas.stop()
    c = np.asarray(info["p_B"], float)
    up = R_B[:, 1]
    tilt = math.degrees(math.acos(float(np.clip(up[2], -1.0, 1.0))))
    return c, up, tilt, info


def cmd_sphere_plan(a):
    FT, HS, SL = S3.imports()
    import s1b_tags as S1
    T0, mode = S1.arm_T_BP(FT, HS)
    st, s, M, joints = S3A.state(HS, a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{time.strftime('%Y%m%d-%H%M%S')}_sphere_{a.finger}"
    c, up, tilt, info = locate_sphere(HS, a.map)
    if tilt > 5.0:
        raise SystemExit(f"the sphere stands {tilt:.1f} deg off vertical: not on its block as expected")
    h = HS.map_height(a.map)
    loc = S3.locate_block(FT, a.edge, 2.0, frames_to=str(stem))
    mdl = S3A.model(loc, a.tape, a.noise_l, a.noise_z, 2.0) if S3.block_frame_B(loc) is not None else None
    f = a.finger
    q = [float(joints[f][k]) for k in JOINTS]
    _, _, tip, con = FT.fk(f, M[f], q)
    n_P = (con - tip) / np.linalg.norm(con - tip)                  # the apex's outward normal in P
    n_B = T0[:3, :3] @ n_P
    off = math.degrees(math.acos(float(np.clip(-n_B[2], -1.0, 1.0))))
    if off > a.max_off:
        raise SystemExit(f"the {f}'s pad faces {off:.1f} deg off straight down at {joints[f]} (limit {a.max_off:g}): "
                         "curl it first (s3_press.py pose --finger F --mcp M --pip P --go, MCP + PIP about 87)")
    top = c + R_BALL * up
    target = top + np.array([0.0, 0.0, a.gap])
    con_B = T0[:3, :3] @ con + T0[:3, 3]
    delta = target - con_B
    if delta[2] > 0:
        raise SystemExit(f"the target lies {delta[2]:.1f} mm above the apex: plan from higher")
    T_l = S3A.shifted(T0, [delta[0], delta[1], 0.0])
    T_d = S3A.shifted(T0, delta)
    pts = S3A.hand_points(FT, HS, M, joints)
    X_P, R, lab = pts
    caps = sphere_caps(c, up, h, HS)
    pad_i = [i for i, lb in enumerate(lab) if lb == (f, "tip pad")]
    # two descents, each guarded from its own start: the hanging cable bundle adds force over a long descent (a 149 mm
    # descent tripped 3 N with nothing touched on 11-19 at 12:1x)
    T_m = S3A.shifted(T_d, [0.0, 0.0, min(a.last_leg, float(T_l[2, 3] - T_d[2, 3]))])
    legs = [{"name": "level", "kind": "lin", "T0": T0, "T1": T_l},
            {"name": "descend", "kind": "lin", "T0": T_l, "T1": T_m},
            {"name": "descend", "kind": "lin", "T0": T_m, "T1": T_d}]
    ok, rec_legs = True, []
    for leg in legs:
        least = {k: (math.inf, None) for k in ("pad to ball", "ball", "stem", "block", "table", "S3 block")}

        def keep(key, g, i, u):
            if g < least[key][0]:
                least[key] = (float(g), f"{lab[i][0]} {lab[i][1]} at {u:.2f}")

        for u, T in S3A.leg_poses(leg):
            X = X_P @ T[:3, :3].T + T[:3, 3]
            for name, pa, pb, cr in caps:
                g = seg_dists(X, pa, pb) - cr - R
                if name == "ball":
                    gp = g.copy()
                    gp[[i for i in range(len(g)) if i not in pad_i]] = np.inf
                    i = int(np.argmin(gp))
                    keep("pad to ball", gp[i], i, u)
                    g[pad_i] = np.inf
                i = int(np.argmin(g))
                keep(name, g[i], i, u)
            gz = X[:, 2] - R - S3A.TABLE_Z
            i = int(np.argmin(gz))
            keep("table", gz[i], i, u)
            if mdl is not None:
                p_, nh, w, boxes = mdl
                for bname, box in boxes.items():
                    g, _ = S3A.box_gaps(X, R, p_, nh, w, box)
                    i = int(np.argmin(g))
                    keep("S3 block", g[i], i, u)
        cl = {k: {"mm": None if v[1] is None else round(v[0], 2), "where": v[1]} for k, v in least.items()}
        floor = {"pad to ball": a.gap - 1.0, "table": a.min_table}
        bad = [k for k, v in cl.items() if v["mm"] is not None and v["mm"] < floor.get(k, a.min_clear)]
        if leg is legs[-1] and cl["ball"]["mm"] is not None and cl["ball"]["mm"] < a.gap + a.rigid_margin:
            bad.append(f"rigid parts {cl['ball']['mm']:.1f} mm from the ball at the target (need gap + "
                       f"{a.rigid_margin:g})")
        ok &= not bad
        rec_legs.append({**{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in leg.items()}, "clear": cl,
                         "bad": bad})
        move = np.round(leg["T1"][:3, 3] - leg["T0"][:3, 3], 1).tolist()
        print(f"  {leg['name']:7s} {move} mm: " + "; ".join(f"{k} {S3A.num(v['mm'])} ({v['where']})"
                                                          for k, v in cl.items()) + (f"  -- TOO CLOSE: {bad}" if bad else ""))
    print(f"arm mode {mode}; palm origin B {np.round(T0[:3, 3], 1).tolist()}; sphere centre B {np.round(c, 1).tolist()} "
          f"({c[2] - S3A.TABLE_Z:.1f} mm above the table, tilt {tilt:.1f} deg), top {np.round(top, 1).tolist()}")
    print(f"{f} at {joints[f]}: the apex faces {off:.1f} deg off straight down; it goes {a.gap:g} mm above the ball's "
          f"top; the palm ends at {np.round(T_d[:3, 3], 1).tolist()}")
    rec = {"what": "arm press sphere plan", "t": time.time(), "finger": f, "indenter": "sphere top", "gap_mm": a.gap,
           "travel_mm": a.travel, "sphere_B": c.tolist(), "sphere_up_B": up.tolist(), "sphere_tilt_deg": tilt,
           "sphere_r_mm": R_BALL, "map": str(a.map), "map_height_mm": h, "apex_off_deg": off, "arm_mode": mode,
           "T_BP0": T0.tolist(), "mounts_station": s.get("mount_positions"), "mounts_P": M, "joints": joints,
           "tag_B": None if mdl is None else mdl[0].tolist(), "edge_mm": a.edge, "lead_P": con.tolist(),
           "target_B": target.tolist(), "press_dir_B": [0.0, 0.0, -1.0], "T_BP_end": T_d.tolist(), "legs": rec_legs,
           "ok": ok}
    stem.with_suffix(".json").write_text(json.dumps(rec, indent=1, default=float))
    print(("plan clear" if ok else "plan NOT clear") + f" -> {stem}.json")
    return 0 if ok else 1


def cmd_legs(a, back):
    rec = json.loads(Path(a.plan).read_text())
    if rec.get("what") == "arm press sphere plan":
        FT, HS, SL = S3.imports()
        c, up, tilt, info = locate_sphere(HS, rec["map"])
        moved = float(np.linalg.norm(c - np.array(rec["sphere_B"])))
        print(f"sphere {moved:.2f} mm from the plan's, tilt {tilt:.1f} deg")
        if moved > 1.5 or tilt > 5.0:
            raise SystemExit("the sphere moved since the plan: plan again (or pick it up)")
    return S3A.run_legs(a, back)


def press_dir(rec):
    if rec.get("press_dir_B") is not None:
        return np.array(rec["press_dir_B"], float)
    nh = np.array(rec["nh_B"], float)
    return nh if rec.get("plate", "pad") == "pad" else -nh        # into the plate (s3_approach.cmd_plan's d_in)


def cmd_press(a):
    rec = json.loads(Path(a.plan).read_text())
    if not rec.get("ok"):
        raise SystemExit("the plan is not clear")
    FT, HS, SL = S3.imports()
    import s1b_tags as S1
    from sr2hand.geometry.rotations import tf_to_ur
    P, SV = SL._driver()
    f = rec["finger"]
    d = press_dir(rec)
    d = d / np.linalg.norm(d)
    T_end = np.array(rec["T_BP_end"])
    gap = float(rec["gap_mm"])
    T_BP, mode = S1.arm_T_BP(FT, HS)
    dp = float(np.linalg.norm(T_BP[:3, 3] - T_end[:3, 3]))
    dr = math.degrees(math.acos(float(np.clip((np.trace(T_BP[:3, :3].T @ T_end[:3, :3]) - 1) / 2, -1, 1))))
    indenter = a.indenter or rec.get("indenter") or f"{rec.get('plate', 'pad')} plate face" + (
        f" (dl {rec['dl']:+g} mm)" if rec.get("dl") else "")
    print(f"{f}: {indenter}; press along {np.round(d, 3).tolist()} in B from the plan's end ({dp:.2f} mm, {dr:.2f} deg "
          f"from here; arm mode {mode}); {a.v_fast:g} mm/s for {gap - a.gap_air:.1f} mm, then {a.v:g} mm/s to "
          f"{a.target:g} N along the press (at most {gap + a.travel:.1f} mm in all), hold {a.hold:g} s, back; "
          f"x{a.reps}, rest {a.rest:g} s")
    if mode != 7:
        raise SystemExit("the arm is not running")
    if dp > 1.0 or dr > 0.5:
        raise SystemExit("the hand is not at the plan's end pose: run go first")
    if not a.go:
        return 0
    import tip_contacts as TC
    pids, how = TC.ensure_splitter(start=True)
    st = HS.Station(a.station, a.token_file)
    s = st.state()
    if s.get("servo_torque") != 1:
        raise SystemExit("servo torque is not on")
    sids = {(ff, k): str(SV.FINGER_JOINTS[P.FINGER_ID[ff]][P.SIM_JOINT_TO_SERVO[k]][0]) for ff in FINGERS for k in JOINTS}
    mine = [sids[(f, k)] for k in JOINTS]
    arm = HS.Arm(False, 0.4)
    arm.connect(True)
    ft = arm.ft
    time.sleep(1.5)
    R_PH = S3.hexe_in_P()
    samp = S3.Sampler(HS, st)
    samp.start()
    tcp = TcpForceSampler(FT.HT.ROBOT)
    tcp.start()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{time.strftime('%Y%m%d-%H%M%S')}_{f}_arm_press"
    fh = open(f"{stem}.jsonl", "w")
    raw = open(f"{stem}_raw.jsonl", "w")

    def write(fobj, r):
        fobj.write(json.dumps(r, default=float) + "\n")
        fobj.flush()
        os.fsync(fobj.fileno())

    start_joints = HS.sim_joints(s)
    write(fh, {"what": "arm press", "finger": f, "indenter": indenter, "plan": Path(a.plan).name, "press_dir_B": d.tolist(),
               "gap_mm": gap, "gap_air_mm": a.gap_air, "travel_mm": a.travel, "target_N": a.target, "max_n": a.max_n,
               "max_load": a.max_load, "v_mm_s": a.v, "v_fast_mm_s": a.v_fast, "hold_s": a.hold, "reps": a.reps,
               "rest_s": a.rest, "R_PH": R_PH.tolist(), "T_BP_end": T_end.tolist(), "joints": start_joints,
               "splitter": how, "note": a.note, "t": time.time()})
    T_EP_inv = np.linalg.inv(HS.T_EP)
    t_last_diag = [0.0]

    def F_B(F0):
        """The HEX-E's force change in B, the mean of its last 0.1 s (single samples scatter +-0.2 N at rest; the free
        air check of 16:55 tripped a 0.5 N stop on one), or None if the newest sample is over 0.1 s old."""
        t, v = ft.ring.since(time.time() - 0.1)
        if len(t) < 3 or time.time() - float(np.max(t)) > 0.1:
            return None
        R_BP = arm.T_BP()[:3, :3]
        return R_BP @ R_PH @ (np.asarray(v, float)[:, :3].mean(0) - F0)

    def servo_check():
        rows = samp.since(time.time() - 0.3)
        if rows:
            loads = rows[-1][2]
            hi = {k: v for k, v in loads.items() if k in mine and v is not None and abs(float(v)) > a.max_load}
            if hi:
                return f"load {hi} over {a.max_load:g}"
        if time.time() - t_last_diag[0] > 0.5:
            t_last_diag[0] = time.time()
            dg = S3.diag(st)
            bad = {k: (dg.get("status") or {}).get(k) for k in mine if (dg.get("status") or {}).get(k) not in (None, 0, 0.0)}
            off = {k: (dg.get("torque_enable") or {}).get(k) for k in mine
                   if (dg.get("torque_enable") or {}).get(k) not in (None, 1, 1.0)}
            if bad or off:
                return f"flags {bad}, torque off {off}"
        return None

    def line_to(dist_mm, v_mm_s, F0, stop_along=None, label=""):
        """moveL from the plan's end to dist_mm along d at v; stops on the force along d reaching stop_along, |dF| past
        max_n, a servo, or the end. Returns (reason or None, (t, s mm, F along d) samples, moved to s mm)."""
        p0 = T_end[:3, 3]
        T_goal = S3A.shifted(T_end, d * dist_mm) @ T_EP_inv
        s_from = float((arm.T_BP()[:3, 3] - p0) @ d)
        max_s = abs(dist_mm - s_from) / v_mm_s + 10.0
        arm.c.moveL(tf_to_ur(T_goal).tolist(), v_mm_s / 1e3, a.acc, True)
        why, rows = None, []
        t0 = time.time()
        time.sleep(0.02)
        while True:
            Fb = F_B(F0)
            pos = arm.T_BP()[:3, 3]
            sd = float((pos - p0) @ d)
            if Fb is None:
                why = "no HEX-E sample for 100 ms"
            else:
                fa = float(-Fb @ d)
                rows.append((time.time(), sd, fa))
                if np.linalg.norm(Fb) > a.max_n:
                    why = f"|dF| {np.linalg.norm(Fb):.2f} N over {a.max_n:g}"
                elif stop_along is not None and abs(fa) >= stop_along and time.time() - t0 > a.blank_s:
                    why = "target"
            if why is None:
                why = servo_check()
            if why is None and (arm.r.isProtectiveStopped() or arm.r.isEmergencyStopped()):
                why = "protective or emergency stop"
            if why is None and time.time() - t0 > max_s:
                why = f"move over {max_s:.0f} s"
            if why:
                arm.c.stopL(2.0)
                break
            if arm.c.getAsyncOperationProgress() < 0 and time.time() - t0 > 0.2:
                break
            time.sleep(0.01)
        time.sleep(0.15)
        s_end = float((arm.T_BP()[:3, 3] - p0) @ d)
        print(f"   {label}: {why or 'end'} at s {s_end:.2f} mm", flush=True)
        return why, rows, s_end

    ended = None
    try:
        for rep in range(1, a.reps + 1):
            if rep > 1:
                time.sleep(a.rest)
            t_start = time.time()
            F0, sd0, n0 = S3.hexe_mean(ft, 1.0)
            if F0 is None:
                ended = "the HEX-E stopped"
                break
            why1, r1, s1 = line_to(gap - a.gap_air, a.v_fast, F0, stop_along=a.air_n, label=f"press {rep} air")
            if why1 not in (None, "target"):
                ended = f"press {rep}, air leg: {why1}"
            r2, s2, why2 = [], s1, why1
            if ended is None:
                why2, r2, s2 = line_to(gap + a.travel, a.v, F0, stop_along=a.target, label=f"press {rep} load")
                if why2 not in ("target",):
                    ended = f"press {rep}, load leg: {why2 or 'no target force within the travel'}"
            t_hold = time.time()
            if why2 == "target":
                time.sleep(a.hold)
            Fh, _, _ = S3.hexe_mean(ft, min(1.0, a.hold))
            rows = r1 + r2
            s_touch = next((sv for _, sv, fa in rows if abs(fa) > 0.1), None)
            back_to = max(0.0, (s_touch if s_touch is not None else s2) - 0.5)
            why3, r3, s3 = line_to(back_to, a.v, F0, stop_along=None, label=f"press {rep} unload")
            why4, r4, s4 = line_to(0.0, a.v_fast, F0, stop_along=None, label=f"press {rep} back")
            t_end = time.time()
            F1, _, _ = S3.hexe_mean(ft, 1.0)
            peak = max((abs(fa) for _, _, fa in rows), default=0.0)
            write(fh, {"rep": rep, "t_start": t_start, "t_hold": t_hold, "t_end": t_end, "F0": F0.tolist(),
                       "F0_sd": sd0.tolist(), "F_after": None if F1 is None else F1.tolist(),
                       "F_hold_B": None if Fh is None else (arm.T_BP()[:3, :3] @ R_PH @ (Fh - F0)).tolist(),
                       "s_touch_mm": s_touch, "s_peak_mm": s2, "peak_along_N": peak,
                       "stops": [why1, why2, why3, why4], "unload_to_mm": back_to})
            tt, vv = ft.ring.since(t_start - 1.0, time.time())
            write(raw, {"rep": rep, "t0": t_start - 1.0, "t1": time.time(), "hexe_t": np.asarray(tt).tolist(),
                        "hexe": np.round(np.asarray(vv), 4).tolist(), "tcp": tcp.since(t_start - 1.0),
                        "station": [(r[0], r[1].get(f), {k: r[2].get(sids[(f, k)]) for k in JOINTS})
                                    for r in samp.since(t_start - 1.0) if r[1]]})
            print(f"press {rep}: touch at s {S3A.num(s_touch)} mm, peak {peak:.2f} N at s {s2:.2f} mm; back "
                  f"({', '.join(str(x) for x in (why3, why4) if x)})", flush=True)
            if ended or why3 not in (None,) or why4 not in (None,):
                ended = ended or f"press {rep}: back legs {why3}, {why4}"
                break
    finally:
        samp.halt = tcp.halt = True
        write(fh, {"end": ended or "done", "t": time.time()})
        fh.close()
        raw.close()
        arm.close()
        ft.stop = True
    print(f"{'ENDED: ' + ended if ended else 'done'} -> {stem}.jsonl", flush=True)
    return 1 if ended else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sphere-plan", "go", "back", "press"):
        p = sub.add_parser(name)
        p.add_argument("--station", default="http://10.99.99.2:8765")
        p.add_argument("--token-file", default=str(Path.home() / ".sr2_station_token"))
        p.add_argument("--out", default=str(ROOT / "logs" / time.strftime("%Y%m%d-sr2_s3_press")))
        if name == "sphere-plan":
            p.add_argument("--finger", required=True, choices=FINGERS)
            p.add_argument("--map", default=str(SPHERE_MAP))
            p.add_argument("--gap", type=float, default=6.0)
            p.add_argument("--travel", type=float, default=8.0, help="mm past the gap the press may go (recorded)")
            p.add_argument("--rigid-margin", type=float, default=3.0,
                           help="mm: at the target the finger's rigid parts stand at least gap + this from the ball")
            p.add_argument("--last-leg", type=float, default=40.0, help="mm: the last descent, guarded on its own")
            p.add_argument("--max-off", type=float, default=8.0, help="deg: the apex's normal off straight down")
            p.add_argument("--min-clear", type=float, default=5.0)
            p.add_argument("--min-table", type=float, default=15.0)
            p.add_argument("--tape", type=float, default=1.0)
            p.add_argument("--noise-l", type=float, default=4.0)
            p.add_argument("--noise-z", type=float, default=3.0)
            p.add_argument("--edge", type=float, default=S3.BLOCK_EDGE)
        else:
            p.add_argument("--plan", required=True)
            p.add_argument("--go", action="store_true")
        if name in ("go", "back"):
            p.add_argument("--v-up", type=float, default=0.015)
            p.add_argument("--v-level", type=float, default=0.015)
            p.add_argument("--v-down", type=float, default=0.008)
            p.add_argument("--acc", type=float, default=0.05)
            p.add_argument("--stop-n", type=float, default=3.0)
            p.add_argument("--stop-down", type=float, default=2.0)
            p.add_argument("--ur-ft", action="store_true")
            p.add_argument("--blank-s", type=float, default=0.4)
        if name == "press":
            p.add_argument("--target", type=float, default=2.5, help="N along the press")
            p.add_argument("--max-n", type=float, default=3.0, help="N: |dF| absolute guard")
            p.add_argument("--max-load", type=float, default=500.0, help="a finger servo's load that stops the press")
            p.add_argument("--air-n", type=float, default=0.5, help="N along the press that ends the fast leg early")
            p.add_argument("--gap-air", type=float, default=2.5, help="mm short of the plan's gap where the slow leg starts")
            p.add_argument("--travel", type=float, default=3.0, help="mm past the plan's gap at most")
            p.add_argument("--v", type=float, default=0.2, help="mm/s of the load and unload")
            p.add_argument("--v-fast", type=float, default=1.0, help="mm/s of the air legs")
            p.add_argument("--acc", type=float, default=0.05, help="m/s^2")
            p.add_argument("--hold", type=float, default=2.0)
            p.add_argument("--blank-s", type=float, default=0.4, help="s at a move's start when the force stop waits")
            p.add_argument("--reps", type=int, default=3)
            p.add_argument("--rest", type=float, default=5.0)
            p.add_argument("--indenter", default=None)
            p.add_argument("--note", default="")
    a = ap.parse_args()
    if a.cmd == "sphere-plan":
        return cmd_sphere_plan(a)
    if a.cmd in ("go", "back"):
        return cmd_legs(a, a.cmd == "back")
    return cmd_press(a)


if __name__ == "__main__":
    sys.exit(main())
