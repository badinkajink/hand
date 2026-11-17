"""HOM's press in simulation against the hand's two press trials of 2026-11-16 (autopilot hand-20261116-press_sim_real,
2026-11-17, step 3; owner, 2026-11-16 23:0x: "sure, send off a background autopilot run to churn on the press task
calibration"; "we have like 1.5 good trials").

The trials (docs/experiments/20261115-hom_bench_trials/runs/20261116-131243_* and 20261116-131932_*, layout (70, 110) mm,
the 25 mm cylinder with the sphere's stem and tag block): the generalist press (plans/20261115-generalist_press_plan.json:
the task-conflict study's S3 grasp, the pinch 15.9 deg below the cylinder's axis, the middle pressing from 30.1 deg
below, HOM's forces f_pinch 1.87 N and f_press 0.37 N) and the pinch raised to the axis (plans/
20261116-press_x70_y110_lat0_plan.json: the same grasp at 0 deg, the palm 6.3 mm higher, HOM's default 3.0 and 0.5 N).
Each runs here as the bench server ran it: HomBench of the plan rebuilt from its GPU world (hom_bench_world.
world_from_plan) at the bench's 50 Hz, HOM converting its forces with the bench's servo gain
(HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011), no torque budget, no fingertip contacts (the hand streamed none), and the
server's 0.5 s after the schedule's end (hom_server --tail-s).

  run        the calibrated plant of the twists (rev_clutch2.py --stiff; rev_predict.py's): gears the load cannot backdrive
             (forward friction 0.6 |servo torque|, joints held against any backdriving load, joint friction that does
             not creep), kp 0.55, damping 0.011, the index's and the middle's yaw at kp 0.35, friction 1.0, on the press
             scenes of press_stem.py (pads or point contact, impratio 1 or 100, noslip 0 or 30): the object stands on
             its block on the table from the start, nothing holds it, and the table goes at 1.6 s (the arm's lift).
  --station  the station's refusals as on the hand: a frame whose target lies outside the station's servo range
             (manta_hand.servos.FINGER_JOINTS, compared as runtime._validate_sim_pose does, after the link's float32)
             is not applied, the servos keep the last accepted targets, and HOM's next joints are those read back after
             the last accepted frame, or HOM's own reference once they are older than 60 ms (hom_server
             --max-joint-age-ms). HOM sets a joint at its model's limit, -16.110026 deg for the middle PIP, beyond the
             station's -16.11.
  pipeline   the design pipeline's model, as blind_pipeline.py runs it for the twist: HOM converting at 0.72 and the
             SCS0009 at 0.72 N m/rad that the load can backdrive (HOS_PLANT=servo), the scenes' stiff contact (pads,
             impratio 100), the bare 25 g cylinder on its post, held at its start pose up to the plan's hold_s, HOM at
             the GPU world's 100 Hz with the plant's contacts.

Per run: the tracking score as scripts/hardware/hom_trial_tracking.py defines it for the press (err = RMS angle between
the axis and its 2.0 s pose turned by -theta_ref about x_P, over 2.0 s to t_end; track = max(0, 1 - err/ref)); every
0.1 s the press angle (the axis's turn about x_P from its first pose, + in theta_ref's sense), the axis, the object's
position and the middle finger's MCP and PIP (measured and target, servo degrees); held at t_end and at the run's end
(hom_tasks.metrics: not 15 mm below its start, two fingers loaded); the object's lowest height after 2.0 s against its
height at 2.0 s, and the first time it is 5 mm below its start; the frames refused and the periods HOM ran on its
reference. Rows are appended and fsynced as they finish, one per run.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 OMP_NUM_THREADS=1 .venv/bin/python \\
        logs/20261116-sim_real_reversal/press_predict.py run OUT.jsonl [--scenes NAME ...] [--station] [--contacts] \\
        [--plans generalist lat0] [--seeds 0 1 2] [--workers 8]
    HOS_PLANT=servo HOM_ROWS_TAG=_sv25 OMP_NUM_THREADS=1 .venv/bin/python \\
        logs/20261116-sim_real_reversal/press_predict.py pipeline OUT.jsonl [--workers 6]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("HOS_SCENES", "logs/20261102-hom_servo_plant/scenes")     # hom_server.SCENES_2P5
os.environ.setdefault("HOM_ROWS_TAG", "_bench")
os.environ["CUDA_VISIBLE_DEVICES"] = ""                                           # hom_gpu imports warp; no kernel runs
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")
import numpy as np  # noqa: E402

ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
for p in (D, ROOT / "scripts/hardware", ROOT / "scripts/hom", ROOT / "scripts"):
    sys.path.insert(0, str(p))

import press_stem as PS  # noqa: E402

PLANS = PS.PLANS
SCENES = [PS.name(*v) for v in PS.VARIANTS]
TAIL_S = 0.5                     # s after t_end: hom_server's --tail-s default, the hand's last 25 periods
MAX_AGE_S = 0.060                # s: hom_server --max-joint-age-ms
READ_LAG_S = 0.007               # s: a read-back's age at the next state (the station reads after it writes the goals)
T_TASK = 2.0                     # s: the task's start (hom_tasks.T_LIFT + 0.4), the start of the score's window
_W = {}


def world(key, dt, plant):
    """The plan's GPU world (hom_bench_world.world_from_plan), once per process."""
    import hom_bench_world as BW
    k = (key, dt, plant)
    if k not in _W:
        _W[k] = BW.world_from_plan(json.loads(PLANS[key].read_text()), dt=dt, plant=plant)
    return _W[k]


def station_accepts(HB, out):
    """True when the station takes the frame: every target, sent as the link's float32 sim degrees and mapped back to
    servo degrees, inside FINGER_JOINTS' range (manta_hand.runtime._validate_sim_pose)."""
    P, SV = HB._driver()
    for f in HB.FINGERS:
        for j in HB.SIM_JOINTS:
            sv = P.SIM_JOINT_TO_SERVO[j]
            sign = P.JOINT_SIGN[(f, j)]
            value = float(np.float32(float(out[f][sv]) / sign))                  # sim degrees through the link
            lo, hi = SV.FINGER_JOINTS[P.FINGER_ID[f]][sv][2]
            if not lo <= P.servo_deg(f, j, value) <= hi:
                return False
    return True


def press_angle(a0, A):
    """The turn (deg) of axes A about x_P from a0, + in theta_ref's sense (hom_trial_tracking's reference turns a0 by
    -theta_ref about x)."""
    return -np.degrees(np.arctan2(a0[1] * A[:, 2] - a0[2] * A[:, 1], a0[1] * A[:, 1] + a0[2] * A[:, 2]))


def simulate(bench, plant, seed, station=False, contacts=False, tail=TAIL_S, hold_s=0.0, hook=None):
    """HOM on the plant through the bench units at the bench's period, to t_end + tail (module docstring). Returns the
    per-period records and the error that stopped the run, if any (as the server, which then holds its last
    targets and stops). hook(t, plant), if given, is called after every period's physics (the films)."""
    from sr2hand.control import hom_bench as HB
    T = bench.T
    plant.reset(bench.q_open, seed)
    plant.hold_s = hold_s
    bench.reset()
    dt = bench.prm.dt
    t_stop = bench.t_end + tail
    t, lifted, rows, error = 0.0, False, [], None
    acc, last_ref, accepted_prev = None, None, True
    while t < t_stop - 1e-9:
        s = plant.state()
        if t >= T.T_LIFT and not lifted:
            plant.post_off()
            lifted = True
        smp = HB.bench_sample(bench, s)
        joints, src = smp["joints"], "bus"
        if station:
            if accepted_prev:                    # the last frame's read-back: the joints now, a few ms old at use
                acc = (t - READ_LAG_S, smp["joints"])
            if acc is not None and t - acc[0] <= MAX_AGE_S + 1e-9:
                joints = acc[1]
            elif last_ref is not None:
                joints, src = last_ref, "reference"
            else:
                joints, src = HB.servo_from_sim(bench.q_open), "open"
        try:
            out, info = bench.step(t, joints, smp["obj_pos_mm"], smp["obj_quat"], smp["contacts"] if contacts else None)
        except Exception as e:                                       # noqa: BLE001  HOM's solve on a diverged state
            error = f"t {t:.2f} s: {e!r}"[:200]
            break
        hom = info.get("hom") or {}
        if all(f in hom and "theta_ref" in hom[f] for f in HB.FINGERS):
            last_ref = HB.servo_from_sim({f: hom[f]["theta_ref"] for f in HB.FINGERS})
        ok = station_accepts(HB, out) if station else True
        if ok:
            plant.set_targets(HB.sim_from_servo(out))
        accepted_prev = ok
        sch = info.get("schedule") or {}
        rows.append({"t": round(t, 4), "pos": smp["obj_pos_mm"], "quat": smp["obj_quat"], "phase": info["phase"],
                     "theta_ref": sch.get("theta_ref"), "src": src, "ok": ok,
                     "q": smp["joints"], "tg": out, "loaded": sum(1 for f in HB.FINGERS
                                                                  if s["Ftot"][f] >= 0.025 * 9.81 / 2)})
        t += dt
        plant.advance(t)
        if hook is not None:
            hook(t, plant)
    if not error:
        s = plant.state()
        smp = HB.bench_sample(bench, s)
        rows.append({"t": round(t, 4), "pos": smp["obj_pos_mm"], "quat": smp["obj_quat"], "phase": "end",
                     "theta_ref": None, "src": None, "ok": None, "q": smp["joints"], "tg": None,
                     "loaded": sum(1 for f in HB.FINGERS if s["Ftot"][f] >= 0.025 * 9.81 / 2)})
    return rows, error


def summarize(rows, t_end, prm, error=None):
    """The run's row (module docstring) from simulate's records."""
    import hom_trial_tracking as TT
    from sr2hand.geometry.rotations import quat_to_mat
    t = np.array([r["t"] for r in rows])
    P = np.array([r["pos"] for r in rows], float)
    Q = np.array([r["quat"] for r in rows], float)
    A = quat_to_mat(Q)[:, :, 2]
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    win = [k for k in range(len(rows)) if T_TASK - 1e-9 <= t[k] < t_end - 1e-9]
    out = {"periods": int(sum(1 for r in rows if r["phase"] != "end")), "t_last": round(float(t[-1]), 3),
           "error": error}
    if len(win) > 1:
        tr = TT.track("press", [{"t": rows[k]["t"], "obj_quat": rows[k]["quat"],
                                 "ref": {"theta_ref": rows[k]["theta_ref"]}} for k in win], prm, 0.02)
        out.update(track=tr.get("score"), err_rms_deg=tr.get("err_rms_deg"), ref_rms_deg=tr.get("ref_rms_deg"),
                   window=[float(t[win[0]]), float(t[win[-1]])], window_periods=len(win))
    else:
        out.update(track=0.0 if error else None, err_rms_deg=None, ref_rms_deg=None, window=None, window_periods=0)
    z0 = P[0, 2]

    def held_at(k):
        return bool(P[k, 2] >= z0 - 15.0 and rows[k]["loaded"] >= 2)
    k_end = max([k for k in range(len(rows)) if t[k] <= t_end + 1e-9] or [0])
    out["held_t_end"] = held_at(k_end) if not error or t[-1] >= t_end - 1e-9 else False
    out["held_end"] = held_at(len(rows) - 1) if not error else False
    k2 = int(np.searchsorted(t, T_TASK - 1e-9))
    if k2 < len(rows):
        out["drop_mm"] = round(float(P[k2:, 2].min() - P[k2, 2]), 2)
        out["rise_mm"] = round(float(P[k2:, 2].max() - P[k2, 2]), 2)
    low = np.flatnonzero(P[:, 2] < z0 - 5.0)
    out["t_low5_s"] = round(float(t[low[0]]), 2) if len(low) else None
    out["frames_refused"] = int(sum(1 for r in rows if r["ok"] is False))
    out["periods_reference"] = int(sum(1 for r in rows if r["src"] == "reference"))
    th = press_angle(A[0], A)
    ref = np.array([np.nan if r["theta_ref"] is None else r["theta_ref"] for r in rows], float)
    out["press_max_deg"] = round(float(np.nanmax(th[k2:] - th[k2])), 2) if k2 < len(rows) else None
    keep = np.arange(0, len(rows), max(1, int(round(0.1 / (t[1] - t[0])))) if len(t) > 1 else 1)
    keep = keep[t[keep] <= t[-1] + 1e-9]
    mid = lambda key, j: [round(float(rows[k][key]["middle"][j]), 2) if rows[k][key] else None for k in keep]  # noqa: E731
    out["trace"] = {"t": t[keep].round(3).tolist(), "press_deg": th[keep].round(2).tolist(),
                    "theta_ref_deg": [None if not np.isfinite(x) else round(float(x), 2) for x in ref[keep]],
                    "pos_mm": P[keep].round(2).tolist(), "axis": A[keep].round(5).tolist(),
                    "middle_mcp": mid("q", "fe1"), "middle_pip": mid("q", "fe2"),
                    "middle_mcp_target": mid("tg", "fe1"), "middle_pip_target": mid("tg", "fe2")}
    w = np.array([k for k in win if rows[k]["tg"]], int)
    if len(w):
        out["middle_pip_median"] = round(float(np.median([rows[k]["q"]["middle"]["fe2"] for k in w])), 2)
        out["middle_pip_target_median"] = round(float(np.median([rows[k]["tg"]["middle"]["fe2"] for k in w])), 2)
        out["middle_mcp_median"] = round(float(np.median([rows[k]["q"]["middle"]["fe1"] for k in w])), 2)
        # HOM's target less the measured joint over the task (servo degrees), as hom_trial_tracking's joint_gap
        out["gap_median_deg"] = {f"{f}.{j}": round(float(np.median([float(rows[k]["tg"][f][j]) - float(rows[k]["q"][f][j])
                                                                     for k in w])), 1)
                                 for f in ("thumb", "index", "middle") for j in ("aa", "fe1", "fe2")}
    return out


# ------------------------------------------------------------------------------------------------ jobs

def calibrated(plan, scene, mu=1.0):
    """(HomBench of the plan as the bench server builds it, the calibrated plant on the scene at the plan's layout); mu
    the plant's sliding friction (1.0 for the grip tape; the twist's reversal fitted 0.5 with stiff contact as well)."""
    import rev_clutch2 as RC2
    import rev_rescore as RR
    import hom_bench_world as BW
    from sr2hand.control import hom_bench as HB
    RC2.STIFF, RC2.KP_YAW_IM, RC2.BREAKAWAY = True, 0.35, 0.0
    w = world(plan, 0.02, None)
    bench = BW.make_bench(w)
    GP = RC2.clutch_class()
    plant = GP(D / "scenes" / scene, w["setup"], None, {**RR.SERVER, "mu": float(mu)}, 0.0, c=0.6)
    for f in HB.FINGERS:                                               # the plan's layout (the world's mount bodies)
        plant.m.body_pos[plant.m.body(f"{f}_mount").id] = w["body_mounts"][f]
    return bench, plant


SOFT_GL = {"MUJOCO_GL": "egl", "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
           "EGL_PLATFORM": "surfaceless", "LP_NUM_THREADS": "2"}          # Mesa's llvmpipe: no GPU


def frames(plan, scene, seed, station, times, out, size=(360, 480)):
    """One run's frames at the given times (s), seen along the pinch axis (from +x_P) with the finger links see-through,
    tiled in a row with the press angle under each; returns the row as an image array."""
    import mujoco
    import hom_contact_rig as HR
    from sr2hand.geometry.rotations import quat_to_mat
    bench, plant = calibrated(plan, scene)
    m = plant.m
    g = m.vis.global_
    g.offwidth, g.offheight = max(int(g.offwidth), size[1]), max(int(g.offheight), size[0])
    rend = mujoco.Renderer(m, size[0], size[1])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0.0, 0.0, 0.10)
    cam.distance, cam.azimuth, cam.elevation = 0.30, 180.0, -12.0
    for gg in range(m.ngeom):
        bn = m.body(m.geom_bodyid[gg]).name
        if any(bn.startswith(f + "_") for f in ("thumb", "index", "middle")) and "_pad" not in (m.geom(gg).name or ""):
            m.geom_rgba[gg, 3] = min(float(m.geom_rgba[gg, 3]), 0.35)
    shots, a0 = [], []
    todo = sorted(times)

    def hook(t, p):
        if todo and t >= todo[0] - 1e-9:
            todo.pop(0)
            ax = quat_to_mat(p.d.qpos[p.qa + 3:p.qa + 7])[:, 2]
            if not a0:
                a0.append(ax)
            th = float(press_angle(a0[0], ax[None, :])[0])
            rend.update_scene(p.d, cam)
            img = HR.annotate(rend.render().copy(), f"{t:.1f} s", size=16)
            shots.append(HR.annotate_bottom(img, f"press {th:+5.1f} deg, z {1e3 * (p.d.qpos[p.qa + 2] - 0.12):+5.1f} mm",
                                            size=14))
    simulate(bench, plant, seed, station=station, contacts=False, hook=hook)
    rend.close()
    row = np.concatenate(shots, axis=1)
    if out:
        import imageio.v2 as imageio
        imageio.imwrite(out, HR.annotate(row, f"{plan}, {scene}, seed {seed}" + (", station" if station else ""), size=18))
    return row


def set_table_mu(plant, mu):
    """The sliding friction of the table, the stem and the block (PLA on the lab table); a contact takes the larger of
    its two geoms' friction, so all three are set. None keeps the plant's mu."""
    if mu is None:
        return
    m = plant.m
    for g in ("post", "obj_stem", "obj_block"):
        m.geom_friction[m.geom(g).id, 0] = float(mu)


# ------------------------------------------------------------------------------------------------ the hand's records

BENCH = ROOT / "docs/experiments/20261115-hom_bench_trials"
HAND = {"generalist": ("20261116-131243", "20261116-131239"), "lat0": ("20261116-131932", "20261116-131927")}
DELAY_S = 0.012                  # s: a frame reaches the servos about 12 ms after its state (corvus's 3-5 ms round trip
                                 # to the HOM server, then the station's write)


def read_jsonl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def hand_record(plan):
    """The hand's trial of this plan, one item per period of the HOM server's record: t (s, HOM's clock), HOM's targets
    (servo degrees), whether the station took them (station_accepts: on the hand every frame with a target out of
    range failed, 102 refused and 51 dropped connections of 153 in lat0, 36 of 38 in the generalist's fall, while 147
    of 148 in range arrived), theta_ref, corvus's measured joints (servo degrees, the latest read-back) and their time,
    the tracked pose in P and its capture time (both in s on HOM's clock), and the lift's start."""
    from sr2hand.control import hom_bench as HB
    ss, cs = HAND[plan]
    srv = read_jsonl(BENCH / "runs" / f"{ss}_press_d25_trial000.jsonl")
    cor = read_jsonl(BENCH / "runs_corvus" / f"{cs}_press_corvus.jsonl")
    head = srv[0]
    t0 = float(head["t0_ms"])
    by = {r["seq"]: r for r in cor if "kind" not in r and "seq" in r}
    end = next(r for r in cor if r.get("kind") == "end")
    out = []
    for r in srv:
        if "kind" in r:
            continue
        c = by.get(r["seq"], {})
        j = c.get("joints")
        out.append({"t": float(r["t"]), "targets": r["targets"], "ok": station_accepts(HB, r["targets"]),
                    "theta_ref": (r.get("ref") or {}).get("theta_ref"),
                    "joints": None if j is None else {f: dict(zip(("aa", "fe1", "fe2"), j[3 * k:3 * k + 3]))
                                                      for k, f in enumerate(HB.FINGERS)},
                    "t_joints": None if not c.get("t_joints_ms") else (float(c["t_joints_ms"]) - t0) / 1e3,
                    "pos": r.get("obj_pos_mm"), "quat": r.get("obj_quat"),
                    "t_capture": None if r.get("t_capture_ms") is None else (float(r["t_capture_ms"]) - t0) / 1e3})
    lift = (end.get("lift") or {}).get("t_start_ms")
    return {"head": head, "rows": out, "lift_s": (float(lift) - t0) / 1e3 if lift else 1.6, "stamp": ss}


def replay(plan, scene, table_mu=None, delay=DELAY_S, hook=None, mu=1.0):
    """The hand's accepted targets on the calibrated plant, open loop, from the hand's start: its first measured joints,
    the object at the plan's height with the tracked x-y and the axis's tracked heading; the table off at the arm's lift.
    Returns per-period records (as simulate's) and the joints' RMS against corvus's read-backs."""
    from sr2hand.control import hom_bench as HB
    from sr2hand.geometry.rotations import quat_mul, quat_to_mat, quat_yaw
    bench, plant = calibrated(plan, scene, mu)
    set_table_mu(plant, table_mu)
    rec = hand_record(plan)
    R = rec["rows"]
    first = next(r for r in R if r["pos"] is not None and r["quat"] is not None)
    st = bench.setup
    a_n = quat_to_mat(np.asarray(st["quat"], float))[:, 2]
    a_h = quat_to_mat(np.asarray(first["quat"], float))[:, 2]
    if a_n[:2] @ a_h[:2] < 0:                                     # the tracked axis points the other way along the line
        a_h = -a_h
    yaw = math.atan2(a_n[0] * a_h[1] - a_n[1] * a_h[0], a_n[0] * a_h[0] + a_n[1] * a_h[1])
    quat = quat_mul(quat_yaw(yaw), np.asarray(st["quat"], float))
    palm = np.asarray(bench.palm, float)
    pos = np.array([palm[0] + first["pos"][0] / 1e3, palm[1] + first["pos"][1] / 1e3, 0.12])
    q0 = HB.sim_from_servo(R[0]["joints"])
    plant.reset_to(q0, pos, quat)
    plant.hold_s = 0.0
    rows, lifted, t = [], False, 0.0
    applied = 0
    for k, r in enumerate(R):
        t_k = r["t"]
        if t_k + delay > t + 1e-9:
            plant.advance(t_k + delay)
            t = t_k + delay
        if not lifted and t >= rec["lift_s"] - 1e-9:
            plant.post_off()
            lifted = True
        if r["ok"]:
            plant.set_targets(HB.sim_from_servo(r["targets"]))
            applied += 1
        t_next = R[k + 1]["t"] if k + 1 < len(R) else t_k + 0.02
        while t < t_next - 1e-9:                                  # to the next state, the lift on time
            t1 = min(t_next, rec["lift_s"]) if not lifted and rec["lift_s"] > t + 1e-9 else t_next
            plant.advance(t1)
            t = t1
            if not lifted and t >= rec["lift_s"] - 1e-9:
                plant.post_off()
                lifted = True
        s = plant.state()
        smp = HB.bench_sample(bench, s)
        rows.append({"t": round(t_next, 4), "pos": smp["obj_pos_mm"], "quat": smp["obj_quat"], "phase": None,
                     "theta_ref": R[k + 1]["theta_ref"] if k + 1 < len(R) else None, "src": "replay", "ok": r["ok"],
                     "q": smp["joints"], "tg": r["targets"],
                     "loaded": sum(1 for f in HB.FINGERS if s["Ftot"][f] >= 0.025 * 9.81 / 2)})
        if hook is not None:
            hook(t, plant)
    # the joints against corvus's read-backs, each compared with the replay's joints at the read-back's time
    tr = np.array([x["t"] for x in rows])
    err = {f: [] for f in HB.FINGERS}
    seen = set()
    for r in R:
        if r["joints"] is None or r["t_joints"] is None or r["t_joints"] < 0 or r["t_joints"] in seen:
            continue
        seen.add(r["t_joints"])
        if r["t_joints"] > tr[-1]:
            continue
        for f in HB.FINGERS:
            for j in ("aa", "fe1", "fe2"):
                q = np.interp(r["t_joints"], tr, [x["q"][f][j] for x in rows])
                err[f].append(q - float(r["joints"][f][j]))
    joint_rms = {f: round(float(np.sqrt(np.mean(np.square(e)))), 2) for f, e in err.items() if e}
    start = [{"t": 0.0, "pos": HB.bench_sample(bench, {"q": q0, "pos": pos, "quat": quat, "cop": {}})["obj_pos_mm"],
              "quat": quat.tolist(), "phase": None, "theta_ref": None, "src": "replay", "ok": None,
              "q": R[0]["joints"], "tg": None, "loaded": 0}]
    return start + rows, {"joint_rms_deg": joint_rms, "read_backs": len(seen), "frames_applied": applied,
                          "frames": len(R), "start_yaw_deg": round(math.degrees(yaw), 2), "bench": bench}


def job_replay(job):
    plan, scene, table_mu, *rest = job
    mu = rest[0] if rest else 1.0
    w0 = time.perf_counter()
    try:
        rows, info = replay(plan, scene, table_mu, mu=mu)
        bench = info.pop("bench")
        out = summarize(rows, bench.t_end, bench.task_prm)
        out["trace_all"] = {"t": [x["t"] for x in rows], "pos_mm": [[round(v, 2) for v in x["pos"]] for x in rows],
                            "quat": [[round(v, 6) for v in x["quat"]] for x in rows]}
        out.update(info)
    except Exception as e:                                           # noqa: BLE001
        import traceback
        out = {"error": f"{e!r}"[:200] + " | " + traceback.format_exc()[-300:], "track": None}
    return {"model": "replay", "plan": plan, "scene": scene, "seed": 0, "station": True, "contacts": False,
            "table_mu": table_mu, "mu": mu, "hos_plant": os.environ.get("HOS_PLANT"), **out,
            "wall_s": round(time.perf_counter() - w0, 1)}


def job_calibrated(job):
    plan, scene, seed, station, contacts, *rest = job
    table_mu = rest[0] if rest else None
    mu = rest[1] if len(rest) > 1 else 1.0
    w0 = time.perf_counter()
    try:
        bench, plant = calibrated(plan, scene, mu)
        set_table_mu(plant, table_mu)
        rows, err = simulate(bench, plant, seed, station=station, contacts=contacts)
        out = summarize(rows, bench.t_end, bench.task_prm, err)
    except Exception as e:                                           # noqa: BLE001
        out = {"error": f"setup: {e!r}"[:200], "track": None}
    return {"model": "calibrated", "plan": plan, "scene": scene, "seed": seed, "station": station,
            "contacts": contacts, "table_mu": table_mu, "mu": mu, "hos_plant": os.environ.get("HOS_PLANT"), **out,
            "wall_s": round(time.perf_counter() - w0, 1)}


def job_pipeline(job):
    plan, seed = job
    w0 = time.perf_counter()
    try:
        import hom_bench_world as BW
        w = world(plan, None, "plan")                                    # the GPU world's 100 Hz and plant (sv25)
        bench = BW.make_bench(w)
        plant = bench.T.Plant(bench.p_pads, bench.setup)
        rows, err = simulate(bench, plant, seed, station=False, contacts=True, hold_s=bench.hold_s)
        out = summarize(rows, bench.t_end, bench.task_prm, err)
        out["hold_s"] = bench.hold_s
        out["dt"] = bench.prm.dt
    except Exception as e:                                           # noqa: BLE001
        out = {"error": f"setup: {e!r}"[:200], "track": None}
    return {"model": "pipeline", "plan": plan, "scene": "pads, impratio 100, bare cylinder on its post", "seed": seed,
            "station": False, "contacts": True, "hos_plant": os.environ.get("HOS_PLANT"), **out,
            "wall_s": round(time.perf_counter() - w0, 1)}


def done_keys(out):
    keys = set()
    if out.exists():
        for x in out.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                keys.add((r["model"], r["plan"], r["scene"], r["seed"], r["station"], r["contacts"], r.get("table_mu"),
                          r.get("mu", 1.0)))
    return keys


def pool_run(fn, jobs, out, workers, key):
    import multiprocessing as mp
    done = done_keys(out)
    jobs = [j for j in jobs if key(j) not in done]
    print(f"{len(jobs)} runs ({len(done)} done before), {workers} workers -> {out}", flush=True)
    with mp.get_context("fork").Pool(min(workers, max(1, len(jobs)))) as pool, out.open("a") as fh:
        for k, r in enumerate(pool.imap_unordered(fn, jobs), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            print(f"  {k}/{len(jobs)} {r['model']} {r['plan']} {r['scene'][:34]} s{r['seed']} station {int(r['station'])}: "
                  f"track {r.get('track')} held {r.get('held_t_end')}/{r.get('held_end')} drop {r.get('drop_mm')} "
                  f"press max {r.get('press_max_deg')} refused {r.get('frames_refused')} "
                  f"{('ERROR ' + r['error']) if r.get('error') else ''} ({r.get('wall_s')} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("out")
    p.add_argument("--scenes", nargs="+", default=SCENES)
    p.add_argument("--plans", nargs="+", default=list(PLANS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--station", action="store_true", help="the station's refusals and stale joints (docstring)")
    p.add_argument("--contacts", action="store_true", help="HOM gets the plant's contacts (the hand streamed none)")
    p.add_argument("--table-mu", type=float, default=None, help="the table's, stem's and block's friction (set_table_mu)")
    p.add_argument("--mu", type=float, default=1.0, help="the plant's sliding friction")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("replay", help="the hand's accepted targets open loop (replay)")
    p.add_argument("out")
    p.add_argument("--scenes", nargs="+", default=SCENES)
    p.add_argument("--plans", nargs="+", default=list(PLANS))
    p.add_argument("--table-mu", type=float, nargs="+", default=[-1.0], help="-1: the plant's mu (1.0)")
    p.add_argument("--mu", type=float, default=1.0, help="the plant's sliding friction")
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("pipeline")
    p.add_argument("out")
    p.add_argument("--plans", nargs="+", default=list(PLANS))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--workers", type=int, default=6)
    p = sub.add_parser("frames", help="one run per scene, frames at --times along the pinch axis, one row per scene")
    p.add_argument("out")
    p.add_argument("--plan", default="lat0", choices=list(PLANS))
    p.add_argument("--scenes", nargs="+", default=[SCENES[0]])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--station", action="store_true")
    p.add_argument("--times", type=float, nargs="+", default=[0.02, 1.5, 1.9, 2.5, 3.5, 4.5, 5.5])
    a = ap.parse_args()
    out = Path(a.out)
    if a.cmd == "frames":
        os.environ.update(SOFT_GL)
        import imageio.v2 as imageio
        rows = [frames(a.plan, sc, a.seed, a.station, a.times, None) for sc in a.scenes]
        import hom_contact_rig as HR
        img = np.concatenate([HR.annotate(r, f"{a.plan}, {sc}" + (", station" if a.station else ""), size=18)
                              for r, sc in zip(rows, a.scenes)], axis=0)
        imageio.imwrite(out, img)
        print(out, img.shape, flush=True)
        return
    if a.cmd == "run":
        if os.environ.get("HOS_PLANT") != "kp0.55_kv0_fr0.18_fl0_dp0.011":
            raise SystemExit("run: set HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 (HOM converting at the bench's gain)")
        for sc in a.scenes:
            if not (D / "scenes" / sc).exists():
                raise SystemExit(f"{sc}: no such scene; run press_stem.py")
        jobs = [(pl, sc, s, a.station, a.contacts, a.table_mu, a.mu) for pl in a.plans for sc in a.scenes for s in a.seeds]
        pool_run(job_calibrated, jobs, out, a.workers,
                 lambda j: ("calibrated", j[0], j[1], j[2], j[3], j[4], j[5], j[6]))
    elif a.cmd == "replay":
        if os.environ.get("HOS_PLANT") != "kp0.55_kv0_fr0.18_fl0_dp0.011":
            raise SystemExit("replay: set HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 (the bench server's world)")
        mus = [None if m < 0 else m for m in a.table_mu]
        jobs = [(pl, sc, mu, a.mu) for pl in a.plans for sc in a.scenes for mu in mus]
        pool_run(job_replay, jobs, out, a.workers, lambda j: ("replay", j[0], j[1], 0, True, False, j[2], j[3]))
    else:
        if os.environ.get("HOS_PLANT") != "servo":
            raise SystemExit("pipeline: set HOS_PLANT=servo (the design pipeline's SCS0009 at 0.72 N m/rad)")
        jobs = [(pl, s) for pl in a.plans for s in a.seeds]
        pool_run(job_pipeline, jobs, out, a.workers,
                 lambda j: ("pipeline", j[0], "pads, impratio 100, bare cylinder on its post", j[1], False, True, None,
                            1.0))


if __name__ == "__main__":
    main()
