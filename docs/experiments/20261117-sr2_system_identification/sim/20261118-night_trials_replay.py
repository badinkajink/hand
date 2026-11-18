"""Open-loop replays of the night's g0 twists on the calibrated plant, per contact model (hand-4a, 2026-11-18; owner,
09:1x: "when you report these trial results, also, please make sure to also report the simulation prediction and
replay ... across multiple contact model configurations"; owner, 11:0x: "ok yea please do that").

The trials: the twist at g0 (70, 110) mm on 2026-11-17, the first A/B at 19:42-20:08 and the second at 21:06-21:25,
as the lab scored them (docs/experiments/20261115-hom_bench_trials/data/20261117-night_repeats_tracking.json); the five
one-period fragment records are left out. Each replay starts from the hand's start: the first sample's measured joints,
and the object at the plan's height with the tracked x-y and heading, standing on the table under its block. Then every
period's targets, as the HOM server recorded them, reach the servos 12 ms after their state (press_predict.DELAY_S), and
the table goes at the arm's lift (corvus's lift_start_s). Nothing is closed around the plant: the targets are the hand's.

Plant: rev_clutch2's (gears the load cannot backdrive, forward friction 0.6 |servo torque|, joint friction that does not
creep), as the blind predictions of 2026-11-17 18:30 ran it, at two supplies:
  7v5  kp 0.825 N m/rad, tau_max 0.27 N m, the index's and middle's yaw at kp 0.525 (the 5 V values x 1.5, the
       predictions' assumption for 7.5 V)
  5v   kp 0.55, tau_max 0.18, yaw 0.35 (the plant fitted on 2026-11-16 at 5 V)
  7v5y 7v5 with the index's and middle's yaw at the other joints' 0.825 (S2 at 7.5 V: the yaws' free-air shortfalls
       0.2-0.85 deg, against 0.75-3.05 at 5 V, where the 0.35 was fitted)
the MCPs stopped at S1(a)'s measured lows at 7.5 V (index -10.3, middle -8.5 deg, simulator degrees), damping 0.011,
friction 1.0 (the grip tape). Contact models: the stem scenes of replan.SCENES (pads soft: impratio 1; point soft;
pads stiff: impratio 100; pads stiff with noslip 30 iterations; pads soft with noslip).

Per replay, with the hand's record beside it:
  track, twist    hom_trial_tracking.track over the task's window (2.0 s to t_end) on the replayed object, sampled at
                  the tracker's capture times and scored against the trial's own w_ref; hom_trial_tracking.motion over
                  the whole record (rise along z_P, twist about z_P and tilt from the first sample, 3-D drift)
  joints          RMS of the replayed minus the measured joint angle (servo degrees, at each read-back's time), per
                  finger and joint, in three windows: approach (0-0.6 s, free air), grasp (0.6-2.0 s), task (2.0 s on)
  object          RMS of the orientation error (deg) and of the position error (mm) against the tracked pose, task
                  window
  traces          every 0.1 s: the twist about z_P (hand and replay) and the index yaw, middle yaw and thumb MCP
                  (measured, target, replayed; servo degrees)

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 SCALE_HAND=cad_v2 OMP_NUM_THREADS=1 \\
        .venv/bin/python logs/20261117-g13_open_pose/night_replay.py OUT.jsonl [--contacts pads point ...] \\
        [--volts 7v5 5v] [--trials STAMP ...] [--workers 6]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replan as RPL  # noqa: E402  (paths, rev_* imports, the 7.5 V plant)

ROOT = RPL.ROOT
BENCH = ROOT / "docs/experiments/20261115-hom_bench_trials"
SCORED = BENCH / "data/20261117-night_repeats_tracking.json"
STOPS = {"index": -10.3, "middle": -8.5}            # S1(a) at 7.5 V: the MCP lows' stall (simulator degrees)
DELAY_S = 0.012                                    # s: a frame reaches the servos ~12 ms after its state
PLANTS = {"7v5": ({"kp": 0.55 * 1.5, "damping": 0.011, "tau_max": 0.18 * 1.5}, 0.35 * 1.5),
          "5v": ({"kp": 0.55, "damping": 0.011, "tau_max": 0.18}, 0.35),
          "7v5y": ({"kp": 0.55 * 1.5, "damping": 0.011, "tau_max": 0.18 * 1.5}, None)}   # the yaws as strong as the rest
CONTACTS = ["pads", "point", "pads_stiff", "pads_stiff_ns", "pads_soft_ns"]
WINDOWS = {"approach": (0.0, 0.6), "grasp": (0.6, 2.0), "task": (2.0, 99.0)}
TRACE_J = (("index", "aa"), ("middle", "aa"), ("thumb", "fe1"))


def trials():
    """The scored records of the night (fragments left out), with corvus's lift start and the arm of each."""
    d = json.loads(SCORED.read_text())
    out = []
    for t in d["trials"]:
        if not t.get("scored"):
            continue
        out.append({"stamp": t["stamp"], "file": t["file"], "lift_s": (t.get("corvus") or {}).get("lift_start_s") or 1.6,
                    "hand_track": (t.get("track") or {}).get("score"), "ended": (t.get("ended") or {}).get("short")})
    return out


def arm_of(head):
    tips = head.get("tips") or "off"
    flags = []
    if head.get("filter_step_s"):
        flags.append("fstep")
    if head.get("filter_exact"):
        flags.append("exact")
    if head.get("contact_force_off_N") is not None:
        flags.append("hyst")
    block = "A/B 1" if head["started"] < "2026-11-17T21:00" else "A/B 2"
    return {"tips": tips, "flags": "+".join(flags) or "none", "block": block}


def start_pose(setup, first):
    """The object's world pose at the hand's start: the plan's height, the tracked x-y (P is the world frame shifted to
    the palm) and the tracked heading about z (the stem stays vertical); and the start's tilt from that (deg)."""
    import hand_object_scale_sim as S
    from sr2hand.geometry.rotations import quat_mul, quat_yaw, to_scipy
    palm = np.asarray(setup["palm"], float)
    R_n, R_h = to_scipy(np.asarray(setup["quat"], float)), to_scipy(np.asarray(first["obj_quat"], float))
    a_n, a_h = R_n.apply([1.0, 0.0, 0.0]), R_h.apply([1.0, 0.0, 0.0])
    yaw = math.atan2(a_n[0] * a_h[1] - a_n[1] * a_h[0], a_n[0] * a_h[0] + a_n[1] * a_h[1])
    quat = quat_mul(quat_yaw(yaw), np.asarray(setup["quat"], float))
    tilt = float(np.degrees((R_h * to_scipy(quat).inv()).magnitude()))
    pos = np.array([palm[0] + first["obj_pos_mm"][0] / 1e3, palm[1] + first["obj_pos_mm"][1] / 1e3, S.Z_OBJ])
    return pos, quat, math.degrees(yaw), tilt


def make_plant(contact, volts):
    RP = RPL.RG.RF._G["RP"]
    RP.TOY_ROWS = RPL.RG.RF._G["toy_rows"]
    params, kp_yaw = PLANTS[volts]
    RPL.RC2.KP_YAW_IM = kp_yaw
    setup, mxy, _ = RP.toy_setup("twist", 40.0, 162, 0, "grid2dgr")
    GP = RPL.RG._G["GP"]
    p = GP(str(RPL.D / "scenes" / RPL.SCENES[contact]), setup, mxy, {**params, "mu": 1.0}, 0.0, c=0.6)
    for f, lo in STOPS.items():
        jid = p.m.joint(f"{f}_mcp").id
        p.m.jnt_range[jid, 0] = max(p.m.jnt_range[jid, 0], math.radians(lo))
        p.m.jnt_limited[jid] = 1
    return p, setup


def _interp_rows(tq, rt, X):
    return np.stack([np.interp(tq, rt, X[:, i]) for i in range(X.shape[1])], 1)


def run(job):
    try:
        return run1(job)
    except Exception as e:                                       # noqa: BLE001
        import traceback
        return {**job, "error": repr(e)[:200], "tb": traceback.format_exc()[-800:]}


def run1(job):
    import hom_trial_tracking as TT
    from sr2hand.control import hom_bench as HB
    from sr2hand.geometry.rotations import from_scipy, quat_angle, to_scipy
    from scipy.spatial.transform import Slerp
    w0 = time.perf_counter()
    head, rows, _end = TT.load_server(ROOT / job["file"])
    t0_ms = float(head["t0_ms"])
    plant, setup = make_plant(job["contact"], job["volts"])
    palm = np.asarray(setup["palm"], float)
    assert np.allclose(palm, head["palm_m"], atol=2e-4), (palm, head["palm_m"])
    first = next(r for r in rows if r.get("obj_quat") is not None and r.get("obj_pos_mm") is not None)
    pos0, quat0, yaw0, tilt0 = start_pose(setup, first)
    q0 = HB.sim_from_servo(rows[0]["joints"])
    plant.reset_to(q0, pos0, quat0)
    plant.hold_s = 0.0
    bad0 = plant.d.warning[plant.mj.mjtWarning.mjWARN_BADQACC].number
    lift_s = float(job["lift_s"])
    lifted = False
    rec_t, rec_q, rec_x = [], [], []

    def record():
        s = plant.state()
        if rec_t and plant.d.time <= rec_t[-1] + 1e-12:     # no time passed (no delay): keep the newer state
            rec_t.pop()
            rec_q.pop()
            rec_x.pop()
        rec_t.append(float(plant.d.time))
        sv = HB.servo_from_sim(s["q"])
        rec_q.append(np.array([sv[f][j] for f in HB.FINGERS for j in ("aa", "fe1", "fe2")], float))
        rec_x.append(np.concatenate([1e3 * (s["pos"] - palm), s["quat"]]))

    def go(t1):
        nonlocal lifted
        while plant.d.time < t1 - 1e-9:
            seg = min(t1, lift_s) if not lifted and lift_s > plant.d.time + 1e-9 else t1
            plant.advance(seg)
            if not lifted and plant.d.time >= lift_s - 1e-9:
                plant.post_off()
                lifted = True
    record()
    t_rec = [float(r["t"]) for r in rows]
    for k, r in enumerate(rows):
        go(t_rec[k] + DELAY_S)
        record()
        plant.set_targets(HB.sim_from_servo(r["targets"]))
    go(t_rec[-1] + 0.02 + DELAY_S)
    record()
    rt, RQ, RX = np.array(rec_t), np.array(rec_q), np.array(rec_x)
    diverged = bool(plant.d.warning[plant.mj.mjtWarning.mjWARN_BADQACC].number > bad0 or not np.isfinite(RX).all())
    if diverged:
        return {**job, "diverged": True, "wall_s": round(time.perf_counter() - w0, 1)}
    # the replayed object at the tracker's capture times, in the server record's fields
    t = np.array(t_rec)
    t_cap = np.array([(float(r["t_capture_ms"]) - t0_ms) / 1e3 if r.get("t_capture_ms") is not None else t[k]
                      for k, r in enumerate(rows)])
    t_cap = np.clip(t_cap, rt[0], rt[-1])
    P_sim = _interp_rows(t_cap, rt, RX[:, :3])
    Q_sim = from_scipy(Slerp(rt, to_scipy(RX[:, 3:7]))(t_cap))
    sim_rows = [{**r, "obj_pos_mm": P_sim[k].tolist(), "obj_quat": Q_sim[k].tolist()} for k, r in enumerate(rows)]
    Qall = np.array([r["obj_quat"] if r.get("obj_quat") is not None else [np.nan] * 4 for r in rows], float)
    okq = np.isfinite(Qall).all(1)
    t_end = float(head.get("t_end") or 7.8)
    win = [k for k in range(len(rows)) if 2.0 - 1e-9 <= t[k] < t_end - 1e-9]
    dt_ = float(head.get("dt") or 0.02)
    prm = head.get("prm") or {}
    out = {**job, **arm_of(head), "diverged": False, "delay_s": DELAY_S, "start_yaw_deg": round(yaw0, 2),
           "start_tilt_deg": round(tilt0, 2),
           "hand": {"track": TT.track("twist", [rows[k] for k in win], prm, dt_), "motion": TT.motion(rows)},
           "replay": {"track": TT.track("twist", [sim_rows[k] for k in win], prm, dt_), "motion": TT.motion(sim_rows)}}
    # joints against the read-backs, each at its own time
    seen, jt, jm = set(), [], []
    for r in rows:
        tj = r.get("t_joints_ms")
        if tj is None or r.get("joints_src", "bus") != "bus" or tj in seen:
            continue
        seen.add(tj)
        jt.append((float(tj) - t0_ms) / 1e3)
        jm.append(np.concatenate([[float(r["joints"][f][j]) for j in ("aa", "fe1", "fe2")] for f in HB.FINGERS]))
    jt, jm = np.array(jt), np.array(jm)
    keep = (jt >= rt[0]) & (jt <= rt[-1])
    jt, jm = jt[keep], jm[keep]
    js = _interp_rows(jt, rt, RQ)
    names = [f"{f}.{j}" for f in HB.FINGERS for j in ("aa", "fe1", "fe2")]
    jr = {}
    for wn, (a, b) in WINDOWS.items():
        sel = (jt >= a) & (jt < b)
        if sel.sum() < 3:
            continue
        e = js[sel] - jm[sel]
        jr[wn] = {"all": round(float(np.sqrt(np.mean(e ** 2))), 2),
                  **{n: round(float(np.sqrt(np.mean(e[:, i] ** 2))), 2) for i, n in enumerate(names)},
                  "bias": {n: round(float(np.mean(e[:, i])), 2) for i, n in enumerate(names)}}
    out["joint_rms_deg"] = jr
    out["read_backs"] = int(len(jt))
    # the object against the tracked pose over the task's window
    win_t = [k for k in win if okq[k] and rows[k].get("obj_pos_mm") is not None]
    Ph = np.array([rows[k]["obj_pos_mm"] for k in win_t], float)
    eo = np.degrees(quat_angle(Q_sim[win_t], Qall[win_t]))
    ep = np.linalg.norm(P_sim[win_t] - Ph, axis=1)
    out["object_task"] = {"rot_rms_deg": round(float(np.sqrt(np.mean(eo ** 2))), 2),
                          "pos_rms_mm": round(float(np.sqrt(np.mean(ep ** 2))), 2)}
    # traces every 0.1 s
    tw_h = np.full(len(rows), np.nan)
    tw_h[okq] = np.degrees((to_scipy(Qall[okq]) * to_scipy(np.asarray(first["obj_quat"], float)).inv()).as_rotvec()[:, 2])
    tw_s = np.degrees((to_scipy(Q_sim) * to_scipy(Q_sim[0]).inv()).as_rotvec()[:, 2])
    idx = np.arange(0, len(rows), 5)
    tg = np.array([np.concatenate([[float(r["targets"][f][j]) for j in ("aa", "fe1", "fe2")] for f in HB.FINGERS])
                   for r in rows])
    jh = np.array([np.concatenate([[float(r["joints"][f][j]) for j in ("aa", "fe1", "fe2")] for f in HB.FINGERS])
                   for r in rows])
    jsim = _interp_rows(t, rt, RQ)
    tr = {"t": t[idx].round(2).tolist(), "twist_hand": tw_h[idx].round(1).tolist(),
          "twist_replay": tw_s[idx].round(1).tolist()}
    for f, j in TRACE_J:
        i = names.index(f"{f}.{j}")
        tr[f"{f}.{j}"] = {"hand": jh[idx, i].round(1).tolist(), "target": tg[idx, i].round(1).tolist(),
                          "replay": jsim[idx, i].round(1).tolist()}
    sel = (t >= 2.0) & okq
    out["twist_trace_rms_deg"] = round(float(np.sqrt(np.mean((tw_s[sel] - tw_h[sel]) ** 2))), 2)
    out["trace"] = tr
    out["locked_steps"] = int(getattr(plant, "n_locked", 0))
    out["wall_s"] = round(time.perf_counter() - w0, 1)
    return out


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--contacts", nargs="+", default=CONTACTS[:4], choices=list(RPL.SCENES))
    ap.add_argument("--volts", nargs="+", default=["7v5", "5v"], choices=list(PLANTS))
    ap.add_argument("--trials", nargs="*", default=None, help="stamps (default: every scored trial of the night)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--delay", type=float, default=None, help="s from a state to its targets at the servos (default 0.012)")
    a = ap.parse_args()
    global DELAY_S
    if a.delay is not None:
        DELAY_S = a.delay
    RPL.RG.gear_class = RPL.RC2.clutch_class
    RPL.RC2.STIFF, RPL.RC2.BREAKAWAY = True, 0.0
    RPL.RG.RF.RL.rows_file()
    tl = [t for t in trials() if not a.trials or t["stamp"] in a.trials]
    out = Path(a.out)
    done = set()
    if out.exists():
        for x in out.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if "error" not in r:
                    done.add((r["stamp"], r["contact"], r["volts"]))
    jobs = [{**t, "contact": c, "volts": v} for v in a.volts for c in a.contacts for t in tl
            if (t["stamp"], c, v) not in done]
    print(f"{len(jobs)} replays on {a.workers} workers ({len(done)} done)", flush=True)
    with open(out, "a") as fo, mp.get_context("fork").Pool(a.workers, initializer=RPL.RG.init,
                                                           initargs=(str(RPL.RG.RF.SCENE),), maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(run, jobs):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            os.fsync(fo.fileno())
            if "error" in r:
                print(r["stamp"], r["contact"], r["volts"], "ERROR", r["error"], r.get("tb", "")[-400:], flush=True)
                continue
            print(f"{r['stamp']} {r['block']} tips {r['tips']:3s} {r['flags']:10s} {r['contact']:13s} {r['volts']}: "
                  f"track hand {r['hand']['track'].get('score')} replay {r['replay']['track'].get('score')} | twist "
                  f"{r['hand']['motion'].get('twist_end_deg')} / {r['replay']['motion'].get('twist_end_deg')} | rise "
                  f"{r['hand']['motion'].get('rise_max_mm')} / {r['replay']['motion'].get('rise_max_mm')} | joints task "
                  f"{(r['joint_rms_deg'].get('task') or {}).get('all')} | {r['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
