#!/usr/bin/env python3
"""Short-horizon gap between the hand and the simulated plant, per contact model and servo model (hand-4a, 2026-11-19;
owner 17:2x: "for each of these controls, we should be able to measure the gap between reality and our system, use sysid
methods to minimize this per-control step gap across the knobs we can turn (contact model, pose filter, servo model)").

A full open-loop replay accumulates its error: once the replayed sphere lags the hand's by 10 deg, every later period is
compared from a different state. Here each recorded twist is cut into windows of SEG seconds that start every STEP
seconds. Each window starts the plant from the hand's own state at its first sample: the nine measured joint angles (the
read-back) and the tracked sphere pose, at rest, the support post off once the arm has lifted. The plant first holds the
targets then in force for SETTLE seconds, so that small overlaps between the tracked sphere and the model's fingertips
(tracker and joint errors of about 1 mm) resolve before the window; the sphere's motion during that settle is recorded as
the reset's inconsistency. Then the window replays the hand's targets as the full replay does (each period's targets 12
ms after its state). At the window's end the plant is compared with the hand:
  twist error      the sphere's rotation about z_P over the window, simulated minus tracked (deg)
  rotation error   the angle between the simulated and the tracked rotation over the window (deg)
  position error   the distance between the simulated and the tracked displacement over the window (mm)
  joint error      RMS over the nine joints of the simulated minus the measured angle at the window's end (servo deg)
Windows from 0.8 s (the fingers close from about 0.6 s) to the trial's end; "grasp" windows start before 2.0 s, "task"
windows after.

Trials: the 20 scored g0 twists at (70, 110) mm of 2026-11-17 at 7.5 V (data/20261117-night_repeats_tracking.json of the
bench trials), as the full replays of 2026-11-18 used them (sim/20261118-night_trials_replay.py, imported for its plant
and trial list). Plants: rev_clutch2's gears that the load cannot backdrive, the 7.5 V servos (7v5: kp 0.825 N m/rad,
torque limit 0.27 N m, index and middle yaws 0.525; 7v5y: the yaws as the rest), the MCP lows of S1(a); contact models of
replan.SCENES (pads soft and point soft at impratio 1, pads stiff at impratio 100, with or without noslip); friction 1.0
(the grip tape) or MU.

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 OMP_NUM_THREADS=1 .venv/bin/python \\
        docs/experiments/20261119-twist_layout_ranking/sim/20261119-segment_gap.py OUT.jsonl \\
        [--contacts pads point pads_stiff] [--volts 7v5 7v5y] [--mu 1.0 0.6] [--workers 6]
    .venv/bin/python .../20261119-segment_gap.py OUT.jsonl --summary          # per model, from the rows
"""
import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("HOS_PLANT", "kp0.55_kv0_fr0.18_fl0_dp0.011")
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "logs/20261117-g13_open_pose"))
sys.path.insert(0, str(ROOT / "scripts"))
SEG, STEP, SETTLE, T_FIRST = 0.5, 0.25, 0.04, 0.8
JN = ("aa", "fe1", "fe2")


def make_plant(NR, contact, volts, mu):
    """night_replay.make_plant with the plant's sliding friction mu."""
    RPL = NR.RPL
    RP = RPL.RG.RF._G["RP"]
    RP.TOY_ROWS = RPL.RG.RF._G["toy_rows"]
    params, kp_yaw = NR.PLANTS[volts]
    RPL.RC2.KP_YAW_IM = kp_yaw
    setup, mxy, _ = RP.toy_setup("twist", 40.0, 162, 0, "grid2dgr")
    GP = RPL.RG._G["GP"]
    p = GP(str(RPL.D / "scenes" / RPL.SCENES[contact]), setup, mxy, {**params, "mu": mu}, 0.0, c=0.6)
    for f, lo in NR.STOPS.items():
        jid = p.m.joint(f"{f}_mcp").id
        p.m.jnt_range[jid, 0] = max(p.m.jnt_range[jid, 0], math.radians(lo))
        p.m.jnt_limited[jid] = 1
    return p, setup


def run(job):
    try:
        return run1(job)
    except Exception as e:                                       # noqa: BLE001
        import traceback
        return {**job, "error": repr(e)[:200], "tb": traceback.format_exc()[-800:]}


def run1(job):
    import night_replay as NR
    import hom_trial_tracking as TT
    from sr2hand.control import hom_bench as HB
    from sr2hand.geometry.rotations import quat_angle, to_scipy
    w0 = time.perf_counter()
    head, rows, _end = TT.load_server(ROOT / job["file"])
    plant, setup = make_plant(NR, job["contact"], job["volts"], job["mu"])
    palm = np.asarray(setup["palm"], float)
    t = np.array([float(r["t"]) for r in rows])
    t_end = min(float(head.get("t_end") or t[-1]), float(t[-1]))
    lift_s = float(job["lift_s"])
    ok = [r.get("obj_pos_mm") is not None and r.get("obj_quat") is not None and r.get("joints") is not None
          for r in rows]
    mj = plant.mj
    segs = []
    for ts in np.arange(T_FIRST, t_end - SEG + 1e-9, STEP):
        i = next((k for k in range(len(rows)) if t[k] >= ts - 1e-9 and ok[k]), None)
        if i is None or i == 0:
            continue
        e = next((k for k in range(len(rows) - 1, i, -1) if t[k] <= t[i] + SEG + 1e-9 and ok[k]), None)
        if e is None or t[e] - t[i] < 0.6 * SEG:
            continue
        q_i = HB.sim_from_servo(rows[i]["joints"])
        pos_i = palm + np.asarray(rows[i]["obj_pos_mm"], float) / 1e3
        quat_i = np.asarray(rows[i]["obj_quat"], float)
        plant.reset_to(q_i, pos_i, quat_i)
        plant.hold_s = 0.0
        if t[i] >= lift_s:
            plant.post_off()
        bad0 = plant.d.warning[mj.mjtWarning.mjWARN_BADQACC].number
        plant.set_targets(HB.sim_from_servo(rows[i - 1]["targets"]))   # the targets in force at sample i
        plant.advance(SETTLE)
        s0 = plant.state()
        settle_mm = float(np.linalg.norm(s0["pos"] - pos_i) * 1e3)
        settle_deg = float(np.degrees(quat_angle(np.asarray(s0["quat"], float), quat_i)))
        lifted = t[i] >= lift_s
        for k in range(i, e):                            # targets k reach the servos 12 ms after sample k
            tk = SETTLE + (t[k] - t[i]) + NR.DELAY_S
            if not lifted and t[k] + NR.DELAY_S >= lift_s:
                plant.advance(SETTLE + (lift_s - t[i]))
                plant.post_off()
                lifted = True
            plant.advance(tk)
            plant.set_targets(HB.sim_from_servo(rows[k]["targets"]))
        if not lifted and t[e] >= lift_s:
            plant.advance(SETTLE + (lift_s - t[i]))
            plant.post_off()
        plant.advance(SETTLE + (t[e] - t[i]))                 # the state at sample e
        s1 = plant.state()
        diverged = bool(plant.d.warning[mj.mjtWarning.mjWARN_BADQACC].number > bad0
                        or not np.isfinite(s1["pos"]).all())
        Rh = to_scipy(np.asarray(rows[e]["obj_quat"], float)) * to_scipy(quat_i).inv()
        Rs = to_scipy(np.asarray(s1["quat"], float)) * to_scipy(np.asarray(s0["quat"], float)).inv()
        dP_h = (np.asarray(rows[e]["obj_pos_mm"], float) - np.asarray(rows[i]["obj_pos_mm"], float))
        dP_s = (np.asarray(s1["pos"], float) - np.asarray(s0["pos"], float)) * 1e3
        sv = HB.servo_from_sim(s1["q"])
        js = np.array([sv[f][j] for f in HB.FINGERS for j in JN], float)
        jm = np.array([float(rows[e]["joints"][f][j]) for f in HB.FINGERS for j in JN], float)
        segs.append({"t0": round(float(t[i]), 3), "t1": round(float(t[e]), 3), "phase": "grasp" if t[i] < 2.0 else "task",
                     "lifted": bool(t[i] >= lift_s), "diverged": diverged,
                     "settle_mm": round(settle_mm, 2), "settle_deg": round(settle_deg, 2),
                     "twist_hand": round(float(np.degrees(Rh.as_rotvec()[2])), 2),
                     "twist_sim": round(float(np.degrees(Rs.as_rotvec()[2])), 2),
                     "rot_err": round(float(np.degrees((Rs.inv() * Rh).magnitude())), 2),
                     "pos_err": round(float(np.linalg.norm(dP_s - dP_h)), 2),
                     "drop_sim_mm": round(float(-dP_s[2]), 2),
                     "joint_rms": round(float(np.sqrt(np.mean((js - jm) ** 2))), 2),
                     "joint_err": {f"{f}.{j}": round(float(js[3 * a + b] - jm[3 * a + b]), 2)
                                   for a, f in enumerate(HB.FINGERS) for b, j in enumerate(JN)}})
    return {**job, "segments": segs, "wall_s": round(time.perf_counter() - w0, 1)}


def summary(path):
    by = defaultdict(list)
    for x in Path(path).read_text().splitlines():
        if not x.strip():
            continue
        r = json.loads(x)
        if "error" in r:
            continue
        for s in r["segments"]:
            by[(r["contact"], r["volts"], r["mu"])].append(s)
    out = {}
    for key, ss in sorted(by.items()):
        rec = {}
        for ph in ("grasp", "task", "all"):
            sel = [s for s in ss if (ph == "all" or s["phase"] == ph) and not s["diverged"]]
            if not sel:
                continue
            tw = np.array([s["twist_sim"] - s["twist_hand"] for s in sel])
            rec[ph] = {"windows": len(sel), "diverged": sum(s["diverged"] for s in ss if ph in ("all", s["phase"])),
                       "twist_err_rms": round(float(np.sqrt(np.mean(tw ** 2))), 2),
                       "twist_err_mean": round(float(np.mean(tw)), 2),
                       "twist_hand_mean": round(float(np.mean([s["twist_hand"] for s in sel])), 2),
                       "rot_err_rms": round(float(np.sqrt(np.mean([s["rot_err"] ** 2 for s in sel]))), 2),
                       "pos_err_rms": round(float(np.sqrt(np.mean([s["pos_err"] ** 2 for s in sel]))), 2),
                       "joint_rms_mean": round(float(np.mean([s["joint_rms"] for s in sel])), 2),
                       "settle_mm_median": round(float(np.median([s["settle_mm"] for s in sel])), 2),
                       "settle_deg_median": round(float(np.median([s["settle_deg"] for s in sel])), 2)}
        out[f"{key[0]} {key[1]} mu {key[2]:g}"] = rec
    return out


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--contacts", nargs="+", default=["pads", "point", "pads_stiff", "pads_stiff_ns"])
    ap.add_argument("--volts", nargs="+", default=["7v5"])
    ap.add_argument("--mu", type=float, nargs="+", default=[1.0])
    ap.add_argument("--trials", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        print(json.dumps(summary(a.out), indent=1))
        return 0
    import night_replay as NR
    RPL = NR.RPL
    RPL.RG.gear_class = RPL.RC2.clutch_class
    RPL.RC2.STIFF, RPL.RC2.BREAKAWAY = True, 0.0
    RPL.RG.RF.RL.rows_file()
    tl = [t for t in NR.trials() if not a.trials or t["stamp"] in a.trials]
    out = Path(a.out)
    done = set()
    if out.exists():
        for x in out.read_text().splitlines():
            if x.strip():
                r = json.loads(x)
                if "error" not in r:
                    done.add((r["stamp"], r["contact"], r["volts"], r["mu"]))
    jobs = [{**t, "contact": c, "volts": v, "mu": mu} for mu in a.mu for v in a.volts for c in a.contacts for t in tl
            if (t["stamp"], c, v, mu) not in done]
    print(f"{len(jobs)} trials x models on {a.workers} workers ({len(done)} done)", flush=True)
    with open(out, "a") as fo, mp.get_context("fork").Pool(a.workers, initializer=RPL.RG.init,
                                                           initargs=(str(RPL.RG.RF.SCENE),), maxtasksperchild=4) as pool:
        for r in pool.imap_unordered(run, jobs):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            os.fsync(fo.fileno())
            if "error" in r:
                print(r["stamp"], r["contact"], r["volts"], r["mu"], "ERROR", r["error"], r.get("tb", "")[-400:], flush=True)
                continue
            ss = [s for s in r["segments"] if not s["diverged"]]
            tw = np.array([s["twist_sim"] - s["twist_hand"] for s in ss]) if ss else np.array([np.nan])
            print(f"{r['stamp']} {r['contact']:13s} {r['volts']} mu {r['mu']:g}: {len(ss)}/{len(r['segments'])} windows, "
                  f"twist err rms {np.sqrt(np.nanmean(tw ** 2)):.2f} deg, rot rms "
                  f"{np.sqrt(np.mean([s['rot_err'] ** 2 for s in ss])) if ss else float('nan'):.2f}, settle median "
                  f"{np.median([s['settle_mm'] for s in ss]) if ss else float('nan'):.2f} mm | {r['wall_s']} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
