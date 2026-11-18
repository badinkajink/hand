"""The lat0 press's open pose inside the hand's measured MCP stops with the pads off the cylinder, HOM's press from each
candidate in the calibrated simulation, and the pushing finger's target chatter (hand-4a, 2026-11-17 18:4x; lab request
18:3x: "an open pose for press_x70_y110_lat0 inside the measured MCP stops with the pads off the cylinder, as you did for
g13; and a look at why HOM's press targets chatter on the pushing finger").

The plan (plans/20261116-press_x70_y110_lat0_plan.json) wants the index MCP at -14.8 deg; on the hand it stops at -11.4.
Candidates: the index and middle MCPs 0.5 deg inside the stops, their yaws turned dyaw toward zero, their PIPs opened dpip;
the thumb as planned. 'plan@sim' is the plan's pose in the unstopped model, the grasp HOM was planned to close into.

    poses  each candidate's pad clearances from the cylinder on the model
    sim    press_predict.simulate on rev_clutch2's calibrated plant (stopped MCPs), pads with soft contact, contacts to HOM
           off, at 5 V or 7.5 V (gain and torque limit x1.5, assumed), seeds 0-2: the tracking score (press_predict.
           summarize), the grasp at 1.40-1.58 s, and the middle finger's MCP and PIP target chatter from 2.0 to 4.0 s
           (rms about a 0.5 s running mean, and its dominant frequency)

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 SCALE_HAND=cad_v2 OMP_NUM_THREADS=1 .venv/bin/python press_replan.py poses
    HOS_PLANT=... press_replan.py sim --cands plan@sim plan y-1_p4 ... --volts 5 7.5 [--out FILE]
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
OUT = Path(__file__).resolve().parent
for p in (D, ROOT / "scripts/hardware", ROOT / "scripts/hom", ROOT / "scripts"):
    sys.path.insert(0, str(p))
import _paths  # noqa: E402,F401
import press_predict as PP  # noqa: E402

STOP_DEG = {"index": -11.4, "middle": -10.0}
MARGIN_DEG = 0.5
SCENE = os.environ.get("PRESS_SCENE", "cyl025_stem_block_pads_imp1.xml")
FING = ("thumb", "index", "middle")


def world():
    return PP.world("lat0", 0.02, None)


def clearance(w, q):
    import mujoco
    import hand_object_scale_sim as S
    m = w["m_pt"]
    S.set_palm(m, w["setup"]["palm"])
    d = mujoco.MjData(m)
    tb = m.body(S.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    d.qpos[qa:qa + 7] = [0.0, 0.0, S.Z_OBJ, *w["setup"]["quat"]]
    og = int(np.flatnonzero(m.geom_bodyid == tb)[0])
    for f in FING:
        for k, j in enumerate(("yaw", "mcp", "pip")):
            d.qpos[m.jnt_qposadr[m.joint(f"{f}_{j}").id]] = q[f][k]
    mujoco.mj_forward(m, d)
    return {f: round(1000 * mujoco.mj_geomDistance(m, d, m.geom(f"{f}_tipgeom").id, og, 0.05, None), 2) for f in FING}


def candidates(dyaws=(-4, -3, -2, -1, 0), dpips=(0, 2, 4, 6, 9, 12)):
    w = world()
    q0 = {f: np.asarray(w["q_open"][f], float) for f in FING}
    out = {"plan": {"deg": {f: [round(float(v), 3) for v in np.degrees(q0[f])] for f in FING},
                    "clearance_mm": clearance(w, q0)}}
    for dy in dyaws:
        for dp in dpips:
            q = {f: q0[f].copy() for f in FING}
            for f in STOP_DEG:
                q[f][0] += math.radians(dy) * math.copysign(1.0, q[f][0])
                q[f][1] = max(q[f][1], math.radians(STOP_DEG[f] + MARGIN_DEG))
                q[f][2] -= math.radians(dp)
            name = f"y{dy:+d}_p{dp}"
            out[name] = {"deg": {f: [round(float(v), 3) for v in np.degrees(q[f])] for f in FING},
                         "clearance_mm": clearance(w, q)}
    for dp in (2, 4, 6):                                 # the thumb's PIP opened as well (its pad starts 1.4 mm off)
        for tp in (4, 8, 12):
            q = {f: q0[f].copy() for f in FING}
            for f in STOP_DEG:
                q[f][1] = max(q[f][1], math.radians(STOP_DEG[f] + MARGIN_DEG))
                q[f][2] -= math.radians(dp)
            q["thumb"][2] -= math.radians(tp)
            name = f"y+0_p{dp}_t{tp}"
            out[name] = {"deg": {f: [round(float(v), 3) for v in np.degrees(q[f])] for f in FING},
                         "clearance_mm": clearance(w, q)}
    for k, v in out.items():
        print(k, v["deg"]["thumb"], v["deg"]["index"], v["deg"]["middle"], v["clearance_mm"], flush=True)
    return out


def _bench_plant(q_deg, volts, stops):
    import hom_bench_world as BW
    import rev_clutch2 as RC2
    import rev_rescore as RR
    from sr2hand.control import hom_bench as HB
    RC2.STIFF, RC2.BREAKAWAY = True, 0.0
    RC2.KP_YAW_IM = 0.35 * (1.5 if volts == 7.5 else 1.0)
    w = world()
    bench = BW.make_bench(w)
    bench.q_open = {f: np.radians(np.asarray(q_deg[f], float)) for f in FING}
    params = {"kp": 0.55 * 1.5, "damping": 0.011, "tau_max": 0.18 * 1.5} if volts == 7.5 else dict(RR.SERVER)
    GP = RC2.clutch_class()
    plant = GP(D / "scenes" / SCENE, w["setup"], None, {**params, "mu": 1.0}, 0.0, c=0.6)
    for f in HB.FINGERS:
        plant.m.body_pos[plant.m.body(f"{f}_mount").id] = w["body_mounts"][f]
    if stops:
        for f, lo in STOP_DEG.items():
            jid = plant.m.joint(f"{f}_mcp").id
            plant.m.jnt_range[jid, 0] = max(plant.m.jnt_range[jid, 0], math.radians(lo))
            plant.m.jnt_limited[jid] = 1
    return bench, plant


def _osc(t, x, t0=2.0, t1=4.0):
    sel = (t >= t0) & (t <= t1) & np.isfinite(x)
    x = x[sel]
    if len(x) < 40:
        return None, None
    k = 25
    d = (x - np.convolve(x, np.ones(k) / k, mode="same"))[k // 2:-k // 2]
    f = np.fft.rfftfreq(len(d), 0.02)
    A = np.abs(np.fft.rfft(d - d.mean()))
    return round(float(np.sqrt(np.mean(d ** 2))), 2), round(float(f[1:][np.argmax(A[1:])]), 1)


def run(job):
    try:
        w0 = time.perf_counter()
        bench, plant = _bench_plant(job["q_deg"], job["volts"], job["stops"])
        rows, err = PP.simulate(bench, plant, job["seed"], station=False, contacts=False)
        s = PP.summarize(rows, bench.t_end, bench.task_prm, err)
        t = np.array([r["t"] for r in rows])
        tg = lambda f, j: np.array([r["tg"][f][j] if r.get("tg") else np.nan for r in rows], float)  # noqa: E731
        qq = lambda f, j: np.array([r["q"][f][j] if r.get("q") else np.nan for r in rows], float)   # noqa: E731
        P = np.array([r["pos"] for r in rows], float)
        g = [r for r in rows if 1.40 <= r["t"] <= 1.58]
        grasp = {f: [round(float(np.median([r["q"][f][j] for r in g])), 1) for j in ("aa", "fe1", "fe2")] for f in FING}
        chat = {"middle_fe1_target": _osc(t, tg("middle", "fe1")), "middle_fe2_target": _osc(t, tg("middle", "fe2")),
                "middle_fe1_joint": _osc(t, qq("middle", "fe1")), "obj_x": _osc(t, P[:, 0]), "obj_z": _osc(t, P[:, 2])}
        return {**{k: job[k] for k in ("cand", "volts", "seed", "stops")}, "hos_plant": os.environ.get("HOS_PLANT"),
                "track": s.get("track"), "held": s.get("held"), "error": s.get("error") or err, "grasp_joints": grasp,
                "chatter": chat, "summary": {k: v for k, v in s.items() if not isinstance(v, (list, dict))},
                "wall_s": round(time.perf_counter() - w0, 1)}
    except Exception as e:                                       # noqa: BLE001
        return {**{k: job[k] for k in ("cand", "volts", "seed", "stops")}, "error": repr(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("poses", "sim"))
    ap.add_argument("--cands", nargs="+", default=["plan@sim", "plan"])
    ap.add_argument("--volts", type=float, nargs="+", default=[5.0, 7.5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pf = OUT / "press_poses.json"
    if a.cmd == "poses":
        pf.write_text(json.dumps(candidates(), indent=1) + "\n")
        return
    cands = json.loads(pf.read_text())
    cands["plan@sim"] = cands["plan"]
    jobs = [{"cand": c, "q_deg": cands[c]["deg"], "volts": v, "seed": s, "stops": c != "plan@sim"}
            for c in a.cands for v in a.volts for s in a.seeds]
    import multiprocessing as mp
    out = Path(a.out) if a.out else OUT / "press_sim.jsonl"
    with open(out, "w") as fo, mp.get_context("fork").Pool(a.workers) as pool:
        for r in pool.imap_unordered(run, jobs):
            fo.write(json.dumps(r) + "\n")
            fo.flush()
            print(json.dumps({k: r.get(k) for k in ("cand", "volts", "seed", "track", "held", "error")}), flush=True)


if __name__ == "__main__":
    main()
