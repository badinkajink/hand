"""Grasp 13's open pose re-planned inside the hand's measured MCP stops, and HOM's close and twist from each candidate in
the calibrated simulation (hand-4a, 2026-11-17 18:4x; owner via the lab session, 18:0x: "simulation on the workstation
(re-plan g13's open pose; replay the twists with pads and with point contact)"; W1 of
docs/experiments/20261117-sr2_system_identification/QUEUE.md).

On 2026-11-17 the g13 plan sent the index and middle MCPs to -15.0 and -13.7 deg (simulator degrees); they stopped
against the mechanism at -11.4 and -10.0 deg, the index pad 2.6 mm from the sphere on the model (bench README, "State on
2026-11-17 17:22"). Candidates:
  plan      the plan's open pose (the 8 mm open pose of hom_tasks.open_pose, MCPs clipped at the model's limits)
  mcpstop   the plan's pose with the two MCPs at -11.0 and -9.5 (the lab's 17:09-17:11 start)
  open9     the lab's pose that kept both pads 7 mm off (MCPs -9, PIPs 37.2 and 41.4)
  reach     each finger's pad centre as close as the stops allow to the point 8 mm off the object along the contact
            normal (bounded least squares over yaw, MCP, PIP), with the MCP stops 0.5 deg inside the measured ones

    poses  the candidates' joint angles and each pad's clearance from the sphere on the model (no simulation)
    sim    closed loop from each candidate: rev_clutch2's calibrated plant (gears the load cannot backdrive, index and
           middle yaw gains 0.35 at 5 V), HOM converting at the server's 0.55 under the budget 0.56, the stem scene with
           soft contact; the plant's MCPs stopped where the hand's stop (except 'plan@sim', the plan in the unstopped
           model); plant at 5 V or with gain and torque limit x1.5 for 7.5 V; contacts to HOM off or on; seeds 0-2

    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 OMP_NUM_THREADS=1 .venv/bin/python logs/20261117-g13_open_pose/replan.py poses
    HOS_PLANT=kp0.55_kv0_fr0.18_fl0_dp0.011 OMP_NUM_THREADS=1 .venv/bin/python logs/20261117-g13_open_pose/replan.py sim
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
sys.path.insert(0, str(D))
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401
import rev_clutch2 as RC2  # noqa: E402
import rev_gear as RG  # noqa: E402
import rev_rescore as RR  # noqa: E402

PLANS = ROOT / "docs/experiments/20261115-hom_bench_trials/plans"
PLAN = PLANS / "20261116-film_x105_y70_twist_gr_plan.json"
OPEN9 = PLANS / "20261117-film_x105_y70_twist_gr_g13_open9_plan.json"
STOP_DEG = {"index": -11.4, "middle": -10.0}       # where the MCPs stopped on 11-17 (simulator degrees)
MARGIN_DEG = 0.5
SRC, CALL, K = "grid2dgr", 329, 13
SCENES = {"pads": "sph040_stem_block_tape0_imp1.xml", "point": "sph040_stem_block_pt_imp1.xml",
          "pads_stiff": "sph040_stem_block_tape0_imp100.xml"}
PLANT_75 = {"kp": 0.55 * 1.5, "damping": 0.011, "tau_max": 0.18 * 1.5}
JN = ("yaw", "mcp", "pip")


def world():
    """g13's setup and the point-contact model with its mounts, as rev_gear._run builds them."""
    import mujoco
    RG.init(str(RG.RF.SCENE))
    RP = RG.RF._G["RP"]
    RP.TOY_ROWS = RG.RF._G["toy_rows"]
    setup, mxy, _ = RP.toy_setup("twist", 40.0, CALL, K, SRC)
    p_pads, p_pt = RP.ref_scene("sphere", 40.0)
    m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
    RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
    return setup, mxy, m_pt


def clearance(setup, m_pt, q):
    """Each pad's signed distance from the object (mm) at joint angles q (radians per finger)."""
    import mujoco
    import hand_object_scale_sim as S
    import hom_tasks as T
    S.set_palm(m_pt, setup["palm"])
    d = mujoco.MjData(m_pt)
    tb = m_pt.body(S.OBJ).id
    qa = m_pt.jnt_qposadr[m_pt.body_jntadr[tb]]
    d.qpos[qa:qa + 7] = [0.0, 0.0, S.Z_OBJ, *setup["quat"]]
    og = int(np.flatnonzero(m_pt.geom_bodyid == tb)[0])
    for f in T.FINGERS:
        for k, j in enumerate(T.JOINTS):
            d.qpos[m_pt.jnt_qposadr[m_pt.joint(f"{f}_{j}").id]] = q[f][k]
    mujoco.mj_forward(m_pt, d)
    return {f: round(1000 * mujoco.mj_geomDistance(m_pt, d, m_pt.geom(f"{f}_tipgeom").id, og, 0.05, None), 2)
            for f in T.FINGERS}


def reach_pose(setup, m_pt, q_plan):
    """Per finger, the pad-centre point 8 mm off the object along the contact normal, reached as closely as the
    limits allow (the model's limits with the measured MCP stops 0.5 deg inside), by bounded least squares."""
    from scipy.optimize import least_squares
    import hand_object_scale_kin as KIN
    import hom_tasks as T
    lim = KIN.limits(False)
    _q_open, q_touch = T.open_pose(setup, m_pt)
    out = {}
    for f in T.FINGERS:
        lo, hi = lim[f][:, 0].copy(), lim[f][:, 1].copy()
        if f in STOP_DEG:
            lo[1] = max(lo[1], math.radians(STOP_DEG[f] + MARGIN_DEG))
        n = np.asarray(setup["n_out"][f], float)
        goal = KIN.fk(f, np.asarray(q_touch[f], float)[None])["pad"][0] + 0.008 * n   # 8 mm out along n
        x0 = np.clip(np.asarray(q_plan[f], float), lo + 1e-6, hi - 1e-6)
        r = least_squares(lambda q: KIN.fk(f, q[None])["pad"][0] - goal, x0, bounds=(lo, hi))
        out[f] = r.x
    return out


def candidates():
    import hom_tasks as T
    setup, mxy, m_pt = world()
    q_plan, _ = T.open_pose(setup, m_pt)
    plan = json.loads(PLAN.read_text())
    deg = plan["poses"][0]["joints"]
    for f in T.FINGERS:                                   # the plan file is this pose
        assert np.allclose(np.degrees(q_plan[f]), [deg[f][j] for j in JN], atol=1e-6), f
    cand = {"plan": {f: np.array(q_plan[f]) for f in T.FINGERS}}
    q = {f: np.array(q_plan[f]) for f in T.FINGERS}
    q["index"][1], q["middle"][1] = math.radians(-11.0), math.radians(-9.5)
    cand["mcpstop"] = q
    if OPEN9.exists():
        o9 = json.loads(OPEN9.read_text())["poses"][0]["joints"]
        cand["open9"] = {f: np.radians([o9[f][j] for j in JN]) for f in T.FINGERS}
    cand["reach"] = reach_pose(setup, m_pt, q_plan)
    rows = {}
    for name, q in cand.items():
        rows[name] = {"deg": {f: [round(float(v), 2) for v in np.degrees(q[f])] for f in T.FINGERS},
                      "clearance_mm": clearance(setup, m_pt, q)}
        print(f"{name:8s} " + "  ".join(f"{f} {rows[name]['deg'][f]} {rows[name]['clearance_mm'][f]:+.1f} mm"
                                        for f in T.FINGERS), flush=True)
    return rows


# ------------------------------------------------------------------------------------------------ closed loop

def _run(job):
    try:
        return _run1(job)
    except Exception as e:                                       # noqa: BLE001
        return {**job, "error": repr(e)[:160]}


def _run1(job):
    import mujoco
    from sr2hand.control import hom_bench as HB
    import hand_object_scale_sim as S
    import hom_tasks as T
    from rev_sweep import feats_arrays
    RP = RG.RF._G["RP"]
    RP.TOY_ROWS = RG.RF._G["toy_rows"]
    v75 = job["volts"] == 7.5
    RC2.KP_YAW_IM = 0.35 * (1.5 if v75 else 1.0)
    plant_params = {**(PLANT_75 if v75 else RR.SERVER), "mu": 1.0}
    setup, mxy, _layout = RP.toy_setup("twist", 40.0, CALL, K, SRC)
    p_pads, p_pt = RP.ref_scene("sphere", 40.0)
    m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
    RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
    q_open = {f: np.radians(job["q_deg"][f]) for f in T.FINGERS}
    S.set_palm(m_pt, setup["palm"])
    bench = HB.HomBench("twist", 40.0, setup, m_pt, q_open, dt=HB.DT_BENCH, p_pads=p_pads, torque_budget=0.56)
    GP = RG._G["GP"]
    scene = D / "scenes" / SCENES[job["contact"]]
    stops = job["stops"]

    def plant(path, st):
        p = GP(str(scene), st, mxy, plant_params, 0.0, c=0.6)
        if stops:                                        # the hand's MCP stops, 0 deg margin
            for f, lo in STOP_DEG.items():
                jid = p.m.joint(f"{f}_mcp").id
                p.m.jnt_range[jid, 0] = max(p.m.jnt_range[jid, 0], math.radians(lo))
                p.m.jnt_limited[jid] = 1
        return p
    bench.T = RP._Tasks(plant)
    w0 = time.perf_counter()
    stream, res = HB.simulate(bench, job["seed"], contacts=job["contacts"])
    t = np.array([float(s["t"]) for s in stream])
    P = np.array([s["obj_pos_mm"] for s in stream], float)
    Q = np.array([s["obj_quat"] for s in stream], float)
    f = feats_arrays(t, P, Q)
    grasp = [s for s in stream if 1.40 <= float(s["t"]) <= 1.58]
    err, ref = res.get("track_err_rms_deg"), res.get("track_ref_rms_deg")
    med = (lambda fi, j: round(float(np.median([float(s["joints"][fi][j]) for s in grasp])), 1) if grasp else None)
    from scipy.spatial.transform import Rotation as Rot
    ok = np.isfinite(P).all(1) & (np.linalg.norm(Q, axis=1) > 0.5)
    R = Rot.from_quat(Q[ok][:, [1, 2, 3, 0]])
    tw = np.degrees((R * R[0].inv()).as_rotvec()[:, 2])
    keep = np.arange(0, int(ok.sum()), 5)
    return {**{k: job[k] for k in ("cand", "volts", "contacts", "contact", "stops", "seed")},
            "held": bool(res.get("held")),
            "track": None if not err or not ref else round(max(0.0, 1.0 - err / ref), 3),
            "twist_end": round(float(f["twist_end"]), 1), "rise_max": round(float(f["rise_max"]), 2),
            "drift_max": round(float(np.nanmax(np.linalg.norm(P - P[0], axis=1))), 1),
            "grasp_joints": {fi: [med(fi, j) for j in ("aa", "fe1", "fe2")] for fi in ("thumb", "index", "middle")},
            "trace": {"t": t[ok][keep].round(3).tolist(), "twist_deg": tw[keep].round(2).tolist()},
            "wall_s": round(time.perf_counter() - w0, 1)}


def sim(cands, a):
    import multiprocessing as mp
    RG.gear_class = RC2.clutch_class
    RC2.STIFF, RC2.BREAKAWAY = True, 0.0
    RG.RF.RL.rows_file()
    jobs = []
    for name in a.cands:
        for volts in a.volts:
            for contacts in a.contacts:
                for contact in a.contact_models:
                    for seed in a.seeds:
                        q = cands[name]["deg"]
                        jobs.append({"cand": name, "q_deg": q, "volts": volts, "contacts": contacts,
                                     "contact": contact, "stops": name != "plan@sim", "seed": seed})
    print(f"{len(jobs)} runs on {a.workers} workers", flush=True)
    with mp.get_context("fork").Pool(a.workers, initializer=RG.init, initargs=(str(RG.RF.SCENE),)) as pool:
        rows = []
        for r in pool.imap_unordered(_run, jobs):
            rows.append(r)
            print(json.dumps({k: r.get(k) for k in ("cand", "volts", "contacts", "contact", "seed", "held", "track",
                                                   "twist_end", "drift_max", "grasp_joints", "error")}), flush=True)
    return rows


def grid_poses(dyaws=(-8, -4, -3, -2, -1, 0, 4, 8), dpips=(0, 2, 3, 4, 6, 9)):
    """Index and middle at their MCP stops + 0.5 deg, the plan's yaw moved by dyaw away from the palm's centre line
    (sign of the plan's yaw) and the plan's PIP less dpip; the thumb as planned. Clearance on the model per pose."""
    import hom_tasks as T
    setup, mxy, m_pt = world()
    q_plan, _ = T.open_pose(setup, m_pt)
    out = {}
    for dy in dyaws:
        for dp in dpips:
            q = {f: np.array(q_plan[f]) for f in T.FINGERS}
            for f in STOP_DEG:
                q[f][0] += math.radians(dy) * math.copysign(1.0, q[f][0])
                q[f][1] = math.radians(STOP_DEG[f] + MARGIN_DEG)
                q[f][2] -= math.radians(dp)
            name = f"y{dy:+d}_p{dp}"
            out[name] = {"deg": {f: [round(float(v), 2) for v in np.degrees(q[f])] for f in T.FINGERS},
                         "clearance_mm": clearance(setup, m_pt, q)}
            print(name, out[name]["clearance_mm"], flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("poses", "sim", "grid"))
    ap.add_argument("--cands", nargs="+", default=["plan@sim", "plan", "mcpstop", "open9", "reach"])
    ap.add_argument("--volts", type=float, nargs="+", default=[5.0, 7.5])
    ap.add_argument("--contacts", type=lambda s: s == "on", nargs="+", default=[False, True])
    ap.add_argument("--contact-models", nargs="+", default=["pads"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    poses_file = OUT / "poses.json"
    if a.cmd == "poses":
        rows = candidates()
        poses_file.write_text(json.dumps(rows, indent=1) + "\n")
        return
    if a.cmd == "grid":
        g = grid_poses()
        rows = {**json.loads(poses_file.read_text()), **g}
        poses_file.write_text(json.dumps(rows, indent=1) + "\n")
        return
    cands = json.loads(poses_file.read_text())
    cands["plan@sim"] = cands["plan"]
    rows = sim(cands, a)
    out = Path(a.out) if a.out else OUT / "sim.json"
    meta = {"hos_plant": os.environ.get("HOS_PLANT"), "budget": 0.56, "stop_deg": STOP_DEG, "plant_75": PLANT_75,
            "scenes": SCENES, "kp_yaw_im_5v": 0.35, "c_f": 0.6, "stiff": True, "written": time.strftime("%F %H:%M")}
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
