"""Gear friction that grows with the torque a servo transmits (hand-4a, 2026-11-16 15:1x, for the corvus session). A
constant dry friction of 0.15 N m on every finger joint reproduced the hand's tracking over the latitude series, but it
would stop a free-air joint about 16 deg short (0.15 / 0.55 rad), where the hand's joints came within 1-4 deg
(data/20261116-sweep/20261116_joint_sag.json). A spur train's friction grows with its load: here each finger joint's
frictionloss is set before every physics step to c |tau| (tau that joint's servo torque at the previous step), so it is
small in free air and near c x 0.18 N m when the servo pushes at its limit under the grip.

Two checks per c: the free-air sag steps of rev_joint_sag.py on the server plant, and the closed-loop series of
rev_fric2.py (grasp 10 at 18.2/12/8/4 deg and the two gr grasps, seeds 0-2, stem scene, soft contact at friction 1.0).

    OMP_NUM_THREADS=1 .venv/bin/python rev_gear.py gear_soft.json --c 0.5 0.8 [--workers 8]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Programs/hand"
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(D))
import rev_fric2 as RF  # noqa: E402
import rev_joint_sag as JS  # noqa: E402
import rev_rescore as RR  # noqa: E402

_G = {}
BUDGET = None          # N m: hom_bench.torque_budget on HOM's targets, as the bench server's --torque-budget (rev_predict)
TRACE = False          # keep each run's twist about z_P over time, every 0.1 s (rev_predict --trace)
MU = 1.0               # the plant's sliding friction (rev_predict --mu)


def gear_class():
    import sr2_hom_replay as RP

    class GearPlant(RP.SR2Plant):
        """SR2Plant with each finger joint's frictionloss c |servo torque|, updated every physics step."""

        def __init__(self, *a, c=0.8, **k):
            super().__init__(*a, **k)
            self.c = c

        def advance(self, t1: float):
            m, d = self.m, self.d
            h = float(m.opt.timestep)
            while d.time < t1 - 1e-9:
                m.dof_frictionloss[self.vadr9] = self.c * np.abs(d.actuator_force[self.acts])
                super().advance(min(t1, d.time + h))
    return GearPlant


def free_air(c):
    """rev_joint_sag's steps on the server plant with the gear friction, the sim's shortfall per sequence."""
    from sr2hand.control import hom_bench as HB
    GP = gear_class()
    out = []
    for f, j, steps, clear in JS.SEQS:
        import sr2_hom_replay as RP
        import mujoco
        import hand_object_scale_sim as S
        import hom_tasks as T
        setup, mxy, _ = RP.toy_setup("twist", 40.0, 329, 10, "grid2dff")
        p_pads, p_pt = RP.ref_scene("sphere", 40.0)
        m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
        RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
        q_open, _ = T.open_pose(setup, m_pt)
        S.set_palm(m_pt, setup["palm"])
        p = GP(p_pads, setup, mxy, {**RR.SERVER, **JS.PLANTS["server"], "mu": 1.0}, 0.0, c=c)
        p.reset(q_open, 0)
        m = p.m
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == p.tb or g == p.post:
                m.geom_contype[g], m.geom_conaffinity[g] = 0, 0
        servo = {fi: {"aa": 0.0, "fe1": 0.0, "fe2": 0.0} for fi in ("thumb", "index", "middle")}
        for fi, js in clear.items():
            servo[fi].update(js)
        t = 2.0
        p.set_targets(HB.sim_from_servo(servo))
        p.advance(t)
        sag = []
        for x in steps:
            servo[f][j] = x
            p.set_targets(HB.sim_from_servo(servo))
            t += 2.0
            p.advance(t)
            sag.append(round(abs(x) - abs(float(HB.servo_from_sim(p.state()["q"])[f][j])), 2))
        sign = 1 if steps[0] > 0 else -1
        out.append({"finger": f, "joint": j, "sign": sign, "sim": sag, "hand": JS.HAND.get((f, j, sign))})
    return out


def init(path):
    RF.init(path)
    _G["GP"] = gear_class()


def run(job):
    try:
        return _run(*job)
    except Exception as e:                                       # noqa: BLE001
        return {"grasp": job[0], "seed": job[4], "c": job[5], "error": repr(e)[:120]}


def _run(name, src, call, k, seed, c):
    RP = RF._G["RP"]
    RP.TOY_ROWS = RF._G["lat_rows"] if src == "lat" else RF._G["toy_rows"]
    from sr2hand.control import hom_bench as HB
    import mujoco
    import hand_object_scale_sim as S
    import hom_tasks as T
    from rev_sweep import feats_arrays
    setup, mxy, _layout = RP.toy_setup("twist", 40.0, call, k, "grid2dff" if src == "lat" else src)
    p_pads, p_pt = RP.ref_scene("sphere", 40.0)
    m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
    RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
    q_open, _ = T.open_pose(setup, m_pt)
    S.set_palm(m_pt, setup["palm"])
    bench = HB.HomBench("twist", 40.0, setup, m_pt, q_open, dt=HB.DT_BENCH, p_pads=p_pads, torque_budget=BUDGET)
    params = {**RR.SERVER, "mu": MU}
    GP = _G["GP"]
    bench.T = RP._Tasks(lambda path, st: GP(path, st, mxy, params, 0.0, c=c))
    w0 = time.perf_counter()
    stream, res = HB.simulate(bench, seed)
    t = np.array([float(s["t"]) for s in stream])
    P = np.array([s["obj_pos_mm"] for s in stream], float)
    Q = np.array([s["obj_quat"] for s in stream], float)
    f = feats_arrays(t, P, Q)
    task = [s for s in stream if 2.0 - 1e-9 <= float(s["t"]) <= bench.t_end + 1e-9]
    gap = (lambda fi, j: round(float(np.median([float(s["targets"][fi][j]) - float(s["joints"][fi][j]) for s in task])), 1))
    err, ref = res.get("track_err_rms_deg"), res.get("track_ref_rms_deg")
    bs = [float(s["budget_scale"]) for s in stream if s.get("budget_scale") is not None]
    trace = None
    if TRACE:                                   # the twist about z_P from the first sample, as the hand's records give it
        from scipy.spatial.transform import Rotation as Rot
        ok = np.isfinite(P).all(1) & (np.linalg.norm(Q, axis=1) > 0.5)
        R = Rot.from_quat(Q[ok][:, [1, 2, 3, 0]])
        tw = np.degrees((R * R[0].inv()).as_rotvec()[:, 2])
        tt = t[ok]
        keep = np.arange(0, len(tt), 5)
        trace = {"t": tt[keep].round(3).tolist(), "twist_deg": tw[keep].round(2).tolist()}
    return {"grasp": name, "seed": seed, "c": c, "held": bool(res.get("held")),
            "budget_scale_min": round(min(bs), 3) if bs else None, "trace": trace,
            "budget_scale_median": round(float(np.median(bs)), 3) if bs else None,
            "track": None if not err or not ref else round(max(0.0, 1.0 - err / ref), 3),
            "twist_end": round(float(f["twist_end"]), 1), "rise_max": round(float(f["rise_max"]), 2),
            "gap_middle_aa": gap("middle", "aa"), "gap_index_aa": gap("index", "aa"), "gap_thumb_fe1": gap("thumb", "fe1"),
            "thumb_pip_median": round(float(np.median([float(s["joints"]["thumb"]["fe2"]) for s in task])), 1),
            "wall_s": round(time.perf_counter() - w0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--c", type=float, nargs="+", default=[0.5, 0.8])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = {"free_air": {}, "series": []}
    for c in a.c:
        fa = free_air(c)
        out["free_air"][str(c)] = fa
        for r in fa:
            print(f"free air c {c}: {r['finger']:6s} {r['joint']:3s} {'+' if r['sign'] > 0 else '-'}: sim "
                  + "/".join(f"{v:.2f}" for v in r["sim"]) + (" | hand " + "/".join(f"{v:.1f}" for v in r["hand"])
                                                              if r["hand"] else ""), flush=True)
    RF.RL.rows_file()
    import multiprocessing as mp
    jobs = [(n, s, cl, k, seed, c) for c in a.c for (n, s, cl, k) in RF.GRASPS for seed in (0, 1, 2)]
    with mp.get_context("fork").Pool(a.workers, initializer=init, initargs=(str(RF.SCENE),)) as pool:
        rows = pool.map(run, jobs)
    out["series"] = rows
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for c in a.c:
        print(f"series c {c}", flush=True)
        for n, *_ in RF.GRASPS:
            rs = [r for r in rows if r["c"] == c and r["grasp"] == n and "error" not in r]
            bad = sum(1 for r in rows if r["c"] == c and r["grasp"] == n and "error" in r)
            if not rs:
                print(f"   {n:16s} all failed", flush=True)
                continue
            mm = lambda key: np.mean([r[key] for r in rs])                              # noqa: E731
            print(f"   {n:16s} held {sum(r['held'] for r in rs)}/{len(rs) + bad} track {np.mean([r['track'] or 0 for r in rs]):.2f}"
                  f" twist {mm('twist_end'):5.1f} rise {mm('rise_max'):5.1f} | gap middle.aa {mm('gap_middle_aa'):+6.1f} "
                  f"index.aa {mm('gap_index_aa'):+5.1f} thumb.fe1 {mm('gap_thumb_fe1'):+5.1f} | thumb PIP med "
                  f"{mm('thumb_pip_median'):+6.1f}" + (f" | {bad} failed" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
