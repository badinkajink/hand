"""A softer servo against the hand's joint gaps (hand-4a, 2026-11-16, for the corvus session: "if the sim's sag is several
times smaller, rerun g10 at 4/8 deg and gr g13/g0 under the soft contact with kp 0.15-0.25 (tau 0.18), reporting twist
at the end, tracking, rise and the middle-yaw gap"). In free air the hand's middle yaw sagged 0.9-4.3 deg under the
finger's weight at -10 to -45 deg; the plant matched that at kp 0.20 (rev_free_air.py), while the hand's index yaw sagged
less than the plant's at 0.55. Variants: every servo at kp 0.25, 0.20 or 0.15, and only the middle's yaw at 0.20.

Closed-loop HOM through hom_bench.simulate on the stem scene with the soft tangential contact (impratio 1, friction 1.0)
and the server's plant (kp 0.55, damping 0.011, tau_max 0.18), with sr2_hom_replay's deadband_deg (the servo as a motor
that drives only outside the band) or frictionloss (N m, on every finger joint). Per run over the task's periods (2.0 s
to the schedule's end), in servo degrees: per finger and joint the target minus the measured angle, its median and largest
magnitude; the measured and target medians of the middle's yaw; the thumb PIP's median, minimum and share below -10;
the object's rise, tilt and twist at the end (rev_sweep.feats_arrays) and the tracking 1 - err/ref.

    OMP_NUM_THREADS=1 .venv/bin/python rev_joint_gaps2.py OUT.json [--workers 8]
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
import rev_lat as RL  # noqa: E402
import rev_rescore as RR  # noqa: E402

FINGERS, JOINTS = ("thumb", "index", "middle"), ("aa", "fe1", "fe2")
GRASPS = [("g10_lat8", "lat", 329, 3), ("g10_lat4", "lat", 329, 4), ("gr_g13_x105_y70", "grid2dgr", 329, 13),
          ("gr_g0_x70_y110", "grid2dgr", 162, 0)]
MODELS = [("soft, kp 0.55", {}), ("soft, kp 0.25", {"kp": 0.25}), ("soft, kp 0.20", {"kp": 0.20}),
          ("soft, kp 0.15", {"kp": 0.15}), ("soft, middle yaw kp 0.20", {"kp_middle_aa": 0.20})]
SCENE = D / "scenes" / "sph040_stem_block_tape0_imp1.xml"      # never rebuilt here
_G = {}


def init(path):
    import sr2_hom_replay as RP
    toy = RP.TOY_ROWS                              # the grids' rows, before rev_lat points them at its own file
    RL.init(path)                                  # the stem scene for spheres; toy rows from rev_lat's rows file
    _G["RP"], _G["lat_rows"], _G["toy_rows"] = RP, RP.TOY_ROWS, toy


def run(job):
    try:
        return _run(*job)
    except Exception as e:                                       # noqa: BLE001  HOM's least squares on a dropped sphere
        return {"grasp": job[0], "seed": job[4], "params": job[5], "error": repr(e)[:120]}


def _run(name, src, call, k, seed, params):
    RP = _G["RP"]
    RP.TOY_ROWS = _G["lat_rows"] if src == "lat" else _G["toy_rows"]
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
    bench = HB.HomBench("twist", 40.0, setup, m_pt, q_open, dt=HB.DT_BENCH, p_pads=p_pads)
    kp_mid = params.get("kp_middle_aa")
    base = {k: v for k, v in params.items() if k != "kp_middle_aa"}

    def plant(path, st):
        p = RP.SR2Plant(path, st, mxy, base, 0.0)
        if kp_mid:                                 # the middle's yaw servo alone at its own gain (position servo)
            a_ = int(p.act["middle"][0])
            p.m.actuator_gainprm[a_, 0] = kp_mid
            p.m.actuator_biasprm[a_, 1] = -kp_mid
        return p
    bench.T = RP._Tasks(plant)
    w0 = time.perf_counter()
    stream, res = HB.simulate(bench, seed)
    t = np.array([float(s["t"]) for s in stream])
    P = np.array([s["obj_pos_mm"] for s in stream], float)
    Q = np.array([s["obj_quat"] for s in stream], float)
    f = feats_arrays(t, P, Q)
    task = [s for s in stream if 2.0 - 1e-9 <= float(s["t"]) <= bench.t_end + 1e-9]
    jt = lambda fi, j: np.array([float(s["joints"][fi][j]) for s in task])      # noqa: E731
    tg = lambda fi, j: np.array([float(s["targets"][fi][j]) for s in task])     # noqa: E731
    gap = {f"{fi}.{j}": [round(float(np.median(tg(fi, j) - jt(fi, j))), 1),
                         round(float(np.abs(tg(fi, j) - jt(fi, j)).max()), 1)] for fi in FINGERS for j in JOINTS}
    pip = jt("thumb", "fe2")
    err, ref = res.get("track_err_rms_deg"), res.get("track_ref_rms_deg")
    return {"grasp": name, "seed": seed, "params": params, "held": bool(res.get("held")),
            "track": None if not err or not ref else round(max(0.0, 1.0 - err / ref), 3),
            "rise_max": round(float(f["rise_max"]), 2), "twist_end": round(float(f["twist_end"]), 1),
            "tilt_end": round(float(f["tilt_end"]), 1), "gap_median_max": gap,
            "middle_aa_measured": round(float(np.median(jt("middle", "aa"))), 1),
            "middle_aa_target": round(float(np.median(tg("middle", "aa"))), 1),
            "thumb_pip_median": round(float(np.median(pip)), 1), "thumb_pip_min": round(float(pip.min()), 1),
            "thumb_pip_below10": round(float(np.mean(pip < -10.0)), 3), "periods": len(task),
            "wall_s": round(time.perf_counter() - w0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    assert SCENE.exists(), SCENE
    RL.rows_file()
    import multiprocessing as mp
    jobs = [(n, s, c, k, seed, {**RR.SERVER, "mu": 1.0, **p}) for _l, p in MODELS for (n, s, c, k) in GRASPS
            for seed in (0, 1, 2)]
    with mp.get_context("fork").Pool(a.workers, initializer=init, initargs=(str(SCENE),)) as pool:
        rows = pool.map(run, jobs)
    for r in rows:
        p = {k: v for k, v in r["params"].items() if k in ("kp_middle_aa",) or (k == "kp" and v != RR.SERVER["kp"])}
        r["model"] = next(lab for lab, q in MODELS if q == p)
    Path(a.out).write_text(json.dumps(rows, indent=1) + "\n")
    for lab, _p in MODELS:
        print(lab, flush=True)
        for n, *_ in GRASPS:
            rs = [r for r in rows if r["model"] == lab and r["grasp"] == n and "error" not in r]
            bad = sum(1 for r in rows if r["model"] == lab and r["grasp"] == n and "error" in r)
            if not rs:
                print(f"   {n:16s} all failed", flush=True)
                continue
            m = lambda key: np.mean([r[key] for r in rs])                              # noqa: E731
            g = lambda key: np.mean([r["gap_median_max"][key][0] for r in rs])         # noqa: E731
            print(f"   {n:16s} held {sum(r['held'] for r in rs)}/{len(rs) + bad} track {np.mean([r['track'] or 0 for r in rs]):.2f}"
                  f" twist {m('twist_end'):5.1f} rise {m('rise_max'):5.1f} | gap middle.aa {g('middle.aa'):+6.1f} "
                  f"(meas {m('middle_aa_measured'):+6.1f}, tgt {m('middle_aa_target'):+6.1f}) index.aa {g('index.aa'):+5.1f} "
                  f"thumb.fe1 {g('thumb.fe1'):+5.1f} | thumb PIP med {m('thumb_pip_median'):+6.1f} min "
                  f"{min(r['thumb_pip_min'] for r in rs):+6.1f}" + (f" | {bad} failed" if bad else ""), flush=True)


if __name__ == "__main__":
    main()
