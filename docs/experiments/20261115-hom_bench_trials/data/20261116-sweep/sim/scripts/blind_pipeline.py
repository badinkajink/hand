"""The design pipeline's simulation of the blind twist grasps, for the comparison with the hand and the calibrated model
(hand-4a, 2026-11-16 20:5x; owner: "i want to end the presentation strong, with some preliminary evidence that we can
get approximately close to real in simulation first"). Each grasp of grid2dgr at layout (105, 70), seeds 0-2, rolled out
on the CPU as the landscapes' scores and the talk's drawings are (hom_tasks.rollout as hom_abstract_film.py record does:
the SCS0009 plant at 0.72 N m/rad that the load can backdrive, HOM converting at 0.72, the scenes' stiff contact, the
sphere on its post). Per run: the tracking score from 2.0 s (1 - err/ref, as hom_tasks.metrics), and the twist about
the vertical every 0.1 s from the rows' angular velocity (yaw_angles of hom_abstract_film).

    HOS_PLANT=servo HOS_SCENES=logs/20261102-hom_servo_plant/scenes HOM_ROWS_TAG=_sv25 OMP_NUM_THREADS=1 \\
        .venv/bin/python logs/20261116-sim_real_reversal/blind_pipeline.py OUT.json --grasps 4 8 12 13 [--workers 6]
"""
import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("SCALE_HAND", "cad_v2")
import numpy as np  # noqa: E402

ROOT = Path.home() / "Programs/hand"
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401


def run(job):
    """One rollout; a run whose state diverged in HOM's least squares (the sphere already dropped) is kept with its
    error and tracking 0, as rev_rescore.py keeps them."""
    try:
        return _run(*job)
    except Exception as e:                                       # noqa: BLE001
        return {"grasp": job[0], "seed": job[1], "held": False, "track": 0.0, "J": 0.0, "twist_end": None,
                "error": repr(e)[:160], "trace": None}


def _run(grasp, seed):
    import mujoco
    import hom_films as HF
    import hom_gpu as HG
    import hom_landscape as L
    T, S, H = HG.T, HG.S, HG.H
    L.GRID_SRC = "grid2dgr"
    call = HF.call_at("twist", 40.0, 105.0, 70.0)
    _key, mxy, setup = HF.find("twist", "grid2dgr_size", 40.0, call, grasp)
    scene = HG.Scene(T.SHAPE["twist"], 40.0)
    bm, q_open = scene.prepare(mxy, setup)
    plant = T.Plant(scene.pads_path, setup)
    m_pt = mujoco.MjModel.from_xml_path(str(scene.pt_path))
    for m in (plant.m, m_pt):
        for f in HG.FINGERS:
            m.body_pos[m.body(f"{f}_mount").id] = bm[f]
    S.set_palm(m_pt, setup["palm"])
    fs, os_ = T.sr2_specs(setup)
    mirror = H.Mirror(m_pt, fs, os_)
    over, noise = T.VARIANTS["base"]
    res, rows, _frames = T.rollout("twist", setup, plant, mirror, q_open, seed, T.PRM,
                                   replace(H.HomParams(dt=T.DT_C), **over), None, noise, 0.0)
    h = T.DT_C
    t = np.array([r["t"] for r in rows], float)
    w = np.array([r["w_dps"][2] for r in rows], float)
    wr = np.array([(r.get("w_ref_dps") or [0.0, 0.0, 0.0])[2] for r in rows], float)
    tw, cmd = np.cumsum(w) * h, np.cumsum(wr) * h
    keep = np.arange(0, len(t), max(1, int(round(0.1 / h))))
    err, ref = res.get("track_err_rms_deg"), res.get("track_ref_rms_deg")
    return {"grasp": grasp, "seed": seed, "call": call, "held": bool(res.get("held")),
            "track": None if not err or not ref else round(max(0.0, 1.0 - err / ref), 3),
            "J": round(float(L.track_score("twist", res)), 3), "contact_all_frac": res.get("contact_all_frac"),
            "twist_end": round(float(tw[-1]), 1), "plant": os.environ.get("HOS_PLANT"),
            "trace": {"t": t[keep].round(3).tolist(), "twist_deg": tw[keep].round(2).tolist(),
                      "cmd_deg": cmd[keep].round(2).tolist()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--grasps", type=int, nargs="+", default=[4, 8, 12, 13])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    import multiprocessing as mp
    jobs = [(g, s) for g in a.grasps for s in a.seeds]
    with mp.get_context("fork").Pool(a.workers) as pool:
        rows = pool.map(run, jobs)
    Path(a.out).write_text(json.dumps({"meta": {"plant": os.environ.get("HOS_PLANT"),
                                                "scenes": os.environ.get("HOS_SCENES")}, "rows": rows}) + "\n")
    for g in a.grasps:
        rs = [r for r in rows if r["grasp"] == g]
        tw = [r["twist_end"] for r in rs if r["twist_end"] is not None]
        print(f"g{g}: track {np.mean([r['track'] for r in rs]):.3f} [{min(r['track'] for r in rs):.3f}, "
              f"{max(r['track'] for r in rs):.3f}] twist {np.mean(tw) if tw else float('nan'):.1f} held "
              f"{sum(r['held'] for r in rs)}/{len(rs)} J {np.mean([r['J'] for r in rs]):.3f} diverged "
              f"{sum('error' in r for r in rs)}", flush=True)


if __name__ == "__main__":
    main()
