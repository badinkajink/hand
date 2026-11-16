"""The thumb's PIP and the yaw joints' gaps in closed-loop simulation of grasp 10 at (105, 70) and its raised variants
(hand-4a, 2026-11-16, for the corvus session: "run the deep g10 (18.2 and 12 deg) on the stem scene and report the thumb
PIP's minimum and the time it spends below 0 deg ... and the yaw joints' target-minus-measured over the task").

Per run, over the task's periods from 2.0 s to the schedule's end, in servo degrees as scripts/hardware/
hom_trial_tracking.py reports the hand: the thumb PIP's (fe2) smallest measured angle and the share of periods below 0
and below -10 deg; per finger and joint the target minus the measured angle, its median and its largest magnitude. The
runs are rev_closed.run's (hom_bench.simulate on the stem scene with the server's plant, seeds 0-2), with the stream kept.

    OMP_NUM_THREADS=1 .venv/bin/python rev_joint_gaps.py OUT.json [--workers 2]
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
GRASPS = [(0, "lat18.2"), (2, "lat12"), (3, "lat8"), (4, "lat4")]          # indices in rev_lat's rows file
MODELS = [("mu 1.0", {"mu": 1.0}, None), ("mu 0.5", {"mu": 0.5}, None), ("mu 1.0, impratio 1", {"mu": 1.0}, 1.0)]
_G = {}


def init(path):
    RL.init(path)
    import sr2_hom_replay as RP
    _G["RP"] = RP


def run(job):
    name, k, seed, params = job
    try:
        return _run(name, k, seed, params)
    except Exception as e:                                       # noqa: BLE001  HOM's least squares on a dropped sphere
        return {"grasp": name, "seed": seed, "params": params, "error": repr(e)[:120]}


def _run(name, k, seed, params):
    RP = _G["RP"]
    from sr2hand.control import hom_bench as HB
    import mujoco
    import hand_object_scale_sim as S
    import hom_tasks as T
    setup, mxy, _layout = RP.toy_setup("twist", 40.0, 329, k, "grid2dff")
    p_pads, p_pt = RP.ref_scene("sphere", 40.0)
    m_pt = mujoco.MjModel.from_xml_path(str(p_pt))
    RP.move_mounts(m_pt, mxy, RP.mount_base(m_pt))
    q_open, _ = T.open_pose(setup, m_pt)
    S.set_palm(m_pt, setup["palm"])
    bench = HB.HomBench("twist", 40.0, setup, m_pt, q_open, dt=HB.DT_BENCH, p_pads=p_pads)
    bench.T = RP._Tasks(lambda path, st: RP.SR2Plant(path, st, mxy, params, 0.0))
    w0 = time.perf_counter()
    stream, res = HB.simulate(bench, seed)
    task = [s for s in stream if 2.0 - 1e-9 <= float(s["t"]) <= bench.t_end + 1e-9]
    pip = np.array([float(s["joints"]["thumb"]["fe2"]) for s in task])
    gap = {}
    for f in FINGERS:
        for j in JOINTS:
            g = np.array([float(s["targets"][f][j]) - float(s["joints"][f][j]) for s in task])
            gap[f"{f}.{j}"] = [round(float(np.median(g)), 1), round(float(np.abs(g).max()), 1)]
    err, ref = res.get("track_err_rms_deg"), res.get("track_ref_rms_deg")
    return {"grasp": name, "seed": seed, "params": params, "held": bool(res.get("held")),
            "track": None if not err or not ref else round(max(0.0, 1.0 - err / ref), 3),
            "thumb_pip_min": round(float(pip.min()), 1), "thumb_pip_below0": round(float(np.mean(pip < 0.0)), 3),
            "thumb_pip_below10": round(float(np.mean(pip < -10.0)), 3), "gap_median_max": gap,
            "periods": len(task), "wall_s": round(time.perf_counter() - w0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    RL.rows_file()
    import multiprocessing as mp
    out = []
    for label, params, imp in MODELS:
        path = D / "scenes" / ("sph040_stem_block.xml" if imp is None else f"sph040_stem_block_tape0_imp{imp:g}.xml")
        assert path.exists(), path                 # never rebuilt here: the 100-layout re-score reads these files
        jobs = [(name, k, seed, {**RR.SERVER, **params}) for k, name in GRASPS for seed in (0, 1, 2)]
        with mp.get_context("fork").Pool(a.workers, initializer=init, initargs=(str(path),)) as pool:
            rows = pool.map(run, jobs)
        for r in rows:
            r["model"], r["scene"] = label, path.name
        out += rows
        print(label, flush=True)
        for _k, name in GRASPS:
            rs = [r for r in rows if r["grasp"] == name and "error" not in r]
            bad = sum(1 for r in rows if r["grasp"] == name and "error" in r)
            if not rs:
                print(f"   {name:8s} all {bad} runs failed", flush=True)
                continue
            g = lambda key: np.mean([r["gap_median_max"][key][0] for r in rs])    # noqa: E731
            print(f"   {name:8s} held {sum(r['held'] for r in rs)}/{len(rs) + bad} track "
                  f"{np.mean([r['track'] or 0 for r in rs]):.2f} | thumb PIP min {min(r['thumb_pip_min'] for r in rs):6.1f}"
                  f" below 0 {np.mean([r['thumb_pip_below0'] for r in rs]):.2f} below -10 "
                  f"{np.mean([r['thumb_pip_below10'] for r in rs]):.2f} | yaw gap median middle {g('middle.aa'):+.1f} "
                  f"index {g('index.aa'):+.1f} thumb {g('thumb.aa'):+.1f} | MCP gap thumb {g('thumb.fe1'):+.1f} "
                  f"index {g('index.fe1'):+.1f}" + (f" | {bad} failed" if bad else ""), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
