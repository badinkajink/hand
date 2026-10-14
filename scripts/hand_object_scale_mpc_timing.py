#!/usr/bin/env python3
"""Cost of the sphere-pad fingertip in the MPC's rollouts, against point contact on the TPU block, on one scale-study
cell with all three fingers on the cylinder.

For each tip model (pads: 1 mm sphere pads; pt: the TPU block as one convex mesh with point contact) and each state
(open: the hand open, nothing touching; grip: 0.8 s after the grip command, three fingers pressing), it reports
  - single-thread mj_step time, with its split into mj_fwdPosition (kinematics, collision, constraint assembly), the
    constraint solve (mj_fwdConstraint) and the rest, over 500 steps;
  - contacts (ncon) and constraint rows (nefc);
  - mujoco.rollout wall time for 64 plans of 0.5 s (the three-pad MPC's replanning) at 1, 6 and 12 threads.

  .venv/bin/python scripts/hand_object_scale_mpc_timing.py --plant kp0.72_kv0_fr0.18_fl0_dp0.0144 --cell cylinder:16:0.625
Writes docs/experiments/20261013-scale_turn_controllers/planners/timing.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hand_object_scale_mpc as M  # noqa: E402
import hand_object_scale_sim as S  # noqa: E402
import hom_turn3 as H  # noqa: E402

OUT = M.OUT / "planners" / "timing.json"


def step_split(mj, m, d, n=500):
    """Mean seconds per mj_step and per phase over n steps from the current state (state restored afterwards)."""
    spec = mj.mjtState.mjSTATE_FULLPHYSICS
    x0 = np.empty(mj.mj_stateSize(m, spec))
    mj.mj_getState(m, d, x0, spec)
    t0 = time.perf_counter()
    for _ in range(n):
        mj.mj_step(m, d)
    full = (time.perf_counter() - t0) / n
    mj.mj_setState(m, d, x0, spec)
    tp = tc = 0.0
    nc, ne = [], []
    for _ in range(n):
        a = time.perf_counter()
        mj.mj_fwdPosition(m, d)
        b = time.perf_counter()
        mj.mj_fwdVelocity(m, d)
        mj.mj_fwdActuation(m, d)
        mj.mj_fwdAcceleration(m, d)
        c = time.perf_counter()
        mj.mj_fwdConstraint(m, d)
        e = time.perf_counter()
        nc.append(d.ncon)
        ne.append(d.nefc)
        tp += b - a
        tc += e - c
        mj.mj_step(m, d)
    mj.mj_setState(m, d, x0, spec)
    return {"step_us": 1e6 * full, "fwd_position_us": 1e6 * tp / n, "constraint_us": 1e6 * tc / n,
            "ncon_mean": float(np.mean(nc)), "ncon_max": int(np.max(nc)), "nefc_mean": float(np.mean(ne))}


def rollout_time(mj, m, d, nthread, nsample=64, nstep=500, sigma=0.08, act=None):
    from mujoco import rollout
    spec = mj.mjtState.mjSTATE_FULLPHYSICS
    x0 = np.empty(mj.mj_stateSize(m, spec))
    mj.mj_getState(m, d, x0, spec)
    rng = np.random.default_rng(0)
    ctrl = np.tile(d.ctrl, (nsample, nstep, 1))
    ctrl[:, :, act] += rng.normal(0, sigma, (nsample, 1, len(act)))
    pool = rollout.Rollout(nthread=nthread)
    datas = [mj.MjData(m) for _ in range(nthread)]
    pool.rollout(m, datas, x0[None], ctrl[:2, :20], nstep=20)          # warm up the pool
    t0 = time.perf_counter()
    pool.rollout(m, datas, x0[None], ctrl, nstep=nstep)
    wall = time.perf_counter() - t0
    pool.close() if hasattr(pool, "close") else None
    return {"nthread": nthread, "wall_s": wall, "us_per_step_thread": 1e6 * wall * nthread / (nsample * nstep)}


def main() -> int:
    import mujoco as mj
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plant", default="kp0.72_kv0_fr0.18_fl0_dp0.0144")
    ap.add_argument("--cell", default="cylinder:16:0.625")
    ap.add_argument("--hom-rows", default=str(M.OUT / "hom_postoff_servo/turn.jsonl"))
    a = ap.parse_args()
    S.PLANT = a.plant
    shape, dd, t = a.cell.split(":")
    dd, t = float(dd), float(t)
    cell, _, _ = next(c for c in S.cell_list("diag") if c[1] == shape and c[2] == dd and c[0].get("t") == t)
    ci = M.hom_best(Path(a.hom_rows))[(cell["tag"], shape, dd)]["cand"]
    plant, meta, geo, sc, gg = M.setup(cell, shape, dd, ci)
    out = {"plant": a.plant, "cell": a.cell, "tag": cell["tag"], "cand": ci, "models": {}}
    for mdl in ("pads", "pt"):
        path = sc[mdl][0]
        p = H.MjPlant(path, meta)
        S.set_palm(p.m, gg["palm"])
        m, d = p.m, p.d
        res = {"ngeom": int(m.ngeom), "nbody": int(m.nbody)}
        p.reset(0)
        q_open = d.ctrl[p.act].copy()
        res["open"] = step_split(mj, m, d)
        grip = np.array([meta["grip"][f][k] for f in S.FINGERS for k in range(3)])
        p.set_targets(grip)
        p.advance(0.8)
        s = p.state()
        res["grip_fingers"] = {f: round(s["F"][f], 2) for f in S.FINGERS}
        res["grip"] = step_split(mj, m, d)
        res["rollout"] = [rollout_time(mj, m, d, n, act=p.act) for n in (1, 6, 12)]
        out["models"][mdl] = res
        print(mdl, json.dumps({k: v for k, v in res.items() if k != "rollout"}), flush=True)
        for r in res["rollout"]:
            print("   rollout", r, flush=True)
        del q_open
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
