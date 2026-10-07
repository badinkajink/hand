#!/usr/bin/env python3
"""The bench reorientation of reorient_backends.py on MuJoCo-Warp: one batch of seeds per hand and condition.

Same scene file (reorient_backends.build_scene), same schedule and scoring; each world is one seed (tool jitter
of reorient_backends.jitter, seed 0 unjittered, so world k pairs with the CPU and Drake rows of seed k). A pad
force is the sum of the finger's contact normal forces on the tool (elliptic cone: the efc row at efc_address[c, 0]),
and a finger counts as engaged when one of its contacts carries more than 1e-6 N, as in reorient_backends.run_mujoco
(rows before 2026-10-07 hold contact counts in the force fields and count contacts at penetration below zero). This
is the backend the RL pipeline trains on (mjlab, 2026-09).

    uv run --extra rl --extra gpu python scripts/reorient_backends_gpu.py --hands D7 --tips tpu6 --worlds 64
Rows: docs/experiments/20261006-fingertip_backends/reorient_gpu.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import reorient_backends as RB  # noqa: E402

OUT = ROOT / "docs/experiments/20261006-fingertip_backends/reorient_gpu.jsonl"


def run_batch(hand, tip, model, plant, ir, mu, seeds, nconmax=768, njmax=3072, block=20):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    plan, traj = RB.load_plan(hand)
    scene, meta = RB.build_scene(hand, tip, model, plant, "bed", ir, mu)
    m = mujoco.MjModel.from_xml_path(str(scene))
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC, "normal force is read from the first efc row"
    d = mujoco.MjData(m)
    tb = m.body(RB.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    tool7 = np.asarray(meta["tool7"], float)
    act = {f"{f}_{j}": m.actuator(f"a_{f}_{j}").id for f in RB.FINGERS for j in RB.JOINTS}
    for n_ in act:
        d.qpos[m.jnt_qposadr[m.joint(n_).id]] = meta["q0"][n_]
    d.qpos[qa:qa + 7] = tool7
    mujoco.mj_forward(m, d)
    nw = len(seeds)
    Q = np.tile(d.qpos, (nw, 1))
    for w, s in enumerate(seeds):
        dx, dy, dyaw = RB.jitter(s)
        Q[w, qa:qa + 3] = tool7[:3] + [dx, dy, 0.0]
        Q[w, qa + 3:qa + 7] = RB._yaw_quat(tool7[3:], dyaw)
    wm = mjw.put_model(m)
    wd = mjw.make_data(m, nworld=nw, nconmax=nconmax, njmax=njmax)
    wd.qpos.assign(Q.astype(np.float32))
    wd.qvel.zero_()
    tool_geoms = np.array([i for i in range(m.ngeom) if m.geom_bodyid[i] == tb])
    finger_of = np.full(m.ngeom, -1)
    for i in range(m.ngeom):
        bn = m.body(m.geom_bodyid[i]).name
        for k, f in enumerate(RB.FINGERS):
            if bn.startswith(f + "_"):
                finger_of[i] = k
    is_tool = np.zeros(m.ngeom, bool)
    is_tool[tool_geoms] = True
    segs = RB.schedule(plan, traj)
    dt = m.opt.timestep
    ctrl = np.tile(d.ctrl, (nw, 1)).astype(np.float32)
    traces = [[] for _ in range(nw)]
    mjw.forward(wm, wd)
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        for _ in range(block):
            mjw.step(wm, wd)
    graph = cap.graph
    # the capture advanced the state by `block` steps; restart from the initial state
    wd.qpos.assign(Q.astype(np.float32))
    wd.qvel.zero_()
    wd.time.zero_()
    if hasattr(wd, "qacc_warmstart"):
        wd.qacc_warmstart.zero_()
    mjw.forward(wm, wd)
    wp.synchronize()

    njmax_ = wd.efc.force.shape[1]

    def snap(t):
        q = wd.qpos.numpy()
        nacon = int(wd.nacon.numpy()[0])
        g = wd.contact.geom.numpy()[:nacon]
        wid = wd.contact.worldid.numpy()[:nacon]
        adr = wd.contact.efc_address.numpy()[:nacon, 0]
        efc_f = wd.efc.force.numpy()
        cnt = np.zeros((nw, 3), int)
        frc = np.zeros((nw, 3))
        sel = (is_tool[g[:, 0]] | is_tool[g[:, 1]]) & (adr >= 0) & (adr < njmax_)
        for (g0, g1), w, a in zip(g[sel], wid[sel], adr[sel]):
            o = g1 if is_tool[g0] else g0
            k = finger_of[o]
            fn = float(efc_f[w, a])
            if k >= 0 and fn > 1e-6:
                cnt[w, k] += 1
                frc[w, k] += fn
        for w in range(nw):
            qw = q[w, qa + 3:qa + 7].astype(float)
            qw /= np.linalg.norm(qw)
            R9 = np.zeros(9)
            mujoco.mju_quat2Mat(R9, qw)
            traces[w].append({"t": round(t, 4), "cos": float(R9[8]), "x": float(q[w, qa]), "y": float(q[w, qa + 1]),
                              "z": float(q[w, qa + 2]), "F": {f: round(float(frc[w, k]), 4) for k, f in enumerate(RB.FINGERS)},
                              "n": {f: int(cnt[w, k]) for k, f in enumerate(RB.FINGERS)}})

    t = 0.0
    W = []
    k_tr = max(1, int(round(RB.TRACE_DT / dt)))
    assert k_tr % block == 0 or block % k_tr == 0
    for t_end, tgt, _ in segs:
        for n_, a in act.items():
            ctrl[:, a] = tgt[n_]
        wd.ctrl.assign(ctrl)
        while t < t_end - 1e-9:
            w0 = time.perf_counter()
            wp.capture_launch(graph)
            wp.synchronize()
            W.append(time.perf_counter() - w0)
            t += block * dt
            if abs(t / RB.TRACE_DT - round(t / RB.TRACE_DT)) < 1e-6:
                snap(t)
    rows = []
    for w, s in enumerate(seeds):
        tr = traces[w]
        bad = (not all(np.isfinite([x["z"] for x in tr]))) or any(abs(x["z"]) > 2.0 for x in tr)
        res = RB.score(tr, segs) if not bad else {}
        res.update(status="ejected" if bad else "complete", seed=s, jitter=RB.jitter(s), F_unit="N",
                   trace_coarse=[[x["t"], round(x["cos"], 4), round(x["z"], 5), round(sum(x["F"].values()), 3),
                                  sum(1 for f in RB.FINGERS if x["n"][f] > 0)] for x in tr[::5]])
        rows.append(res)
    us_world_step = float(np.median(W) / block / nw * 1e6)
    return rows, scene, meta, us_world_step, float(np.median(W) / block * 1e6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", nargs="+", default=list(RB.HANDS))
    ap.add_argument("--tips", nargs="+", default=["tpu6"])
    ap.add_argument("--models", nargs="+", default=["pads"])
    ap.add_argument("--plants", nargs="+", default=["fast"])
    ap.add_argument("--ir", nargs="+", type=float, default=[100.0])
    ap.add_argument("--mu", nargs="+", default=["scene"])
    ap.add_argument("--worlds", type=int, default=64, help="seeds 0 .. worlds-1")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    seeds = list(range(a.worlds))
    for hand in a.hands:
        for tip in a.tips:
            for model in a.models:
                for plant in a.plants:
                    for ir in a.ir:
                        for mu in a.mu:
                            t0 = time.time()
                            base = {"sim": "mjwarp", "hand": hand, "tag": RB.HANDS[hand][0], "tip": tip, "model": model,
                                    "plant": plant, "numerics": "bed", "impratio": ir, "mu": mu, "jitter_mode": "j2",
                                    "n_worlds": len(seeds)}
                            try:
                                rows, scene, meta, usw, uss = run_batch(hand, tip, model, plant, ir, mu, seeds)
                                for r in rows:
                                    r.update(base)
                                    r["case_id"] = f"mjwarp|{hand}|{tip}|{model}|{plant}|bed|{ir:g}|{mu}|j2|{r['seed']}"
                                    r.update(us_per_world_step=usw, us_per_batch_step=uss, scene=str(scene.relative_to(ROOT)),
                                             script="scripts/reorient_backends_gpu.py", git_rev=RB.git_rev(),
                                             when=time.strftime("%Y-%m-%d %H:%M"), wall_s_batch=time.time() - t0)
                                    RB.append_row(a.out, r)
                                held = sum(1 for r in rows if r.get("held_hold"))
                                tt = [r["turn_hold_deg"] for r in rows if r.get("status") == "complete"]
                                print(f"mjwarp {hand} {tip} {model} {plant} ir{ir:g} mu{mu}: held {held}/{len(rows)} "
                                      f"turn {np.mean(tt):+.1f} sd {np.std(tt):.1f}  {usw:.1f} us/world-step, "
                                      f"{uss:.0f} us/batch-step, wall {time.time() - t0:.0f} s", flush=True)
                            except Exception as e:
                                r = dict(base, status="failed", error=repr(e), traceback=traceback.format_exc()[-2000:],
                                         case_id=f"mjwarp|{hand}|{tip}|{model}|{plant}|bed|{ir:g}|{mu}|j2|batch")
                                RB.append_row(a.out, r)
                                print("FAILED", hand, tip, repr(e), flush=True)
                                print(traceback.format_exc()[-1500:], flush=True)


if __name__ == "__main__":
    main()
