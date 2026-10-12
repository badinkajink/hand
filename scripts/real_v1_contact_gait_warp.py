#!/usr/bin/env python3
"""Replay held-state contact gaits in MuJoCo-Warp parallel worlds."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_v1_d1_continuation as d1  # noqa: E402
from real_v1_d1_continuation import FINGERS, HORIZON_MS, Scene, hold_vector  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--tip", choices=("sphere", "tpu6"), default="tpu6")
    parser.add_argument("--contact-model", choices=("pt", "padsT"), default="padsT")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", default=["hold", "direct_gradient", "cem", "hybrid"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import mujoco_warp as mjw
    import warp as wp

    source = json.loads(args.plans.read_text())
    plans = {"hold": hold_vector().tolist()}
    plans.update({name: source["candidates"][name]["x"] for name in args.methods if name != "hold"})
    names = list(args.methods)
    scenes = {seed: Scene(args.tip, args.contact_model, seed) for seed in args.seeds}
    scene = scenes[args.seeds[0]]
    model = scene.model
    jobs = [(seed, name) for seed in args.seeds for name in names]
    nw = len(jobs)
    qpos = np.stack([scenes[seed].held.qpos for seed, _ in jobs]).astype(np.float32)
    qvel = np.stack([scenes[seed].held.qvel for seed, _ in jobs]).astype(np.float32)
    ctrl0 = np.stack([scenes[seed].ctrl0 for seed, _ in jobs]).astype(np.float32)
    act = np.stack([scenes[seed].held.act for seed, _ in jobs]).astype(np.float32)
    wm = mjw.put_model(model)
    wd = mjw.make_data(model, nworld=nw, nconmax=768, njmax=3072)

    def reset():
        wd.qpos.assign(qpos)
        wd.qvel.assign(qvel)
        wd.ctrl.assign(ctrl0)
        if act.size:
            wd.act.assign(act)
        wd.time.zero_()
        if hasattr(wd, "qacc_warmstart"):
            wd.qacc_warmstart.zero_()
        mjw.forward(wm, wd)
        wp.synchronize()

    reset()
    with wp.ScopedCapture() as cap:
        for _ in range(10):
            mjw.step(wm, wd)
    graph = cap.graph
    reset()
    obj = model.body(d1.OBJ).id
    qa, _ = d1.tool_addresses(model)
    is_tool = np.array([model.geom_bodyid[i] == obj for i in range(model.ngeom)])
    finger_of = d1.finger_of_geom(model)
    traces = [[] for _ in jobs]
    wall_step = []

    def snapshot(t_ms):
        q = wd.qpos.numpy()
        nacon = int(wd.nacon.numpy()[0])
        geoms = wd.contact.geom.numpy()[:nacon]
        world = wd.contact.worldid.numpy()[:nacon]
        address = wd.contact.efc_address.numpy()[:nacon, 0]
        efc = wd.efc.force.numpy()
        forces = np.zeros((nw, 3))
        selected = ((is_tool[geoms[:, 0]] | is_tool[geoms[:, 1]])
                    & (address >= 0) & (address < efc.shape[1]))
        for (g0, g1), w, a in zip(geoms[selected], world[selected], address[selected]):
            other = g1 if is_tool[g0] else g0
            finger = finger_of[other]
            if finger >= 0:
                forces[int(w), finger] += max(0.0, float(efc[int(w), int(a)]))
        for w in range(nw):
            quat = q[w, qa + 3:qa + 7].astype(float)
            quat /= np.linalg.norm(quat)
            cos = 1 - 2*(quat[1]**2 + quat[2]**2)
            traces[w].append({"t_s": t_ms/1000, "vertical_cos": float(cos),
                              "tool_z_m": float(q[w, qa + 2]),
                              "finger_normal_force_N": {f: float(forces[w, j])
                                                        for j, f in enumerate(FINGERS)}})

    def set_controls(k):
        ctrl = np.stack([d1.plan_ctrl(np.asarray(plans[name], float), k, ctrl0[w], scene.low, scene.high,
                                      scene.yaw_ids, scene.flex_ids) for w, (_, name) in enumerate(jobs)])
        wd.ctrl.assign(ctrl.astype(np.float32))

    for k in range(0, HORIZON_MS, 10):
        if k in d1.CONTROL_CHANGE_MS:
            set_controls(k)
        t0 = time.perf_counter()
        wp.capture_launch(graph)
        wp.synchronize()
        wall_step.append((time.perf_counter() - t0)/10)
        snapshot(k + 10)

    rows = []
    for w, (seed, name) in enumerate(jobs):
        trace = traces[w]
        initial = scenes[seed]
        final = trace[-1]
        min_z = min(initial.z0, *(item["tool_z_m"] for item in trace))   # sampled every 10 ms here
        force = final["finger_normal_force_N"]
        release = d1.release_stats(trace, force)
        rows.append({"seed": seed, "plan": name, "initial_cos": initial.cos0, "initial_z_m": initial.z0,
                     **d1.score_and_held(initial.cos0, initial.z0, final["vertical_cos"], min_z, force),
                     "min_tool_z_m": min_z, "index_release_ms": release["index_release_ms"],
                     "index_recontact": release["index_recontact"], "contact_trace": trace})
    output = {"backend": "MuJoCo-Warp", "mujoco_version": scene.mujoco.__version__,
              "source_file": str(args.plans), "scene": str(scene.scene), "tip": args.tip,
              "contact_model": args.contact_model, "worlds": nw, "seeds": args.seeds,
              "methods": names, "mean_batch_step_s": float(np.mean(wall_step)), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"worlds": nw, "mean_batch_step_s": output["mean_batch_step_s"],
                      "rows": [{k: row[k] for k in ("seed", "plan", "score", "cos_gain",
                                                        "held_final", "index_release_ms", "index_recontact")}
                               for row in rows]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
