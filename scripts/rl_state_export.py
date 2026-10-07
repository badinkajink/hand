#!/usr/bin/env python3
"""Export a held state of the RL training env for timing other engines on it (2026-10-06).

The D6 reorientation env of the RL pipeline (rl_contact_throughput.base_cfg: the calibrated-plant run of each fingertip
variant, zero residual actions, so the scripted grasp closes the fingers and the palm lifts the tool) steps `--steps`
policy steps at `--envs` envs; the lift ends at policy step 32. The scene is written with mjlab's Scene.write
(scene.xml + assets/), the trainer's solver options are written into its <option> (mjlab applies them to the compiled
model only), sensors are stripped, and per world the npz holds qpos, qvel, ctrl, act, qacc_warmstart, the tool's
position, its contact count with each fingertip and the world's contact and constraint counts.

    WARP_CACHE_PATH=$(mktemp -d) uv run --extra rl --extra gpu python scripts/rl_state_export.py --variants legacy mesh pads1
Out: logs/20261006-rl_contact/state_<variant>/{scene.xml, assets/, state.npz, state.json}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rl_contact_throughput as T  # noqa: E402

OUT = ROOT / "logs/20261006-rl_contact"


def export(variant, num_envs, steps):
    import mujoco
    import torch
    cfg = T.base_cfg(ROOT / T.RUNS[variant])
    env = T.make_env(cfg, num_envs, 1024, 4096)
    act = torch.zeros((num_envs, env.action_manager.total_action_dim), device="cuda:0")
    env.reset()
    for _ in range(steps):
        env.step(act)
    torch.cuda.synchronize()
    mjm, wd = env.sim.mj_model, env.sim.wp_data
    out = OUT / f"state_{variant}"
    env.scene.write(out)
    # solver options of the trainer, as compiled
    xml_p = out / "scene.xml"
    root = ET.parse(xml_p).getroot()
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
        root.remove(opt)
        root.insert(0, opt)
    o = mjm.opt
    cone = {0: "pyramidal", 1: "elliptic"}[int(o.cone)]
    integ = {0: "Euler", 1: "RK4", 2: "implicit", 3: "implicitfast"}[int(o.integrator)]
    opt.attrib.update(timestep=f"{o.timestep:g}", iterations=str(o.iterations), ls_iterations=str(o.ls_iterations),
                      tolerance=f"{o.tolerance:g}", impratio=f"{o.impratio:g}", cone=cone, integrator=integ,
                      gravity=" ".join(f"{g:g}" for g in o.gravity))
    for s in list(root.iter("sensor")):
        for parent in root.iter():
            if s in list(parent):
                parent.remove(s)
    ET.ElementTree(root).write(xml_p)
    m2 = mujoco.MjModel.from_xml_path(str(xml_p))
    names = [mjm.body(i).name for i in range(mjm.nbody)]
    assert (m2.nq, m2.nv, m2.nu) == (mjm.nq, mjm.nv, mjm.nu), "exported model differs in size"
    tool = next(i for i, x in enumerate(names) if x.split("/")[-1] == "cube")
    tips = {f: next(i for i, x in enumerate(names) if x.endswith(f"{f}_tip")) for f in ("thumb", "index", "middle")}
    n = int(wd.nacon.numpy()[0])
    geom, wid = wd.contact.geom.numpy()[:n], wd.contact.worldid.numpy()[:n]
    tipc = np.zeros((num_envs, 3), int)
    for (g1, g2), w in zip(geom, wid):
        b = {int(mjm.geom_bodyid[g1]), int(mjm.geom_bodyid[g2])}
        for k, tb in enumerate(tips.values()):
            if b == {tb, tool}:
                tipc[w, k] += 1
    arrays = {k: getattr(wd, k).numpy().copy() for k in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart")}
    arrays["tool_pos"] = wd.xpos.numpy()[:, tool, :].copy()
    # mjlab spaces the envs on a grid: the robot's root body is moved per world in the GPU model and free bodies carry
    # the origin in their qpos. The MJCF has the robot at its own origin, so free-joint positions are written relative
    # to each world's offset (the root body's GPU position minus its position in a plain compile at that qpos).
    d0 = mujoco.MjData(m2)
    root_b = next(i for i in range(1, mjm.nbody) if mjm.body_parentid[i] == 0 and names[i].startswith("robot/"))
    off = np.zeros((num_envs, 3))
    for w in range(num_envs):
        d0.qpos[:] = arrays["qpos"][w]
        mujoco.mj_kinematics(m2, d0)
        off[w] = wd.xpos.numpy()[w, root_b] - d0.xpos[root_b]
    for j in range(m2.njnt):
        if m2.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = m2.jnt_qposadr[j]
            arrays["qpos"][:, a:a + 3] -= off
    arrays["tool_pos"] -= off
    arrays["env_offset"] = off
    arrays["tip_tool_contacts"] = tipc
    arrays["ncon_world"] = np.bincount(wid, minlength=num_envs)
    arrays["nefc_world"] = wd.nefc.numpy().copy()
    np.savez(out / "state.npz", **arrays)
    held = (tipc > 0).sum(axis=1) >= 2
    meta = {"variant": variant, "run": T.RUNS[variant], "num_envs": num_envs, "policy_steps": steps,
            "tool_body": names[tool], "tip_bodies": {f: names[b] for f, b in tips.items()},
            "nq": mjm.nq, "nv": mjm.nv, "nu": mjm.nu, "ngeom": mjm.ngeom,
            "held_worlds": int(held.sum()), "tool_z_mm_median": float(1e3 * np.median(arrays["tool_pos"][:, 2])),
            "ncon_world_mean": float(arrays["ncon_world"].mean()), "nefc_world_mean": float(arrays["nefc_world"].mean()),
            "tip_tool_contacts_mean": tipc.mean(axis=0).tolist(), "root_body": names[root_b],
            "env_offset_range_m": [float(off.min()), float(off.max())],
            "option": dict(opt.attrib)}
    dc = mujoco.MjData(m2)
    dc.qpos[:] = arrays["qpos"][0]
    mujoco.mj_forward(m2, dc)
    meta["cpu_contacts_world0"] = int(dc.ncon)
    (out / "state.json").write_text(json.dumps(meta, indent=1))
    env.close()
    print(json.dumps(meta), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", default=["legacy", "mesh", "pads1"])
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    for v in a.variants:
        export(v, a.envs, a.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
