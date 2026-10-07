#!/usr/bin/env python3
"""Regression check of the RL env's contact sensors after the switch to reduce="netforce" (2026-10-06).

env_build._build_sensors now sums every matched contact per fingertip (and for the palm) and the sim allocates
contact_sensor_maxmatch 256. Two checks on the D6 reorientation env of the RL pipeline (zero residual actions:
the scripted grasp and lift):

  legacy   the box tip with point contact. The old sensors (reduce "none", one slot) are added next to the new ones
           in the same env, so both read the same physics state. For 200 policy steps at 64 envs, every reward and
           termination term that takes a `sensor_name` is evaluated with the new and with the old sensor, and the
           per-tip force norms and found flags are compared.
  pads1    the TPU block with 1 mm sphere pads. At a few steps after the grasp the tip-tool contact forces of each
           world's GPU solve are summed per finger (efc_force of each contact, elliptic cone, rotated to the world
           frame by the contact frame); the sensor's netforce norm is compared with that sum. The same sum after a CPU
           re-solve (qpos, qvel, ctrl and warm start copied into MjData, mj_forward) is reported beside it.
           mujoco_warp 3.6's get_data_into copies the contacts of the wrong world, so the GPU arrays are read directly.

mjlab caches a sensor's data at its first read in a step (the reward terms, before the env's closing forward); both
sensors' caches are invalidated before each comparison so that both read the same sensordata.

    WARP_CACHE_PATH=$(mktemp -d) uv run --extra rl --extra gpu python scripts/rl_contact_sensor_check.py
Rows: docs/experiments/20261006-rl_contact/sensor_check.jsonl (one per check, fsynced).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import rl_contact_throughput as T  # noqa: E402

OUT = ROOT / "docs/experiments/20261006-rl_contact/sensor_check.jsonl"
OLD = {"fingertip_cube_contact": "fingertip_cube_contact_none", "palm_cube_contact": "palm_cube_contact_none"}


def make_env(variant, num_envs, with_old):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.sensor import ContactSensorCfg
    from morphohand.rl.env_cfg import to_mjlab_cfg
    cfg = T.base_cfg(ROOT / T.RUNS[variant])
    mj = to_mjlab_cfg(dataclasses.replace(cfg, num_envs=num_envs))
    mj.sim.nconmax, mj.sim.njmax = 1024, 4096
    if with_old:
        extra = []
        for s in mj.scene.sensors:
            if s.name in OLD:
                extra.append(dataclasses.replace(s, name=OLD[s.name], reduce="none", num_slots=1))
        mj.scene.sensors = tuple(mj.scene.sensors) + tuple(extra)
    return ManagerBasedRlEnv(cfg=mj, device="cuda:0", render_mode=None)


def sensor_terms(env):
    """(manager, name, cfg) of every reward and termination term with a sensor_name parameter."""
    out = []
    for kind, mgr in (("reward", env.reward_manager), ("termination", env.termination_manager)):
        for name, tc in zip(mgr._term_names, mgr._term_cfgs):
            if isinstance(tc.params, dict) and tc.params.get("sensor_name") in OLD:
                out.append((kind, name, tc))
    return out


def check_legacy(steps, num_envs):
    import torch
    env = make_env("legacy", num_envs, with_old=True)
    terms = sensor_terms(env)
    act = torch.zeros((num_envs, env.action_manager.total_action_dim), device="cuda:0")
    env.reset()
    worst = {f"{k}:{n}": 0.0 for k, n, _ in terms}
    n_eval = {f"{k}:{n}": 0 for k, n, _ in terms}
    stateful = {"terminate_tip_lost"}            # keeps a per-env counter; evaluated once per step only
    fmax = {s: 0.0 for s in OLD}
    found_diff = {s: 0 for s in OLD}
    multi = {s: 0 for s in OLD}
    tot = {s: 0 for s in OLD}
    for step in range(steps):
        env.step(act)
        for s in list(OLD) + list(OLD.values()):
            env.scene.sensors[s]._invalidate_cache()
        for kind, name, tc in terms:
            if getattr(tc.func, "__name__", "") in stateful:
                continue
            p = dict(tc.params)
            a = tc.func(env, **p).float()
            p["sensor_name"] = OLD[p["sensor_name"]]
            b = tc.func(env, **p).float()
            worst[f"{kind}:{name}"] = max(worst[f"{kind}:{name}"], float((a - b).abs().max()))
            n_eval[f"{kind}:{name}"] += 1
        for s, so in OLD.items():
            dn, do = env.scene.sensors[s].data, env.scene.sensors[so].data
            fn, fo = dn.force.norm(dim=-1), do.force.norm(dim=-1)
            fmax[s] = max(fmax[s], float((fn - fo).abs().max()))
            found_diff[s] += int(((dn.found > 0) != (do.found > 0)).sum())
            multi[s] += int((dn.found > 1).sum())
            tot[s] += int((dn.found > 0).sum())
    env.close()
    return {"check": "legacy", "num_envs": num_envs, "steps": steps,
            "term_max_abs_diff": worst, "term_evals": n_eval, "skipped_stateful": sorted(stateful),
            "force_norm_max_abs_diff_N": fmax, "found_flag_mismatches": found_diff,
            "tip_steps_in_contact": tot, "tip_steps_with_multiple_matches": multi}


def tip_tool_sums(mjm, wd, w, tips, tool_body):
    """Per finger: world-frame sum of the tip-tool contact forces on the tip in world w of the GPU data, and the
    number of contacts (elliptic cone: component j is efc_force[efc_address[i, j]]; rows need not be contiguous)."""
    n = int(wd.nacon.numpy()[0])
    geom, wid = wd.contact.geom.numpy()[:n], wd.contact.worldid.numpy()[:n]
    fr, adr, ef = wd.contact.frame.numpy()[:n], wd.contact.efc_address.numpy()[:n], wd.efc.force.numpy()[w]
    fsum = {f: np.zeros(3) for f in tips}
    cnt = {f: 0 for f in tips}
    for i in np.flatnonzero(wid == w):
        b1, b2 = mjm.geom_bodyid[geom[i][0]], mjm.geom_bodyid[geom[i][1]]
        for f, tb in tips.items():
            if {b1, b2} == {tb, tool_body} and adr[i][0] >= 0:
                fw = fr[i].reshape(3, 3).T @ ef[adr[i][:3]]       # one efc row per force component
                fsum[f] += fw if b1 == tb else -fw
                cnt[f] += 1
    return fsum, cnt


def cpu_sums(mjm, wd, w, tips, tool_body):
    """The same sum after CPU MuJoCo re-solves world w's state."""
    import mujoco
    d = mujoco.MjData(mjm)
    for k in ("qpos", "qvel", "ctrl", "act", "qacc_warmstart", "mocap_pos", "mocap_quat"):
        a = getattr(wd, k).numpy()
        if a.size:
            getattr(d, k)[:] = a[w]
    mujoco.mj_forward(mjm, d)
    fsum = {f: np.zeros(3) for f in tips}
    cnt = {f: 0 for f in tips}
    c6 = np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        b1, b2 = mjm.geom_bodyid[con.geom1], mjm.geom_bodyid[con.geom2]
        for f, tb in tips.items():
            if {b1, b2} == {tb, tool_body}:
                mujoco.mj_contactForce(mjm, d, i, c6)
                fw = con.frame.reshape(3, 3).T @ c6[:3]
                fsum[f] += fw if b1 == tb else -fw
                cnt[f] += 1
    return fsum, cnt


def check_pads(num_envs, at_steps):
    import torch
    env = make_env("pads1", num_envs, with_old=False)
    mjm = env.sim.mj_model
    names = [mjm.body(i).name for i in range(mjm.nbody)]
    tips = {f: next(i for i, x in enumerate(names) if x.endswith(f"{f}_tip")) for f in ("thumb", "index", "middle")}
    tool = next(i for i, x in enumerate(names) if x.split("/")[-1] == "cube")
    act = torch.zeros((num_envs, env.action_manager.total_action_dim), device="cuda:0")
    env.reset()
    rows = []
    sens = env.scene.sensors["fingertip_cube_contact"]
    for step in range(max(at_steps) + 1):
        env.step(act)
        if step not in at_steps:
            continue
        sens._invalidate_cache()
        F = sens.data.force.detach().cpu().numpy()            # (B, 3, 3): world-frame net force per tip
        found = sens.data.found.detach().cpu().numpy()
        for w in range(num_envs):
            fs, n = tip_tool_sums(mjm, env.sim.wp_data, w, tips, tool)
            fs2, n2 = cpu_sums(mjm, env.sim.wp_data, w, tips, tool)
            for k, f in enumerate(tips):
                s_norm = float(np.linalg.norm(F[w, k]))
                rows.append({"step": step, "world": w, "finger": f, "sensor_N": s_norm, "found": int(found[w, k]),
                             "sum_same_solve_N": float(np.linalg.norm(fs[f])), "contacts": n[f],
                             "sum_cpu_resolve_N": float(np.linalg.norm(fs2[f])), "contacts_cpu": n2[f],
                             "vec_err_N": float(min(np.linalg.norm(F[w, k] - fs[f]), np.linalg.norm(F[w, k] + fs[f])))})
    env.close()
    loaded = [r for r in rows if r["sum_same_solve_N"] > 0.05]
    rel = [abs(r["sensor_N"] - r["sum_same_solve_N"]) / r["sum_same_solve_N"] for r in loaded]
    rel2 = [abs(r["sensor_N"] - r["sum_cpu_resolve_N"]) / r["sum_same_solve_N"] for r in loaded]
    return {"check": "pads1", "num_envs": num_envs, "at_steps": list(at_steps), "n_tip_samples": len(rows),
            "n_loaded": len(loaded), "rel_err_same_solve_max": max(rel) if rel else None,
            "rel_err_same_solve_median": float(np.median(rel)) if rel else None,
            "rel_err_cpu_resolve_median": float(np.median(rel2)) if rel2 else None,
            "rel_err_cpu_resolve_max": max(rel2) if rel2 else None,
            "found_max": max(r["found"] for r in rows), "contacts_max": max(r["contacts"] for r in rows),
            "found_equals_contacts": all(r["found"] == r["contacts"] for r in rows), "samples": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks", nargs="+", default=["legacy", "pads1"])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--pad-envs", type=int, default=8)
    ap.add_argument("--pad-steps", type=int, nargs="+", default=[60, 90, 120, 160])
    a = ap.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    from morphohand.rl import env_build
    for c in a.checks:
        row = check_legacy(a.steps, a.envs) if c == "legacy" else check_pads(a.pad_envs, a.pad_steps)
        row.update(reduce=env_build.CONTACT_SENSOR_REDUCE, maxmatch=env_build.CONTACT_SENSOR_MAXMATCH,
                   when=time.strftime("%Y-%m-%d %H:%M"))
        with open(OUT, "a") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
