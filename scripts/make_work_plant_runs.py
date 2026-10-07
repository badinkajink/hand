#!/usr/bin/env python3
"""Morphology runs for the RL contact comparison on the working plant (2026-10-06).

The calibrated RL runs (`20260920-sv1_u0308_b050_cal_tip` and its tip variants) carry the 2026-09-02 servo plant: finger
kp 0.5 N m/rad, kv 0.02, torque limit 0.35 N m, joint damping 0.5 (a 1 s time constant), frictionloss 0.0035. The bench
readbacks give the working plant instead (docs/experiments/20261006-servo_recalibration/, reorient_backends spec
"kp4_kv0_fr1_fl0_dp0.08"): kp 4 N m/rad, kv 0, torque limit 1 N m, damping 0.08 (tau 0.02 s), frictionloss 0, mu 1.
This copies the box-tip run, writes that plant into the scene's `ctrl` default class (finger actuators and joints;
the palm and everything else unchanged), and derives the two fingertip variants with make_pad_morphology_run.py:
the TPU block as one convex mesh (point contact) and as 1 mm sphere pads (pad d0 from this scene's inverse weights).
`--check` then runs the RL env with zero residual actions (scripted grasp and lift) at 64 envs and reports, at policy
steps 60 and 150, the tool height, the worlds whose tool is held by at least two fingertips, and the tip contacts.

    uv run --extra rl --extra gpu python scripts/make_work_plant_runs.py --check
Out: results/phase1/real_v1/20261006-sv1_u0308_b050_work_tip{,_tpu2.7mesh,_tpu2.7pads1}/
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results/phase1/real_v1/20260920-sv1_u0308_b050_cal_tip"
OUT = ROOT / "results/phase1/real_v1/20261006-sv1_u0308_b050_work_tip"
PLANT = dict(kp=4.0, kv=0.0, forcerange=1.0, damping=0.08, frictionloss=0.0)


def write_base():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in SRC.iterdir():
        if f.is_file() and not f.name.startswith(("frozen_scene", "plant_")):
            shutil.copy(f, OUT / f.name)
    tree = ET.parse(SRC / "frozen_scene.xml")
    n = 0
    for d in tree.getroot().iter("default"):
        if d.get("class") != "ctrl":
            continue
        for pos in d.findall("position"):
            pos.set("kp", f"{PLANT['kp']:g}")
            pos.set("kv", f"{PLANT['kv']:g}")
            pos.set("forcerange", f"-{PLANT['forcerange']:g} {PLANT['forcerange']:g}")
            n += 1
        for jt in d.findall("joint"):
            jt.set("damping", f"{PLANT['damping']:g}")
            jt.set("frictionloss", f"{PLANT['frictionloss']:g}")
            n += 1
    if n != 2:
        raise SystemExit(f"expected one ctrl position and one ctrl joint default, changed {n}")
    tree.write(OUT / "frozen_scene.xml")
    s = json.load(open(OUT / "summary.json"))
    s["frozen_scene_xml"] = str(OUT / "frozen_scene.xml")
    s["plant"] = dict(s.get("plant", {}), **PLANT, spec="kp4_kv0_fr1_fl0_dp0.08")
    s["derived_from"] = str(SRC)
    json.dump(s, open(OUT / "summary.json", "w"), indent=1)
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(OUT / "frozen_scene.xml"))
    a = m.actuator("a_thumb_mcp").id
    j = m.joint("thumb_mcp").id
    print(f"{OUT.name}: thumb_mcp kp {m.actuator_gainprm[a, 0]:g}, kv {-m.actuator_biasprm[a, 2]:g}, "
          f"forcerange {m.actuator_forcerange[a]}, damping {m.dof_damping[m.jnt_dofadr[j]]:g}, "
          f"frictionloss {m.dof_frictionloss[m.jnt_dofadr[j]]:g}", flush=True)


def variants():
    for suffix, model in (("_tpu2.7mesh", "pt"), ("_tpu2.7pads1", "pads")):
        out = OUT.with_name(OUT.name + suffix)
        subprocess.run([sys.executable, str(ROOT / "scripts/make_pad_morphology_run.py"), "--src", str(OUT),
                        "--out", str(out), "--model", model], check=True)


def check(runs, num_envs=64):
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    import rl_contact_throughput as T
    rows = []
    for run in runs:
        cfg = T.base_cfg(run)
        env = T.make_env(cfg, num_envs, 512, 1024)
        mjm, wd = env.sim.mj_model, env.sim.wp_data
        names = [mjm.body(i).name for i in range(mjm.nbody)]
        tool = next(i for i, x in enumerate(names) if x.split("/")[-1] == "cube")
        tips = [next(i for i, x in enumerate(names) if x.endswith(f"{f}_tip")) for f in ("thumb", "index", "middle")]
        act = torch.zeros((num_envs, env.action_manager.total_action_dim), device="cuda:0")
        env.reset()
        row = {"run": str(run.relative_to(ROOT))}
        for step in range(151):
            env.step(act)
            if step not in (60, 150):
                continue
            n = int(wd.nacon.numpy()[0])
            geom, wid = wd.contact.geom.numpy()[:n], wd.contact.worldid.numpy()[:n]
            tc = np.zeros((num_envs, 3), int)
            for (g1, g2), w in zip(geom, wid):
                b = {int(mjm.geom_bodyid[g1]), int(mjm.geom_bodyid[g2])}
                for k, tb in enumerate(tips):
                    if b == {tb, tool}:
                        tc[w, k] += 1
            z = wd.xpos.numpy()[:, tool, 2]
            row[f"step{step}"] = {"tool_z_mm_median": float(1e3 * np.median(z)),
                                  "held_worlds": int(((tc > 0).sum(1) >= 2).sum()),
                                  "tip_contacts_mean": tc.mean(0).round(1).tolist()}
        env.close()
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    if not a.check_only:
        write_base()
        variants()
    if a.check or a.check_only:
        runs = [OUT, OUT.with_name(OUT.name + "_tpu2.7mesh"), OUT.with_name(OUT.name + "_tpu2.7pads1")]
        rows = check(runs)
        out = ROOT / "docs/experiments/20261006-rl_contact/work_plant_runs.json"
        out.write_text(json.dumps({"plant": PLANT, "checks": rows}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
