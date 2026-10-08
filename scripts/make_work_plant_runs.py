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
import math
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
FINGERS = ("thumb", "index", "middle")

# 2026-10-08 arms of the contact-model policy comparison (docs/handoff/20261007-contact_model_policy_training.md).
# (d) condim 4 on the TPU block mesh. MuJoCo's elliptic cone bounds the spin torque by mu_spin N, so mu_spin = mu r_on,
# r_on the pads' twist-onset arm on the 2.7 mm TPU tip: 2.52 / 2.74 / 2.96 / 3.12 mm at 1 / 3 / 10 / 20 N
# (docs/experiments/20261008-contact_model_policies/twist_onset/tip_T2.jsonl); the 2026-10-06 policies grip with
# 5-27 N per finger, about 12 N typical, where r_on = 3.0 mm. Priority 1 on the tip so that its condim and friction,
# not the tool's (1 0.2 0.02), set the contact.
C4_DIR = ROOT / "results/phase1/real_v1/20261008-sv1_u0308_b050_work_tip_tpu2.7mesh_c4"
C4_SPIN = 0.0030
# (e) The compliant skin kept by the native-compliance study (scripts/contact_bed_candidates.py SKIN; page
# docs/experiments/20261007-native_compliance/): each fingertip's pad spheres on a child body `<f>_tipskin` joined to the
# tip by two slides tangent to the palmar face (y, z) and a hinge about its normal (x) through the face centre, with
# Mindlin's tangential and Lubkin's torsional initial stiffness and critical damping (2 g with 18 g of armature on the
# slides and 5e-7 kg m^2 on the hinge on the bed; see the armature note below). On the bed the springs were sized on the Hertz radius of the
# 10.55 mm sphere pad at 1 N (13.8 kN/m, 8.55 mN m/rad). On the hand they are sized with the same formulas on the
# pads' own patch at the policies' ~12 N grip: the scripted grasp presses 60 / 39 / 17 spheres of 1 mm^2 at 16.5 / 11.0
# / 5.6 N (thumb, index, middle; docs/experiments/20261006-rl_contact/work_plant_runs.json), 3.5 mm^2/N, so 42 mm^2
# and an equivalent radius a = 3.66 mm at 12 N: k_t = 8 G a / (2 - nu) = 65 kN/m, k_theta = 16 G a^3 / 3 =
# 0.90 N m/rad (G 3.45 MPa, nu 0.45, the stated TPU). The skin's mass is taken off the tip's inertial so each finger
# weighs what it does in the other arms. Joint ranges (+-1.5 mm, +-0.5 rad, 8x and 12x the deflection at full friction
# and at the twist onset) keep mjlab's soft joint limits, which treat an unlimited joint as range [0, 0], off the skin.
# The bed ran the skin at impratio 1000 so that the friction rows, not the skin, stay stiff. Under the trainer's solver
# (10 iterations, 2 ms, elliptic) impratio 1000 diverges for the pads and for the skin, and the skin also at 100 and 300
# (logs/20261008-contact_model_policies/skin_cpu_diag*.py: tool ejected within 0.04-1.3 s); at the env's impratio 10 the
# skin grips and lifts like the pads. Arm (e) therefore keeps the env's impratio 10 and differs from (c) by the skin.
SKIN_DIR = ROOT / "results/phase1/real_v1/20261008-sv1_u0308_b050_work_tip_tpu2.7skin"
SKIN_NREF, SKIN_AREA_PER_N = 12.0, 3.5e-6
# Armature: the bed's 18 g / 5e-7 kg m^2 leave the skin's spring periods at omega dt = 3.6 / 2.6 at the trainer's 2 ms
# step; CPU MuJoCo holds the scripted grasp there, MuJoCo-Warp (float32) throws the tool in the first 0.1 s in every
# world (logs/20261008-contact_model_policies/skin_mjw_variants.py; 10x armature or 10x softer springs both hold). The
# armature is set to omega dt = 1 on each spring (k dt^2: 0.26 kg on the slides, 3.6e-6 kg m^2 on the hinge), damping
# critical on it. Armature acts on the skin's own coordinates only, i.e. on its micrometre motion relative to the tip,
# so the static compliance is unchanged and the inertia the tool sees while it sticks is the skin's 2 g.
SKIN = dict(m=0.002, I=5e-8, arm=0.26, arm_t=3.6e-6, zeta=1.0, range_t=0.0015, range_th=0.5)


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


def _copy_run(src: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("frozen_scene"):
            shutil.copy(f, out / f.name)


def _write_summary(out: Path, src: Path, **extra):
    s = json.load(open(out / "summary.json"))
    s["frozen_scene_xml"] = str(out / "frozen_scene.xml")
    s["derived_from"] = str(src)
    s.update(extra)
    json.dump(s, open(out / "summary.json", "w"), indent=1)


def condim4_variant(spin: float = C4_SPIN):
    """Arm (d): the TPU block mesh run with condim 4, friction (mu, spin, 1e-4) and priority 1 on each tip geom."""
    src = OUT.with_name(OUT.name + "_tpu2.7mesh")
    _copy_run(src, C4_DIR)
    tree = ET.parse(src / "frozen_scene.xml")
    for f in FINGERS:
        g = tree.getroot().find(f".//body[@name='{f}_tip']/geom[@name='{f}_tipgeom']")
        mu = float(g.get("friction").split()[0])
        g.set("condim", "4")
        g.set("priority", "1")
        g.set("friction", f"{mu:g} {spin:g} 0.0001")
    tree.write(C4_DIR / "frozen_scene.xml")
    _write_summary(C4_DIR, src, fingertip=dict(json.load(open(src / "summary.json")).get("fingertip", {}),
                                               condim=4, mu_spin=spin, priority=1))
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(C4_DIR / "frozen_scene.xml"))
    g = m.geom("thumb_tipgeom")
    print(f"{C4_DIR.name}: thumb_tipgeom condim {g.condim[0]}, friction {g.friction}, priority {g.priority[0]}")


def skin_sizing(N_ref: float = SKIN_NREF):
    sys.path.insert(0, str(ROOT / "scripts"))
    import contact_reference_laws as L
    mat = L.mat()
    a = math.sqrt(SKIN_AREA_PER_N * N_ref / math.pi)
    kt = 8 * mat["G"] * a / (2 - mat["nu"])
    kth = 16 * mat["G"] * a ** 3 / 3
    ct = 2 * SKIN["zeta"] * math.sqrt(kt * (SKIN["m"] + SKIN["arm"]))
    cth = 2 * SKIN["zeta"] * math.sqrt(kth * (SKIN["I"] + SKIN["arm_t"]))
    return dict(N_ref=N_ref, a=a, kt=kt, kth=kth, ct=ct, cth=cth, G=mat["G"], nu=mat["nu"], E=mat["E"])


def _column_maps(root, src_xml: Path):
    """For the joints added under `root` (the skin's), the source column of every new qpos and qvel entry (-1 for an
    added joint, at rest 0). Source joints are matched by name; the unnamed tool free joint through its body."""
    import mujoco
    src = mujoco.MjModel.from_xml_path(str(src_xml))
    bare = ET.fromstring(ET.tostring(root, encoding="unicode"))
    for kf in bare.findall("keyframe"):
        bare.remove(kf)
    new = mujoco.MjModel.from_xml_string(ET.tostring(bare, encoding="unicode"))
    maps = {False: [], True: []}
    for j in range(new.njnt):
        name = new.joint(j).name
        t = new.jnt_type[j]
        try:
            js = (src.joint(name).id if name else
                  int(src.body_jntadr[src.body(new.body(new.jnt_bodyid[j]).name).id]))
        except KeyError:
            js = None
        for vel in (False, True):
            w = (6 if vel else 7) if t == 0 else (3 if vel else 4) if t == 1 else 1
            a = None if js is None else int(src.jnt_dofadr[js] if vel else src.jnt_qposadr[js])
            maps[vel] += [-1] * w if a is None else list(range(a, a + w))
    return np.array(maps[False]), np.array(maps[True])


def _remap(x: np.ndarray, cols: np.ndarray) -> np.ndarray:
    out = np.zeros(x.shape[:-1] + (len(cols),), dtype=x.dtype)
    keep = cols >= 0
    out[..., keep] = x[..., cols[keep]]
    return out


def _extend_keyframes(root, src_xml: Path, qcols, vcols):
    """Rewrite every keyframe's qpos/qvel through the column maps."""
    for key in root.iter("key"):
        for attr, cols in (("qpos", qcols), ("qvel", vcols)):
            if key.get(attr) is not None:
                old = np.array([float(v) for v in key.get(attr).split()])
                key.set(attr, " ".join(f"{v:.9g}" for v in _remap(old, cols)))


def skin_variant(N_ref: float = SKIN_NREF):
    """Arm (e): the 1 mm pad run with each tip's pad spheres moved onto a spring-mounted skin body."""
    import mujoco
    sys.path.insert(0, str(ROOT / "scripts"))
    import fingertip_geometry as G
    src = OUT.with_name(OUT.name + "_tpu2.7pads1")
    _copy_run(src, SKIN_DIR)
    sz = skin_sizing(N_ref)
    tree = ET.parse(src / "frozen_scene.xml")
    root = tree.getroot()
    for f in FINGERS:
        sgn = G.FACE_SIGN[f]
        tipb = root.find(f".//body[@name='{f}_tip']")
        inert = tipb.find("inertial")
        m_tip = float(inert.get("mass"))
        scale = (m_tip - SKIN["m"]) / m_tip
        inert.set("mass", f"{m_tip - SKIN['m']:.9g}")
        inert.set("diaginertia", " ".join(f"{float(v) * scale:.9g}" for v in inert.get("diaginertia").split()))
        c = np.array([sgn * G.REACH, 0.0, G.CENTRE[2]])
        skin = ET.SubElement(tipb, "body", name=f"{f}_tipskin", pos=f"{c[0]:.7f} {c[1]:.7f} {c[2]:.7f}")
        ET.SubElement(skin, "inertial", pos="0 0 0", mass=f"{SKIN['m']:g}",
                      diaginertia=f"{SKIN['I']:g} {SKIN['I']:g} {SKIN['I']:g}")
        for ax, axis in (("y", "0 1 0"), ("z", "0 0 1")):
            ET.SubElement(skin, "joint", name=f"{f}_skin_{ax}", type="slide", axis=axis, stiffness=f"{sz['kt']:.6g}",
                          damping=f"{sz['ct']:.6g}", armature=f"{SKIN['arm']:g}",
                          range=f"{-SKIN['range_t']:g} {SKIN['range_t']:g}")
        ET.SubElement(skin, "joint", name=f"{f}_skin_t", type="hinge", axis="1 0 0", stiffness=f"{sz['kth']:.6g}",
                      damping=f"{sz['cth']:.6g}", armature=f"{SKIN['arm_t']:g}",
                      range=f"{-SKIN['range_th']:g} {SKIN['range_th']:g}")
        for g in [g for g in tipb.findall("geom") if g.get("name", "").startswith(f"{f}_pad")]:
            tipb.remove(g)
            p = np.array([float(v) for v in g.get("pos").split()]) - c
            g.set("pos", f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f}")
            skin.append(g)
    # pad stiffness through d0 with the skin body's inverse weight in place of the tip's (reorient_backends.replace_tips)
    m0 = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d0s = {}
    for f in FINGERS:
        tipb = root.find(f".//body[@name='{f}_tip']")
        skin = tipb.find(f"body[@name='{f}_tipskin']")
        pads = skin.findall("geom")
        tc = float(pads[0].get("solref").split()[0])
        d0_old = float(pads[0].get("solimp").split()[0])
        diag_old = m0.body_invweight0[m0.body(f"{f}_tip").id, 0] + m0.body_invweight0[m0.body("screwdriver_medium").id, 0]
        K = 1.0 / (tc ** 2 * (1.0 - d0_old) * diag_old)     # the pad stiffness the source scene reaches
        diag = m0.body_invweight0[m0.body(f"{f}_tipskin").id, 0] + m0.body_invweight0[m0.body("screwdriver_medium").id, 0]
        d0 = 1.0 - 1.0 / (tc ** 2 * K * diag)
        if d0 < 0.05:
            raise ValueError(f"{f}: skin pad d0 {d0:.3f} below 0.05")
        d0s[f] = d0
        for g in pads:
            sol = g.get("solimp").split()
            g.set("solimp", " ".join([f"{d0:.6g}", f"{d0:.6g}"] + sol[2:]))
    qcols, vcols = _column_maps(root, src / "frozen_scene.xml")
    _extend_keyframes(root, src / "frozen_scene.xml", qcols, vcols)
    tree.write(SKIN_DIR / "frozen_scene.xml")
    ref = dict(np.load(src / "best_rollout.npz"))       # the CEM reference rollout the observations read
    ref["qpos"], ref["qvel"] = _remap(ref["qpos"], qcols), _remap(ref["qvel"], vcols)
    np.savez(SKIN_DIR / "best_rollout.npz", **ref)
    _write_summary(SKIN_DIR, src, fingertip=dict(json.load(open(src / "summary.json")).get("fingertip", {}),
                                                 skin=dict(SKIN, **sz, pad_d0_skin=d0s)))
    m = mujoco.MjModel.from_xml_path(str(SKIN_DIR / "frozen_scene.xml"))
    print(f"{SKIN_DIR.name}: {m.njnt} joints, impratio {m.opt.impratio:g} (env: 10), k_t {sz['kt']:.0f} N/m, "
          f"k_theta {sz['kth']:.3f} N m/rad, a {sz['a'] * 1e3:.2f} mm, pad d0 {d0s}")


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
        # a tip and its skin child body (arm e) both count as that fingertip
        tips = [{i for i, x in enumerate(names) if x.endswith((f"{f}_tip", f"{f}_tipskin"))} for f in FINGERS]
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
                    if tool in b and len(b & tb) == 1:
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
    ap.add_argument("--arms-20261008", action="store_true", help="write only the 2026-10-08 arms (d) and (e)")
    a = ap.parse_args()
    if a.arms_20261008:
        condim4_variant()
        skin_variant()
        if a.check:
            rows = check([C4_DIR, SKIN_DIR])
            out = ROOT / "docs/experiments/20261008-contact_model_policies/arm_scene_checks.json"
            out.write_text(json.dumps({"checks": rows}, indent=1))
        return 0
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
