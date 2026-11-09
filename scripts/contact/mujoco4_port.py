#!/usr/bin/env python3
"""MuJoCo 3.6.0 against 4.0.0 for the hand's contact simulation (2026-11-08). Every subcommand runs under either
environment and writes rows tagged with the MuJoCo version to docs/experiments/20261108-mujoco4_port/data/.

    3.6.0: .venv/bin/python (the repository's environment, MuJoCo and MuJoCo Warp 3.6.0)
    4.0.0: logs/20261108-mujoco4_port/venv/bin/python (uv venv, Python 3.12: mujoco 4.0.0, mujoco-warp 4.0.0,
           warp-lang 1.18.0, numpy, scipy, trimesh, matplotlib, imageio, imageio-ffmpeg, pillow, pyyaml, tyro, shapely)

roll   A cylinder lying at rest on a level box, the screwdriver and mat of the cad_v3 scene
       (assets/mjcf/experimental/20261104-cad_v3/ur5_full_k1.xml: r 12.5 mm, 100 mm, density 500, friction 2.4 on 1.8,
       solref 0.006 1, solimp 0.97 0.995 0.0005, 2 ms, implicitfast, pyramidal cone), stepped 1 s from rest under
       variants of the collider and solver. Per variant: roll about the cylinder's axis, travel, contacts, and at 50 ms
       each contact's offset from the vertical plane through the axis, the tilt of its normal and the torque of the
       contact forces about the axis.
"""
from __future__ import annotations

import argparse
import json
import time
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import mujoco  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/20261108-mujoco4_port"
DATA = OUT / "data"
VER = mujoco.__version__


def emit(path: Path, row: dict):
    """Append one row and fsync it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"mujoco": VER, **row}
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(row), flush=True)


# ------------------------------------------------------------------------------------------------ roll

R_CYL, HL_CYL = 0.0125, 0.05


def roll_xml(support="box", dt=0.002, cone="pyramidal", flags="", spin_deg=0.0, shape="cylinder", offset=(0.0, 0.0),
             condim=3, ccd_tol=None):
    """The cad_v3 screwdriver on its mat; the cylinder's axis along world -y as in the scene, turned about that axis
    by spin_deg before the run. The mat's top face is z = 0 and the cylinder starts touching it."""
    a = np.radians(spin_deg) / 2
    # scene quat (0.70711, 0.70711, 0, 0) puts body z along world -y; spin about body z first
    q_scene = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0])
    q_spin = np.array([np.cos(a), 0, 0, np.sin(a)])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, q_scene, q_spin)
    sup = (f'<geom name="mat" type="box" pos="{-offset[0]} {-offset[1]} -0.0164" size="0.35 0.25 0.0164"/>' if support == "box"
           else '<geom name="mat" type="plane" size="1 1 0.1"/>')
    size = f"{R_CYL} {HL_CYL}"
    return f"""<mujoco model="roll">
  <option timestep="{dt}" integrator="implicitfast" cone="{cone}"{f' ccd_tolerance="{ccd_tol}"' if ccd_tol else ''}>
    <flag {flags}/>
  </option>
  <default><geom friction="1.8 0.15 0.01" solref="0.006 1" solimp="0.97 0.995 0.0005"/></default>
  <worldbody>
    {sup}
    <body name="tool" pos="0 0 {R_CYL}" quat="{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}">
      <freejoint/>
      <geom name="tool" type="{shape}" size="{size}" density="500" friction="2.4 0.2 0.02" condim="{condim}"/>
    </body>
  </worldbody>
</mujoco>"""


ROLL_VARIANTS = {
    "scene": {},
    "plane": {"support": "plane"},
    "elliptic": {"cone": "elliptic"},
    "dt0.5ms": {"dt": 0.0005},
    "multiccd_on": {"flags": 'multiccd="enable"'},
    "multiccd_off": {"flags": 'multiccd="disable"'},
    "nativeccd_off": {"flags": 'nativeccd="disable"'},
    "spin15": {"spin_deg": 15.0},
    "spin45": {"spin_deg": 45.0},
    "capsule": {"shape": "capsule"},
    # the cad_v3 scene's mat centre sits 30 mm along x and 70 mm along y from the screwdriver
    "offset": {"offset": (0.03, 0.07)},
    "offset_elliptic": {"offset": (0.03, 0.07), "cone": "elliptic"},
    "offset_ccdtol1e-10": {"offset": (0.03, 0.07), "ccd_tol": 1e-10},
    "offset_condim6": {"offset": (0.03, 0.07), "condim": 6},
    "offset_multiccd_off": {"offset": (0.03, 0.07), "flags": 'multiccd="disable"'},
    "offset_multiccd_on": {"offset": (0.03, 0.07), "flags": 'multiccd="enable"'},
}


def contact_report(m, d, tool_body):
    """Per contact on the tool: offset of the point from the vertical plane through the axis (mm, signed along the
    horizontal normal of that plane), tilt of the contact normal from vertical (deg), and the torque of the contact
    force about the axis through the centre of mass (N mm). Force on the tool is taken with its normal part upward."""
    c = d.xipos[tool_body]
    ax = d.xmat[tool_body].reshape(3, 3)[:, 2]
    side = np.cross(ax, [0, 0, 1.0])
    side /= np.linalg.norm(side)
    rows, tau = [], 0.0
    f6 = np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        mujoco.mj_contactForce(m, d, i, f6)
        fr = con.frame.reshape(3, 3)
        F = fr.T @ f6[:3]
        n = fr[0]
        if F[2] < 0:     # orient to the force the mat exerts on the tool
            F = -F
        p = con.pos
        t = float(np.cross(p - c, F) @ ax)
        tau += t
        rows.append({"offset_mm": round(1e3 * float((p - c) @ side), 4), "tilt_deg": round(float(np.degrees(np.arccos(min(1.0, abs(n[2]))))), 4),
                     "depth_mm": round(-1e3 * con.dist, 4), "Fz_N": round(float(F[2]), 5), "Fside_N": round(float(F @ side), 5),
                     "tau_Nmm": round(1e3 * t, 5)})
    return rows, 1e3 * tau


def roll_case(name, kw, T=1.0):
    m = mujoco.MjModel.from_xml_string(roll_xml(**kw))
    d = mujoco.MjData(m)
    b = m.body("tool").id
    mujoco.mj_forward(m, d)
    q0 = d.xquat[b].copy()
    p0 = d.xpos[b].copy()
    early = None
    n = int(round(T / m.opt.timestep))
    k50 = int(round(0.05 / m.opt.timestep))
    for k in range(1, n + 1):
        mujoco.mj_step(m, d)
        if k == k50:
            mujoco.mj_forward(m, d)
            early = contact_report(m, d, b)
    dq = np.zeros(3)
    mujoco.mju_subQuat(dq, d.xquat[b], q0)   # rotation from q0 to q, in q0's body frame
    roll = float(np.degrees(dq[2]))            # about the body z axis (the cylinder's axis)
    return {"case": name, **{k: v for k, v in kw.items()}, "roll_deg_1s": round(roll, 3),
            "travel_mm_1s": round(1e3 * float(np.linalg.norm((d.xpos[b] - p0)[:2])), 3), "ncon_1s": int(d.ncon),
            "contacts_50ms": early[0], "tau_axis_50ms_Nmm": round(early[1], 5)}


def cmd_roll(a):
    out = DATA / "roll.jsonl"
    for name, kw in ROLL_VARIANTS.items():
        if name.endswith("multiccd_on") and VER.startswith("4."):
            continue                                 # the default on 4.0
        if name.endswith("multiccd_off") and VER.startswith("3."):
            continue                                 # the default on 3.6
        if a.only and name not in a.only:
            continue
        emit(out, roll_case(name, kw))


# ------------------------------------------------------------------------------------------------ pinch rig

# The two-pad pinch rig of docs/experiments/20261005-contact_bed/PROTOCOL.md (scripts/hom/hom_contact_rig.py): two
# fingertip spheres (r 10.55 mm, 20 g) on force-controlled rails squeeze the screwdriver (cylinder r 12.5 mm, 100 mm,
# 24.5 g), mu 1, gravity off. Pads of 1 mm spheres at the current modulus 2.5 MPa and relaxation time 0.04 s, the rig's
# impratio 100.
PADS = "mj:spheres:s1:rs0.75:ir100:tr0.04:E2.5e6"
VARIANTS = {
    "implicitfast": {},                               # the repository's integrator
    "noslip10": {"noslip_iterations": 10},            # noslip post-pass, 10 iterations
    "discrete": {"integrator": "discrete"},           # 4.0's discrete integrator (constraint springs at the step's end)
    "diagexact": {"flag": 'diagexact="enable"'},      # exact constraint-space inertia diagonal in the regulariser
}
ON_36 = ("implicitfast", "noslip10")
_VAR: dict = {}


def install_variants():
    """Route every rig MJCF through the variant held in _VAR (integrator, noslip iterations or an enable flag)."""
    import re
    import hom_contact_rig as H
    if getattr(H, "_mj4_patched", False):
        return H
    orig = H.mj_xml

    def mj_xml(*args, **kw):
        xml, info = orig(*args, **kw)
        v = _VAR
        if "integrator" in v:
            xml = xml.replace('integrator="implicitfast"', f'integrator="{v["integrator"]}"')
        if "noslip_iterations" in v:
            xml = xml.replace("<option ", f'<option noslip_iterations="{v["noslip_iterations"]}" ', 1)
        if "flag" in v:
            xml = re.sub(r"(<option[^>]*?)/>", lambda mt: mt.group(1) + f'><flag {v["flag"]}/></option>', xml, count=1)
        return xml, info

    H.mj_xml = mj_xml
    H._mj4_patched = True
    return H


def variants_here(names):
    return [v for v in names if VER.startswith("4.") or v in ON_36]


def slim(row):
    """Drop the per-step traces from a task row."""
    return {k: v for k, v in row.items() if not (isinstance(v, list) and len(v) > 40)}


def cmd_rig(a):
    install_variants()
    import contact_bed_common as B
    import contact_bed_pull as P
    import contact_bed_twist as T
    P.MODELS["pads"] = PADS
    B.MODELS["pads"] = (PADS, PADS)
    for v in variants_here(a.variants):
        _VAR.clear()
        _VAR.update(VARIANTS[v])
        for dt in a.dt:
            for N in a.N:
                for task, fn in (("pull", lambda: P.run_case("pads", N, dt)), ("twist", lambda: T.run_case("pads", N, dt))):
                    if task not in a.tasks:
                        continue
                    try:
                        r = slim(fn())
                    except Exception as e:                  # a variant that fails to compile or diverges is a result
                        r = {"N": N, "dt_ms": dt, "status": "error", "error": repr(e)[:300]}
                    emit(DATA / f"rig_{task}_{VER}.jsonl", {"variant": v, "spec": PADS, **r})


def cmd_gpu(a):
    """MuJoCo Warp on the pinch rig: world-steps per second and creep. Settle 0.4 s at N on the CPU, copy the state to
    every world, apply an axial force mu N on the tool (half the two pads' slip force 2 mu N) and step 1 s in 10-step
    CUDA graphs. Creep is the slope of the tool's axial displacement over the last 0.8 s; cost is the median graph time."""
    H = install_variants()
    import contact_bed_common as B
    import mujoco_warp as mjw
    import warp as wp
    for v in variants_here(a.variants):
        _VAR.clear()
        _VAR.update(VARIANTS[v])
        for dt_ms in a.dt:
            dt = dt_ms * 1e-3
            rig = B.new_rig(PADS, dt)
            rig.set_pad_force(a.N)
            rig.set_tool_wrench(np.zeros(3), np.zeros(3))
            rig.step(0.4)
            m, d = rig.m, rig.d
            ax = rig.tool_state()["axis"].copy()
            # the same hold on the CPU, for the creep MuJoCo itself gives under this protocol
            p0 = d.xpos[rig.tool].copy()
            rig.set_tool_wrench(H.MU * a.N * ax, np.zeros(3))
            u, t = [], []
            for k in range(int(round(1.0 / dt))):
                rig.step(dt)
                u.append(float((d.xpos[rig.tool] - p0) @ ax))
                t.append((k + 1) * dt)
            u, t = np.array(u), np.array(t)
            w = t >= 0.2
            cpu_creep = float(np.polyfit(t[w], u[w], 1)[0]) * 1e6 if np.all(np.isfinite(u)) else float("nan")
            # back to the settled state for the GPU
            rig = B.new_rig(PADS, dt)
            rig.set_pad_force(a.N)
            rig.set_tool_wrench(np.zeros(3), np.zeros(3))
            rig.step(0.4)
            m, d = rig.m, rig.d
            try:
                wm = mjw.put_model(m)
            except Exception as e:
                emit(DATA / f"rig_gpu_{VER}.jsonl", {"variant": v, "dt_ms": dt_ms, "N": a.N, "status": "unsupported",
                                              "error": repr(e)[:300], "cpu_creep_um_s": cpu_creep})
                continue
            for nw in a.nworld:
                wp.synchronize()
                dev = wp.get_device("cuda:0")
                free0 = dev.free_memory
                wd = mjw.make_data(m, nworld=nw, nconmax=512, njmax=2048)
                wd.qpos.assign(np.tile(d.qpos, (nw, 1)).astype(np.float32))
                wd.qvel.assign(np.tile(d.qvel, (nw, 1)).astype(np.float32))
                wd.ctrl.assign(np.tile(d.ctrl, (nw, 1)).astype(np.float32))
                xf = np.zeros((nw, m.nbody, 6), np.float32)
                xf[:, rig.tool, :3] = H.MU * a.N * ax
                wd.xfrc_applied.assign(xf)
                mjw.forward(wm, wd)
                wp.synchronize()
                block = 10
                with wp.ScopedCapture() as cap:
                    for _ in range(block):
                        mjw.step(wm, wd)
                x0 = wd.xpos.numpy()[:, rig.tool].astype(float)
                ticks, U, T = [], [], []
                nblk = int(round(1.0 / (dt * block)))
                for b in range(nblk):
                    wp.synchronize()
                    t0 = time.perf_counter()
                    wp.capture_launch(cap.graph)
                    wp.synchronize()
                    ticks.append(time.perf_counter() - t0)
                    xb = wd.xpos.numpy()[:, rig.tool].astype(float)
                    U.append(((xb - x0) @ ax))
                    T.append((b + 1) * block * dt)
                U, T = np.array(U), np.array(T)
                fin = np.all(np.isfinite(wd.qpos.numpy()), axis=1)
                w = T >= 0.2
                creep = [float(np.polyfit(T[w], U[w, i], 1)[0]) * 1e6 for i in np.flatnonzero(fin)[:64]]
                per = float(np.median(ticks)) / block
                emit(DATA / f"rig_gpu_{VER}.jsonl", {
                    "variant": v, "dt_ms": dt_ms, "N": a.N, "nworld": nw, "status": "ok",
                    "world_steps_per_s": nw / per, "us_per_world_step": per / nw * 1e6,
                    "sim_s_per_wall_s": nw * dt / per, "finite_fraction": float(fin.mean()),
                    "gpu_creep_um_s": float(np.median(creep)) if creep else float("nan"), "cpu_creep_um_s": cpu_creep,
                    "nacon_per_world": float(wd.nacon.numpy()[0]) / nw, "vram_mib": (free0 - dev.free_memory) / 2 ** 20})
                del wd
            del wm


# ------------------------------------------------------------------------------------------------ hand grasp

def cmd_grasp(a):
    """The scripted grasp of scripts/ur5/ur5_shear_impratio.py `mjw` (the servo plant's pad scene of the RL runs: keyframe
    open_ik, fingers closed over 0.16 s, the palm raised 0.1 m from 0.48 s, 2 s in all; slip of the tool against the
    fingertips from 1 to 2 s) with its rows written here, tagged with the MuJoCo version and the integrator."""
    import ur5_shear_impratio as U
    integ = a.integrator
    set_rl0, append0 = U.set_rl, U.append

    def set_rl(m, ir, noslip=0):
        set_rl0(m, ir, noslip)
        if integ == "discrete":
            m.opt.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE

    U.set_rl = set_rl
    U.append = lambda out, row: append0(DATA / f"grasp_{VER}.jsonl", {"mujoco": VER, "integrator": integ, **row})
    ns = argparse.Namespace(E=a.E, ir=a.ir, sims=a.sims, n=64, n_cpu=a.n_cpu, n_gpu=a.n_gpu, nconmax=640, njmax=3072,
                            njmax_soft=8192, run="pads", min_d0=0.05, tc=None, noslip=a.noslip, skin_from_slides=None)
    U.cmd_mjw(ns)


def cmd_hand_gpu(a):
    """MuJoCo Warp throughput on the hand: the servo plant's pad scene of the RL runs with its pads at E MPa, at the
    trainer's numerics (ur5_shear_impratio.rl_scene: 2 ms, implicitfast, elliptic cone, Newton, 10 iterations), from
    keyframe open_ik (the screwdriver on the table under three fingertips), the keyframe's controls held. Steps in
    10-step CUDA graphs; cost is the median graph time over 30 graphs (0.6 s)."""
    import ur5_shear_impratio as U
    import mujoco_warp as mjw
    import warp as wp
    for v in variants_here(a.variants):
        if v == "diagexact":
            continue
        m, info = U.rl_scene(a.E, a.ir)
        if v == "noslip10":
            m.opt.noslip_iterations = 10
        if v == "discrete":
            m.opt.integrator = mujoco.mjtIntegrator.mjINT_DISCRETE
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        try:
            wm = mjw.put_model(m)
        except Exception as e:
            emit(DATA / f"hand_gpu_{VER}.jsonl", {"variant": v, "E_MPa": a.E, "ir": a.ir, "status": "unsupported",
                                                  "error": repr(e)[:300]})
            continue
        for nw in a.nworld:
            wd = mjw.make_data(m, nworld=nw, nconmax=320, njmax=2048)
            wd.qpos.assign(np.tile(d.qpos, (nw, 1)).astype(np.float32))
            wd.qvel.zero_()
            wd.ctrl.assign(np.tile(d.ctrl, (nw, 1)).astype(np.float32))
            mjw.forward(wm, wd)
            wp.synchronize()
            block = 10
            with wp.ScopedCapture() as cap:
                for _ in range(block):
                    mjw.step(wm, wd)
            ticks = []
            for _ in range(30):
                wp.synchronize()
                t0 = time.perf_counter()
                wp.capture_launch(cap.graph)
                wp.synchronize()
                ticks.append(time.perf_counter() - t0)
            per = float(np.median(ticks)) / block
            emit(DATA / f"hand_gpu_{VER}.jsonl", {
                "variant": v, "E_MPa": a.E, "ir": a.ir, "nworld": nw, "status": "ok", "dt_ms": m.opt.timestep * 1e3,
                "world_steps_per_s": nw / per, "us_per_world_step": per / nw * 1e6,
                "finite_fraction": float(np.all(np.isfinite(wd.qpos.numpy()), axis=1).mean()),
                "nacon_per_world": float(wd.nacon.numpy()[0]) / nw, "nefc_max": int(wd.nefc.numpy().max())})
            del wd
        del wm


# ------------------------------------------------------------------------------------------------ film

CAD_V3 = ROOT / "assets/mjcf/experimental/20261104-cad_v3/ur5_full_k1.xml"
FILM_DIR = ROOT / "logs/20261108-mujoco4_port/film"
MEDIA = OUT / "media"


def cmd_film(a):
    """Render the cad_v3 scene from keyframe 0 for 3 s at 25 frames per second, looking along the screwdriver's axis;
    a massless dark stripe along the screwdriver shows its roll. --tol sets ccd_tolerance."""
    import xml.etree.ElementTree as ET
    root = ET.parse(CAD_V3).getroot()
    for b in root.iter("body"):
        if b.get("name") == "screwdriver_medium":
            ET.SubElement(b, "geom", type="box", size="0.0126 0.0016 0.0502", pos="0 0 0", density="0", contype="0",
                          conaffinity="0", rgba="0.1 0.1 0.12 1")
    for c in root.iter("compiler"):
        for key in ("meshdir", "texturedir"):
            c.set(key, str((CAD_V3.parent / c.get(key, ".")).resolve()))
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if a.tol:
        m.opt.ccd_tolerance = a.tol
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    b = m.body("screwdriver_medium").id
    r = mujoco.Renderer(m, 300, 360)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = d.xpos[b] + np.array([0.025, 0, 0])
    cam.distance, cam.azimuth, cam.elevation = 0.20, 90.0, -12.0
    frames, every = [], int(round(0.04 / m.opt.timestep))
    for k in range(int(round(3.0 / m.opt.timestep)) + 1):
        if k % every == 0:
            r.update_scene(d, cam)
            frames.append(r.render().copy())
        mujoco.mj_step(m, d)
    r.close()
    FILM_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FILM_DIR / f"{a.label}.npz", frames=np.array(frames))
    print(a.label, len(frames), "frames", flush=True)


def cmd_film_tile(a):
    """Side by side with a label bar over each panel; H.264 and a JPEG poster at the last frame."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    panels = [(lab, np.load(FILM_DIR / f"{key}.npz")["frames"]) for key, lab in zip(a.keys, a.labels)]
    n = min(len(f) for _, f in panels)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    out = []
    for k in range(n):
        tiles = []
        for lab, fr in panels:
            im = Image.new("RGB", (fr.shape[2], fr.shape[1] + 30), "white")
            im.paste(Image.fromarray(fr[k]), (0, 30))
            dr = ImageDraw.Draw(im)
            dr.text((8, 5), lab, fill="black", font=font)
            dr.text((fr.shape[2] - 62, 5), f"{k * 0.04:.2f} s", fill="black", font=font)
            tiles.append(np.asarray(im))
        out.append(np.concatenate(tiles, axis=1))
    MEDIA.mkdir(parents=True, exist_ok=True)
    w = imageio.get_writer(str(MEDIA / f"{a.out}.mp4"), fps=25, codec="libx264", quality=None, pixelformat="yuv420p",
                           macro_block_size=8, ffmpeg_params=["-crf", "27", "-preset", "medium"], ffmpeg_log_level="error")
    for f in out:
        w.append_data(f)
    w.close()
    Image.fromarray(out[-1]).save(MEDIA / f"{a.out}.jpg", quality=85)
    print("wrote", MEDIA / f"{a.out}.mp4", n, "frames", flush=True)


# ------------------------------------------------------------------------------------------------ components

SMOKE_SCENES = [
    "assets/mjcf/real_v1/hand_frozen_morphology.xml",
    "results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip_tpu2.7pads1/frozen_scene.xml",
    "logs/20261030-hand_object_scale_cad_v2/scenes/cyl008.0/x092.5_y102.5_pads_bv2.xml",
    "logs/20261030-hand_object_scale_cad_v2/scenes/cyl008.0/x092.5_y102.5_pt.xml",
    "assets/mjcf/experimental/20261030-hand_object_scale_cad_v2/base/scene_cyl025.0.xml",
    "assets/mjcf/experimental/20261104-cad_v3/ur5_full_k1.xml",
]
CLEARANCE_PLAN = ("rv05_manual_a75", "docs/experiments/20260916-tip_gait/deploy")


def _step(m, d, T=0.2):
    for _ in range(int(round(T / m.opt.timestep))):
        mujoco.mj_step(m, d)
    return bool(np.all(np.isfinite(d.qpos)))


def chk_scenes():
    ok = 0
    for p in SMOKE_SCENES:
        m = mujoco.MjModel.from_xml_path(str(ROOT / p))
        d = mujoco.MjData(m)
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, d, 0)
        ok += _step(m, d)
    return ok == len(SMOKE_SCENES), f"{ok} of {len(SMOKE_SCENES)} scenes compile and step 0.2 s"


def chk_pads():
    import reorient_backends as RB
    out = ROOT / "logs/20261108-mujoco4_port" / f"scenes_{VER}"
    p, meta = RB.build_scene("D6", "tcd_ell7.4x8.5x11x22", "pads", "fast", "bed", 100.0, "scene", out_dir=out, builder="v2")
    m = mujoco.MjModel.from_xml_path(str(p))
    d = mujoco.MjData(m)
    npad = int(np.sum((m.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) & np.isclose(m.geom_size[:, 0], RB.PAD_RS)))
    return _step(m, d), f"D6 bench scene with the v4b tip as pads ({npad} spheres) builds, compiles and steps"


def chk_render():
    m = mujoco.MjModel.from_xml_path(str(ROOT / SMOKE_SCENES[0]))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, 240, 320)
    r.update_scene(d)
    img = r.render()
    r.close()
    return float(img.mean()) > 1.0, f"EGL frame 320 x 240, mean level {img.mean():.0f}"


def chk_clearance():
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "scripts/hardware/real_v1_trajectory_clearance.py"), "--plan",
                        CLEARANCE_PLAN[0], "--deploy-dir", str(ROOT / CLEARANCE_PLAN[1])], capture_output=True, text=True,
                       timeout=900, cwd=ROOT)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    return "Traceback" not in r.stderr, f"exit {r.returncode}; {last[:150]}"


def chk_mjw():
    import mujoco_warp as mjw
    m = mujoco.MjModel.from_xml_path(str(ROOT / SMOKE_SCENES[1]))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    wm = mjw.put_model(m)
    wd = mjw.make_data(m, nworld=8, nconmax=640, njmax=3072)
    wd.qpos.assign(np.tile(d.qpos, (8, 1)).astype(np.float32))
    wd.ctrl.assign(np.tile(d.ctrl, (8, 1)).astype(np.float32))
    for _ in range(100):
        mjw.step(wm, wd)
    ok = bool(np.all(np.isfinite(wd.qpos.numpy())))
    return ok, "pad scene of the RL runs: put_model, 8 worlds, 100 steps"


def chk_comfree():
    sys.path.insert(0, str(ROOT / "external/comfree_warp"))
    import comfree_warp as cf
    m = mujoco.MjModel.from_xml_path(str(ROOT / SMOKE_SCENES[0]))
    wm = cf.put_model(m)
    wd = cf.make_data(m, nworld=4)
    for _ in range(50):
        cf.step(wm, wd)
    return bool(np.all(np.isfinite(wd.qpos.numpy()))), "real_v1 hand: put_model, 4 worlds, 50 steps"


def chk_mink():
    import mink
    m = mujoco.MjModel.from_xml_path(str(ROOT / SMOKE_SCENES[0]))
    mink.Configuration(m)
    from importlib.metadata import version
    return True, f"mink {version('mink')}: Configuration of the real_v1 hand"


def chk_mjx():
    from mujoco import mjx
    m = mujoco.MjModel.from_xml_path(str(ROOT / SMOKE_SCENES[0]))
    mx = mjx.put_model(m)
    dx = mjx.step(mx, mjx.make_data(mx))
    return bool(np.all(np.isfinite(np.asarray(dx.qpos)))), "real_v1 hand: put_model, one step"


def chk_mjlab():
    import mjlab.sim  # noqa: F401
    from importlib.metadata import version
    return True, f"mjlab {version('mjlab')} imports"


def chk_newton():
    import newton  # noqa: F401
    from importlib.metadata import version
    return True, f"newton {version('newton')} imports"


CHECKS = {"scenes": chk_scenes, "pads": chk_pads, "render": chk_render, "clearance": chk_clearance, "mjw": chk_mjw,
          "comfree": chk_comfree, "mink": chk_mink, "mjx": chk_mjx, "mjlab": chk_mjlab, "newton": chk_newton}


def cmd_compat(a):
    for name in a.only or list(CHECKS):
        t0 = time.time()
        try:
            ok, detail = CHECKS[name]()
            status = "runs" if ok else "fails"
        except ModuleNotFoundError as e:
            status, detail = "not installed", str(e)[:200]
        except Exception as e:
            status, detail = "fails", f"{type(e).__name__}: {str(e).strip().splitlines()[0][:200] if str(e).strip() else ''}"
        emit(DATA / f"compat_{VER}.jsonl", {"check": name, "status": status, "detail": detail,
                                             "wall_s": round(time.time() - t0, 1)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roll")
    r.add_argument("--only", nargs="+")
    g = sub.add_parser("rig")
    g.add_argument("--variants", nargs="+", default=list(VARIANTS))
    g.add_argument("--tasks", nargs="+", default=["pull", "twist"])
    g.add_argument("--N", nargs="+", type=float, default=[0.5, 1.0, 3.0])
    g.add_argument("--dt", nargs="+", type=float, default=[1.0, 5.0])
    u = sub.add_parser("gpu")
    u.add_argument("--variants", nargs="+", default=list(VARIANTS))
    u.add_argument("--dt", nargs="+", type=float, default=[1.0, 5.0])
    u.add_argument("--N", type=float, default=1.0)
    u.add_argument("--nworld", nargs="+", type=int, default=[1024, 4096])
    h = sub.add_parser("grasp")
    h.add_argument("--integrator", choices=["implicitfast", "discrete"], default="implicitfast")
    h.add_argument("--E", nargs="+", type=float, default=[2.5])
    h.add_argument("--ir", nargs="+", type=float, default=[10.0, 100.0])
    h.add_argument("--noslip", nargs="+", type=int, default=[0, 10])
    h.add_argument("--sims", nargs="+", default=["cpu", "mjw"])
    h.add_argument("--n-cpu", type=int, default=16)
    h.add_argument("--n-gpu", type=int, default=256)
    hg = sub.add_parser("hand-gpu")
    hg.add_argument("--variants", nargs="+", default=list(VARIANTS))
    hg.add_argument("--E", type=float, default=2.5)
    hg.add_argument("--ir", type=float, default=10.0)
    hg.add_argument("--nworld", nargs="+", type=int, default=[256, 1024, 2048])
    f = sub.add_parser("film")
    f.add_argument("--label", required=True)
    f.add_argument("--tol", type=float, default=None)
    ft = sub.add_parser("film-tile")
    ft.add_argument("--keys", nargs="+", required=True)
    ft.add_argument("--labels", nargs="+", required=True)
    ft.add_argument("--out", required=True)
    c = sub.add_parser("compat")
    c.add_argument("--only", nargs="+", choices=list(CHECKS))
    a = ap.parse_args()
    {"roll": cmd_roll, "rig": cmd_rig, "gpu": cmd_gpu, "grasp": cmd_grasp, "hand-gpu": cmd_hand_gpu,
     "film": cmd_film, "film-tile": cmd_film_tile, "compat": cmd_compat}[a.cmd](a)


if __name__ == "__main__":
    main()
