#!/usr/bin/env python3
"""Physics time per world-step of MuJoCo-Warp and Newton from one held state of the RL training env (2026-10-06).

The state is the D6 reorientation env after its scripted grasp and lift (scripts/rl_state_export.py, 60 policy steps,
64 worlds): logs/20261006-rl_contact/state_<variant>/{scene.xml, state.npz}. Every engine loads the same MJCF with the
same body inertias (each body's compiled mass, centre and principal inertia written as an <inertial>, so that geoms
added or dropped in Newton cannot change the mass distribution), the trainer's options (2 ms, implicitfast, elliptic
cone, impratio 10, 10 solver / 20 line-search iterations; with --mjlab-opts, the default, also mjlab's parallel line
search and 50 CCD iterations, where MuJoCo-Warp defaults to an iterative line search and 35), the 64 exported
configurations tiled over the batch, the
exported actuator targets, and zero velocity. Blocks of 50 steps: one uncaptured block (kernel compile), one captured
CUDA graph, then `--blocks` timed replays.

  engine   mjw        raw MuJoCo-Warp (put_model; broadphase as put_model chooses it, as in mjlab: NXN over the
                      contype-filtered geom pairs when they number under 250,000; --broadphase sap forces
                      SAP_SEGMENTED, which newton_hand_throughput.py used)
           nt_pt      Newton SolverMuJoCo, Newton's collision pipeline, point contact; 0.5 mm contact gap, margin 0
           nt_hydro   Newton's pipeline with the tips' TPU block meshes and the tool as hydroelastic SDF shapes,
                      kh = E/h = 1.18e9 N/m^3 multiplied by 1/m_eff = invweight0[tip] + invweight0[tool] of the
                      solver's model (contact_bed_newton.py, newton_turn.py --mass-correct), tool kh 100x the largest
                      tip kh, 0.5 mm voxels, +-6 mm band, contact reduction, buffers sized as in newton_hand_throughput
           mjlab      reference: the live RL env itself (rl_contact_throughput.make_env, 60 zero-action policy steps,
                      then env.sim.step for the same number of physics steps), uv environment only
  variant  legacy     the box tip (point contact), mesh: TPU block as one convex mesh, pads1: TPU block, 1 mm pads
           (pads touch only the tool; in Newton their solimp d0 is recomputed from Newton's inverse weights,
            d0' = 1 - (1 - d0) (w_tip + w_tool)_mjcf / (w_tip + w_tool)_newton, newton_hand_throughput.pad_d0_from)

Newton's MJCF importer needs trimesh for mesh files, which its environment lacks: mesh geoms are dropped from the MJCF
and the meshes are added back as newton.Mesh from the OBJ vertices (tips: the TPU block; tool: the screw-tip cone),
except the pads' visual block (contype 4: floor and finger contacts, none in the held state).

Measured: us per world-step (block wall time / 50 / worlds), contacts and constraint rows per world (mean, max), GPU
memory, and the fraction of worlds whose tool is still within 20 mm of its exported position after the timed run.

    uv run --extra rl --extra gpu python scripts/same_state_timing.py --engine mjw --variant pads1 --nworld 1024 2048
    PY=logs/20261004-contact-transfer/venv/bin/python   # newton 1.7.0.dev0, mujoco_warp 3.14
    WARP_CACHE_PATH=$(mktemp -d) $PY scripts/same_state_timing.py --engine nt_hydro --variant mesh --nworld 1024
Rows: docs/experiments/20261006-rl_contact/same_state_timing.jsonl, one fsynced line per engine x variant x batch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STATES = ROOT / "logs/20261006-rl_contact"
OUT = ROOT / "docs/experiments/20261006-rl_contact/same_state_timing.jsonl"
BLOCK = 50
KH_TIP = 1e7 / 0.0085
SDF_VOXEL, SDF_BAND = 0.0005, 0.006
FINGERS = ("thumb", "index", "middle")


def gpu_used_mb():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return None


def read_obj(path):
    V, F = [], []
    for line in open(path):
        t = line.split()
        if not t:
            continue
        if t[0] == "v":
            V.append([float(x) for x in t[1:4]])
        elif t[0] == "f":
            idx = [int(x.split("/")[0]) - 1 for x in t[1:]]
            for j in range(1, len(idx) - 1):
                F += [idx[0], idx[j], idx[j + 1]]
    return np.array(V, np.float32), np.array(F, np.int32)


def load_state(variant):
    d = STATES / f"state_{variant}"
    meta = json.loads((d / "state.json").read_text())
    st = dict(np.load(d / "state.npz"))
    return d, meta, st


def engine_xml(sdir):
    """scene.xml with every massive body's compiled inertia pinned and one-value solrefs given damping ratio 1."""
    import mujoco
    out = sdir / "scene_engines.xml"
    src = (sdir / "scene.xml").read_text()
    m = mujoco.MjModel.from_xml_string(src)
    root = ET.fromstring(src)
    for b in root.iter("body"):
        name = b.get("name")
        if not name:
            continue
        i = m.body(name).id
        if m.body_mass[i] <= 0:
            continue
        for old in b.findall("inertial"):
            b.remove(old)
        inn = ET.Element("inertial", pos=" ".join(f"{x:.9g}" for x in m.body_ipos[i]),
                         quat=" ".join(f"{x:.9g}" for x in m.body_iquat[i]), mass=f"{m.body_mass[i]:.9g}",
                         diaginertia=" ".join(f"{x:.9g}" for x in m.body_inertia[i]))
        b.insert(0, inn)
    xml = ET.tostring(root, encoding="unicode")
    xml = re.sub(r'solref="([0-9.eE+-]+)"', r'solref="\1 1"', xml)
    out.write_text(xml)
    m2 = mujoco.MjModel.from_xml_string(xml)
    assert np.allclose(m2.body_mass, m.body_mass) and np.allclose(m2.body_inertia, m.body_inertia, rtol=1e-6)
    return xml, m2


def tiled(a, nworld):
    reps = int(np.ceil(nworld / a.shape[0]))
    return np.tile(a, (reps,) + (1,) * (a.ndim - 1))[:nworld]


OPTS = {"mjlab": True, "broadphase": "auto", "broadphase_used": None, "warm": False, "nconmax": None,
        "njmax": None}


def set_ls_parallel(opt):
    """mjlab's parallel line search; the option was removed in MuJoCo-Warp 3.9.1."""
    try:
        opt.ls_parallel = True
    except AttributeError:
        pass


def run_mjw(xml, m, st, meta, nworld, nblocks):
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    d = mujoco.MjData(m)
    d.qpos[:] = st["qpos"][0]
    d.ctrl[:] = st["ctrl"][0]
    mujoco.mj_forward(m, d)
    wm = mjw.put_model(m)
    if OPTS["broadphase"] == "sap":
        wm.opt.broadphase = mjw.BroadphaseType.SAP_SEGMENTED
    OPTS["broadphase_used"] = str(wm.opt.broadphase).split(".")[-1]
    if OPTS["mjlab"]:
        set_ls_parallel(wm.opt)
    ncon = int(st["ncon_world"].max())
    nefc = int(st["nefc_world"].max())
    # the broadphase writes its candidate pairs into the contact buffer: on the pads they number ~6x the contacts
    # (250+ per world for 41 contacts), and a buffer sized from the contacts overflows and drops pad contacts
    nconmax, njmax = OPTS["nconmax"] or max(64, 12 * ncon), OPTS["njmax"] or max(256, 8 * nefc)
    wd = mjw.put_data(m, d, nworld=nworld, nconmax=nconmax, njmax=njmax)
    wd.qpos.assign(tiled(st["qpos"], nworld).astype(np.float32))
    wd.ctrl.assign(tiled(st["ctrl"], nworld).astype(np.float32))
    if OPTS["warm"]:
        wd.qvel.assign(tiled(st["qvel"], nworld).astype(np.float32))
        wd.qacc_warmstart.assign(tiled(st["qacc_warmstart"], nworld).astype(np.float32))
    else:
        wd.qvel.zero_()
    if st["act"].size:
        wd.act.assign(tiled(st["act"], nworld).astype(np.float32))
    tb = m.body(meta["tool_body"]).id
    x0 = tiled(st["tool_pos"], nworld)

    def block():
        for _ in range(BLOCK):
            mjw.step(wm, wd)
    block()
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        block()
    t0 = time.perf_counter()
    for _ in range(nblocks):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    el = time.perf_counter() - t0
    xp = wd.xpos.numpy()[:, tb, :]
    nacon = int(wd.nacon.numpy()[0])
    nefc_w = wd.nefc.numpy()
    return dict(us_per_world_step=1e6 * el / (nblocks * BLOCK * nworld), contacts_per_world=nacon / nworld,
                nefc_world_mean=float(nefc_w.mean()), nefc_world_max=int(nefc_w.max()),
                held_frac=float(np.mean(np.linalg.norm(xp - x0, axis=1) < 0.02)), nconmax=nconmax, njmax=njmax,
                sim_s=round((nblocks + 2) * BLOCK * float(m.opt.timestep), 3))


def run_mjlab(variant, nworld, nblocks):
    import torch
    sys.path.insert(0, str(ROOT / "scripts"))
    import rl_contact_throughput as T
    cfg = T.base_cfg(ROOT / T.RUNS[variant])
    env = T.make_env(cfg, nworld, 512, 1024)
    try:
        act = torch.zeros((nworld, env.action_manager.total_action_dim), device="cuda:0")
        env.reset()
        for _ in range(60):
            env.step(act)
        mjm, wd = env.sim.mj_model, env.sim.wp_data
        tb = next(i for i in range(mjm.nbody) if mjm.body(i).name.split("/")[-1] == "cube")
        x0 = wd.xpos.numpy()[:, tb].copy()
        for _ in range(2 * BLOCK):
            env.sim.step()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(nblocks * BLOCK):
            env.sim.step()
        torch.cuda.synchronize()
        el = time.perf_counter() - t0
        nefc_w = wd.nefc.numpy()
        return dict(us_per_world_step=1e6 * el / (nblocks * BLOCK * nworld),
                    contacts_per_world=int(wd.nacon.numpy()[0]) / nworld, nefc_world_mean=float(nefc_w.mean()),
                    nefc_world_max=int(nefc_w.max()),
                    held_frac=float(np.mean(np.linalg.norm(wd.xpos.numpy()[:, tb] - x0, axis=1) < 0.02)),
                    nconmax=512, njmax=1024, sim_s=round((nblocks + 2) * BLOCK * float(mjm.opt.timestep), 3),
                    broadphase_used=str(env.sim.wp_model.opt.broadphase).split(".")[-1])
    finally:
        env.close()


def strip_meshes(xml):
    """Newton's MJCF: drop mesh geoms (its importer would need trimesh) and return their info; move plane geoms of
    joint-less top-level bodies (mjlab's terrain) to the worldbody, which Newton requires of planes; and make the
    mocap base a fixed body (it is never moved here, so MuJoCo's mocap and Newton's fixed body coincide)."""
    root = ET.fromstring(xml)
    wb = root.find("worldbody")
    for b in list(wb.findall("body")):
        if b.find("joint") is None and b.find("freejoint") is None:
            if b.get("mocap") == "true":
                del b.attrib["mocap"]
            for g in list(b.findall("geom")):
                if g.get("type") == "plane" and not b.get("pos") and not b.get("quat"):
                    b.remove(g)
                    wb.insert(0, g)
    files = {ms.get("name"): ms.get("file") for ms in root.iter("mesh")}
    dropped = []
    for b in root.iter("body"):
        for g in list(b.findall("geom")):
            if g.get("type") == "mesh":
                dropped.append(dict(body=b.get("name"), mesh=g.get("mesh"), file=files.get(g.get("mesh")),
                                    contype=int(g.get("contype", 1)), conaffinity=int(g.get("conaffinity", 1)),
                                    pos=g.get("pos"), quat=g.get("quat"), friction=g.get("friction")))
                b.remove(g)
    return ET.tostring(root, encoding="unicode"), dropped


def solver_invweights(model, solver, names):
    mjm = solver.mj_model
    labels = [x.split("/")[-1] for x in model.body_label]
    m2n = solver.mjc_body_to_newton.numpy()[0]
    w, mass = {}, {}
    for bi in range(mjm.nbody):
        nb = int(m2n[bi])
        nm = labels[nb] if 0 <= nb < len(labels) else ""
        if nm in names:
            w[nm], mass[nm] = float(mjm.body_invweight0[bi, 0]), float(mjm.body_mass[bi])
    return w, mass


def build_newton(xml, m, meta, nworld, engine, variant, buffer_fraction, buffer_mult_broad, extra):
    import newton
    hydro = engine == "nt_hydro"
    xml_n, dropped = strip_meshes(xml)
    b0 = newton.ModelBuilder(gravity=tuple(float(g) for g in m.opt.gravity))
    b0.add_mjcf(xml_n, ctrl_direct=True, parse_sites=False, parse_visuals=False)
    for i in range(len(b0.shape_gap)):
        b0.shape_gap[i] = 0.0005
        b0.shape_margin[i] = 0.0
    body_of = {x.split("/")[-1]: k for k, x in enumerate(b0.body_label)}
    tool = meta["tool_body"].split("/")[-1]
    tips = [meta["tip_bodies"][f].split("/")[-1] for f in FINGERS]
    for dg in dropped:
        bname = dg["body"].split("/")[-1]
        if dg["contype"] == 4:                  # the pads' block: floor and finger contacts only
            continue
        if bname not in body_of or not dg["file"]:
            continue
        V, F = read_obj(dg["file"])
        mesh = newton.Mesh(V, F)
        on_hydro = hydro and (bname in tips or bname == tool)
        if on_hydro:
            mesh.build_sdf(target_voxel_size=SDF_VOXEL, narrow_band_range=(-SDF_BAND, SDF_BAND), margin=SDF_BAND)
        mu = float(dg["friction"].split()[0]) if dg["friction"] else 1.0
        cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, gap=0.0 if on_hydro else 0.0005, mu=mu, kh=KH_TIP,
                                              density=0.0, is_hydroelastic=on_hydro, mu_torsional=0.0, mu_rolling=0.0)
        xform = None
        if dg["pos"] or dg["quat"]:
            import warp as wp
            p = [float(x) for x in (dg["pos"] or "0 0 0").split()]
            q = [float(x) for x in (dg["quat"] or "1 0 0 0").split()]
            xform = wp.transform(p, wp.quat(q[1], q[2], q[3], q[0]))
        k = b0.add_shape_mesh(body_of[bname], mesh=mesh, cfg=cfg, label=f"{bname}_{dg['mesh'].split('/')[-1]}",
                              **({"xform": xform} if xform is not None else {}))
        if on_hydro:
            b0.shape_material_kf[k] = 10.0
    if hydro:
        for i in range(len(b0.shape_label)):
            bb = b0.shape_body[i]
            if bb < 0:
                continue
            bn = b0.body_label[bb].split("/")[-1]
            if bn != tool or b0.shape_flags[i] & int(newton.ShapeFlags.HYDROELASTIC):
                continue
            b0.shape_flags[i] |= int(newton.ShapeFlags.HYDROELASTIC)
            b0.shape_sdf_target_voxel_size[i] = SDF_VOXEL
            b0.shape_sdf_narrow_band_range[i] = (-SDF_BAND, SDF_BAND)
            b0.shape_sdf_padding[i] = SDF_BAND
            b0.shape_margin[i] = 0.0
            b0.shape_gap[i] = 0.0
            b0.shape_material_kh[i] = KH_TIP * 100.0
            b0.shape_material_kf[i] = 10.0
    scene = newton.ModelBuilder(gravity=tuple(float(g) for g in m.opt.gravity))
    scene.replicate(b0, nworld)
    model = scene.finalize(device="cuda:0")
    nconmax = 512 if variant == "pads1" else (320 if hydro else 64)
    njmax = 1536 if variant == "pads1" else (1024 if hydro else 256)
    kw = {}
    if hydro:
        from newton.geometry import HydroelasticSDF
        kw["sdf_hydroelastic_config"] = HydroelasticSDF.Config(reduce_contacts=True, anchor_contact=True,
                                                               buffer_fraction=buffer_fraction,
                                                               buffer_mult_broad=int(buffer_mult_broad))
    pipe = newton.CollisionPipeline(model, reduce_contacts=True, rigid_contact_max=nworld * nconmax,
                                    broad_phase="explicit", **kw)
    o = m.opt
    solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, disable_sensors=True, njmax=njmax,
                                         nconmax=nconmax, iterations=int(o.iterations),
                                         ls_iterations=int(o.ls_iterations), cone="elliptic",
                                         integrator="implicitfast", tolerance=float(o.tolerance),
                                         impratio=float(o.impratio), ccd_iterations=int(o.ccd_iterations))
    if OPTS["mjlab"]:
        set_ls_parallel(solver.mjw_model.opt)
    w, mass = solver_invweights(model, solver, set(tips) | {tool})
    extra.update(invweight0_newton=w, body_mass_newton=mass, nconmax=nconmax, njmax=njmax)
    if hydro:
        kh_tip = {t: KH_TIP * (w[t] + w[tool]) for t in tips}
        kh_tool = 100.0 * max(kh_tip.values())
        shape_body = model.shape_body.numpy()
        flags = model.shape_flags.numpy()
        kh = model.shape_material_kh.numpy()
        labels = [x.split("/")[-1] for x in model.body_label]
        for i in range(len(kh)):
            bb = int(shape_body[i])
            if bb < 0 or not flags[i] & int(newton.ShapeFlags.HYDROELASTIC):
                continue
            kh[i] = kh_tool if labels[bb] == tool else kh_tip.get(labels[bb], kh[i])
        model.shape_material_kh.assign(kh)
        extra.update(kh_tip=kh_tip, kh_tool=kh_tool, buffer_fraction=buffer_fraction,
                     buffer_mult_broad=int(buffer_mult_broad))
    return model, pipe, solver


def pads_d0_newton(xml, m, meta, extra):
    """Rewrite the pads' solimp d0 from Newton's inverse weights (one-world build)."""
    import newton  # noqa: F401
    w_new = {}
    model, pipe, solver = build_newton(xml, m, meta, 1, "nt_pt", "pads1", 1.0, 1, w_new)
    w_new = w_new["invweight0_newton"]
    tool = meta["tool_body"].split("/")[-1]
    w_mj = {n.split("/")[-1]: float(m.body_invweight0[m.body(n).id, 0])
            for n in list(meta["tip_bodies"].values()) + [meta["tool_body"]]}
    d0 = {}
    for f in FINGERS:
        t = meta["tip_bodies"][f].split("/")[-1]
        mt = re.search(rf'<geom name="[^"]*{f}_pad0"[^>]*?solimp="([0-9.eE+-]+)', xml)
        d_old = float(mt.group(1))
        d_new = 1.0 - (1.0 - d_old) * (w_mj[t] + w_mj[tool]) / (w_new[t] + w_new[tool])
        d0[f] = (d_old, d_new)
        xml = re.sub(rf'(<geom name="[^"]*{f}_pad\d+"[^>]*?solimp=")[0-9.eE+-]+ [0-9.eE+-]+',
                     lambda mm, d=d_new: f"{mm.group(1)}{d:.6g} {d:.6g}", xml)
    extra.update(invweight0_mjcf=w_mj, pad_d0={f: {"mjcf": a, "newton": b} for f, (a, b) in d0.items()})
    return xml


def joint_map(m, mm):
    """Solver-model joint j -> MJCF-compile joint j. Newton keeps the MJCF's joint order and splits a merged D6
    joint back into one MuJoCo joint per axis, so the two lists align; type and range are checked joint by joint."""
    if m.njnt != mm.njnt:
        raise ValueError(f"joint count differs: MJCF {m.njnt}, Newton's MuJoCo model {mm.njnt}")
    for j in range(m.njnt):
        if m.jnt_type[j] != mm.jnt_type[j] or not np.allclose(m.jnt_range[j], mm.jnt_range[j], atol=1e-6):
            raise ValueError(f"joint {j}: {m.joint(j).name} type {m.jnt_type[j]} range {m.jnt_range[j]} vs "
                             f"{mm.joint(j).name} type {mm.jnt_type[j]} range {mm.jnt_range[j]}")
    return list(range(m.njnt))


def run_newton(xml, m, st, meta, nworld, nblocks, engine, variant, buffer_fraction, buffer_mult_broad):
    import newton
    import warp as wp
    extra = {}
    if variant == "pads1":
        xml = pads_d0_newton(xml, m, meta, extra)
    model, pipe, solver = build_newton(xml, m, meta, nworld, engine, variant, buffer_fraction, buffer_mult_broad, extra)
    mm = solver.mj_model
    # state: MJCF qpos -> Newton joint_q through the solver's joint -> Newton dof map (free joints: wxyz -> xyzw)
    import mujoco
    jmap = joint_map(m, mm)
    qs = model.joint_q_start.numpy()
    qds = model.joint_qd_start.numpy()
    jq = model.joint_q.numpy()
    nj = model.joint_count
    Q = tiled(st["qpos"], nworld)
    j2d = solver.mjc_jnt_to_newton_dof.numpy()                 # (nworld, njnt)
    dof_joint = np.zeros(int(model.joint_dof_count), int)
    for jn in range(nj):
        dof_joint[qds[jn]:qds[jn + 1] if jn + 1 < nj else len(dof_joint)] = jn
    for w in range(nworld):
        for j, j0 in enumerate(jmap):
            dof = int(j2d[w, j])
            jn = dof_joint[dof]
            a = m.jnt_qposadr[j0]
            if m.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE:
                q = Q[w, a:a + 7]
                jq[qs[jn]:qs[jn] + 3] = q[:3]
                jq[qs[jn] + 3:qs[jn] + 7] = [q[4], q[5], q[6], q[3]]
            else:
                jq[qs[jn] + (dof - qds[jn])] = Q[w, a]
    model.joint_q.assign(jq)
    model.joint_qd.zero_()
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    # actuators through their transmission joint (Newton's MuJoCo model has no actuator names)
    act_of_joint = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)}
    order = [act_of_joint[jmap[int(mm.actuator_trnid[a, 0])]] for a in range(mm.nu)]
    C = tiled(st["ctrl"], nworld)[:, order].astype(np.float32)
    ctrl.mujoco.ctrl.assign(C.reshape(ctrl.mujoco.ctrl.shape))
    cc = pipe.contacts()
    labels = [x.split("/")[-1] for x in model.body_label]
    tool = meta["tool_body"].split("/")[-1]
    tool_ids = np.array([i for i, x in enumerate(labels) if x == tool])
    x0 = s0.body_q.numpy()[tool_ids, :3].copy()
    st_ = {"s0": s0, "s1": s1}
    dt = float(m.opt.timestep)

    def block():
        for _ in range(BLOCK):
            st_["s0"].clear_forces()
            pipe.collide(st_["s0"], cc)
            solver.step(st_["s0"], st_["s1"], ctrl, cc, dt)
            st_["s0"], st_["s1"] = st_["s1"], st_["s0"]
    block()
    wp.synchronize()
    hs = getattr(pipe, "hydroelastic_sdf", None)
    if hs is not None:
        hs._host_warning_poll_interval = 10 ** 12
    with wp.ScopedCapture() as cap:
        block()
    t0 = time.perf_counter()
    for _ in range(nblocks):
        wp.capture_launch(cap.graph)
    wp.synchronize()
    el = time.perf_counter() - t0
    x1 = st_["s0"].body_q.numpy()[tool_ids, :3]
    md = solver.mjw_data
    na = int(md.nacon.numpy()[0])
    nefc_w = md.nefc.numpy()
    return dict(us_per_world_step=1e6 * el / (nblocks * BLOCK * nworld), contacts_per_world=na / nworld,
                nefc_world_mean=float(nefc_w.mean()), nefc_world_max=int(nefc_w.max()),
                held_frac=float(np.mean(np.linalg.norm(x1 - x0, axis=1) < 0.02)),
                tool_start_err_mm=float(1e3 * np.abs(x0 - tiled(st["tool_pos"], nworld)).max()),
                sim_s=round((nblocks + 2) * BLOCK * dt, 3), **extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, choices=["mjlab", "mjw", "nt_pt", "nt_hydro"])
    ap.add_argument("--variant", required=True, choices=["legacy", "mesh", "pads1"])
    ap.add_argument("--nworld", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--blocks", type=int, default=20)
    ap.add_argument("--buffer-fraction", type=float, default=0.013)
    ap.add_argument("--buffer-mult-broad", type=int, default=40)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-mjlab-opts", action="store_true", help="MuJoCo-Warp's own line search and CCD defaults")
    ap.add_argument("--broadphase", choices=["auto", "sap"], default="auto", help="mjw: put_model's choice or SAP")
    ap.add_argument("--warm", action="store_true", help="mjw: the exported qvel and solver warm start, not rest")
    ap.add_argument("--nconmax", type=int, default=None, help="mjw: contacts per world (default 12x the exported)")
    ap.add_argument("--njmax", type=int, default=None, help="mjw: constraint rows per world (default 8x the exported)")
    a = ap.parse_args()
    OPTS["mjlab"] = not a.no_mjlab_opts
    OPTS["broadphase"] = a.broadphase
    OPTS["warm"] = a.warm
    OPTS["nconmax"], OPTS["njmax"] = a.nconmax, a.njmax
    sdir, meta, st = load_state(a.variant)
    xml, m = engine_xml(sdir)
    if OPTS["mjlab"]:
        m.opt.ccd_iterations = 50
    import mujoco_warp
    import warp
    for nw in a.nworld:
        row = {"engine": a.engine, "variant": a.variant, "nworld": nw, "state": str(sdir.relative_to(ROOT)),
               "dt": float(m.opt.timestep), "iterations": int(m.opt.iterations), "impratio": float(m.opt.impratio),
               "warp": warp.__version__, "mujoco_warp": getattr(mujoco_warp, "__version__", "?"),
               "when": time.strftime("%Y-%m-%d %H:%M"), "gpu_before_mb": gpu_used_mb(),
               "ls_parallel": OPTS["mjlab"] and getattr(mujoco_warp, "__version__", "0") < "3.9.1",
               "ccd_iterations": int(m.opt.ccd_iterations), "warm": OPTS["warm"],
               "exported_ncon_world": float(st["ncon_world"].mean()), "exported_nefc_world": float(st["nefc_world"].mean())}
        try:
            if a.engine == "mjlab":
                r = run_mjlab(a.variant, nw, a.blocks)
                OPTS["broadphase_used"] = r.pop("broadphase_used")
            elif a.engine == "mjw":
                r = run_mjw(xml, m, st, meta, nw, a.blocks)
            else:
                r = run_newton(xml, m, st, meta, nw, a.blocks, a.engine, a.variant, a.buffer_fraction,
                               a.buffer_mult_broad)
                import newton
                row["newton"] = newton.__version__
            row.update(r, status="ok", gpu_used_mb=gpu_used_mb(), broadphase=OPTS["broadphase_used"])
            print(f"{a.engine:<8} {a.variant:<6} {nw:>5}: {r['us_per_world_step']:.2f} us/world-step, "
                  f"{r['contacts_per_world']:.1f} contacts, {r['nefc_world_mean']:.0f} rows/world, held "
                  f"{r['held_frac']:.3f}, GPU {row['gpu_used_mb']} MB", flush=True)
        except Exception as e:
            row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
            print(f"{a.engine} {a.variant} {nw}: {row['error']}", flush=True)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return 0


if __name__ == "__main__":
    sys.exit(main())
