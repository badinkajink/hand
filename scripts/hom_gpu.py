#!/usr/bin/env python3
"""HOM control of the sphere tasks (roll, twist) on MuJoCo-Warp: many hand layouts and seeds as parallel GPU worlds,
with the controller of src/morphohand/control/hom.py batched in torch on the GPU (owner, 2026-10-29: "can we leverage
GPU acceleration for higher throughput evaluation, since our mjwarp sphere packed model is ok enough?").

The scale study's layouts compile to models that differ only in the positions of the three mount bodies (checked
field by field, 2026-10-29), so one MuJoCo-Warp model serves every layout: body_pos gets one row per world, carrying
that world's mounts and palm. Each world is one (layout, seed); a layout's grasp is either the task study's equator
grasp (symmetric tripods, `--layouts 2d`) or the foundational pose that scripts/fp_sr2_toy.py solved for it (any of the
six-coordinate layouts, `--layouts rrt6d|sobol6d`).

The controller is hom.HomController.step for these tasks, per world and finger in torch:
  - the GCF from the tip's centre of pressure (its pads' contacts with the sphere, normal force from the elliptic
    cone's first row, gathered by a Warp kernel) at 0.02 N or more, else from the witness points of the rounded TPU
    block and the sphere (analytic here; the CPU uses mj_geomDistance on the block's convex mesh);
  - Eq. (1) from the joint axes and anchors; Eq. (2) solved exactly by enumerating the 27 active-bound patterns of the
    three joint rates (the CPU uses SciPy's BVLS on the same problem);
  - the complementary filter at k_min: every commanded roll and slide of these tasks is zero, so Eqs. (4)-(6) reduce to
    a low-pass of the contact point on the block's surface, stepped along the tangent and projected back;
  - the servo impedance in the 'reference' mode, gravity from qfrc_bias.
The physics differs from the CPU runner in float32, a solver tolerance of 1e-6 with 20 iterations, and pads that
collide only with the sphere. Scores come from hom_tasks.metrics on the same per-tick rows, so they compare directly.

  .venv/bin/python scripts/hom_gpu.py run --task twist --layouts 2d --seeds 0 1 2 --out hom2d_gpu
  .venv/bin/python scripts/hom_gpu.py run --task twist --layouts sobol6d --max 600 --seeds 0 1 2 --out hom6d_sobol_gpu
Rows: docs/experiments/20261029-foundational_pose_review/rows/<out>.jsonl.
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402  (module level: kernel annotations resolve against these globals)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import fp_hom_check as C  # noqa: E402
import hom_tasks as T  # noqa: E402

S, H, K, G = T.S, T.H, T.K, T.G
FINGERS, JOINTS = T.FINGERS, T.JOINTS
D_MM = 40.0
REF = (92.5, 102.5)                    # the reference layout whose scenes are compiled (the task study's sphere grasp)
NCONMAX, NJMAX = 1024, 2048
ROWS = C.ROWS
W_TASK = np.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0])     # spin, roll_y, roll_z, sep, slide_y, slide_z (free rolling)


@wp.kernel
def finger_forces(nacon: wp.array(dtype=int), worldid: wp.array(dtype=int), geom: wp.array(dtype=wp.vec2i),
                  pos: wp.array(dtype=wp.vec3), efc_addr: wp.array2d(dtype=int), efc_force: wp.array2d(dtype=float),
                  objmask: wp.array(dtype=int), fidx: wp.array(dtype=int), tidx: wp.array(dtype=int),
                  ftip: wp.array2d(dtype=float), ftot: wp.array2d(dtype=float), cops: wp.array3d(dtype=float)):
    """Per world and finger: the normal force of all the finger's contacts with the object, and of its tip pads' with
    their force-weighted contact positions (hom_tasks.Plant.contacts)."""
    i = wp.tid()
    if i >= nacon[0]:
        return
    g = geom[i]
    o1 = objmask[g[0]]
    o2 = objmask[g[1]]
    if o1 == o2:
        return
    other = g[0]
    if o1 == 0:
        other = g[0]
    else:
        other = g[1]
    k = fidx[other]
    if k < 0:
        return
    adr = efc_addr[i, 0]
    if adr < 0:
        return
    w = worldid[i]
    f = efc_force[w, adr]
    if f <= 0.0:
        return
    wp.atomic_add(ftot, w, k, f)
    if tidx[other] >= 0:
        p = pos[i]
        wp.atomic_add(ftip, w, k, f)
        wp.atomic_add(cops, w, k, 0, f * p[0])
        wp.atomic_add(cops, w, k, 1, f * p[1])
        wp.atomic_add(cops, w, k, 2, f * p[2])


# ------------------------------------------------------------------------------------------------ layouts and grasps

def mount_xy(d_mm):
    """Palm-frame (x, y) mount positions (m) of a six-coordinate layout (mm)."""
    d = np.asarray(d_mm, float) / 1000.0
    return {"thumb": d[0:2], "index": d[2:4], "middle": d[4:6]}


def tripod_setup(task, mounts_xy, grasp):
    """hom_tasks' setup dict for an equator-type sphere grasp: grasp = (centre in the palm frame, {f: outward normal})."""
    centre, n_out = grasp
    r = D_MM / 2000.0
    rr = r + K.R + K.GAP - K.GAP                       # the task study's touching grasp (gap 0; open_pose adds 8 mm)
    tg = {f: centre + rr * n_out[f] for f in FINGERS}
    palm = np.array([-centre[0], -centre[1], S.Z_OBJ - centre[2]])
    mounts = {f: np.array([mounts_xy[f][0], mounts_xy[f][1], 0.0]) for f in FINGERS}
    return {"task": task, "shape": "sphere", "d_mm": D_MM, "r": r, "hl": 0.0, "palm": palm,
            "quat": np.array(T.SPHERE_Q), "design": {}, "targets": tg, "n_out": n_out, "mounts": mounts}


def grasp_2d(xs, ys):
    """The task study's equator grasp on a symmetric tripod (hom_tasks.design_sphere's rule)."""
    cell = C.cell_of(xs, ys)
    des = C.design(cell)
    if des is None:
        return None, None
    gg = S.grasp_geometry(cell, "sphere", D_MM, des["h"], des["spread"])
    n_out = {f: np.asarray(gg["n_out"][f], float) for f in FINGERS}
    return (np.asarray(gg["centre_palm"], float), n_out), des


def grasp_fp(row):
    """The tripod foundational pose that fp_sr2_toy.py solved for a layout (sphere centre and contact normals)."""
    import fp_sr2_toy as FP
    rec = row["fp"].get("tripod", {})
    if not rec.get("ok"):
        return None
    M = FP.mounts6(row["d"])
    obj, _tg, n, _pc, _ax = FP.geometry("tripod", np.asarray(rec["v"], float), M)
    return np.asarray(obj["centre"], float), {f: np.asarray(n[f], float) for f in FINGERS}


class Layouts:
    """Per-layout mounts, palm, grasp and open pose, from one reference pt model whose mount bodies are moved."""

    def __init__(self, task):
        import mujoco
        self.mj = mujoco
        self.task = task
        cell = C.cell_of(*REF)
        sc = S.cell_scenes(dict(cell), "sphere", D_MM, models=("pads", "pt"))
        self.pads_path, self.pt_path = Path(sc["pads"][0]), Path(sc["pt"][0])
        self.m_pt = mujoco.MjModel.from_xml_path(str(self.pt_path))
        self.mount_bid = {f: self.m_pt.body(f"{f}_mount").id for f in FINGERS}
        # body_pos of a mount = its nominal position + the layout's palm-frame offset from REAL_V1_MOUNTS
        ref_xy = mount_xy([-REF[0] / 2, 0.0, REF[0] / 2, REF[1] / 2, REF[0] / 2, -REF[1] / 2])
        self.base = {f: self.m_pt.body_pos[self.mount_bid[f]].copy() - np.array([*ref_xy[f], 0.0]) for f in FINGERS}

    def body_mounts(self, mxy):
        return {f: self.base[f] + np.array([mxy[f][0], mxy[f][1], 0.0]) for f in FINGERS}

    def prepare(self, mxy, grasp):
        setup = tripod_setup(self.task, mxy, grasp)
        bm = self.body_mounts(mxy)
        for f in FINGERS:
            self.m_pt.body_pos[self.mount_bid[f]] = bm[f]
        q_open, _q_touch = T.open_pose(setup, self.m_pt)
        return setup, bm, q_open


def check_mounts():
    """The mount positions built from offsets against a scene compiled for another layout."""
    import mujoco
    L = Layouts("twist")
    for xs, ys in ((70.0, 80.0), (115.0, 140.0)):
        sc = S.cell_scenes(dict(C.cell_of(xs, ys)), "sphere", D_MM, models=("pt",))
        m = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
        bm = L.body_mounts(mount_xy([-xs / 2, 0.0, xs / 2, ys / 2, xs / 2, -ys / 2]))
        err = max(float(np.abs(m.body_pos[L.mount_bid[f]] - bm[f]).max()) for f in FINGERS)
        print(f"layout {xs} x {ys}: mount body_pos error {err:.2e} m")


# ------------------------------------------------------------------------------------------------ the batch

def mocap_model(path):
    """The scene compiled with the palm and the post as mocap bodies (both are welded to the world and have no joints,
    so the physics is unchanged; MuJoCo-Warp then recomputes their geoms' poses every step)."""
    import mujoco
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    for name in ("palm_pose", "post"):
        b = root.find(f".//worldbody/body[@name='{name}']")
        if b is None:
            raise RuntimeError(f"{name} is not a child of the world body")
        b.set("mocap", "true")
    tmp = Path(path).with_name(f"{Path(path).stem}.mocap.{os.getpid()}.xml")
    ET.ElementTree(root).write(tmp)
    try:
        return mujoco.MjModel.from_xml_path(str(tmp))
    finally:
        tmp.unlink()


class Batch:
    """nworld (layout, seed) worlds of one task on MuJoCo-Warp with the batched HOM controller."""

    def __init__(self, L: Layouts, worlds, prm=None):
        import mujoco
        import mujoco_warp as mjw
        import torch
        self.mj, self.mjw, self.torch = mujoco, mjw, torch
        self.L, self.task, self.worlds = L, L.task, worlds
        self.prm = H.HomParams(dt=T.DT_C) if prm is None else prm
        W = self.W = len(worlds)
        m = mocap_model(L.pads_path)
        self.m = m
        self.ob = m.body(S.OBJ).id
        self.post_bid = m.body("post").id
        pads = np.array([g for g in range(m.ngeom) if "_pad" in (m.geom(g).name or "")])
        tool = np.flatnonzero(m.geom_bodyid == self.ob)
        m.geom_contype[pads], m.geom_conaffinity[pads] = 2, 0         # pads meet the sphere only
        m.geom_conaffinity[tool] = m.geom_conaffinity[tool] | 2
        m.opt.tolerance, m.opt.iterations, m.opt.ls_iterations = 1e-6, 20, 10
        self.wm = mjw.put_model(m)
        self.wm.opt.contact_sensor_maxmatch = max(int(self.wm.opt.contact_sensor_maxmatch), 256)
        # MuJoCo-Warp computes the poses of geoms welded to the world once, in make_data; the palm and the post are
        # mocap bodies here (mocap_model), so the mounts' geoms below the palm move with each world's body_pos and
        # the palm and the post move through mocap_pos
        bp = np.repeat(m.body_pos[None].astype(np.float32), W, 0)
        for w, wd_ in enumerate(worlds):
            for f in FINGERS:
                bp[w, L.mount_bid[f]] = wd_["body_mounts"][f]
        self.wm.body_pos = wp.array(bp, dtype=wp.vec3)
        self.wd = mjw.make_data(m, nworld=W, nconmax=NCONMAX, njmax=NJMAX)
        self.mo_palm = int(m.body_mocapid[m.body("palm_pose").id])
        self.mo_post = int(m.body_mocapid[self.post_bid])
        # indices
        self.qa = int(m.jnt_qposadr[m.body_jntadr[self.ob]])
        self.va = int(m.jnt_dofadr[m.body_jntadr[self.ob]])
        jid = np.array([[m.joint(f"{f}_{j}").id for j in JOINTS] for f in FINGERS])
        self.jid = jid
        self.qadr = np.array([[m.jnt_qposadr[i] for i in r] for r in jid])
        self.vadr = np.array([[m.jnt_dofadr[i] for i in r] for r in jid])
        self.act = np.array([[m.actuator(f"a_{f}_{j}").id for j in JOINTS] for f in FINGERS])
        self.tip = np.array([m.body(f"{f}_tip").id for f in FINGERS])
        self.pip = np.array([m.joint(f"{f}_pip").id for f in FINGERS])
        self.lo = m.jnt_range[jid, 0]
        self.hi = m.jnt_range[jid, 1]
        self.kp = np.array([[m.actuator_gainprm[m.actuator(f"a_{f}_{j}").id][0] for j in JOINTS] for f in FINGERS])
        objmask = np.zeros(m.ngeom, np.int32)
        objmask[tool] = 1
        fidx = -np.ones(m.ngeom, np.int32)
        tidx = -np.ones(m.ngeom, np.int32)
        for g in range(m.ngeom):
            bn = m.body(m.geom_bodyid[g]).name
            for k, f in enumerate(FINGERS):
                if bn.startswith(f + "_"):
                    fidx[g] = k
                    if bn == f"{f}_tip":
                        tidx[g] = k
        self.objmask, self.fidx, self.tidx = (wp.array(objmask, dtype=int), wp.array(fidx, dtype=int),
                                              wp.array(tidx, dtype=int))
        self.ftip = wp.zeros((W, 3), dtype=float)
        self.ftot = wp.zeros((W, 3), dtype=float)
        self.cops = wp.zeros((W, 3, 3), dtype=float)
        # fingertip surfaces (hom_tasks.sr2_specs): rounded boxes in each tip body's frame
        specs, _ = T.sr2_specs(worlds[0]["setup"])
        boxes = [sp.surface.s if hasattr(sp.surface, "s") else sp.surface for sp in specs]
        self.box_c = np.array([b.c for b in boxes])
        self.box_a = np.array([b.a for b in boxes])
        self.box_r = float(boxes[0].r)
        self._init_state()
        self._capture()

    # ---------------------------------------------------------------- state
    def _init_state(self):
        m, W = self.m, self.W
        qpos = np.repeat(m.qpos0[None], W, 0).astype(np.float64)
        ctrl = np.zeros((W, m.nu))
        for w, wd_ in enumerate(self.worlds):
            seed = wd_["seed"]
            rng = np.random.default_rng(seed)
            jit = np.zeros(3) if seed == 0 else np.array([rng.uniform(-1, 1) * 1e-3, rng.uniform(-1, 1) * 1e-3, 0.0])
            yaw = 0.0 if seed == 0 else math.radians(rng.uniform(-2, 2))
            qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
            q = np.zeros(4)
            self.mj.mju_mulQuat(q, qz, np.asarray(wd_["setup"]["quat"], float))
            qpos[w, self.qa:self.qa + 3] = np.array([0.0, 0.0, S.Z_OBJ]) + jit
            qpos[w, self.qa + 3:self.qa + 7] = q
            for k, f in enumerate(FINGERS):
                qpos[w, self.qadr[k]] = wd_["q_open"][f]
                ctrl[w, self.act[k]] = wd_["q_open"][f]
        mp = np.repeat(m.body_pos[m.body_mocapid >= 0][None], W, 0)
        mp = np.zeros((W, m.nmocap, 3))
        for b in range(m.nbody):
            if m.body_mocapid[b] >= 0:
                mp[:, m.body_mocapid[b]] = m.body_pos[b]
        for w, wd_ in enumerate(self.worlds):
            mp[w, self.mo_palm] = wd_["setup"]["palm"]
        self.wd.mocap_pos.assign(mp.astype(np.float32))
        self.wd.qpos.assign(qpos.astype(np.float32))
        self.wd.qvel.zero_()
        self.wd.ctrl.assign(ctrl.astype(np.float32))
        self.mjw.forward(self.wm, self.wd)
        wp.synchronize()

    def _capture(self):
        mjw = self.mjw
        self.nsub = int(round(self.prm.dt / self.m.opt.timestep))
        has_kin = hasattr(mjw, "kinematics")
        self.has_kin = has_kin

        def block():
            for _ in range(self.nsub):
                mjw.step(self.wm, self.wd)
            if has_kin:
                mjw.kinematics(self.wm, self.wd)
            self.ftip.zero_()
            self.ftot.zero_()
            self.cops.zero_()
            c = self.wd.contact
            wp.launch(finger_forces, dim=c.pos.shape[0],
                      inputs=[self.wd.nacon, c.worldid, c.geom, c.pos, c.efc_address, self.wd.efc.force, self.objmask,
                              self.fidx, self.tidx, self.ftip, self.ftot, self.cops])
        block()                                                   # compile every kernel before the capture
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            block()
        self.graph = cap.graph
        self._init_state()                                         # the warm-up block moved the state: start again
        self._forces_now()

    def _forces_now(self):
        """The finger forces at the present state (before the first control period)."""
        self.ftip.zero_()
        self.ftot.zero_()
        self.cops.zero_()
        c = self.wd.contact
        wp.launch(finger_forces, dim=c.pos.shape[0],
                  inputs=[self.wd.nacon, c.worldid, c.geom, c.pos, c.efc_address, self.wd.efc.force, self.objmask,
                          self.fidx, self.tidx, self.ftip, self.ftot, self.cops])
        wp.synchronize()

    # ---------------------------------------------------------------- the controller
    def _rb_closest(self, x, k=None):
        """Closest point on finger k's rounded box (tip frame) to points x (..., 3), its outward normal and the
        signed distance; k indexes the finger axis of x when None (x shaped (W, 3, 3))."""
        th = self.torch
        c, a = self.c_t, self.a_t
        if k is not None:
            c, a = c[k], a[k]
        xx = x - c
        core = th.maximum(th.minimum(xx, a), -a)
        d = xx - core
        dist = th.linalg.norm(d, dim=-1, keepdim=True)
        out = dist > 1e-12
        n_out = d / dist.clamp_min(1e-12)
        # inside the core: the nearest face
        q = xx.abs() - a
        kk = q.argmax(-1, keepdim=True)
        sg = th.where(th.gather(xx, -1, kk) >= 0, 1.0, -1.0).to(x.dtype)
        n_in = th.zeros_like(x).scatter(-1, kk, sg)
        s_in = xx.clone().scatter(-1, kk, sg * (th.gather(a.expand_as(xx), -1, kk) + self.box_r))
        n = th.where(out, n_out, n_in)
        s = th.where(out, core + self.box_r * n_out, s_in) + c
        sd = th.where(out, dist - self.box_r, th.gather(q, -1, kk) - self.box_r)[..., 0]
        return s, n, sd

    def reset_controller(self):
        th = self.torch
        dev = "cuda"
        f64 = th.float64
        self.dev, self.f64 = dev, f64
        self.c_t = th.tensor(self.box_c, dtype=f64, device=dev)            # (3, 3)
        self.a_t = th.tensor(self.box_a, dtype=f64, device=dev)
        self.lo_t = th.tensor(self.lo, dtype=f64, device=dev)              # (3 fingers, 3 joints)
        self.hi_t = th.tensor(self.hi, dtype=f64, device=dev)
        self.kp_t = th.tensor(self.kp, dtype=f64, device=dev)
        q0 = np.array([[wd_["q_open"][f] for f in FINGERS] for wd_ in self.worlds])
        self.theta_ref = th.tensor(q0, dtype=f64, device=dev)               # (W, 3, 3)
        self.healthy = self.theta_ref.clone()
        self.c1 = None
        pr = self.prm
        self.sw = th.tensor(np.sqrt(W_TASK) * np.array([1.0, 1.0, 1.0] + [1.0 / pr.ell] * 3), dtype=f64, device=dev)
        pats = np.array(np.meshgrid([0, 1, 2], [0, 1, 2], [0, 1, 2], indexing="ij")).reshape(3, -1).T
        self.pats = th.tensor(pats, device=dev)                             # (27, 3): 0 free, 1 lower, 2 upper
        self.n_out = th.tensor(np.array([[wd_["setup"]["n_out"][f] for f in FINGERS] for wd_ in self.worlds]),
                               dtype=f64, device=dev)                       # (W, 3, 3) palm = world axes
        self.r_obj = D_MM / 2000.0

    def views(self):
        to = wp.to_torch
        return {"qpos": to(self.wd.qpos), "qvel": to(self.wd.qvel), "xpos": to(self.wd.xpos), "xmat": to(self.wd.xmat),
                "xanchor": to(self.wd.xanchor), "xaxis": to(self.wd.xaxis), "qfrc_bias": to(self.wd.qfrc_bias),
                "ctrl": to(self.wd.ctrl), "ftip": to(self.ftip), "ftot": to(self.ftot), "cops": to(self.cops),
                "mocap_pos": to(self.wd.mocap_pos)}

    def control(self, V, t, vw):
        """One control period: Eq. (1), Eq. (2), the filter and the impedance for every world and finger. V: the
        object's reference body twist (W, 6). Returns the servo targets (W, 3, 3)."""
        th, f64, pr = self.torch, self.f64, self.prm
        dt = pr.dt
        qpos = vw["qpos"].to(f64)
        theta = qpos[:, th.tensor(self.qadr, device=self.dev)]                         # (W, 3, 3)
        xpos, xmat = vw["xpos"].to(f64), vw["xmat"].to(f64)
        pb, Rb = xpos[:, self.tip], xmat[:, self.tip]                                    # (W, 3, 3), (W, 3, 3, 3)
        o, Ro = xpos[:, self.ob], xmat[:, self.ob]                                       # (W, 3), (W, 3, 3)
        anc = vw["xanchor"].to(f64)[:, th.tensor(self.jid, device=self.dev)]             # (W, 3, 3j, 3)
        axs = vw["xaxis"].to(f64)[:, th.tensor(self.jid, device=self.dev)]
        Fm = vw["ftip"].to(f64)                                                          # (W, 3)
        cop = vw["cops"].to(f64) / Fm.clamp_min(1e-12)[..., None]
        RbT = Rb.transpose(-1, -2)

        def to_tip(p):                                       # world points (W, 3, 3) into each tip frame
            return th.einsum("wfij,wfj->wfi", RbT, p - pb)

        def to_world(p):
            return pb + th.einsum("wfij,wfj->wfi", Rb, p)

        oo = o[:, None, :].expand(-1, 3, -1)
        # GCF from the centre of pressure
        s1c, n1c, _ = self._rb_closest(to_tip(cop))
        p1c = to_world(s1c)
        n1w = th.einsum("wfij,wfj->wfi", Rb, n1c)
        n0c = th.nn.functional.normalize(cop - oo, dim=-1)
        p0c = oo + self.r_obj * n0c
        x1c = th.nn.functional.normalize(n1w - n0c, dim=-1)
        distc = ((p0c - p1c) * x1c).sum(-1)
        # GCF from the witness points (block against sphere)
        s1w, _n1w_, _sd = self._rb_closest(to_tip(oo))
        p1w = to_world(s1w)
        u = th.nn.functional.normalize(p1w - oo, dim=-1)
        p0w = oo + self.r_obj * u
        distw = th.linalg.norm(p1w - oo, dim=-1) - self.r_obj
        x1w = -u
        self.dbg = {"distw": distw, "p1w": p1w, "x1w": x1w, "distc": distc, "Fm": Fm}
        use_cop = (Fm >= pr.contact_force_min)[..., None]
        p1 = th.where(use_cop, p1c, p1w)
        x1 = th.where(use_cop, x1c, x1w)
        dist = th.where(use_cop[..., 0], distc, distw)
        # frame: y is the PIP axis laid into the tangent plane
        ay = axs[:, :, 2]                                                                 # (W, 3, 3) pip axis
        y = ay - (ay * x1).sum(-1, keepdim=True) * x1
        y = th.nn.functional.normalize(y, dim=-1)
        z = th.cross(x1, y, dim=-1)
        R1 = th.stack([x1, y, z], -1)                                                     # columns
        R1T = R1.transpose(-1, -2)
        # Eq. (1)
        w_o = th.einsum("wij,wj->wi", Ro, V[:, :3])
        v_o = th.einsum("wij,wj->wi", Ro, V[:, 3:])
        v_p1 = v_o[:, None] + th.cross(w_o[:, None].expand_as(p1), p1 - oo, dim=-1)
        Jr = axs.transpose(-1, -2)                                                        # (W, 3, 3xyz, 3j)
        Jp = th.cross(axs, p1[:, :, None, :] - anc, dim=-1).transpose(-1, -2)
        A = th.cat([R1T @ Jr, R1T @ Jp], -2)                                               # (W, 3, 6, 3)
        b = -th.cat([th.einsum("wfij,wj->wfi", R1T, w_o), th.einsum("wfij,wfj->wfi", R1T, v_p1)], -1)
        v_ref = th.zeros_like(b)
        if T.T_OPEN <= t < T.T_CLOSE:
            v_ref[..., 3] = T.V_APPROACH
        v_ref[..., 3] = th.where((dist < pr.aw_dist) & (v_ref[..., 3] > 0), th.zeros_like(dist), v_ref[..., 3])
        # Eq. (2): min |sw (A u + b - v_ref)|^2 + w_damp |u|^2 + w_healthy |k (theta_h - theta) - u|^2, box bounds
        As = self.sw[:, None] * A
        rs = self.sw * (v_ref - b)
        I3 = th.eye(3, dtype=f64, device=self.dev)
        Hm = As.transpose(-1, -2) @ As + (pr.w_damp + pr.w_healthy) * I3
        g = th.einsum("wfij,wfi->wfj", As, rs) + pr.w_healthy * pr.k_healthy * (self.healthy - theta)
        lb = th.maximum(th.full_like(theta, -pr.u_max), (self.lo_t - self.theta_ref) / dt)
        ub = th.minimum(th.full_like(theta, pr.u_max), (self.hi_t - self.theta_ref) / dt)
        lb, ub = th.minimum(lb, ub - 1e-12), th.maximum(ub, lb + 1e-12)
        P = self.pats                                                                      # (27, 3)
        fixed = (P > 0)[:, None, None, :]                                                  # (27, 1, 1, 3)
        bound = th.where(P[:, None, None, :] == 1, lb[None], ub[None])                    # (27, W, 3, 3)
        Mm = th.where(fixed[..., None], I3.expand(27, 1, 1, 3, 3) * 1.0, Hm[None].expand(27, -1, -1, -1, -1))
        cc = th.where(fixed, bound, g[None].expand(27, -1, -1, -1))
        uu = th.linalg.solve(Mm, cc[..., None])[..., 0]                                    # (27, W, 3, 3)
        feas = ((uu >= lb[None] - 1e-9) & (uu <= ub[None] + 1e-9)).all(-1)
        obj = 0.5 * th.einsum("pwfi,wfij,pwfj->pwf", uu, Hm, uu) - (uu * g[None]).sum(-1)
        obj = th.where(feas, obj, th.full_like(obj, float("inf")))
        best = obj.argmin(0)                                                               # (W, 3)
        u_sol = th.gather(uu, 0, best[None, ..., None].expand(1, -1, -1, 3))[0]
        u_sol = th.minimum(th.maximum(u_sol, lb), ub)
        tr = self.theta_ref + dt * u_sol
        tr = th.minimum(th.maximum(tr, theta - pr.lead), theta + pr.lead)
        self.theta_ref = th.minimum(th.maximum(tr, self.lo_t), self.hi_t)
        # the filter at k_min: the contact point low-passed on the block's surface
        l1 = to_tip(p1)
        if self.c1 is None:
            self.c1 = self._rb_closest(l1)[0]
        q1, n1, _ = self._rb_closest(self.c1)
        e = l1 - q1
        e_t = e - (e * n1).sum(-1, keepdim=True) * n1
        self.c1 = self._rb_closest(q1 + pr.filter.k_min * dt * e_t)[0]
        s_act, n_act_l, _ = self._rb_closest(self.c1)
        p_act = to_world(s_act)
        n_act = th.einsum("wfij,wfj->wfi", Rb, n_act_l)
        # the impedance of the 'reference' mode, with gravity
        offset = th.zeros_like(theta)
        if t >= T.T_CLOSE:
            F = T.PRM["f_tripod"]
            Ja = th.cross(axs, p_act[:, :, None, :] - anc, dim=-1).transpose(-1, -2)    # (W, 3, 3xyz, 3j)
            gap = (dist - (n_act * th.einsum("wfij,wfj->wfi", Ja, self.theta_ref - theta)).sum(-1)).clamp(0.0, pr.gap_max)
            JJ = Ja @ Ja.transpose(-1, -2) + pr.dls ** 2 * I3
            step = th.linalg.solve(JJ, (gap[..., None] * n_act)[..., None])[..., 0]
            offset = offset + th.einsum("wfij,wfi->wfj", Ja, step) + th.einsum("wfij,wfi->wfj", Ja, F * n_act) / self.kp_t
        if pr.gravity_ff:
            qb = vw["qfrc_bias"].to(f64)[:, th.tensor(self.vadr, device=self.dev)]
            offset = offset + qb / self.kp_t
        return th.minimum(th.maximum(self.theta_ref + offset, self.lo_t), self.hi_t)

    # ---------------------------------------------------------------- the schedule (hom_tasks.Schedule, roll/twist)
    def schedule(self, t):
        """The object's reference angular velocity (world, rad/s) and the phase name at time t."""
        w_ref = np.zeros(3)
        ph = Tsched_phase(t)
        if ph != "task":
            return w_ref, ph
        tt = t - (T.T_LIFT + 0.4)
        if self.task == "roll":
            seq = [(1, +1), (1, -1), (None, 0), (1, -1), (1, +1), (None, 0)]
            durs = [0.5, 0.5, 0.3, 0.5, 0.5, 0.3]
        else:
            seq = [(2, +1), (2, -1), (None, 0)]
            durs = [0.5, 0.5, 0.3]
        edges = np.cumsum([0.0] + durs)
        i = int(np.searchsorted(edges, tt, side="right")) - 1
        if i < len(seq):
            ax, sg = seq[i]
            if ax is not None:
                w_ref[ax] = sg * math.radians(T.PRM["w_step_dps"])
            rep = " 2" if seq[:i].count(seq[i]) else ""
            return w_ref, ("rest" if ax is None else f"w{'xyz'[ax]} {'+' if sg > 0 else '-'}{rep}")
        if self.task == "twist":
            tr = tt - edges[-1]
            if tr < 3.0:
                w_ref[2] = math.radians(T.PRM["ramp_dps"])
                return w_ref, "twist ramp"
        return w_ref, "hold"

    def t_end(self):
        t_task = {"roll": 2 * 1.3, "twist": 1.3 + 4.0}[self.task]
        return T.T_LIFT + 0.4 + t_task + (0.3 if self.task == "roll" else 0.5)

    # ---------------------------------------------------------------- one batch
    def run(self):
        th = self.torch
        self.reset_controller()
        vw = self.views()
        dtc = self.prm.dt
        n = int(round(self.t_end() / dtc))
        W = self.W
        rec_pos = th.zeros((n, W, 3), device=self.dev)
        rec_w = th.zeros((n, W, 3), device=self.dev)
        rec_ft = th.zeros((n, W, 3), device=self.dev)
        rec_fo = th.zeros((n, W, 3), device=self.dev)
        rec_vr = th.zeros((n, W, 3), device=self.dev)
        rec_ax = th.zeros((n, W, 3), device=self.dev)
        phases, wrefs, times = [], [], []
        post_off = False
        t0 = time.perf_counter()
        for k in range(n):
            t = round(k * dtc, 6)
            qpos = vw["qpos"]
            R = vw["xmat"][:, self.ob].to(self.f64)                                      # (W, 3, 3)
            wl = vw["qvel"][:, self.va + 3:self.va + 6].to(self.f64)
            ww = th.einsum("wij,wj->wi", R, wl)
            rec_pos[k] = qpos[:, self.qa:self.qa + 3]
            rec_w[k] = ww.float()
            rec_ft[k] = vw["ftip"]
            rec_fo[k] = vw["ftot"]
            rec_ax[k] = R[:, :, 2].float()
            if t >= T.T_LIFT and not post_off:
                vw["mocap_pos"][:, self.mo_post, 2] -= 1.0                              # the post goes away
                post_off = True
            w_ref, ph = self.schedule(t)
            w_ref_t = th.tensor(w_ref, dtype=self.f64, device=self.dev)
            v_ref = th.zeros((W, 3), dtype=self.f64, device=self.dev)
            if self.task == "roll":
                v_ref = th.cross(w_ref_t.expand(W, 3), -self.r_obj * self.n_out[:, 0], dim=-1)
            rec_vr[k] = v_ref.float()
            RT = R.transpose(-1, -2)
            V = th.cat([th.einsum("wij,j->wi", RT, w_ref_t), th.einsum("wij,wj->wi", RT, v_ref)], -1)
            phases.append(ph)
            wrefs.append(w_ref)
            times.append(t)
            q_cmd = self.control(V, t, vw)
            vw["ctrl"][:, th.tensor(self.act.reshape(-1), device=self.dev)] = q_cmd.reshape(W, 9).float()
            wp.capture_launch(self.graph)
        wp.synchronize()
        wall = time.perf_counter() - t0
        rec = {"pos": rec_pos.cpu().numpy(), "w": rec_w.cpu().numpy(), "ft": rec_ft.cpu().numpy(),
               "fo": rec_fo.cpu().numpy(), "vr": rec_vr.cpu().numpy(), "ax": rec_ax.cpu().numpy()}
        final = {"pos": vw["qpos"][:, self.qa:self.qa + 3].cpu().numpy(), "ftot": vw["ftot"].cpu().numpy()}
        return rec, final, phases, np.array(wrefs), times, wall

    def score(self, rec, final, phases, wrefs, times):
        out = []
        for w, wd_ in enumerate(self.worlds):
            rows = []
            for k, t in enumerate(times):
                rows.append({"t": t, "phase": phases[k], "pos_mm": (1e3 * rec["pos"][k, w]).tolist(),
                             "axis": rec["ax"][k, w].tolist(),
                             "w_dps": np.degrees(rec["w"][k, w]).tolist(),
                             "w_ref_dps": np.degrees(wrefs[k]).tolist(),
                             "v_ref_mms": (1e3 * rec["vr"][k, w]).tolist(),
                             "F": {f: float(rec["ft"][k, w, i]) for i, f in enumerate(FINGERS)},
                             "F_link": {f: float(max(0.0, rec["fo"][k, w, i] - rec["ft"][k, w, i]))
                                        for i, f in enumerate(FINGERS)}})
            se = {"pos": final["pos"][w], "Ftot": {f: float(final["ftot"][w, i]) for i, f in enumerate(FINGERS)}}
            res = T.metrics(self.task, wd_["setup"], rows, {"ok": True}, se)
            out.append(res)
        return out


def Tsched_phase(t):
    if t < T.T_OPEN:
        return "open"
    if t < T.T_CLOSE:
        return "close"
    if t < T.T_LIFT:
        return "squeeze"
    if t < T.T_LIFT + 0.4:
        return "lift"
    return "task"


# ------------------------------------------------------------------------------------------------ commands

def layout_list(kind, nmax):
    """[(key dict, mounts xy, grasp)] for the requested layouts."""
    out = []
    if kind == "2d":
        for xs, ys, kin in C.pick_layouts(0, everyone=True):
            grasp, des = grasp_2d(xs, ys)
            if grasp is None:
                continue
            out.append(({"x_sep_mm": xs, "y_sep_mm": ys, "kin": {k: kin[k] for k in ("tmr_deg", "mft_mNm", "smin_mm")}},
                        mount_xy([-xs / 2, 0.0, xs / 2, ys / 2, xs / 2, -ys / 2]), grasp))
            if nmax and len(out) >= nmax:
                break
    else:
        rows = [json.loads(x) for x in (ROWS / f"{kind}.jsonl").read_text().splitlines() if x.strip()]
        for r in rows:
            if not r.get("ok"):
                continue
            g = grasp_fp(r)
            if g is None:
                continue
            kin = r["fp"]["tripod"]
            out.append(({"d": r["d"], "kin": {k: kin[k] for k in ("tmr_deg", "mft_mNm", "smin_mm")},
                         "kin_pinch": {k: r["fp"]["pinch"][k] for k in ("tmr_deg", "mft_mNm", "smin_mm")}},
                        mount_xy(r["d"]), g))
            if nmax and len(out) >= nmax:
                break
    return out


def cmd_run(a):
    L = Layouts(a.task)
    lays = layout_list(a.layouts, a.max)
    print(f"{len(lays)} layouts with a grasp", flush=True)
    worlds = []
    t0 = time.time()
    for key, mxy, grasp in lays:
        setup, bm, q_open = L.prepare(mxy, grasp)
        for sd in a.seeds:
            worlds.append({"key": key, "seed": sd, "setup": setup, "body_mounts": bm, "q_open": q_open})
    print(f"{len(worlds)} worlds prepared in {time.time() - t0:.1f} s", flush=True)
    path = ROWS / f"{a.out}.jsonl"
    for q in (path, ROWS / f"{a.out}_timing.jsonl"):
        if q.exists():
            q.unlink()
    for b0 in range(0, len(worlds), a.batch):
        chunk = worlds[b0:b0 + a.batch]
        t1 = time.time()
        B = Batch(L, chunk)
        t_build = time.time() - t1
        rec, final, phases, wrefs, times, wall = B.run()
        res = B.score(rec, final, phases, wrefs, times)
        with open(path, "a") as fh:
            for wd_, r in zip(chunk, res):
                row = dict(wd_["key"], task=a.task, seed=wd_["seed"], status="ok", engine="mjwarp", **r)
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        held = sum(bool(r.get("held")) for r in res)
        with open(ROWS / f"{a.out}_timing.jsonl", "a") as fh:
            fh.write(json.dumps({"batch": b0 // a.batch, "worlds": len(chunk), "build_s": round(t_build, 2),
                                 "run_s": round(wall, 2), "task": a.task, "layouts": a.layouts,
                                 "gpu": __import__("torch").cuda.get_device_name(0),
                                 "max_mem_gb": round(__import__("torch").cuda.max_memory_allocated() / 1e9, 2),
                                 "when": time.strftime("%Y-%m-%d %H:%M")}) + "\n")
        print(f"batch {b0 // a.batch}: {len(chunk)} worlds, build {t_build:.1f} s, run {wall:.1f} s "
              f"({wall / len(chunk) * 1000:.0f} ms per rollout), held {held}", flush=True)
        del B
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--task", choices=["twist", "roll"], default="twist")
    p.add_argument("--layouts", default="2d", help="2d, rrt6d or sobol6d")
    p.add_argument("--max", type=int, default=0)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--out", default="hom_gpu")
    sub.add_parser("check-mounts")
    a = ap.parse_args()
    if a.cmd == "check-mounts":
        check_mounts()
        return 0
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
