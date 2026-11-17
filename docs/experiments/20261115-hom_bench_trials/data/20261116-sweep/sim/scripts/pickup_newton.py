#!/usr/bin/env python3
"""The hang of pickup_sim.py --hang in Newton, with hydroelastic contact between the fingertips and the ball (2026-11-16,
for the talk of 2026-11-17).

Scene: pickup_sim's point scene with the real object (scenes/sph040_stem_block_pt_mirror.xml: the reference cell's pt
scene, ball 21.0 g, stem 0.6 g, block 6.5 g; rev_inputs.stem_pt), with the plan's palm (meta.hom.setup.palm_m) and finger
mounts (meta.hom.d6_P_mm) written into the MJCF as SR2Plant.move_mounts places them, and the post 1 m down (no table in
the hang). Newton plant: scripts/contact/steel_hand_compare.py's NewtonBatch on one world, SolverMuJoCo (MuJoCo-Warp,
elliptic cones, impratio 100, 20 iterations, implicitfast), 1 ms physics, friction 1.0 on every shape.
  hydro     the three tip meshes and the ball as hydroelastic SDF shapes (steel_hand_compare.world_builder, plant
            nt_mjc_hydro: kh = E / h = 2.5 MPa / 8.5 mm on the tips, 100 kh on the ball, 0.5 mm voxels, +-6 mm band,
            friction rows at kf 10); the stem and block in point contact (not hydroelastic)
  point     every mesh as its convex hull in Newton's point contact (plant nt_mjc_pt), 0.5 mm contact gap
  *_it100   100 solver iterations, 50 line-search iterations, tolerance 1e-10 (the MuJoCo scenes' options)
  *_imp1    impratio 1
  *_kf2     the hydroelastic shapes' friction-row gain kf 2 (about a 20 ms friction row) in place of 10
  *_ns      MuJoCo's noslip solver after the main one (30 iterations), as pickup_sim's *_ns scenes
Servos: MuJoCo position actuators, kp 0.55 N m/rad (--kp-yaw-im sets the index's and middle's yaw gains, 0.35 in
pickup_sim's plant), joint damping 0.011 N m s/rad, force range 0.18 N m; a plain servo (pickup_sim's plant is
rev_clutch2's servo that the load cannot backdrive). --lock KP TAU (runs *_lock) holds the finger joints at their angles
at the release with a stiff servo (20 N m/rad, 2 N m in the runs of 2026-11-16), set on MuJoCo-Warp's actuators: the
gears that the load cannot backdrive, under constant commands.
Runs of 2026-11-16 (pickup_hang10_newton.json): with the plain servo the pinch at the release is 0.1-0.3 N per tip and
every variant loses the object (with kf 10 the friction rows raise the normal forces to 12-17 N within 25 ms and throw
the ball out; with kf 2 it slides down to hang by its block); with the joints locked, hydro_kf2 holds it at 32.0, 31.9,
31.8 and 31.6 deg 1, 3, 6 and 10 s after the release and point contact swings it between 2 and 42 deg (period 0.56 s).
pickup_plain_servo.py is the MuJoCo control with the plain servo.
Schedule (pickup_sim --hang): the fingers at the plan's touch pose, the ball's centre at P = (-1.66, -17.19, -84.0) mm, the
object tilted 42 deg from vertical about x_P; the targets ramped over 1 s to the grip (thumb and middle PIPs +8 deg) and
held; the object put back at its start pose at rest up to 1.5 s, then free for 10 s. The reset follows every 1 ms physics
step (--hold step, the default); pickup_sim's reset after every 20 ms period (--hold period) lets the stiff hydroelastic
contact kick the ball off the pads within each period (the thumb's mesh is 0.94 mm inside the ball at the start pose), so
that at the release the thumb has no contact (run hydro_hold20ms, 2026-11-16 22:08). Rows every
20 ms: tilt (the object's y axis from world z, deg), z_mm (the ball's centre), pip (thumb, middle; deg), F (the thumb's and
the middle's tip normal force on the object, N), Ftot (all of each finger's bodies), ncon.

    WARP_CACHE_PATH=$(mktemp -d) logs/20261113-steel/venv/bin/python logs/20261116-sim_real_reversal/pickup_newton.py \\
        logs/20261116-sim_real_reversal/pickup_hang10_newton.json --variants hydro point --trace-ms 30 [--lock 20 2]
Each variant's run replaces the run of the same name in the output list, written after each variant. --trace-ms N keeps
every physics step of the first N ms after the release (release_trace: ball centre, velocity, tip forces, centres of
pressure). CUDA graphs are off unless --capture (a capture that began with the ball out of contact failed, error 906).
"""
from __future__ import annotations

import os

os.environ.setdefault("SCALE_HAND", "cad_v2")
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "logs/20261116-sim_real_reversal"
sys.path.insert(0, str(ROOT / "scripts" / "contact"))
import steel_hand_compare as SHC  # noqa: E402  NewtonBatch, world_builder; puts scripts/ and src/ on sys.path
from scipy.spatial.transform import Rotation  # noqa: E402

HG, T = SHC.HG, SHC.T
PLAN = ROOT / "docs/experiments/20261115-hom_bench_trials/plans/20261116-pinch_pickup_thumb_middle_plan.json"
SRC = D / "scenes" / "sph040_stem_block_pt_mirror.xml"
FINGERS, JOINTS = ("thumb", "index", "middle"), ("yaw", "mcp", "pip")
DT = 0.02
SQUEEZE = 8.0
HAND_P_MM = (-1.66, -17.19, -84.0)                   # pickup_sim.HAND_P_MM
HANG = {"tilt": 42.0, "hold_s": 1.5, "held_s": 10.0}  # pickup_sim.HANG
KP, DAMPING, TAU_MAX = 0.55, 0.011, 0.18
IT100 = {"iterations": 100, "ls_iterations": 50, "tolerance": 1e-10}   # the MuJoCo scenes' solver options
VARIANTS = {"hydro": ("nt_mjc_hydro", {}), "point": ("nt_mjc_pt", {}),
            "hydro_ns": ("nt_mjc_hydro", {"noslip_iterations": 30}),
            "point_ns": ("nt_mjc_pt", {"noslip_iterations": 30}),
            "hydro_it100": ("nt_mjc_hydro", IT100), "point_it100": ("nt_mjc_pt", IT100),
            "hydro_it100_ns": ("nt_mjc_hydro", {**IT100, "noslip_iterations": 30}),
            "point_it100_ns": ("nt_mjc_pt", {**IT100, "noslip_iterations": 30}),
            "hydro_it100_imp1": ("nt_mjc_hydro", {**IT100, "impratio": 1.0}),
            "point_it100_imp1": ("nt_mjc_pt", {**IT100, "impratio": 1.0}),
            "hydro_it100_kf2": ("nt_mjc_hydro", IT100, {"kf": 2.0}),
            "hydro_kf2": ("nt_mjc_hydro", {}, {"kf": 2.0}),
            "hydro_it100_kf2_ns": ("nt_mjc_hydro", {**IT100, "noslip_iterations": 30}, {"kf": 2.0}),
            "hydro_it100_imp1_kf2": ("nt_mjc_hydro", {**IT100, "impratio": 1.0}, {"kf": 2.0})}
BUILDER = {}                                       # the running variant's builder settings (kf of the hydroelastic shapes)
NONHYDRO = ("obj_stem", "obj_block")


# ------------------------------------------------------------------------------------------------ scene
def plan_inputs():
    plan = json.loads(PLAN.read_text())
    h = plan["meta"]["hom"]
    poses = {p["name"]: {f: np.radians([p["joints"][f][j] for j in JOINTS]) for f in FINGERS} for p in plan["poses"]}
    d6 = h["d6_P_mm"]
    mxy = {"thumb": (d6[0] / 1e3, d6[1] / 1e3), "index": (d6[2] / 1e3, d6[3] / 1e3), "middle": (d6[4] / 1e3, d6[5] / 1e3)}
    palm = np.asarray(h["setup"]["palm_m"], float)
    return poses, mxy, palm


def scene_xml(mxy, palm, kp_yaw_im=None):
    """The stem scene with the servo plant, the plan's palm and mounts, and the post out of reach."""
    import mujoco
    x = SRC.read_text()
    old_j = '<joint type="hinge" damping="0.0144" armature="0.001" />'
    old_p = '<position kp="0.72" kv="0" gear="1" forcerange="-0.18 0.18" ctrlrange="-1 1" />'
    if x.count(old_j) != 1 or x.count(old_p) != 1:
        raise RuntimeError("the ctrl class of the stem scene changed")
    x = x.replace(old_j, f'<joint type="hinge" damping="{DAMPING:g}" armature="0.001" />')
    x = x.replace(old_p, f'<position kp="{KP:g}" kv="0" gear="1" forcerange="-{TAU_MAX:g} {TAU_MAX:g}" ctrlrange="-1 1" />')
    if kp_yaw_im:
        for f in ("index", "middle"):
            old = f'name="a_{f}_yaw" joint="{f}_yaw"'
            if x.count(old) != 1:
                raise RuntimeError(f"actuator a_{f}_yaw")
            x = x.replace(old, f'{old} kp="{kp_yaw_im:g}"')
    # the mounts as SR2Plant.move_mounts and hom_gpu.Scene.body_mounts place them: compiled position less the reference
    # layout's, plus the layout's
    m0 = mujoco.MjModel.from_xml_string(x)
    ref = HG.mount_xy(HG.sym6(*HG.REF))
    pos = {"palm_pose": palm}
    for f in FINGERS:
        pos[f"{f}_mount"] = m0.body_pos[m0.body(f"{f}_mount").id] - np.array([*ref[f], 0.0]) \
            + np.array([mxy[f][0], mxy[f][1], 0.0])
    pos["post"] = np.array([0.0, 0.0, -1.0])
    root = ET.fromstring(x)
    for b in root.iter("body"):
        n = b.get("name")
        if n in pos:
            b.set("pos", " ".join(f"{v:.9g}" for v in pos[n]))
    xml = ET.tostring(root, encoding="unicode")
    m1 = mujoco.MjModel.from_xml_string(xml)
    for n, v in pos.items():
        if not np.allclose(m1.body_pos[m1.body(n).id], v, atol=1e-9):
            raise RuntimeError(f"body {n} not placed")
    return xml, {n: [round(float(u), 6) for u in v] for n, v in pos.items()}


def mj_reference(xml, q0, c, quat_wxyz):
    """MuJoCo's compiled object mass properties and the tip frames at the start pose, for the Newton model's checks."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    ob = m.body(T.S.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[ob]]
    d.qpos[qa:qa + 3], d.qpos[qa + 3:qa + 7] = c, quat_wxyz
    for f in FINGERS:
        for k, j in enumerate(JOINTS):
            d.qpos[m.jnt_qposadr[m.joint(f"{f}_{j}").id]] = q0[f][k]
    mujoco.mj_forward(m, d)
    iq = m.body_iquat[ob]
    R = Rotation.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()
    return {"mass": float(m.body_mass[ob]), "com": m.body_ipos[ob].copy(), "inertia": R @ np.diag(m.body_inertia[ob]) @ R.T,
            "tips": {f: d.xpos[m.body(f"{f}_tip").id].copy() for f in FINGERS},
            "kp": {a: float(m.actuator_gainprm[m.actuator(f"a_{a}").id, 0]) for a in
                   (f"{f}_{j}" for f in FINGERS for j in JOINTS)},
            "ctrlrange": {f"{f}_{j}": m.actuator_ctrlrange[m.actuator(f"a_{f}_{j}").id].copy()
                          for f in FINGERS for j in JOINTS}}


_WB = SHC.world_builder


def world_builder(newton, xml, plant, pairs="obj"):
    """steel_hand_compare.world_builder; then the stem and block out of the hydroelastic set (point contact, the 0.5 mm
    gap), and the object's mass properties checked against MuJoCo's compile of the same MJCF (set from it if they
    differ)."""
    import mujoco
    import warp as wp
    b, info = _WB(newton, xml, plant, pairs)
    lab = [x.split("/")[-1] for x in b.shape_label]
    for n in NONHYDRO:
        i = lab.index(n)
        b.shape_flags[i] &= ~int(newton.ShapeFlags.HYDROELASTIC)
        b.shape_gap[i] = SHC.GAP
    if "kf" in BUILDER and "hydro" in plant:       # SolverMuJoCo's friction-row gain on the hydroelastic shapes
        for i in range(len(lab)):
            if b.shape_flags[i] & int(newton.ShapeFlags.HYDROELASTIC):
                b.shape_material_kf[i] = float(BUILDER["kf"])
    m = mujoco.MjModel.from_xml_string(xml)
    ob = m.body(T.S.OBJ).id
    iq = m.body_iquat[ob]
    R = Rotation.from_quat([iq[1], iq[2], iq[3], iq[0]]).as_matrix()
    I_mj = R @ np.diag(m.body_inertia[ob]) @ R.T
    k = info["obj_body"]
    mass_nt, com_nt = float(b.body_mass[k]), np.array(b.body_com[k], float).reshape(3)
    I_nt = np.array(b.body_inertia[k], float).reshape(3, 3)
    info["mass_check"] = {"newton_mass": mass_nt, "mujoco_mass": float(m.body_mass[ob]),
                          "newton_com": com_nt.round(6).tolist(), "mujoco_com": m.body_ipos[ob].round(6).tolist(),
                          "inertia_rel_err": float(np.abs(I_nt - I_mj).max() / np.abs(I_mj).max())}
    if abs(mass_nt - m.body_mass[ob]) > 1e-3 * m.body_mass[ob] or np.abs(com_nt - m.body_ipos[ob]).max() > 1e-5 \
            or info["mass_check"]["inertia_rel_err"] > 1e-2:
        b.body_mass[k], b.body_inv_mass[k] = float(m.body_mass[ob]), 1.0 / float(m.body_mass[ob])
        b.body_com[k] = wp.vec3(*map(float, m.body_ipos[ob]))
        b.body_inertia[k] = wp.mat33(*map(float, I_mj.reshape(-1)))
        b.body_inv_inertia[k] = wp.mat33(*map(float, np.linalg.inv(I_mj).reshape(-1)))
        info["mass_check"]["set_from_mujoco"] = True
    return b, info


# ------------------------------------------------------------------------------------------------ run
def tilt_deg(q_xyzw):
    R = Rotation.from_quat(np.asarray(q_xyzw, float)).as_matrix()
    return math.degrees(math.acos(float(np.clip(R[2, 1], -1.0, 1.0))))


def run(variant, kp_yaw_im=None, hold="step", trace_ms=0, capture=False, lock=None):
    import newton
    import warp as wp
    plant, extra, *bx = VARIANTS[variant]
    BUILDER.clear()
    BUILDER.update(bx[0] if bx else {})
    name = variant + (f"_yaw{kp_yaw_im:g}" if kp_yaw_im else "") + ("_hold20ms" if hold == "period" else "") \
        + ("_lock" if lock else "")
    poses, mxy, palm = plan_inputs()
    xml, placed = scene_xml(mxy, palm, kp_yaw_im)
    c = palm + np.asarray(HAND_P_MM) / 1e3
    th = math.radians(HANG["tilt"])
    a = math.pi / 2 - th
    quat = np.array([math.cos(a / 2), math.sin(a / 2), 0.0, 0.0])                    # wxyz, about x
    ref = mj_reference(xml, poses["touch"], c, quat)
    out = {"variant": name, "scene": SRC.name, "simulator": "newton", "plant": plant, "rows": [],
           "placed_bodies": placed, "solver_extra": extra, "builder_extra": dict(BUILDER), "kp_yaw_im": kp_yaw_im,
           "hold": {"step": "per_step", "period": "per_period"}[hold], "release_trace": [], "cuda_graphs": capture}
    t_build = time.time()
    base_solver = newton.solvers.SolverMuJoCo
    SHC.world_builder = world_builder
    if extra:
        class Solver(base_solver):
            def __init__(self, model, **kw):
                super().__init__(model, **{**kw, **extra})
        newton.solvers.SolverMuJoCo = Solver
    try:
        B = SHC.NewtonBatch([xml], plant, capture=capture)
    finally:
        newton.solvers.SolverMuJoCo = base_solver
        SHC.world_builder = _WB
    out["build_s"] = round(time.time() - t_build, 1)
    out["mass_check"] = B.info["mass_check"]
    out["solver_opts"] = {k: str(v) for k, v in B.solver_opts.items()}
    out["versions"] = {"newton": newton.__version__, "warp": wp.__version__}
    fq = B.free_q
    if fq != 0:
        raise RuntimeError("the object's free joint is not the first joint")
    # servo check: the MuJoCo model inside the solver
    mjm = B.solver.mj_model
    acts = {mjm.actuator(i).name: i for i in range(mjm.nu)}
    out["servo_check"] = {n: {"kp": round(float(mjm.actuator_gainprm[i, 0]), 4),
                              "bias": [round(float(v), 4) for v in mjm.actuator_biasprm[i, :3]],
                              "forcerange": [round(float(v), 4) for v in mjm.actuator_forcerange[i]],
                              "damping": round(float(mjm.dof_damping[mjm.jnt_dofadr[mjm.actuator_trnid[i, 0]]]), 5)}
                          for n, i in acts.items()}
    # the start state: fingers at the touch pose, at rest, the object at (c, quat)
    lo = {f: np.array([ref["ctrlrange"][f"{f}_{j}"][0] for j in JOINTS]) for f in FINGERS}
    hi = {f: np.array([ref["ctrlrange"][f"{f}_{j}"][1] for j in JOINTS]) for f in FINGERS}

    def set_targets(q_cmd):
        for f in FINGERS:
            qc = np.clip(q_cmd[f], lo[f], hi[f])
            for k, j in enumerate(JOINTS):
                B.tgt[0, B.dof[(f, j)][0]] = float(qc[k])

    for f in FINGERS:
        for k, j in enumerate(JOINTS):
            qa_, da_ = B.dof[(f, j)]
            B.q[0, qa_] = poses["touch"][f][k]
    B.q[0, fq:fq + 3] = c
    B.q[0, fq + 3:fq + 7] = [quat[1], quat[2], quat[3], quat[0]]                    # Newton: xyzw
    B.qd[0] = 0.0
    B.dirty_q = True
    set_targets(poses["touch"])
    obj0 = B.q[0, fq:fq + 7].copy()
    B.push()
    bq = B.s0.body_q.numpy()
    blab = B.info["blab"]
    out["fk_check_mm"] = {f: round(1e3 * float(np.abs(bq[blab.index(f"{f}_tip"), :3] - ref["tips"][f]).max()), 4)
                          for f in FINGERS}
    ob = B.obj_body
    out["start"] = {"c_mm": (1e3 * c).round(3).tolist(), "quat_wxyz": quat.round(6).tolist(),
                    "tilt": round(tilt_deg(bq[ob, 3:7]), 3)}

    def squeezed(sq):
        q = {f: poses["touch"][f].copy() for f in FINGERS}
        for f in ("thumb", "middle"):
            q[f][2] += math.radians(sq)
        return q

    t = 0.0
    t_run = time.time()

    def hold_reset():
        B.q[0, fq:fq + 7] = obj0
        B.qd[0, 0:6] = 0.0
        B.dirty_q = True

    def row_trace():
        """The state after one physics step: ball centre (mm), its velocity (mm/s, deg/s), each tip's normal force and
        centre of pressure relative to the ball's centre (mm)."""
        o = B.q[0, fq:fq + 7]
        return {"t": round(B.time, 4), "c_mm": (1e3 * o[:3]).round(3).tolist(), "tilt": round(tilt_deg(o[3:7]), 3),
                "qd": [round(float(v), 4) for v in B.qd[0, 0:6]], "ncon": int(B.ncon),
                "F": {f: round(float(B.cop[0][f][1]), 4) for f in B.cop[0]},
                "cop_mm": {f: (1e3 * (np.asarray(B.cop[0][f][0]) - o[:3])).round(3).tolist() for f in B.cop[0]},
                "pip": [round(math.degrees(float(B.q[0, B.dof[(f, "pip")][0]])), 3) for f in ("thumb", "middle")]}

    locked = []

    def lock_servos():
        """The finger servos replaced at the release by a stiff hold of the joints' angles then (gears that the load
        cannot backdrive, pickup_sim's rev_clutch2 plant): MuJoCo-Warp's actuator gain and bias set to lock[0] N m/rad and
        the force range to +-lock[1] N m, the targets to the angles at the release."""
        mm, mj = B.solver.mjw_model, B.solver.mj_model
        g, bp, fr = mm.actuator_gainprm.numpy(), mm.actuator_biasprm.numpy(), mm.actuator_forcerange.numpy()
        names = []
        for i in range(mj.nu):
            jn = mj.joint(int(mj.actuator_trnid[i, 0])).name
            if any(jn.endswith(f"{f}_{j}") for f in FINGERS for j in JOINTS):
                g[..., i, 0], bp[..., i, 1], bp[..., i, 2] = lock[0], -lock[0], 0.0
                fr[..., i, 0], fr[..., i, 1] = -lock[1], lock[1]
                names.append(jn)
        mm.actuator_gainprm.assign(g)
        mm.actuator_biasprm.assign(bp)
        mm.actuator_forcerange.assign(fr)
        qlock = {}
        for f in FINGERS:
            for j in JOINTS:
                qa_ = B.dof[(f, j)][0]
                B.tgt[0, qa_] = float(B.q[0, qa_])
                qlock[f"{f}_{j}"] = round(math.degrees(float(B.q[0, qa_])), 3)
        locked.append(True)
        out["lock"] = {"kp": lock[0], "tau_max": lock[1], "joints": names, "q_deg": qlock}

    def period(phase, q_cmd):
        nonlocal t
        if lock and phase == "held" and not locked:
            lock_servos()
        if not locked:
            set_targets(q_cmd)
        t1 = round(t + DT, 6)
        if phase == "held" and trace_ms and len(out["release_trace"]) < trace_ms:
            out["release_trace"].append(row_trace())            # the released state, then each step
            n_ = int(round(DT / SHC.PHYS_DT))
            for k_ in range(1, n_ + 1):
                B.step_to(round(t + k_ * SHC.PHYS_DT, 6))
                out["release_trace"].append(row_trace())
            t = t1
        elif hold == "step" and t1 <= HANG["hold_s"] + 1e-9:
            # the object pinned: back at its start pose, at rest, after every physics step
            n_ = int(round(DT / SHC.PHYS_DT))
            for k_ in range(1, n_ + 1):
                B.step_to(round(t + k_ * SHC.PHYS_DT, 6))
                if not np.isfinite(B.q[0]).all():
                    raise FloatingPointError(f"non-finite state at {B.time:.3f} s")
                hold_reset()
            t = t1
        else:
            t = t1
            B.step_to(t)
            if not np.isfinite(B.q[0]).all():
                raise FloatingPointError(f"non-finite state at {t:.3f} s")
            if t <= HANG["hold_s"] + 1e-9:                 # SR2Plant.advance's hold: back at the start pose, at rest
                hold_reset()
        q = B.q[0]
        o = q[fq:fq + 7]
        cop = B.cop[0]
        out["rows"].append({"t": round(t, 3), "phase": phase, "drop_mm": 0.0, "squeeze": SQUEEZE,
                            "tilt": round(tilt_deg(o[3:7]), 2), "z_mm": round(1e3 * float(o[2]), 2),
                            "pip": [round(math.degrees(float(q[B.dof[(f, "pip")][0]])), 2) for f in ("thumb", "middle")],
                            "F": [round(float(cop[f][1]) if f in cop else 0.0, 3) for f in ("thumb", "middle")],
                            "Ftot": [round(float(B.Ftot[0][f]), 3) for f in FINGERS], "ncon": int(B.ncon),
                            "cop_mm": {f: (1e3 * (np.asarray(cop[f][0]) - o[:3])).round(2).tolist() for f in cop}})

    try:
        n = int(round(1.0 / DT))
        for k in range(1, n + 1):
            u = k / n
            period("squeeze (held)", {f: (1 - u) * poses["touch"][f] + u * squeezed(SQUEEZE)[f] for f in FINGERS})
        while t < HANG["hold_s"] - 1e-9:
            period("squeeze (held)", squeezed(SQUEEZE))
        for _ in range(int(round(HANG["held_s"] / DT))):
            period("held", squeezed(SQUEEZE))
    except Exception as e:                                                        # noqa: BLE001  a diverged state
        out["error"] = repr(e)[:300]
        out["tb"] = traceback.format_exc()[-1500:]
    wp.synchronize()
    if locked:                                     # the gains as MuJoCo-Warp holds them at the end (not reset by Newton)
        gl = B.solver.mjw_model.actuator_gainprm.numpy()[..., 0].reshape(-1)
        out["lock"]["gains_end"] = sorted({round(float(v), 4) for v in gl})
    out["run_s"] = round(time.time() - t_run, 1)
    out["sim_s"] = round(t, 3)
    rows = out["rows"]
    held = [x for x in rows if x["phase"] == "held"]
    rel = next((x for x in reversed(rows) if x["phase"] == "squeeze (held)"), None)
    z0 = 1e3 * float(c[2])
    out["summary"] = {
        "tilt_start": rows[0]["tilt"] if rows else None,
        "tilt_held": held[-1]["tilt"] if held else None,
        "held_at": {f"{k:g}": next((x["tilt"] for x in held if x["t"] >= HANG["hold_s"] + k - 1e-6), None)
                    for k in (1, 3, 6, 10)},
        "F_release": rel["F"] if rel else None, "Ftot_release": rel["Ftot"] if rel else None,
        "F_release_mean100ms": [round(float(np.mean([x["F"][k] for x in rows if x["phase"] == "squeeze (held)"][-5:])), 3)
                                for k in (0, 1)] if rel else None,
        "both_in_contact_last100ms": sum(1 for x in [x for x in rows if x["phase"] == "squeeze (held)"][-5:]
                                         if x["F"][0] > 0 and x["F"][1] > 0),
        "z_drop_mm_max": round(z0 - min(x["z_mm"] for x in held), 2) if held else None,
        "z_drop_mm_end": round(z0 - held[-1]["z_mm"], 2) if held else None,
        "dropped": bool(held and held[-1]["z_mm"] < z0 - 40.0)}
    del B
    return out


def save(path, run_):
    path = Path(path)
    res = json.loads(path.read_text()) if path.exists() else []
    res = [r for r in res if r.get("variant") != run_["variant"]] + [run_]
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(res) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default=str(D / "pickup_hang10_newton.json"))
    ap.add_argument("--variants", nargs="+", default=["hydro", "point"], choices=list(VARIANTS))
    ap.add_argument("--kp-yaw-im", type=float, default=None,
                    help="the index's and middle's yaw servo gain, N m/rad (pickup_sim's plant: 0.35)")
    ap.add_argument("--hold", choices=["step", "period"], default="step",
                    help="step: the object reset after every 1 ms physics step up to hold_s (default); period: after "
                         "every 20 ms control period, as SR2Plant.advance (the stiff hydroelastic contact kicks the ball "
                         "off the pads within each period)")
    ap.add_argument("--trace-ms", type=int, default=0,
                    help="every physics step of the first N ms after the release in the run's release_trace")
    ap.add_argument("--capture", action="store_true", help="CUDA graphs of 10 steps (a capture that starts with the "
                    "ball out of contact failed with CUDA error 906, 2026-11-16 22:15)")
    ap.add_argument("--lock", type=float, nargs=2, default=None, metavar=("KP", "TAU"),
                    help="at the release, the finger joints held at their angles by a stiff servo (KP N m/rad, +-TAU N m): "
                         "the gears that the load cannot backdrive (pickup_sim's plant); e.g. --lock 20 2")
    a = ap.parse_args()
    for v in a.variants:
        t0 = time.time()
        try:
            r = run(v, a.kp_yaw_im, a.hold, a.trace_ms, a.capture, a.lock)
        except Exception as e:                                                    # noqa: BLE001
            r = {"variant": v + (f"_yaw{a.kp_yaw_im:g}" if a.kp_yaw_im else ""), "error": repr(e)[:300],
                 "tb": traceback.format_exc()[-2000:], "rows": [], "summary": {}}
        r["wall_s"] = round(time.time() - t0, 1)
        save(a.out, r)
        s = r.get("summary", {})
        print(f"{r['variant']:16s} held_at {s.get('held_at')} F_release {s.get('F_release')} z drop max "
              f"{s.get('z_drop_mm_max')} end {s.get('z_drop_mm_end')} mm | build {r.get('build_s')} s run {r.get('run_s')} s"
              f" wall {r['wall_s']} s" + (f" | ERROR {r['error']}" if r.get("error") else ""), flush=True)
        for k in ("mass_check", "fk_check_mm", "start"):
            if k in r:
                print(f"   {k}: {json.dumps(r[k])}", flush=True)
        if "servo_check" in r:
            print("   servo: " + "; ".join(f"{n} kp {x['kp']} bias {x['bias']} fr {x['forcerange']} d {x['damping']}"
                                       for n, x in r["servo_check"].items()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
