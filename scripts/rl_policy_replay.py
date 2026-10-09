#!/usr/bin/env python3
"""Open-loop replay of a trained reorientation policy's finger targets in other simulators (2026-10-06).

The RL contact comparison (scripts/rl_contact_train_queue.sh) trains on MuJoCo-Warp. This replays what a final policy
commanded, from the state at its reorientation onset, in CPU MuJoCo (1 mm pads or the TPU block mesh), Drake (the TPU
block as a compliant hydroelastic convex, reorient_backends.DrakeBench) and Newton (hydroelastic TPU block with kh
divided by the tip-tool effective mass, as newton_turn.py --mass-correct).

  record   (uv environment, GPU) the run's env with the deterministic policy, N worlds: at policy step ONSET (58,
           the reorientation start; the palm has finished its lift) the scene is written (mjlab Scene.write) and each
           world's state saved relative to its grid origin; the finger servo targets of every later policy step and
           the tool's pose are recorded. Out: logs/20261006-rl_contact/replay/<tag>/{scene.xml, rec.npz}.
  bench    turn that scene into a bench-like one: prefixes stripped, the tool renamed screwdriver_medium, the palm's
           six joints removed and the palm body placed at its pose at the onset (it does not move after the lift),
           sensors and the mocap flag dropped. Out: bench.xml and bench_meta.json (plant, mu, TPU meshes, the onset
           finger angles and tool pose) next to the scene.
  replay   step the bench-like scene from the onset state through the recorded targets (one per 20 ms) in one
           engine, and report the tool's final cosine with vertical, whether it is held (above 60 mm, >= 2 fingers
           in contact) and the per-finger pad force, against the recorded MuJoCo-Warp outcome.

    uv run --extra rl --extra gpu python scripts/rl_policy_replay.py record --tag <run tag>
    PY=logs/20261001-hom_contact/venv/bin/python; $PY scripts/rl_policy_replay.py bench --dir <replay dir>
    $PY scripts/rl_policy_replay.py replay --dir <dir> --engine mujoco|drake [--hold 1.0]
    logs/20261004-contact-transfer/venv/bin/python scripts/rl_policy_replay.py replay --dir <dir> --engine newton
Rows: docs/experiments/20261006-rl_contact/policy_replay.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "docs/experiments/20261006-rl_contact/policy_replay.jsonl"
REPLAY = ROOT / "logs/20261006-rl_contact/replay"
# --study 20261008: the contact-model policy study (scripts/rl_contact_train_queue_20261008.sh)
STUDIES = {"20261008": (ROOT / "docs/experiments/20261008-contact_model_policies/policy_replay.jsonl",
                        ROOT / "logs/20261008-contact_model_policies/replay")}


def block_dir(d: Path) -> Path:
    """The replay dir of the TPU block mesh run of the same seed, whose bench.xml gives Drake and Newton the plain
    block geometry (pad and skin runs have sphere tips, the box run the legacy box). A seed whose mesh run was stopped
    before its final checkpoint takes seed 0's: every run lifts the palm to the same pose (to 1e-7 m) and the mesh
    scene is the same file for all seeds."""
    n = d.name
    for a, b in (("pads1", "mesh"), ("tpu27skin", "tpu27mesh"), ("_box_", "_tpu27mesh_")):
        n = n.replace(a, b)
    bd = d.with_name(n)
    if bd != d and "_40M_s" in n and not (ROOT / "results/rl" / n / "tensorboard" / "model_812.pt").exists():
        bd = d.with_name(re.sub(r"_s\d+$", "_s0", n))
    return bd
FINGERS, JOINTS = ("thumb", "index", "middle"), ("yaw", "mcp", "pip")
NAMES = [f"{f}_{j}" for f in FINGERS for j in JOINTS]
TOOL = "screwdriver_medium"
PALM_JOINTS = ("palm_px", "palm_py", "palm_pz", "palm_rx", "palm_ry", "palm_rz")
ONSET, STEPS, DT_POLICY = 58, 250, 0.02
# the trainer's solver options (env_build MujocoCfg); mjlab applies them to the compiled model, not to the spec its
# Scene.write exports, so an exported scene carries MuJoCo's defaults (pyramidal cone, impratio 1, Euler) without them
TRAINER_OPT = dict(timestep="0.002", iterations="10", ls_iterations="20", tolerance="1e-08", impratio="10",
                   cone="elliptic", integrator="implicitfast", gravity="0 0 -9.81")
NEWTON_DIR = [None]
NEWTON_NUM = [None]


# ---------------------------------------------------------------------------------- record (GPU)

def record(tag: str, ckpt: str | None, n: int = 16):
    """Deterministic rollout of a run's checkpoint in its own env; scene + onset state + later finger targets."""
    import mujoco
    import torch
    from morphohand.rl.deploy import act_b, build_actor, finger_ctrl_from_keyframe, make_env_cfg, run_env_overrides
    rd = ROOT / "results/rl" / tag
    if ckpt is None:
        ckpt = max((rd / "tensorboard").glob("model_*.pt"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    ckpt = Path(ckpt)
    trained = run_env_overrides(ckpt)
    import yaml
    morph = Path(yaml.safe_load(open(rd / "config.yaml"))["env"]["foundational_run_dir"])
    frozen = morph / "frozen_scene.xml"
    summ = json.loads((morph / "summary.json").read_text())
    bfc = finger_ctrl_from_keyframe(frozen, "open_ik")
    cfg = make_env_cfg(frozen, summ["keyframe"], morph, bfc, enable_target_axis=True, num_steps=STEPS,
                       finger_residual_scale=0.5, lift_delta=0.1, open_finger_from_keyframe=True, num_envs=n,
                       finger_residual_active_from_step=int(trained.get("finger_residual_active_from_step", ONSET)),
                       reorient_start_step=int(trained.get("reorient_start_step", ONSET)),
                       lift_phase_start_step=trained.get("lift_phase_start_step"))
    out = REPLAY / tag
    out.mkdir(parents=True, exist_ok=True)
    env, wrapped, actor = build_actor(cfg, ckpt, out / "tmp")
    mjm = env.unwrapped.sim.mj_model
    wd = env.unwrapped.sim.wp_data
    acts = [mjm.joint(int(mjm.actuator_trnid[a, 0])).name.split("/")[-1] for a in range(mjm.nu)]
    fa = [acts.index(nm) for nm in NAMES]
    names = [mjm.body(i).name for i in range(mjm.nbody)]
    tool = next(i for i, x in enumerate(names) if x.split("/")[-1] == "cube")
    root_b = next(i for i in range(1, mjm.nbody) if mjm.body_parentid[i] == 0 and names[i].startswith("robot/"))
    obs_td, _ = wrapped.reset()
    rec = {"finger_targets": [], "tool_cos": [], "tool_z": [], "qpos_t": []}
    free_q = []
    with torch.no_grad():
        for k in range(STEPS):
            if k == ONSET:
                env.unwrapped.scene.write(out)
                qpos = wd.qpos.numpy().copy()
                d0 = mujoco.MjData(mujoco.MjModel.from_xml_path(str(out / "scene.xml")))
                m0 = d0.model if hasattr(d0, "model") else mujoco.MjModel.from_xml_path(str(out / "scene.xml"))
                off = np.zeros((n, 3))
                for w in range(n):
                    d0.qpos[:] = qpos[w]
                    mujoco.mj_kinematics(m0, d0)
                    off[w] = wd.xpos.numpy()[w, root_b] - d0.xpos[root_b]
                for j in range(m0.njnt):
                    if m0.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                        a = m0.jnt_qposadr[j]
                        qpos[:, a:a + 3] -= off
                        free_q.append(a)
                rec["qpos"] = qpos
                rec["env_offset"] = off
            obs_td, *_ = wrapped.step(act_b(actor, obs_td, False))
            if k >= ONSET:
                rec["finger_targets"].append(wd.ctrl.numpy()[:, fa].copy())
                xm = wd.xmat.numpy()[:, tool].reshape(n, 3, 3)
                rec["tool_cos"].append(xm[:, 2, 2].copy())
                rec["tool_z"].append(wd.xpos.numpy()[:, tool, 2].copy())
                q = wd.qpos.numpy().copy()          # every world's state after the step, grid offset removed (films)
                for a in free_q:
                    q[:, a:a + 3] -= rec["env_offset"]
                rec["qpos_t"].append(q.astype(np.float32))
    np.savez(out / "rec.npz", qpos=rec["qpos"], env_offset=rec["env_offset"],
             finger_targets=np.stack(rec["finger_targets"], axis=1), tool_cos=np.stack(rec["tool_cos"], axis=1),
             tool_z=np.stack(rec["tool_z"], axis=1), qpos_t=np.stack(rec["qpos_t"], axis=1))
    env.close()
    cz = np.stack(rec["tool_z"], axis=1)[:, -1]
    cc = np.stack(rec["tool_cos"], axis=1)[:, -1]
    print(f"{tag}: recorded {n} worlds from step {ONSET}; MuJoCo-Warp final cos {cc.mean():+.3f}, tool above 60 mm "
          f"in {(cz > 0.06).sum()}/{n}; checkpoint {ckpt.name}", flush=True)
    return {"tag": tag, "checkpoint": str(ckpt.relative_to(ROOT)), "n": n, "mjw_cos_end": cc.tolist(),
            "mjw_z_end_mm": (1e3 * cz).tolist()}


# ---------------------------------------------------------------------------------- bench-like scene

def bench(d: Path, world: int = 0, plant: str = "kp4_kv0_fr1_fl0_dp0.08", mu: float = 1.0):
    """bench.xml + bench_meta.json from d/scene.xml and the onset state of `world` in d/rec.npz (or d/state.npz)."""
    import mujoco
    st = dict(np.load(d / ("rec.npz" if (d / "rec.npz").exists() else "state.npz")))
    qpos = st["qpos"][world] if st["qpos"].ndim == 2 else st["qpos"]
    src = (d / "scene.xml").read_text()
    m0 = mujoco.MjModel.from_xml_string(src)
    d0 = mujoco.MjData(m0)
    d0.qpos[:] = qpos
    mujoco.mj_kinematics(m0, d0)
    palm = m0.body("robot/palm_pose").id
    palm_pos, palm_quat = d0.xpos[palm].copy(), d0.xquat[palm].copy()
    tool_b = next(i for i in range(m0.nbody) if m0.body(i).name.split("/")[-1] == "cube")
    tool7 = np.r_[d0.xpos[tool_b], d0.xquat[tool_b]]
    q0 = {n: float(qpos[m0.jnt_qposadr[m0.joint("robot/" + n).id]]) for n in NAMES}
    root = ET.fromstring(src)
    # bodies, joints, geoms, sites, actuators and their references lose the entity prefix; assets (meshes, textures,
    # materials) and default classes keep it, since the two entities define some under the same names
    act_tags = {"position", "motor", "general", "velocity"}
    for el in root.iter():
        keys = ()
        if el.tag in ("body", "joint", "freejoint", "geom", "site"):
            keys = ("name",)
        elif el.tag in act_tags:
            keys = ("name", "joint")
        elif el.tag in ("exclude", "pair"):
            keys = ("body1", "body2", "geom1", "geom2")
        for k in keys:
            v = el.get(k)
            if v and re.match(r"(robot|cube)/", v):
                el.set(k, v.split("/", 1)[1])
    for b in root.iter("body"):
        if b.get("name") == "cube":
            b.set("name", TOOL)
        if b.get("mocap") == "true":
            del b.attrib["mocap"]
        if b.get("name") == "palm_pose":
            for j in list(b.findall("joint")):
                if j.get("name") in PALM_JOINTS:
                    b.remove(j)
            b.set("pos", " ".join(f"{v:.9g}" for v in palm_pos))
            b.set("quat", " ".join(f"{v:.9g}" for v in palm_quat))
    for j in root.iter("joint"):
        if j.get("name") == "cube_joint":
            j.set("name", f"{TOOL}_joint")
    for act in root.findall("actuator"):
        for a in list(act):
            if a.get("joint") in PALM_JOINTS:
                act.remove(a)
    for tag in ("sensor", "keyframe"):
        for el in root.findall(tag):
            root.remove(el)
    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(0, opt)
    opt.attrib.update(TRAINER_OPT)
    meshes = {}
    for ms in root.iter("mesh"):
        if ms.get("name", "").split("/")[-1] in ("tpu_pos", "tpu_neg"):
            sgn = 1.0 if ms.get("name").endswith("tpu_pos") else -1.0
            meshes[str(sgn)] = meshes[str(int(sgn))] = ms.get("file")
    (d / "bench.xml").write_text(ET.tostring(root, encoding="unicode"))
    m1 = mujoco.MjModel.from_xml_path(str(d / "bench.xml"))
    assert m1.nq == m0.nq - len(PALM_JOINTS) and m1.nu == m0.nu - len(PALM_JOINTS)
    meta = {"hand": "D6", "plant": plant, "mu": mu, "meshes": meshes, "q0": q0, "tool7": tool7.tolist(),
            "palm_pos": palm_pos.tolist(), "palm_quat": palm_quat.tolist(), "world": world}
    (d / "bench_meta.json").write_text(json.dumps(meta, indent=1))
    # check: the same contacts on CPU at the onset state
    d1 = mujoco.MjData(m1)
    t = m1.body(TOOL).id
    a = m1.jnt_qposadr[m1.body_jntadr[t]]
    d1.qpos[a:a + 7] = tool7
    for n, v in q0.items():
        d1.qpos[m1.jnt_qposadr[m1.joint(n).id]] = v
    mujoco.mj_forward(m1, d1)
    d0.qpos[:] = qpos
    mujoco.mj_forward(m0, d0)
    print(f"{d.name}: bench.xml, palm at {np.round(1e3 * palm_pos, 1)} mm; contacts {d1.ncon} (scene {d0.ncon})",
          flush=True)
    return meta


# ---------------------------------------------------------------------------------- engines

def make_plant(engine, d, meta):
    """CPU MuJoCo steps the run's own scene (its pads or its mesh tip). Drake takes the TPU block as a compliant
    convex from the mesh run's scene (a pad scene's tip geoms are spheres), with this run's state."""
    import hom_turn3 as H3
    if engine == "mujoco":
        return H3.MjPlant(d / "bench.xml", meta)
    if engine == "drake":
        if not meta.get("meshes"):       # the box run's scene has no TPU meshes: take the block run's
            meta = dict(meta, meshes=json.loads((block_dir(d) / "bench_meta.json").read_text())["meshes"])
        return H3.DrakePlant(block_dir(d) / "bench.xml", meta)
    raise ValueError(engine)


def replay(d: Path, engine: str, hold: float | None = None, seed: int = 0):
    meta = json.loads((d / "bench_meta.json").read_text())
    rec = dict(np.load(d / "rec.npz")) if (d / "rec.npz").exists() else None
    if hold is not None and rec is not None:
        targets = np.tile(rec["finger_targets"][meta["world"]][0], (int(round(hold / DT_POLICY)), 1))
    elif hold is not None or rec is None:
        targets = np.tile([meta["q0"][n] for n in NAMES], (int(round((hold or 1.0) / DT_POLICY)), 1))
        if rec is None and (d / "state.npz").exists():
            st = dict(np.load(d / "state.npz"))
            sc = ET.parse(d / "scene.xml").getroot()
            acts = [a.get("joint").split("/")[-1] for act in sc.findall("actuator") for a in act]
            row = st["ctrl"][meta["world"]]
            targets = np.tile([row[acts.index(n)] for n in NAMES], (targets.shape[0], 1))
    else:
        targets = rec["finger_targets"][meta["world"]]          # (steps after onset, 9)
    if engine == "newton":
        return replay_newton(d, meta, targets, newton_dir=NEWTON_DIR[0])
    plant = make_plant(engine, d, meta)
    plant.reset(seed)
    s0 = plant.state()
    t, trace = 0.0, []
    w0 = time.perf_counter()
    for k, tg in enumerate(targets):
        plant.set_targets(np.asarray(tg, float))
        t += DT_POLICY
        plant.advance(t)
        s = plant.state()
        trace.append((round(t, 3), float(s["R"][2, 2]), float(s["p"][2]), sum(1 for f in FINGERS if s["n"][f] > 0)))
    s = plant.state()
    nf = sum(1 for f in FINGERS if s["n"][f] > 0)
    return {"engine": engine, "world": meta.get("world"), "cos_start": float(s0["R"][2, 2]), "cos_end": float(s["R"][2, 2]),
            "z_start_mm": 1e3 * float(s0["p"][2]), "z_end_mm": 1e3 * float(s["p"][2]), "fingers_end": nf,
            "held_end": bool(s["p"][2] > 0.06 and nf >= 2), "F_end": {f: round(s["F"][f], 3) for f in FINGERS},
            "steps": len(targets), "wall_s": round(time.perf_counter() - w0, 1),
            "trace": trace[:: max(1, len(trace) // 50)]}


def match_hydro_friction(model, solver, impratio, tc=0.01, d0=0.9):
    """Friction gain kf of the hydroelastic tip and tool shapes for which SolverMuJoCo's elliptic-cone mapping gives
    the friction rows the pads' 10 ms solref time constant: t_f = 2 / (kf w ((1 - d0) / impratio + d0)), w the solver
    model's inverse weights of tip and tool, d0 0.9 the hydroelastic contacts' solimp (contact_bed_newton
    NewtonRig._match_friction). build_newton's kf 10 gives t_f 3.9 ms on the bed. The tool's shapes take the mean of
    the three fingers' values. Returns kf per finger."""
    mjm = solver.mj_model
    labels = [x.split("/")[-1] for x in model.body_label]
    m2n = solver.mjc_body_to_newton.numpy()[0]
    w = {labels[int(m2n[bi])]: float(mjm.body_invweight0[bi, 0]) for bi in range(mjm.nbody)
         if 0 <= int(m2n[bi]) < len(labels)}
    kf_f = {f: 2.0 / (tc * (w[f"{f}_tip"] + w[TOOL]) * ((1.0 - d0) / impratio + d0)) for f in FINGERS}
    kf = model.shape_material_kf.numpy()
    body = model.shape_body.numpy()
    for i in range(len(kf)):
        if body[i] < 0:
            continue
        bl = labels[body[i]]
        for f in FINGERS:
            if bl == f"{f}_tip":
                kf[i] = kf_f[f]
        if bl == TOOL:
            kf[i] = float(np.mean(list(kf_f.values())))
    model.shape_material_kf.assign(kf)
    return kf_f


KF_MATCH = [False]


def replay_newton(d, meta, targets, newton_dir=None, nworld=1, film_cb=None):
    """The bench-like scene in Newton with hydroelastic TPU blocks, kh x (invweight0 tip + invweight0 tool) of the
    solver's model, tool kh 100x (same_state_timing.build_newton, engine nt_hydro). The geometry comes from
    `newton_dir`'s bench.xml (default: the mesh run of the same seed), since a pad run's tips are sphere pads and
    Newton's hydroelastic tip is the plain block; the state and targets are this run's."""
    import mujoco
    import newton
    import warp as wp
    import same_state_timing as SS
    nd = Path(newton_dir) if newton_dir else block_dir(d)
    xml = (nd / "bench.xml").read_text()
    # as same_state_timing.engine_xml: one-value solrefs get damping ratio 1 (Newton's importer would set 0) and every
    # massive body's compiled inertia is pinned, so the shapes Newton drops or re-adds cannot change the masses
    m0 = mujoco.MjModel.from_xml_string(xml)
    root = ET.fromstring(xml)
    for b in root.iter("body"):
        nm = b.get("name")
        if not nm or m0.body_mass[m0.body(nm).id] <= 0:
            continue
        i = m0.body(nm).id
        for old in b.findall("inertial"):
            b.remove(old)
        b.insert(0, ET.Element("inertial", pos=" ".join(f"{x:.9g}" for x in m0.body_ipos[i]),
                               quat=" ".join(f"{x:.9g}" for x in m0.body_iquat[i]), mass=f"{m0.body_mass[i]:.9g}",
                               diaginertia=" ".join(f"{x:.9g}" for x in m0.body_inertia[i])))
    xml = re.sub(r'solref="([0-9.eE+-]+)"', r'solref="\1 1"', ET.tostring(root, encoding="unicode"))
    m = mujoco.MjModel.from_xml_string(xml)
    if NEWTON_NUM[0]:                       # e.g. the contact bed's 1 ms / 100 iterations / 50 line search
        m.opt.timestep, m.opt.iterations, m.opt.ls_iterations = NEWTON_NUM[0]
    sm = {"tool_body": TOOL, "tip_bodies": {f: f"{f}_tip" for f in FINGERS}}
    extra = {}
    model, pipe, solver = SS.build_newton(xml, m, sm, nworld, "nt_hydro", "mesh", 0.013, 40, extra)
    mm = solver.mj_model
    if KF_MATCH[0]:
        extra["kf_tip"] = match_hydro_friction(model, solver, float(m.opt.impratio))
    jmap = SS.joint_map(m, mm)
    qs, qds = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    jq = model.joint_q.numpy()
    nj = model.joint_count
    j2d = solver.mjc_jnt_to_newton_dof.numpy()
    dof_joint = np.zeros(int(model.joint_dof_count), int)
    for jn in range(nj):
        dof_joint[qds[jn]:qds[jn + 1] if jn + 1 < nj else len(dof_joint)] = jn
    q_mj = np.zeros(m.nq)
    for n in NAMES:
        q_mj[m.jnt_qposadr[m.joint(n).id]] = meta["q0"][n]
    ta = m.jnt_qposadr[m.body_jntadr[m.body(TOOL).id]]
    q_mj[ta:ta + 7] = meta["tool7"]
    for w in range(nworld):
        for j, j0 in enumerate(jmap):
            dof = int(j2d[w, j])
            jn = dof_joint[dof]
            a = m.jnt_qposadr[j0]
            if m.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE:
                q = q_mj[a:a + 7]
                jq[qs[jn]:qs[jn] + 3] = q[:3]
                jq[qs[jn] + 3:qs[jn] + 7] = [q[4], q[5], q[6], q[3]]
            else:
                jq[qs[jn] + (dof - qds[jn])] = q_mj[a]
    model.joint_q.assign(jq)
    model.joint_qd.zero_()
    s0, s1 = model.state(), model.state()
    ctrl = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    order = [NAMES.index(m.joint(jmap[int(mm.actuator_trnid[a, 0])]).name) for a in range(mm.nu)]
    cc = pipe.contacts()
    labels = [x.split("/")[-1] for x in model.body_label]
    tool_id = labels.index(TOOL)
    body_finger = np.array([next((f for f in FINGERS if x.startswith(f + "_")), "") for x in labels])
    shape_body = model.shape_body.numpy()
    dt = float(m.opt.timestep)
    sub = int(round(DT_POLICY / dt))
    st = {"s0": s0, "s1": s1}

    def tool_state():
        x, y, z, qx, qy, qz, qw = st["s0"].body_q.numpy()[tool_id]
        return 1.0 - 2.0 * (qx * qx + qy * qy), float(z)

    cos0, z0 = tool_state()
    w0 = time.perf_counter()
    trace = []
    for k, tg in enumerate(targets):
        row = np.asarray(tg, np.float32)[order]
        ctrl.mujoco.ctrl.assign(np.tile(row, nworld).reshape(ctrl.mujoco.ctrl.shape))
        for _ in range(sub):
            st["s0"].clear_forces()
            pipe.collide(st["s0"], cc)
            solver.step(st["s0"], st["s1"], ctrl, cc, dt)
            st["s0"], st["s1"] = st["s1"], st["s0"]
        if k % 10 == 0:
            c, z = tool_state()
            trace.append((round((k + 1) * DT_POLICY, 3), c, z))
        if film_cb is not None and k % film_cb[2] == 0:
            fr, frames, _, head = film_cb
            jq_now = st["s0"].joint_q.numpy() if hasattr(st["s0"], "joint_q") and st["s0"].joint_q is not None else None
            if jq_now is None:
                jq_now = model.joint_q.numpy()
            q9 = np.zeros(9)
            for j, j0 in enumerate(jmap):
                nm = m.joint(j0).name
                if nm in NAMES:
                    dof = int(j2d[0, j])
                    jn = dof_joint[dof]
                    q9[NAMES.index(nm)] = jq_now[qs[jn] + (dof - qds[jn])]
            x, y, z_, qx, qy, qz, qw = st["s0"].body_q.numpy()[tool_id]
            c, _ = tool_state()
            frames.append(fr.draw(q9, [x, y, z_], [qw, qx, qy, qz], head + [f"t {(k + 1) * DT_POLICY:4.2f} s  cos {c:+.2f}"]))
    wp.synchronize()
    cos1, z1 = tool_state()
    n = int(cc.rigid_contact_count.numpy()[0])
    a_, b_ = cc.rigid_contact_shape0.numpy()[:n], cc.rigid_contact_shape1.numpy()[:n]
    touch = set()
    for i, j in zip(a_, b_):
        bi, bj = shape_body[i], shape_body[j]
        for t, o in ((bi, bj), (bj, bi)):
            if t == tool_id and o >= 0 and body_finger[o]:
                touch.add(body_finger[o])
    return {"engine": "newton", "world": meta.get("world"), "newton_scene": str(nd), "dt": dt, "iterations": int(m.opt.iterations), "cos_start": cos0, "cos_end": cos1, "z_start_mm": 1e3 * z0,
            "z_end_mm": 1e3 * z1, "fingers_end": len(touch), "held_end": bool(z1 > 0.06 and len(touch) >= 2),
            "steps": len(targets), "wall_s": round(time.perf_counter() - w0, 1), "kh_tip": extra.get("kh_tip"),
            "invweight0_newton": extra.get("invweight0_newton"), "kf_matched": extra.get("kf_tip"), "trace": trace}


# ---------------------------------------------------------------------------------- films

CAM = dict(distance=0.24, elevation=-6.0, azimuth=0.0)     # feedback_rl_films_rerender_close_up: the turn in the image plane
ENGINE_LBL = {"mjw": "MuJoCo-Warp (training)", "mujoco": "CPU MuJoCo", "drake": "Drake hydroelastic", "newton": "Newton hydroelastic"}


def _label(frame, lines):
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame)
    dr = ImageDraw.Draw(im)
    y = 6
    for ln in lines:
        dr.rectangle([4, y - 1, 8 + 7 * len(ln), y + 13], fill=(255, 255, 255))
        dr.text((6, y), ln, fill=(20, 20, 20))
        y += 15
    return np.asarray(im)


class FilmRenderer:
    """A MuJoCo copy of the bench-like scene that draws any engine's state: the finger angles and the tool's pose."""

    def __init__(self, xml_path: Path, lookat, w=640, h=480):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.m.vis.global_.offwidth, self.m.vis.global_.offheight = max(w, 640), max(h, 480)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, h, w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.elevation, self.cam.azimuth = CAM["distance"], CAM["elevation"], CAM["azimuth"]
        self.cam.lookat[:] = np.asarray(lookat, float)
        self.qadr = [self.m.jnt_qposadr[self.m.joint(n).id] for n in NAMES]
        tb = self.m.body(TOOL).id
        self.qa = self.m.jnt_qposadr[self.m.body_jntadr[tb]]

    def draw(self, q9, p, quat, lines, qfull=None):
        if qfull is not None:
            self.d.qpos[:] = qfull
        else:
            self.d.qpos[self.qadr] = q9
            self.d.qpos[self.qa:self.qa + 3] = p
            self.d.qpos[self.qa + 3:self.qa + 7] = quat
        self.mj.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, self.cam)
        return _label(self.r.render().copy(), lines)


def _scene_to_bench_map(d: Path):
    """Index of every bench.xml qpos entry in the exported scene.xml qpos (the bench drops the palm's six joints)."""
    import mujoco
    m0 = mujoco.MjModel.from_xml_path(str(d / "scene.xml"))
    m1 = mujoco.MjModel.from_xml_path(str(d / "bench.xml"))
    idx = np.zeros(m1.nq, int)
    for j in range(m1.njnt):
        n = m1.joint(j).name
        cand = [k for k in range(m0.njnt) if m0.joint(k).name.split("/", 1)[-1] in (n, "cube_joint" if n == f"{TOOL}_joint" else n)]
        k = cand[0]
        w = {0: 7, 1: 4}.get(int(m1.jnt_type[j]), 1)
        idx[m1.jnt_qposadr[j]:m1.jnt_qposadr[j] + w] = np.arange(m0.jnt_qposadr[k], m0.jnt_qposadr[k] + w)
    return idx


def film(d: Path, engine: str, out: Path, fps: int = 25, w: int = 640, h: int = 480, title: str = ""):
    """One clip of the replay of bench_meta.json's world in `engine` (mjw: the recorded MuJoCo-Warp rollout itself)."""
    import imageio.v2 as imageio
    meta = json.loads((d / "bench_meta.json").read_text())
    rec = dict(np.load(d / "rec.npz"))
    world = int(meta["world"])
    look = np.asarray(meta["tool7"][:3], float) + [0.0, 0.0, 0.01]
    own = engine in ("mjw", "mujoco")
    fr = FilmRenderer((d if own else block_dir(d)) / "bench.xml", look, w, h)
    frames = []
    every = max(1, int(round(1.0 / fps / DT_POLICY)))
    head = [title or d.name, ENGINE_LBL[engine]]
    if engine == "mjw":
        idx = _scene_to_bench_map(d)
        Q = rec["qpos_t"][world]
        for k in range(0, Q.shape[0], every):
            q = Q[k][idx]
            R9 = np.zeros(9)
            fr.mj.mju_quat2Mat(R9, q[fr.qa + 3:fr.qa + 7])
            frames.append(fr.draw(None, None, None, head + [f"t {(k + 1) * DT_POLICY:4.2f} s  cos {R9[8]:+.2f}"], qfull=q))
    elif engine == "newton":
        replay_newton(d, meta, rec["finger_targets"][world], newton_dir=NEWTON_DIR[0], film_cb=(fr, frames, every, head))
    else:
        plant = make_plant(engine, d, meta)
        plant.reset(0)
        t = 0.0
        for k, tg in enumerate(rec["finger_targets"][world]):
            plant.set_targets(np.asarray(tg, float))
            t += DT_POLICY
            plant.advance(t)
            if k % every == 0:
                s = plant.state()
                frames.append(fr.draw(s["q"], s["p"], s["quat"], head + [f"t {t:4.2f} s  cos {s['R'][2, 2]:+.2f}"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    wr = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=8,
                            ffmpeg_params=["-crf", "26", "-preset", "medium"], ffmpeg_log_level="error")
    for f_ in frames:
        wr.append_data(f_)
    wr.close()
    print(f"{out}: {len(frames)} frames ({engine}, world {world})", flush=True)
    return len(frames)


def tile(clips, out: Path, out_jpg: Path | None = None, poster_at: float = 0.75, fps: int = 25):
    """Clips side by side, frame by frame; a shorter clip holds its last frame."""
    import imageio.v2 as imageio
    from PIL import Image
    rd = [imageio.get_reader(str(c)) for c in clips]
    its = [iter(r) for r in rd]
    n = max(r.count_frames() for r in rd)
    last = [None] * len(clips)
    wr = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=None, pixelformat="yuv420p", macro_block_size=8,
                            ffmpeg_params=["-crf", "27", "-preset", "medium"], ffmpeg_log_level="error")
    kp = int(round(poster_at * (n - 1)))
    for k in range(n):
        for i, it in enumerate(its):
            try:
                last[i] = np.asarray(next(it))[:, :, :3]
            except StopIteration:
                pass
        canvas = np.concatenate(last, axis=1)
        wr.append_data(canvas)
        if out_jpg is not None and k == kp:
            Image.fromarray(canvas).save(out_jpg, quality=88)
    wr.close()
    for r in rd:
        r.close()


# ---------------------------------------------------------------------------------- CLI

def main():
    global OUT, REPLAY
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rc = sub.add_parser("record")
    rc.add_argument("--tag", required=True)
    rc.add_argument("--checkpoint", default=None, help="default: the run's newest model_*.pt")
    rc.add_argument("--n", type=int, default=16)
    b = sub.add_parser("bench")
    b.add_argument("--dir", type=Path, required=True)
    b.add_argument("--world", type=int, default=0)
    b.add_argument("--plant", default="kp4_kv0_fr1_fl0_dp0.08")
    r = sub.add_parser("replay")
    r.add_argument("--dir", type=Path, required=True)
    r.add_argument("--engine", required=True, choices=["mujoco", "drake", "newton"])
    r.add_argument("--hold", type=float, default=None, help="hold the onset targets this long instead (a test)")
    r.add_argument("--out", type=Path, default=OUT)
    r.add_argument("--newton-dir", type=Path, default=None, help="newton: the bench.xml to take the geometry from")
    r.add_argument("--newton-num", default=None, help="newton: dt,iterations,ls_iterations (default: the scene's)")
    fm = sub.add_parser("film")
    fm.add_argument("--dir", type=Path, required=True)
    fm.add_argument("--engine", required=True, choices=["mjw", "mujoco", "drake", "newton"])
    fm.add_argument("--out", type=Path, required=True)
    fm.add_argument("--title", default="")
    fm.add_argument("--newton-dir", type=Path, default=None)
    tl = sub.add_parser("tile")
    tl.add_argument("--clips", type=Path, nargs="+", required=True)
    tl.add_argument("--out", type=Path, required=True)
    tl.add_argument("--poster", type=Path, default=None)
    for sp in (rc, b, r, fm, tl):
        sp.add_argument("--study", choices=sorted(STUDIES), default=None,
                        help="write rows and replay dirs to that study's folders")
    a = ap.parse_args()
    if a.study:
        OUT, REPLAY = STUDIES[a.study]
        KF_MATCH[0] = True       # 2026-10-08: Newton's hydroelastic friction rows at the pads' 10 ms
        if a.cmd == "replay" and a.out == ROOT / "docs/experiments/20261006-rl_contact/policy_replay.jsonl":
            a.out = OUT
    if a.cmd == "record":
        os.environ.setdefault("MUJOCO_GL", "egl")
        row = record(a.tag, a.checkpoint, a.n)
        row.update(kind="record", when=time.strftime("%Y-%m-%d %H:%M"))
        with open(OUT, "a") as fh:
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return 0
    if a.cmd == "bench":
        bench(a.dir, a.world, a.plant)
        return 0
    if a.cmd == "film":
        os.environ.setdefault("MUJOCO_GL", "egl")
        NEWTON_DIR[0] = a.newton_dir
        film(a.dir, a.engine, a.out, title=a.title)
        return 0
    if a.cmd == "tile":
        tile(a.clips, a.out, a.poster)
        return 0
    NEWTON_DIR[0] = a.newton_dir
    if a.newton_num:
        dt_, it_, ls_ = a.newton_num.split(",")
        NEWTON_NUM[0] = (float(dt_), int(it_), int(ls_))
    row = {"dir": str(a.dir), "when": time.strftime("%Y-%m-%d %H:%M"), "hold_test_s": a.hold}
    try:
        row.update(replay(a.dir, a.engine, a.hold), status="ok")
    except Exception as e:
        row.update(engine=a.engine, status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
    with open(a.out, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    print({k: v for k, v in row.items() if k not in ("trace", "tb")}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
