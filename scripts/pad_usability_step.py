#!/usr/bin/env python3
"""Largest stable and accurate physics step of the 1 mm sphere pads and of point contact on the real_v1 hand
(experiment U3 of the 2026-10-14 pad-usability study; plan in docs/notes/20261014-contact_model_assessment_plan.md).

threshold  Spectral stability threshold with MuJoCo's refsafe clamp off (the contact time constant t_c stays as
           set at every step). Per state: 0.2 s at a 0.25 ms step to reach rest, then the spectral radius of the
           one-step map, mujoco.mjd_transitionFD(m, d, 1e-9, centred, A), on a grid of steps h = 0.5, 0.75 ... 30 ms,
           and bisection between the last stable and the first unstable grid step to 0.01 ms; unstable means a
           radius above 1 + 1e-3 (a held tool's slow creep sits at radius 1). Per threshold it records the loaded
           tool contacts per finger (n), their impedance d0, time constant t_c and diagonal inverse weight Lambda-hat,
           the impratio kappa, three predictions
             normal-row bound    2 t_c (sqrt(1 + q) - 1),  q = d0 + (1 - d0) / n  (smallest over the fingers)
             normal damper bound q t_c
             friction-row bound  2 / (b lam_t'), lam_t' = largest eigenvalue of A_act (A + R)^-1 over the loaded
                                 tool-contact friction rows (contact_bed_stability.lam_max), b the rows' damping
                                 coefficient (2 / (d0 t_c) for the pads); closed form t_c (d0 + (1 - d0)
                                 Lambda-hat / (kappa lam_t)), lam_t the largest eigenvalue of A over those rows
           the unstable eigenvector's shares (finger-joint kinetic energy; normal against tangential relative
           velocity at the loaded tool contacts; the tool's rotational against translational kinetic energy) and
           2 s rollouts at 0.95 and 1.05 times the threshold.
           States: scale:<shape>:<d_mm>:<t>   grip of a scale-study cell 0.8 s after the grip command (servo plant,
                                              hand_object_scale_mpc.setup), the post's collisions then removed as
                                              the MPC removes them at 0.8 s
                   bench:<hand>:<servo|work>[:<t_s>]  the plan replay of the hand at seed 0, 1 ms, stopped at the end
                                              of the hold after the turn (default) or at t_s
                   rl:<variant>               world 0 of the RL held state U1 exported to
                                              logs/20261014-pad_usability/state_<variant>/ (the trainer's numerics)
                   bed                        the contact bed's two-pad pinch (task 7): 21 pad pairs (d0 x t_c) and 3
                                              point-contact pairs, contact_bed_stability.build(refsafe=False), 1 N per
                                              pad, the tool hanging on friction across its axis
sweep      Plan replays of D1-D8 (seeds 0-4) with the clamp on, at steps given by --dt, through the replay loop of
           scripts/pad_usability_backends.py with reorient_backends.run_mujoco replaced by a copy that also records
           MuJoCo's bad-state warnings (NaN or |qacc| > 1e10: MuJoCo then resets the state and the run stops),
           the tool's largest linear and angular speed and distance from its start, contacts and solver iterations.
           Scenes are built under logs/20261014-pad_usability/scenes/step_<model>_bed_dt<ms>/.
film       D7 seed 0 on the servo plant at 1, 4 and 10 ms for the 1 mm pads and point contact, tiled 3 x 2.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/pad_usability_step.py threshold --states scale:cylinder:16:0.625 bench:D6:servo rl:servo_pads1 bed \\
        --models pads pads2 pt pt4 --out docs/experiments/20261014-pad_usability/step/threshold.jsonl
    $PY scripts/pad_usability_step.py sweep --plants kp0.72_kv0_fr0.18_fl0_dp0.0144 --models pads pt \\
        --dt 1 2 4 5 10 --out docs/experiments/20261014-pad_usability/step/plans_dt.jsonl
    MUJOCO_GL=egl $PY scripts/pad_usability_step.py film

Rows: one fsynced JSON line per threshold (threshold.jsonl, resumable by state and model) or per replay
(plans_dt.jsonl, resumable by case_id) under docs/experiments/20261014-pad_usability/step/; films in step/media/.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import pad_usability_backends as PB  # noqa: E402
import reorient_backends as RB  # noqa: E402

SCRIPT = "scripts/pad_usability_step.py"
STEP = ROOT / "docs/experiments/20261014-pad_usability/step"
MEDIA = STEP / "media"
LOGS = ROOT / "logs/20261014-pad_usability"
SERVO, WORK = PB.SERVO, PB.WORK
PLANT_OF = {"servo": SERVO, "work": WORK}
FINGERS = RB.FINGERS
H_SETTLE, T_SETTLE = 0.25e-3, 0.2            # s
GRID_MS = np.round(np.arange(0.5, 30.0 + 1e-9, 0.25), 4)
TOL_MS = 0.01
RAD_TOL = 1e-3
SLOW_DISC = 0.05
EPS_FD = 1e-9
T_CONFIRM = 2.0
CONFIRM_MM, CONFIRM_DEG = 5.0, 5.0
EJECT_SPEED, EJECT_DIST = 2.0, 0.30         # m/s, m
BED_D0S = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95]
BED_TCS_MS = [5.0, 10.0, 20.0]
BED_POINT = [(0.2, 10.0), (0.5, 10.0), (0.9, 10.0)]

_PB_BUILD = PB.build


def _build_step(*args, prefix="", **kw):
    """PB.build with every scene of this study under a 'step_' folder, so no other study's scene is rewritten."""
    return _PB_BUILD(*args, prefix="step_" + prefix, **kw)


PB.build = _build_step


def append_row(path, row):
    PB.append_row(path, row)


def jl(path):
    path = Path(path)
    return [json.loads(x) for x in path.open() if x.strip()] if path.exists() else []


def git_rev():
    return RB.git_rev()


# ====================================================================================== replays (sweep and bench states)

def _bad(mj, d):
    W = mj.mjtWarning
    return any(d.warning[int(w)].number > 0 for w in (W.mjWARN_BADQACC, W.mjWARN_BADQVEL, W.mjWARN_BADQPOS))


def run_mujoco_diag(scene, meta, plan, traj, seed, film=None, t_stop=None):
    """reorient_backends.run_mujoco with the same initial state, command schedule, stepping and 20 ms trace, plus
    per-step diagnostics read after each mj_step (they do not change the state): MuJoCo bad-state warnings (the run
    stops at the first; MuJoCo has then reset the data), the tool's largest linear speed (m/s), angular speed
    (rad/s) and distance from its start (m), contacts and solver iterations. With t_stop it returns (m, d) at the
    first step boundary at or after t_stop instead of a scored row."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    tb = m.body(RB.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    va = m.jnt_dofadr[m.body_jntadr[tb]]
    tool7 = np.asarray(meta["tool7"], float)
    dx, dy, dyaw = RB.jitter(seed)
    d.qpos[qa:qa + 3] = tool7[:3] + [dx, dy, 0.0]
    d.qpos[qa + 3:qa + 7] = RB._yaw_quat(tool7[3:], dyaw)
    act = {}
    for f in FINGERS:
        for j in RB.JOINTS:
            n = f"{f}_{j}"
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = meta["q0"][n]
            act[n] = m.actuator(f"a_{n}").id
    mujoco.mj_forward(m, d)
    p_start = d.qpos[qa:qa + 3].copy()
    tool_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == tb}
    finger_of = {}
    for i in range(m.ngeom):
        bn = m.body(m.geom_bodyid[i]).name
        for f in FINGERS:
            if bn.startswith(f + "_"):
                finger_of[i] = f
    f6 = np.zeros(6)
    segs = RB.schedule(plan, traj)
    trace, W = [], []

    def snap():
        F = {f: 0.0 for f in FINGERS}
        n = {f: 0 for f in FINGERS}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if g1 not in tool_geoms and g2 not in tool_geoms:
                continue
            o = g2 if g1 in tool_geoms else g1
            f = finger_of.get(o)
            if f is None:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            if f6[0] > 1e-6:
                F[f] += float(f6[0])
                n[f] += 1
        R9 = np.zeros(9)
        mujoco.mju_quat2Mat(R9, d.qpos[qa + 3:qa + 7])
        p = d.qpos[qa:qa + 3]
        trace.append({"t": round(float(d.time), 4), "cos": float(R9[8]), "x": float(p[0]), "y": float(p[1]),
                      "z": float(p[2]), "F": {k: round(v, 4) for k, v in F.items()}, "n": n})

    dt = m.opt.timestep
    k_trace = max(1, int(round(RB.TRACE_DT / dt)))
    step = 0
    bad = nan_state = False
    v_max = w_max = dist_max = 0.0
    ncon_max = 0
    niter_sum = niter_cap = 0
    t_bad = None
    for t_end, tgt, _ in segs:
        for n_, a_ in act.items():
            d.ctrl[a_] = tgt[n_]
        while d.time < t_end - 1e-9:
            if t_stop is not None and d.time >= t_stop - 1e-9:
                return m, d
            w0 = time.perf_counter()
            mujoco.mj_step(m, d)
            W.append(time.perf_counter() - w0)
            step += 1
            if _bad(mujoco, d) or not np.isfinite(d.qvel[va]):
                nan_state, t_bad = True, step * dt
                break
            v = d.qvel[va:va + 6]
            v_max = max(v_max, math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
            w_max = max(w_max, math.sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5]))
            p = d.qpos[qa:qa + 3]
            dist_max = max(dist_max, math.sqrt((p[0] - p_start[0]) ** 2 + (p[1] - p_start[1]) ** 2
                                               + (p[2] - p_start[2]) ** 2))
            ncon_max = max(ncon_max, d.ncon)
            ni = int(d.solver_niter[0])
            niter_sum += ni
            niter_cap += ni >= m.opt.iterations
            if step % k_trace == 0:
                snap()
                if not np.all(np.isfinite(d.qpos)) or abs(d.qpos[qa + 2]) > 2.0:
                    bad = True
                    break
        if bad or nan_state:
            break
    if t_stop is not None:
        return m, d
    res = RB.score(trace, segs) if not (bad or nan_state) and trace else {}
    status = "nan" if nan_state else ("ejected" if bad else "complete")
    W = np.asarray(W)
    sim_s = step * dt
    res.update(status=status, us_per_step_median=float(np.median(W) * 1e6) if len(W) else None,
               cpu_ms_per_sim_s=float(W.sum() / sim_s * 1e3) if sim_s > 0 else None, ncon_max=int(ncon_max),
               n_steps=step, t_bad_s=t_bad, tool_speed_max_m_s=v_max, tool_omega_max_rad_s=w_max,
               tool_dist_max_m=dist_max, solver_niter_mean=niter_sum / max(1, step),
               solver_at_cap_frac=niter_cap / max(1, step),
               ejected_flag=bool(nan_state or bad or v_max > EJECT_SPEED or dist_max > EJECT_DIST),
               refsafe=not bool(m.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)))
    return res, trace


def cmd_sweep(a):
    RB.run_mujoco = run_mujoco_diag
    ns = argparse.Namespace(sim="mujoco", hands=a.hands, tips=["tpu2.7"], models=a.models, plants=a.plants,
                            numerics=["bed"], ir=[100.0], mu=["1.0"], dt=a.dt, spin=None, seeds=a.seeds, jitter="j2",
                            out=a.out, traces=False)
    PB.SCRIPT = SCRIPT
    PB.cmd_run(ns)


# ====================================================================================== threshold states

class State:
    """A model and data at a rest state, with the dof groups and contact bookkeeping the analysis needs."""

    def __init__(self, m, d, tool_body, side_of, groups, info, tip_geoms=None):
        self.m, self.d, self.tb, self.side_of, self.groups, self.info = m, d, tool_body, side_of, groups, info
        self.tool_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == tool_body}
        self.tip_geoms = set(side_of) if tip_geoms is None else tip_geoms


def _hand_groups(m, tool_body, prefix=""):
    import mujoco
    fing, palm = [], []
    for j in range(m.njnt):
        nm = m.joint(j).name or ""
        base = nm[len(prefix):] if nm.startswith(prefix) else nm
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if any(base.startswith(f + "_") for f in FINGERS):
            fing.append(int(m.jnt_dofadr[j]))
        elif base.startswith("palm_"):
            palm += list(range(int(m.jnt_dofadr[j]), int(m.jnt_dofadr[j]) + (3 if m.jnt_type[j] == 1 else 1)))
    jt = m.body_jntadr[tool_body]
    tool = list(range(int(m.jnt_dofadr[jt]), int(m.jnt_dofadr[jt]) + 6))
    side_of, tips = {}, set()
    for i in range(m.ngeom):
        bn = m.body(m.geom_bodyid[i]).name or ""
        bn = bn[len(prefix):] if bn.startswith(prefix) else bn
        for f in FINGERS:
            if bn.startswith(f + "_"):
                side_of[i] = f
                if bn == f + "_tip":
                    tips.add(i)
    return {"finger": fing, "palm": palm, "tool": tool}, side_of, tips


def state_scale(spec, model):
    """Grip of a scale-study cell 0.8 s after the grip command on the servo plant (U1's plants). The post stays;
    with ':postoff' after the cell its collisions are removed at 0.8 s, as the MPC planner's model has no post (the
    cylinder then slips under gravity, a slow mode the radius excludes)."""
    postoff = spec.endswith(":postoff")
    spec = spec[:-len(":postoff")] if postoff else spec
    import mujoco
    import hand_object_scale_mpc as M
    import hand_object_scale_sim as S
    import pad_usability_timing as PT
    S.PLANT = SERVO
    best = M.hom_best(Path(M.OUT / "hom_postoff_servo/turn.jsonl"))
    shape, dd, t = spec.split(":")
    dd, t = float(dd), float(t)
    cell, _, _ = next(c for c in S.cell_list("diag") if c[1] == shape and c[2] == dd and c[0].get("t") == t)
    ci = best[(cell["tag"], shape, dd)]["cand"]
    plant0, meta, geo, sc, gg = M.setup(cell, shape, dd, ci)
    del plant0
    paths = {"pads": sc["pads"][0], "pt": sc["pt"][0]}
    if model == "pads2":
        paths["pads2"] = PT.pads2_scene(cell, shape, dd)[0]
    p, edits = PT.make_plant(model, paths, meta, gg)
    grip = np.array([meta["grip"][f][k] for f in S.FINGERS for k in range(3)])
    p.reset(0)
    p.set_targets(grip)
    p.advance(0.8)
    m, d = p.m, p.d
    post = m.geom("post").id
    n_post = sum(1 for i in range(d.ncon) if post in (int(d.contact[i].geom[0]), int(d.contact[i].geom[1])))
    if postoff:
        m.geom_contype[post] = 0
        m.geom_conaffinity[post] = 0
    groups, side_of, tips = _hand_groups(m, p.tb)
    info = {"cell": spec, "tag": cell["tag"], "cand": ci, "plant": SERVO, "scene": str(Path(p.xml_path).relative_to(ROOT)),
            "t_state_s": float(d.time), "post_contacts_at_0.8s": n_post, "post_removed": postoff,
            "F_fingers_at_0.8s": {f: round(float(v), 3) for f, v in p.state()["F"].items()}, **edits}
    return State(m, d, p.tb, side_of, groups, info, tips), p


def state_bench(hand, plant_short, model, t_s=None, seed=0):
    """The hand's plan replay at seed 0 on the plant at 1 ms (U2's scenes), stopped at t_s (default: the end of the
    hold after the turn, t_traj_end + 1.5 s, where reorient_backends.score reads turn_hold_deg)."""
    plant = PLANT_OF[plant_short]
    plan, traj = RB.load_plan(hand)
    segs = RB.schedule(plan, traj)
    t_traj_end = [e for e, _, k in segs if k == "traj"][-1]
    t_stop = t_s if t_s is not None else t_traj_end + RB.T_HOLD
    scene, meta = PB.build(hand, "tpu2.7", model, plant, "bed", 100.0, "1.0", dt_ms=1.0, prefix="thr_")
    m, d = run_mujoco_diag(scene, meta, plan, traj, seed, None, t_stop=t_stop)
    tb = m.body(RB.OBJ).id
    groups, side_of, tips = _hand_groups(m, tb)
    info = {"hand": hand, "plant": plant, "seed": seed, "scene": str(Path(scene).relative_to(ROOT)),
            "t_state_s": float(d.time), "t_traj_end_s": t_traj_end, "spin_m": meta.get("spin_m"),
            "pad_d0": meta.get("pad_d0"), "n_pads": meta.get("n_pads")}
    return State(m, d, tb, side_of, groups, info, tips), None


def state_rl(variant, world=0):
    """World 0 of the RL held state exported by U1 (qpos, qvel, ctrl and warm start, float32), in the trainer's
    scene with its numerics (2 ms, impratio 10, Newton, 10 iterations, tolerance 1e-8)."""
    import mujoco
    sdir = LOGS / f"state_{variant}"
    if not (sdir / "state.npz").exists():
        raise FileNotFoundError(f"{sdir}/state.npz (U1's export) is missing")
    m = mujoco.MjModel.from_xml_path(str(sdir / "scene_key.xml"))
    d = mujoco.MjData(m)
    z = np.load(sdir / "state.npz")
    d.qpos[:] = z["qpos"][world]
    d.qvel[:] = z["qvel"][world]
    d.ctrl[:] = z["ctrl"][world]
    d.qacc_warmstart[:] = z["qacc_warmstart"][world]
    mujoco.mj_forward(m, d)
    js = json.loads((sdir / "state.json").read_text())
    tb = m.body(js["tool_body"]).id
    groups, side_of, tips = _hand_groups(m, tb, prefix="robot/")
    info = {"variant": variant, "world": world, "run": js.get("run"), "scene": str((sdir / "scene_key.xml").relative_to(ROOT)),
            "policy_steps": js.get("policy_steps"), "option": js.get("option"),
            "tip_tool_contacts_export": [int(v) for v in z["tip_tool_contacts"][world]]}
    return State(m, d, tb, side_of, groups, info, tips), None


def state_bed(model, d0, tc_ms):
    """The bed's two-pad pinch (contact_bed_stability.build with refsafe off), 1 N per pad, weight compensated for
    0.4 s at 0.25 ms (as the bed's settle), then released so the tool hangs on friction."""
    import mujoco
    import contact_bed_stability as CS
    rig = CS.build(model, 1.0, tc_ms * 1e-3, d0, False)
    m, d = rig.m, rig.d
    rig.set_pad_force(CS.N_PINCH)
    m.opt.timestep = H_SETTLE
    rig.set_tool_wrench(np.array([0.0, 0.0, CS.H.M_TOOL * CS.H.G]), np.zeros(3))
    for _ in range(int(round(CS.T_SETTLE / H_SETTLE))):
        mujoco.mj_step(m, d)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    groups = {"finger": [int(m.jnt_dofadr[m.joint(f"rail{s}").id]) for s in "LR"], "palm": [],
              "tool": list(range(int(m.jnt_dofadr[m.body_jntadr[rig.tool]]),
                                 int(m.jnt_dofadr[m.body_jntadr[rig.tool]]) + 6))}
    stored = [r for r in jl(CS.OUT) if r.get("kind") == "summary" and r.get("model") == model
              and abs(r["d0"] - d0) < 1e-9 and abs(r["tc_ms"] - tc_ms) < 1e-9 and not r["refsafe"]]
    info = {"bed_model": model, "d0_set": d0, "tc_ms_set": tc_ms, "N_pinch": CS.N_PINCH,
            "stored_summary": {k: stored[0].get(k) for k in (
                "dt_max_held_ms", "dt_first_fail_ms", "pred_dt_max_simple_ms", "pred_dt_max_pad_ms",
                "pred_dt_max_both_ms", "pred_dt_max_lam_full_ms", "pred_dt_max_friction_ms", "n_L_ref", "n_R_ref",
                "lam_n_ref", "lam_t_ref")} if stored else None}
    return State(m, d, rig.tool, dict(rig.geom_side), groups, info), rig


# ====================================================================================== spectral analysis

def full_state(mj, m, d):
    spec = mj.mjtState.mjSTATE_INTEGRATION
    x = np.empty(mj.mj_stateSize(m, spec))
    mj.mj_getState(m, d, x, spec)
    return x


def set_state(mj, m, d, x):
    mj.mj_setState(m, d, x, mj.mjtState.mjSTATE_INTEGRATION)
    mj.mj_forward(m, d)


def settle(st):
    """refsafe off, then T_SETTLE at H_SETTLE; returns the rest state and the warnings raised on the way."""
    import mujoco
    m, d = st.m, st.d
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_REFSAFE)
    m.opt.timestep = H_SETTLE
    p0 = d.xpos[st.tb].copy()
    for _ in range(int(round(T_SETTLE / H_SETTLE))):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    return full_state(mujoco, m, d), {"bad_state_in_settle": _bad(mujoco, d),
                                      "tool_move_in_settle_mm": float(1e3 * np.linalg.norm(d.xpos[st.tb] - p0))}


def transition(st, x0, h):
    import mujoco
    m, d = st.m, st.d
    m.opt.timestep = h
    set_state(mujoco, m, d, x0)
    n = 2 * m.nv + m.na
    A = np.zeros((n, n))
    mujoco.mjd_transitionFD(m, d, EPS_FD, 1, A, None, None, None)
    return A


def split_eigs(ev):
    """Eigenvalues of the one-step map split into resolved slow modes (|lambda - 1| < SLOW_DISC: a held tool's
    creep, a slip under gravity; their growth per step is sigma h with sigma the physical rate) and the rest, whose
    largest modulus is the radius that decides numerical stability."""
    slow = np.abs(ev - 1.0) < SLOW_DISC
    rest = ev[~slow]
    return (float(np.max(np.abs(rest))) if rest.size else 0.0), (float(np.max(np.abs(ev[slow]))) if slow.any() else None)


def radius(st, x0, h, both=False):
    r, r_slow = split_eigs(np.linalg.eigvals(transition(st, x0, h)))
    return (r, r_slow) if both else r


def scan(st, x0):
    """Spectral radius on GRID_MS, then bisection of the first crossing of 1 + RAD_TOL to TOL_MS."""
    both = [radius(st, x0, h * 1e-3, both=True) for h in GRID_MS]
    rad = [b[0] for b in both]
    uns = [i for i, r in enumerate(rad) if r > 1 + RAD_TOL]
    out = {"grid_ms": [float(h) for h in GRID_MS], "radius_grid": [round(r, 7) for r in rad],
           "slow_radius_grid": [None if b[1] is None else round(b[1], 7) for b in both],
           "n_unstable_grid": len(uns)}
    if not uns:
        out.update(threshold_ms=None, h_unstable_ms=None, note=f"stable to {GRID_MS[-1]:g} ms")
        return out
    i0 = uns[0]
    if i0 == 0:
        lo, hi = 0.01, float(GRID_MS[0])
        if radius(st, x0, lo * 1e-3) > 1 + RAD_TOL:
            out.update(threshold_ms=0.0, h_unstable_ms=lo, note="unstable at 0.01 ms")
            return out
        out["note"] = "unstable at 0.5 ms; bisected below the grid"
    else:
        lo, hi = float(GRID_MS[i0 - 1]), float(GRID_MS[i0])
    while hi - lo > TOL_MS:
        mid = 0.5 * (lo + hi)
        if radius(st, x0, mid * 1e-3) > 1 + RAD_TOL:
            hi = mid
        else:
            lo = mid
    stable_after = [float(GRID_MS[i]) for i in range(i0 + 1, len(GRID_MS)) if rad[i] <= 1 + RAD_TOL]
    out.update(threshold_ms=lo, h_unstable_ms=hi, stable_again_ms=stable_after[:5],
               radius_at_threshold=radius(st, x0, lo * 1e-3), radius_at_unstable=radius(st, x0, hi * 1e-3))
    return out


def efc_jac(m, d):
    """Dense constraint Jacobian (nefc x nv) whether MuJoCo stores it dense or sparse."""
    import mujoco
    if mujoco.mj_isSparse(m):
        J = np.zeros((d.nefc, m.nv))
        mujoco.mju_sparse2dense(J, d.efc_J, d.efc_J_rownnz, d.efc_J_rowadr, d.efc_J_colind)
        return J
    return np.array(d.efc_J[:d.nefc * m.nv]).reshape(d.nefc, m.nv)


def loaded_contacts(st):
    """Every contact carrying normal force, with its kind: 'pad' (tool and fingertip geom or bed pad), 'link' (tool and
    another finger geom), 'support' (tool and anything else: post, floor), 'finger_other' (finger and anything but
    the tool), 'other'; with its rows' damping b, stiffness times impedance and time constant."""
    m, d = st.m, st.d
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        adr = int(c.efc_address)
        if adr < 0 or d.efc_force[adr] <= 1e-9:
            continue
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        t1, t2 = g1 in st.tool_geoms, g2 in st.tool_geoms
        if t1 != t2:
            o = g2 if t1 else g1
            kind = ("pad" if o in st.tip_geoms else "link") if st.side_of.get(o) else "support"
        else:
            kind = "finger_other" if (st.side_of.get(g1) or st.side_of.get(g2)) else "other"
        out.append({"kind": kind, "adr": adr, "dim": int(c.dim), "N": float(d.efc_force[adr]), "tc": float(c.solref[0]),
                    "imp": float(d.efc_KBIP[adr, 2])})
    return out


def loaded_tool_contacts(st):
    """Contacts between the tool and a fingertip geom (or bed pad) carrying normal force, with their rows."""
    m, d = st.m, st.d
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        if (g1 in st.tool_geoms) == (g2 in st.tool_geoms):
            continue
        o = g2 if g1 in st.tool_geoms else g1
        side = st.side_of.get(o)
        adr = int(c.efc_address)
        if side is None or o not in st.tip_geoms or adr < 0 or d.efc_force[adr] <= 1e-9:
            continue
        out.append({"side": side, "adr": adr, "dim": int(c.dim), "tc": float(c.solref[0]),
                    "dmax": float(c.solimp[1]), "imp": float(d.efc_KBIP[adr, 2]), "b": float(d.efc_KBIP[adr, 1]),
                    "k": float(d.efc_KBIP[adr, 0]), "diag": float(d.efc_diagApprox[adr]), "N": float(d.efc_force[adr]),
                    "b_t": float(d.efc_KBIP[adr + 1, 1]) if c.dim > 1 else None})
    return out


def _implicit_damping(m):
    """Velocity derivative that implicitfast takes implicitly on each dof: joint damping and actuator kv."""
    D = np.array(m.dof_damping, float).copy()
    for a in range(m.nu):
        if m.actuator_trntype[a] == 0:                       # joint transmission
            j = int(m.actuator_trnid[a, 0])
            kv = -float(m.actuator_biasprm[a, 2])
            if kv > 0:
                D[int(m.jnt_dofadr[j])] += kv * float(m.actuator_gear[a, 0]) ** 2
    return D


def row_eigs(st, rows, h):
    """lam' = largest eigenvalue of A (A + R)^-1 and of A_act (A + R)^-1 (A_act with M + h D, D the implicit
    damping), and lam = largest eigenvalue of A, over the constraint rows given."""
    import mujoco
    m, d = st.m, st.d
    if not rows:
        return None
    nv = m.nv
    J = efc_jac(m, d)[rows]
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(m, M, d.qM)
    Minv = np.linalg.inv(M)
    Mact = np.linalg.inv(M + h * np.diag(_implicit_damping(m)))
    A = J @ Minv @ J.T
    A_act = J @ Mact @ J.T
    R = np.diag(np.array(d.efc_R)[rows])
    return {"lam_prime": float(np.max(np.linalg.eigvals(np.linalg.solve(A + R, A)).real)),
            "lam_prime_act": float(np.max(np.linalg.eigvals(A_act @ np.linalg.inv(A + R)).real)),
            "lam_A": float(np.max(np.linalg.eigvalsh(0.5 * (A + A.T)))), "R_mean": float(np.mean(np.diag(R)))}


def _servo_stiffness(m):
    """Position gain of each joint's actuator on its dof (kp gear^2): the explicit servo spring."""
    K = np.zeros(m.nv)
    for a in range(m.nu):
        if m.actuator_trntype[a] == 0 and m.actuator_biastype[a] == 1:
            j = int(m.actuator_trnid[a, 0])
            K[int(m.jnt_dofadr[j])] += -float(m.actuator_biasprm[a, 1]) * float(m.actuator_gear[a, 0]) ** 2
    return K


def jointspace_threshold(st, servo=True, imp_deriv=True, lo_ms=0.05, hi_ms=60.0):
    """Threshold of a linear model of MuJoCo's step built from the rest state: forces F_s = -K_s q - D v (servo
    springs, joint damping and actuator kv), contact force f = (A + R)^-1 (a_ref - J M^-1 F_s) over the rows of
    every loaded contact (in the cone's interior), a_ref = -B J v - K imp J q (springs on normal rows only), and
    the implicitfast update v' = v + h (M + h D)^-1 (F_s + J^T f), q' = q + h v'. With imp_deriv the impedance's
    dependence on penetration enters (solimp with d_min < d_max, as the point contact's 0.97-0.995 over 0.5 mm):
    a_ref's spring becomes K (imp + imp' r) and each row's R = (1 - imp) / imp Lambda-hat varies with the normal
    row's penetration, dR / dr = -R imp' / (imp (1 - imp)), which adds -(A + R)^-1 (dR / dr) f to the force.
    Bisection of the step where the largest eigenvalue modulus outside the slow disc passes 1 + 1e-6.
    servo=False drops K_s."""
    import mujoco
    m, d = st.m, st.d
    cons = loaded_contacts(st)
    nv = m.nv
    rows, kmask, nrow = [], [], []
    for c in cons:
        for k in range(c["dim"]):
            rows.append(c["adr"] + k)
            kmask.append(k == 0)
            nrow.append(len(rows) - 1 - k)                     # index (in rows) of this contact's normal row
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(m, M, d.qM)
    Minv = np.linalg.inv(M)
    D = np.diag(_implicit_damping(m))
    Ks = np.diag(_servo_stiffness(m)) if servo else np.zeros((nv, nv))
    if rows:
        J = efc_jac(m, d)[rows]
        KB = np.array(d.efc_KBIP)[rows]
        Rv = np.array(d.efc_R)[rows]
        G = J.T @ np.linalg.inv(J @ Minv @ J.T + np.diag(Rv))
        P = np.eye(nv) - G @ J @ Minv
        kspring = KB[:, 0] * KB[:, 2]
        if imp_deriv:
            pos = np.array(d.efc_pos)[rows]
            kspring = KB[:, 0] * (KB[:, 2] + KB[:, 3] * pos)
        Kq = P @ Ks + G @ np.diag(np.where(kmask, kspring, 0.0)) @ J
        if imp_deriv:
            imp, impP = KB[:, 2], KB[:, 3]
            fr = np.array(d.efc_force)[rows]
            E = np.zeros((len(rows), len(rows)))
            for i_, n_ in enumerate(nrow):
                if 0 < imp[n_] < 1:
                    E[i_, n_] = -Rv[i_] * impP[n_] / (imp[n_] * (1 - imp[n_])) * fr[i_]
            Kq = Kq + G @ E @ J
        Kv = P @ D + G @ np.diag(KB[:, 1]) @ J
    else:
        Kq, Kv = Ks, D
    I = np.eye(nv)

    def rad(h):
        N = np.linalg.inv(M + h * D)
        Mh = np.block([[I - h * h * N @ Kq, h * (I - h * N @ Kv)], [-h * N @ Kq, I - h * N @ Kv]])
        return split_eigs(np.linalg.eigvals(Mh))[0]
    lo, hi = lo_ms * 1e-3, hi_ms * 1e-3
    if rad(hi) <= 1 + 1e-6:
        return None
    if rad(lo) > 1 + 1e-6:
        return 0.0
    while hi - lo > 1e-6:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if rad(mid) <= 1 + 1e-6 else (lo, mid)
    return lo * 1e3


def predictions(st, h):
    """The three bounds of the plan and their numerical versions at the rest state (refsafe off), step h for A_act."""
    import mujoco
    m, d = st.m, st.d
    cons = loaded_tool_contacts(st)
    if not cons:
        return {"n_loaded": 0}
    per = {}
    for c in cons:
        per.setdefault(c["side"], []).append(c)
    fin = {}
    for s, cs in per.items():
        n = len(cs)
        d0 = float(np.mean([c["imp"] for c in cs]))
        tc = float(np.mean([c["tc"] for c in cs]))
        q = d0 + (1 - d0) / n
        fin[s] = {"n": n, "d0": d0, "dmax": float(np.mean([c["dmax"] for c in cs])), "tc_ms": tc * 1e3,
                  "Lambda_hat": float(np.mean([c["diag"] for c in cs])), "N": float(sum(c["N"] for c in cs)),
                  "q": q, "normal_full_ms": 2e3 * tc * (math.sqrt(1 + q) - 1), "normal_damper_ms": 1e3 * q * tc}
    rows_n = [c["adr"] for c in cons]
    rows_t = [c["adr"] + k for c in cons for k in (1, 2) if c["dim"] >= 3]
    en = row_eigs(st, rows_n, h)
    et = row_eigs(st, rows_t, h)
    d0 = float(np.mean([c["imp"] for c in cons]))
    tc = float(np.mean([c["tc"] for c in cons]))
    b = float(np.mean([c["b"] for c in cons]))
    b_t = float(np.mean([c["b_t"] for c in cons if c["b_t"] is not None])) if rows_t else None
    kimp = float(np.mean([c["k"] * c["imp"] for c in cons]))
    lam_hat = float(np.mean([c["diag"] for c in cons]))
    kappa = float(m.opt.impratio)
    out = {"n_loaded": len(cons), "fingers": fin, "d0": d0, "tc_ms": tc * 1e3, "b": b, "b_t": b_t, "kimp": kimp,
           "Lambda_hat": lam_hat, "kappa": kappa, "eig_normal": en, "eig_friction": et,
           "pred_normal_full_ms": min(v["normal_full_ms"] for v in fin.values()),
           "pred_normal_damper_ms": min(v["normal_damper_ms"] for v in fin.values())}
    if et:
        out["pred_friction_ms"] = 2e3 / (b_t * et["lam_prime_act"])
        out["pred_friction_closed_ms"] = 1e3 * tc * (d0 + (1 - d0) * lam_hat / (kappa * et["lam_A"]))
    if en:
        c_ = en["lam_prime_act"] * b
        w2 = en["lam_prime_act"] * kimp
        out["pred_normal_lam_ms"] = 1e3 * (-c_ + math.sqrt(c_ * c_ + 4 * w2)) / w2
    out["pred_jointspace_ms"] = jointspace_threshold(st, servo=True)
    out["pred_jointspace_noservo_ms"] = jointspace_threshold(st, servo=False)
    out["pred_jointspace_constimp_ms"] = jointspace_threshold(st, servo=True, imp_deriv=False)
    allc = loaded_contacts(st)
    out["loaded_by_kind"] = {k: sum(1 for c in allc if c["kind"] == k) for k in sorted({c["kind"] for c in allc})}
    out["tc_ms_by_kind"] = {k: sorted({round(c["tc"] * 1e3, 3) for c in allc if c["kind"] == k})
                            for k in sorted({c["kind"] for c in allc})}
    return out


def mode_shares(st, x0, h):
    """Eigen-decomposition at step h: the eigenvalue of largest modulus and its eigenvector's shares."""
    import mujoco
    m, d = st.m, st.d
    A = transition(st, x0, h)
    ev, V = np.linalg.eig(A)
    cand = np.where(np.abs(ev - 1.0) >= SLOW_DISC)[0]
    i = int(cand[np.argmax(np.abs(ev[cand]))])
    lam = ev[i]
    nv = m.nv
    u = V[nv:2 * nv, i]                                       # velocity part (the position part is proportional)
    set_state(mujoco, m, d, x0)
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(m, M, d.qM)
    ke = {}
    for g, idx in st.groups.items():
        if idx:
            ke[g] = float(np.real(np.conj(u[idx]) @ M[np.ix_(idx, idx)] @ u[idx]))
    jacp, jacr = np.zeros((3, nv)), np.zeros((3, nv))
    mujoco.mj_jacBodyCom(m, d, jacp, jacr, st.tb)
    vc, w = jacp @ u, jacr @ u
    R = d.ximat[st.tb].reshape(3, 3)
    Iw = R @ np.diag(m.body_inertia[st.tb]) @ R.T
    ke_lin = float(m.body_mass[st.tb] * np.real(np.vdot(vc, vc)))
    ke_rot = float(np.real(np.conj(w) @ Iw @ w))
    tot = sum(ke.values())
    J = efc_jac(m, d)
    grp = {"pad_normal": 0.0, "pad_tangential": 0.0, "link": 0.0, "support": 0.0, "finger_other": 0.0, "other": 0.0}
    tors = 0.0
    for c in loaded_contacts(st):
        v2 = np.abs(J[c["adr"]:c["adr"] + c["dim"]] @ u) ** 2
        if c["kind"] == "pad":
            grp["pad_normal"] += float(v2[0])
            grp["pad_tangential"] += float(v2[1:3].sum())
            tors += float(v2[3:].sum())
        else:
            grp[c["kind"]] += float(v2[:3].sum())
    gsum = sum(grp.values())
    shares = {f"cs_{k}": v / gsum for k, v in grp.items()} if gsum > 0 else {}
    sh = {f"ke_{g}": v / tot for g, v in ke.items()} if tot > 0 else {}
    s_f = (ke.get("finger", 0.0) + ke.get("palm", 0.0)) / tot if tot > 0 else None
    s_rot = ke_rot / (ke_rot + ke_lin) if ke_rot + ke_lin > 0 else None
    top = max(grp, key=grp.get) if gsum > 0 else None
    if top is None:
        label = "finger-joint" if (s_f or 0) >= 0.5 else "tool-free"
    elif top == "pad_normal":
        label = "pad-normal"
    elif top == "pad_tangential":
        label = "tool-rotational" if (s_rot or 0) >= 0.5 else "tool-tangential"
    else:
        label = top.replace("_", "-")
    return {"h_ms": h * 1e3, "lambda_re": float(lam.real), "lambda_im": float(lam.imag), "lambda_abs": float(abs(lam)),
            "lambda_arg_deg": float(np.degrees(np.angle(lam))), **sh, "share_hand_ke": s_f, "share_tool_rot_ke": s_rot,
            **shares, "share_contact_normal": (grp["pad_normal"] / (grp["pad_normal"] + grp["pad_tangential"])
                                               if grp["pad_normal"] + grp["pad_tangential"] > 0 else None),
            "torsional_rel": tors / gsum if gsum > 0 else None, "mode": label}


def confirm(st, x0, h, T=T_CONFIRM):
    """Rollout of T seconds at step h from the rest state (refsafe off, commands fixed), started with a velocity
    kick that excites every mode (tool: 1 mm/s and 0.05 rad/s, hand dofs: 0.01 rad/s or 1 mm/s, random directions
    from generator seed 0): bounded when the state stays finite without a MuJoCo bad-state warning and the tool
    stays within 5 mm and 5 deg of its rest pose."""
    import mujoco
    m, d = st.m, st.d
    rho = radius(st, x0, h)                                  # sets the step h and leaves d at x0
    rng = np.random.default_rng(0)
    tool = st.groups["tool"]
    kick = np.zeros(m.nv)
    u = rng.normal(size=3)
    kick[tool[:3]] = 1e-3 * u / np.linalg.norm(u)
    u = rng.normal(size=3)
    kick[tool[3:]] = 0.05 * u / np.linalg.norm(u)
    hand = st.groups["finger"] + st.groups["palm"]
    if hand:
        kick[hand] = 0.01 * rng.choice([-1.0, 1.0], size=len(hand))
    set_state(mujoco, m, d, x0)
    d.qvel[:] += kick
    mujoco.mj_forward(m, d)
    w0 = {int(w): int(d.warning[int(w)].number) for w in (mujoco.mjtWarning.mjWARN_BADQACC,
                                                         mujoco.mjtWarning.mjWARN_BADQVEL,
                                                         mujoco.mjtWarning.mjWARN_BADQPOS)}
    p0 = d.xpos[st.tb].copy()
    R0 = d.xmat[st.tb].reshape(3, 3).copy()
    dp_max = ang_max = 0.0
    why = None
    n = int(round(T / h))
    for k in range(n):
        mujoco.mj_step(m, d)
        if any(int(d.warning[w].number) > c for w, c in w0.items()) or not np.all(np.isfinite(d.qpos)):
            why = f"bad state at {k * h:.3f} s"
            break
        mujoco.mj_kinematics(m, d)
        dp = float(np.linalg.norm(d.xpos[st.tb] - p0)) * 1e3
        R = d.xmat[st.tb].reshape(3, 3)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R0.T @ R) - 1) / 2))))
        dp_max, ang_max = max(dp_max, dp), max(ang_max, ang)
    bounded = why is None and dp_max < CONFIRM_MM and ang_max < CONFIRM_DEG
    if why is None and not bounded:
        why = f"tool moved {dp_max:.2f} mm / {ang_max:.2f} deg"
    return {"h_ms": h * 1e3, "radius": rho, "bounded": bool(bounded), "tool_move_max_mm": dp_max,
            "tool_turn_max_deg": ang_max, "why": why}


def analyse(st, row):
    import mujoco
    x0, info = settle(st)
    row.update(info)
    row["contacts_at_rest"] = {s: sum(1 for c in loaded_tool_contacts(st) if c["side"] == s)
                               for s in sorted(set(st.side_of.values()))}
    row["ncon_at_rest"] = int(st.d.ncon)
    row["nefc_at_rest"] = int(st.d.nefc)
    row["nv"], row["ngeom"] = int(st.m.nv), int(st.m.ngeom)
    row["impratio"] = float(st.m.opt.impratio)
    row["solver"] = {"iterations": int(st.m.opt.iterations), "ls_iterations": int(st.m.opt.ls_iterations),
                     "tolerance": float(st.m.opt.tolerance), "integrator": int(st.m.opt.integrator),
                     "cone": int(st.m.opt.cone)}
    t0 = time.time()
    sc = scan(st, x0)
    row.update(sc)
    row["scan_wall_s"] = time.time() - t0
    h_star = sc.get("threshold_ms")
    set_state(mujoco, st.m, st.d, x0)
    row["pred"] = predictions(st, (h_star or 1.0) * 1e-3)
    if h_star:
        row["mode"] = mode_shares(st, x0, sc["h_unstable_ms"] * 1e-3)
        row["confirm"] = [confirm(st, x0, f * h_star * 1e-3) for f in (0.95, 1.05)]
        p = row["pred"]
        for k in ("pred_normal_full_ms", "pred_normal_damper_ms", "pred_friction_ms", "pred_friction_closed_ms",
                  "pred_normal_lam_ms", "pred_jointspace_ms", "pred_jointspace_noservo_ms",
                  "pred_jointspace_constimp_ms"):
            if p.get(k):
                row.setdefault("rel_err", {})[k] = (h_star - p[k]) / p[k]
                row.setdefault("within10", {})[k] = bool(abs(h_star - p[k]) / p[k] <= 0.10)
    return row


def threshold_jobs(states, models):
    jobs = []
    for s in states:
        if s == "bed":
            jobs += [("bed", "mj_pads1", d0, tc) for tc in BED_TCS_MS for d0 in BED_D0S]
            jobs += [("bed", "mj_point3", d0, tc) for d0, tc in BED_POINT]
        elif s.startswith("rl:"):
            jobs.append((s, {"servo_pads1": "pads", "servo_mesh": "mesh"}.get(s[3:], s[3:]), None, None))
        else:
            jobs += [(s, mdl, None, None) for mdl in models]
    return jobs


def cmd_threshold(a):
    have = {(r["state"], r["model"]) for r in jl(a.out)}
    for state, model, d0, tc in threshold_jobs(a.states, a.models):
        mkey = model if state != "bed" else f"{model}|d0{d0:g}|tc{tc:g}"
        if (state, mkey) in have:
            continue
        t0 = time.time()
        row = {"state": state, "model": mkey, "when": time.strftime("%Y-%m-%d %H:%M")}
        keep = None
        try:
            if state.startswith("scale:"):
                st, keep = state_scale(state[6:], model)
            elif state.startswith("bench:"):
                parts = state.split(":")
                st, keep = state_bench(parts[1], parts[2], model, float(parts[3]) if len(parts) > 3 else None)
            elif state.startswith("rl:"):
                st, keep = state_rl(state[3:])
            else:
                st, keep = state_bed(model, d0, tc)
            row["info"] = st.info
            analyse(st, row)
            row["status"] = "ok"
        except Exception as e:
            row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
        row.update(script=SCRIPT, git_rev=git_rev(), wall_s=time.time() - t0)
        append_row(a.out, row)
        p = row.get("pred", {})
        print(f"{state:28s} {mkey:26s} {row['status']:5s} h* {row.get('threshold_ms')} ms "
              f"(normal {p.get('pred_normal_full_ms', float('nan')):.2f}, friction {p.get('pred_friction_ms', float('nan')):.2f}"
              f", normal-lam {p.get('pred_normal_lam_ms', float('nan')):.2f}, joint {p.get('pred_jointspace_ms') or float('nan'):.2f}/{p.get('pred_jointspace_noservo_ms') or float('nan'):.2f}/{p.get('pred_jointspace_constimp_ms') or float('nan'):.2f}) mode {row.get('mode', {}).get('mode')} "
              f"loaded {row.get('contacts_at_rest')} wall {row['wall_s']:.1f} s"
              + (f"  {row.get('error')}" if row["status"] != "ok" else ""), flush=True)
        del keep
        gc.collect()


# ====================================================================================== films

def cmd_film(a):
    import contact_bed_common as CB
    import reorient_backends_films as RBF
    MEDIA.mkdir(parents=True, exist_ok=True)
    plant = PLANT_OF[a.plant]
    titles = {"pads": "1 mm pads", "pt": "Point contact", "pads2": "2 mm pads", "pt4": "Condim 4"}
    clips = []
    for key in a.models:
        for dt in a.dt:
            out = MEDIA / f"{a.hand}_s{a.seed}_{a.plant}_{key}_dt{dt:g}.mp4"
            if not out.exists():
                t0 = time.time()
                fr = PB.film_mujoco(a.hand, a.seed, key, plant, f"{titles[key]}, {dt:g} ms step", dt_ms=dt)
                CB.write_h264(fr, out)
                print(f"wrote {out.relative_to(ROOT)} ({len(fr)} frames, {time.time() - t0:.0f} s)", flush=True)
                del fr
                gc.collect()
            clips.append(out)
    stem = MEDIA / f"{a.hand}_s{a.seed}_{a.plant}_steps"
    n = RBF.tile_stream(clips, stem.with_suffix(".mp4"), stem.with_name(stem.name + "_poster.png"), cols=len(a.dt),
                        poster_at=a.poster_at)
    print("tile", n, "frames ->", stem.with_suffix(".mp4").relative_to(ROOT), flush=True)


# ====================================================================================== CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("threshold", help="spectral stability thresholds with refsafe off")
    t.add_argument("--states", nargs="+", default=["scale:cylinder:16:0.625", "scale:cylinder:25:0.125",
                                                   "bench:D6:servo", "bench:D6:work", "rl:servo_pads1",
                                                   "rl:servo_mesh", "bed"])
    t.add_argument("--models", nargs="+", default=["pads", "pads2", "pt", "pt4"])
    t.add_argument("--out", type=Path, default=STEP / "threshold.jsonl")
    s = sub.add_parser("sweep", help="plan replays at several physics steps with refsafe on")
    s.add_argument("--hands", nargs="+", default=list(RB.HANDS))
    s.add_argument("--models", nargs="+", default=["pads", "pads2", "pt", "pt4"], choices=PB.MODELS)
    s.add_argument("--plants", nargs="+", default=[SERVO, WORK])
    s.add_argument("--dt", nargs="+", type=float, default=[2.0, 4.0, 5.0, 10.0])
    s.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    s.add_argument("--out", type=Path, default=STEP / "plans_dt.jsonl")
    f = sub.add_parser("film", help="D7 seed 0 at 1, 4 and 10 ms, 1 mm pads and point contact, tiled")
    f.add_argument("--hand", default="D7")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--plant", default="servo", choices=list(PLANT_OF))
    f.add_argument("--models", nargs="+", default=["pads", "pt"])
    f.add_argument("--dt", nargs="+", type=float, default=[1.0, 4.0, 10.0])
    f.add_argument("--poster-at", type=float, default=0.62)
    a = ap.parse_args()
    {"threshold": cmd_threshold, "sweep": cmd_sweep, "film": cmd_film}[a.cmd](a)


if __name__ == "__main__":
    main()
