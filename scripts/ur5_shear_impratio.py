#!/usr/bin/env python3
"""Impratio of the 1 mm sphere pads under the policy trainer's solver: whether impratio 100 or 300 keeps the hand's
replays and the scripted grasp stable at 10 solver iterations and a 2 ms step, how far it lowers the creep, and what it
costs. No training.

replays  CPU MuJoCo with the trainer's numerics (pad_usability_backends 'rl': 2 ms, implicitfast, elliptic cone, Newton,
         10 iterations, 20 line-search iterations, tolerance 1e-8) at impratio 10, 100 and 300. The fingertips are 1 mm
         pads (builder v2) of the UR5 pad's shape, tcd_ell7.4x8.5x11x22_u0.1 (readiness variant ell11_g30), at 10 MPa,
         the scenes' modulus, and at 2.0 MPa, the modulus fitted to the UR5 presses under the 12.5 mm rod. Below a pad
         d0 of 0.05 the solref time constant is lengthened (tr 0.02 s, else 0.03, 0.04 s), since a softer pad is
         reachable only with a longer one (contact overview, step 7). Two command sets on the servo plant: the D1-D8
         bench plans (fe_contact_calibration_sim d1d8, RB.schedule: grip, trajectory, a 1.5 s hold, 4 s more) and the
         readiness cells' logged pinned-iCEM commands for this tip (fe_contact_calibration_sim icem), each from three
         placements (seeds 0-2, RB.jitter). Per run: held, ejected (non-finite state or the tool flung), the tool's
         drift from the end of the hold to the end of the run (D1-D8, the creep), each fingertip's normal force on the
         tool, the share of steps at the iteration cap, us per step.
mjw      The scripted grasp of the 10-08 skin diagnostics (logs/20261008-contact_model_policies/skin_mjw_variants.py:
         keyframe open_ik, finger targets eased to the best rollout's grasp over 0.16 s, the palm raised 0.1 m from
         0.48 s over 0.16 s) on the servo plant's pad scene of the RL runs (results/phase1/real_v1/
         20261011-sv1_u0308_b050_servo_tip_tpu2.7pads1, 10 MPa) and on the same scene with its pads rescaled to 2.0 MPa,
         stepped in MuJoCo-Warp (float32, 256 worlds over 64 placements) and in CPU MuJoCo (16 placements) at the
         trainer's numerics, impratio 10 and 100, to 2 s: held (the tool rose with the palm and two fingers touch it),
         the tool's slip against the fingertips from 1 to 2 s, steps at the iteration cap, constraint rows per world
         against the trainer's njmax, us per step or world-step.
creep    The pads' creep under the UR5 hold forces (the rig of ur5_tip_calibration_pads.py --shear: the pads fixed, the
         sphere or rod on three slides pushed with each press hold's measured force vector for 1 s, friction 1, its
         tangential speed over the last 0.5 s) at impratio 10, 100 and 300, with the bed's numerics (1 ms, 100
         iterations) and the trainer's, at the moduli fitted to the presses (0.90 MPa sphere, 2.0 MPa rod) and at 10 MPa.

Rows: docs/experiments/20261026-shear_friction_impratio/data/20261026-impratio_{replays,mjw,creep}.jsonl, one fsynced
line per run; `replays` and `creep` resume by key. Scenes are cached under logs/20261026-shear_friction_impratio/.

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- .venv/bin/python scripts/ur5_shear_impratio.py replays --E 10
    WARP_CACHE_PATH=$(mktemp -d) .venv/bin/python scripts/ur5_shear_impratio.py mjw
    .venv/bin/python scripts/ur5_shear_impratio.py creep
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
OUT = ROOT / "docs/experiments/20261026-shear_friction_impratio/data"
LOGS = ROOT / "logs/20261026-shear_friction_impratio"
TIP = "ell11_g30"
IRS = (10.0, 100.0, 300.0)
TRS = (0.02, 0.03, 0.04)                 # s: solref relaxation times tried, tc = tr / 2
RL_OPT = dict(timestep=0.002, iterations=10, ls_iterations=20, tolerance=1e-8)
FINGERS = ("thumb", "index", "middle")
SCRIPT = "scripts/ur5_shear_impratio.py"


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rows_of(path):
    out = []
    if Path(path).exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def niter(d):
    """Solver iterations of the last step: the largest over the constraint islands."""
    n = max(1, int(getattr(d, "nisland", 1) or 1))
    return int(np.max(d.solver_niter[:n]))


def diverged(d):
    """MuJoCo resets mjData when an acceleration is not finite or huge (mjWARN_BADQACC), and d.time starts over; a
    replay that only watched d.time would then loop for ever (2026-10-26, D2 at impratio 100)."""
    import mujoco
    return int(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number) > 0


def set_rl(m, ir):
    import mujoco
    m.opt.timestep = RL_OPT["timestep"]
    m.opt.iterations = RL_OPT["iterations"]
    m.opt.ls_iterations = RL_OPT["ls_iterations"]
    m.opt.tolerance = RL_OPT["tolerance"]
    m.opt.impratio = ir
    m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON


# ------------------------------------------------------------------------------------------------ replays

def with_tr(fn):
    """fn(tr) at the shortest relaxation time of TRS whose pads keep d0 >= 0.05 (replace_tips raises below)."""
    last = None
    for tr in TRS:
        try:
            return fn(tr), tr
        except ValueError as e:
            if "d0" not in str(e):
                raise
            last = e
    raise last


def run_d1d8(scene, meta, plan, traj, seed, ir):
    """RB.run_mujoco's bench replay at the trainer's numerics, with the iteration cap and per-finger forces."""
    import mujoco
    import reorient_backends as RB
    m = mujoco.MjModel.from_xml_path(str(scene))
    set_rl(m, ir)
    d = mujoco.MjData(m)
    tb = m.body(RB.OBJ).id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    tool7 = np.asarray(meta["tool7"], float)
    dx, dy, dyaw = RB.jitter(seed)
    d.qpos[qa:qa + 3] = tool7[:3] + [dx, dy, 0.0]
    d.qpos[qa + 3:qa + 7] = RB._yaw_quat(tool7[3:], dyaw)
    act = {}
    for f in RB.FINGERS:
        for j in RB.JOINTS:
            n = f"{f}_{j}"
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = meta["q0"][n]
            act[n] = m.actuator(f"a_{n}").id
    mujoco.mj_forward(m, d)
    tool_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == tb}
    finger_of = {}
    for i in range(m.ngeom):
        bn = m.body(m.geom_bodyid[i]).name
        for f in RB.FINGERS:
            if bn.startswith(f + "_"):
                finger_of[i] = f
    f6 = np.zeros(6)
    trace, W, NI = [], [], []
    t_bad = None

    def snap():
        F = {f: 0.0 for f in RB.FINGERS}
        cnt = {f: 0 for f in RB.FINGERS}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if (g1 in tool_geoms) == (g2 in tool_geoms):
                continue
            f = finger_of.get(g2 if g1 in tool_geoms else g1)
            if f is None:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            if f6[0] <= 1e-6:
                continue
            F[f] += float(f6[0])
            cnt[f] += 1
        R9 = np.zeros(9)
        mujoco.mju_quat2Mat(R9, d.qpos[qa + 3:qa + 7])
        p = d.qpos[qa:qa + 3]
        trace.append({"t": round(float(d.time), 4), "cos": float(R9[8]), "x": float(p[0]), "y": float(p[1]),
                      "z": float(p[2]), "F": {k: round(v, 4) for k, v in F.items()}, "n": cnt})

    segs = RB.schedule(plan, traj)
    k_trace = max(1, int(round(RB.TRACE_DT / m.opt.timestep)))
    step, bad = 0, False
    for t_end, tgt, _ in segs:
        for n_, a in act.items():
            d.ctrl[a] = tgt[n_]
        while d.time < t_end - 1e-9:
            w0 = time.perf_counter()
            mujoco.mj_step(m, d)
            W.append(time.perf_counter() - w0)
            NI.append(niter(d))
            step += 1
            if diverged(d):
                t_bad = float(step * m.opt.timestep)
                bad = True
                break
            if step % k_trace == 0:
                snap()
                if not np.all(np.isfinite(d.qpos)) or abs(d.qpos[qa + 2]) > 2.0:
                    t_bad = float(d.time)
                    bad = True
                    break
        if bad:
            break
    res = RB.score(trace, segs) if not bad else {}
    t_hold = [e for e, _, k in segs if k == "hold"][-1]
    h = next((s for s in trace if s["t"] >= t_hold - 1e-9), trace[-1])
    e = trace[-1]
    NI = np.array(NI)
    res.update(status="ejected" if bad else "complete", t_unstable_s=t_bad, us_per_step_median=float(np.median(W) * 1e6),
               n_steps=step,
               cap_share=float(np.mean(NI >= m.opt.iterations)), niter_mean=float(NI.mean()),
               F_hold_N=h["F"], F_end_N=e["F"], t_hold_s=float(t_hold), t_end_s=float(e["t"]),
               creep_um_s=(res["drift_hold_to_end_mm"] * 1e3 / max(e["t"] - t_hold, 1e-6)) if not bad else None,
               trace=[[s["t"], round(s["x"] * 1e3, 4), round(s["y"] * 1e3, 4), round(s["z"] * 1e3, 4), round(s["cos"], 6),
                       [s["F"][f] for f in FINGERS]] for s in trace[::5]])
    return res


def cmd_replays(a):
    import argparse as AP
    import fe_contact_calibration_sim as FS
    import reorient_backends as RB
    out = OUT / "20261026-impratio_replays.jsonl"
    have = {(r["set"], r["plan"], r["E_MPa"], r["ir"], r["seed"]) for r in rows_of(out)
            if r.get("status") in ("complete", "ejected", "ok", "unstable")}
    key = FS.spec_of(TIP)["key"]
    for E_MPa in a.E:
        E = E_MPa * 1e6
        for ir in a.ir:
            arm = f"rl{ir:g}"
            FS.ARMS[arm] = dict(sim="mujoco", numerics="rl", ir=ir, mu="1.0")
            if "d1d8" in a.sets:
                for hand in a.hands:
                    todo = [s for s in a.seeds if ("d1d8", hand, E_MPa, ir, s) not in have]
                    if not todo:
                        continue
                    plan, traj = RB.load_plan(hand)

                    def build(tr):
                        RB.PAD_TR = tr
                        FS.LOGS = LOGS / f"tr{tr:g}"
                        return FS.d1d8_scene(TIP, key, arm, E, 0.001, None, hand)
                    try:
                        (scene, meta), tr = with_tr(build)
                    except Exception as e:
                        print(f"{hand} E {E_MPa} ir {ir}: scene failed {e!r}", flush=True)
                        traceback.print_exc()
                        continue
                    for sd in todo:
                        t0 = time.time()
                        row = dict(set="d1d8", plan=hand, tip=TIP, key=key, E_MPa=E_MPa, ir=ir, tr_s=tr, seed=sd,
                                   numerics="rl", opt=RL_OPT, plant=FS.SERVO, scene=str(Path(scene).relative_to(ROOT)),
                                   pad_d0=meta.get("pad_d0"))
                        try:
                            row.update(run_d1d8(scene, meta, plan, traj, sd, ir))
                        except Exception as e:
                            row.update(status="failed", error=repr(e), tb=traceback.format_exc()[-1500:])
                        row.update(script=SCRIPT, wall_s=round(time.time() - t0, 2), when=time.strftime("%Y-%m-%d %H:%M"))
                        append(out, row)
                        print(f"d1d8 {hand} E {E_MPa:g} ir {ir:g} tr {tr:g} s{sd}: {row.get('status')} held "
                              f"{row.get('held_end')} turn {row.get('turn_end_deg', float('nan')):+.1f} creep "
                              f"{row.get('creep_um_s')} cap {row.get('cap_share', float('nan')):.3f} "
                              f"{row.get('us_per_step_median', float('nan')):.0f} us {row['wall_s']:.1f} s", flush=True)
                    gc.collect()
            if "icem" in a.sets:
                for pid, tip, cs, ci, log, logged in FS.icem_plans():
                    if tip != TIP:
                        continue
                    todo = [s for s in a.seeds if ("icem", pid, E_MPa, ir, s) not in have]
                    if not todo:
                        continue

                    def build(tr):
                        RB.PAD_TR = tr
                        FS.LOGS = LOGS / f"tr{tr:g}"
                        return FS.plants(tip, cs, ci, E, "pads", 0.001, ir=ir)
                    try:
                        (plant, meta, S), tr = with_tr(build)
                    except Exception as e:
                        print(f"{pid} E {E_MPa} ir {ir}: build failed {e!r}", flush=True)
                        traceback.print_exc()
                        continue
                    set_rl(plant.m, ir)
                    stats = dict(W=[], NI=[])

                    def advance(t1, plant=plant, stats=stats):
                        while plant.d.time < t1 - 1e-9:
                            w0 = time.perf_counter()
                            t_prev = float(plant.d.time)
                            plant.mj.mj_step(plant.m, plant.d)
                            stats["W"].append(time.perf_counter() - w0)
                            stats["NI"].append(niter(plant.d))
                            if diverged(plant.d):
                                raise FloatingPointError(f"unstable at t {t_prev:.3f} s (MuJoCo reset the state)")
                    plant.advance = advance
                    for sd in todo:
                        stats["W"].clear()
                        stats["NI"].clear()
                        t0 = time.time()
                        row = dict(set="icem", plan=pid, tip=tip, cell=cs, cand=ci, E_MPa=E_MPa, ir=ir, tr_s=tr, seed=sd,
                                   numerics="rl", opt=RL_OPT, plant=FS.SERVO, logged=logged)
                        try:
                            res, tr_ = FS.replay(plant, meta, log, sd)
                            NI = np.array(stats["NI"])
                            last = tr_[-1]
                            row.update(res, status="ok", us_per_step_median=float(np.median(stats["W"]) * 1e6),
                                       cap_share=float(np.mean(NI >= plant.m.opt.iterations)), niter_mean=float(NI.mean()),
                                       F_end_N=last["N"], n_end=last["n"],
                                       trace=[[s["t"], s["p"], s["th"], [s["N"][f] for f in FINGERS]] for s in tr_[::5]])
                        except FloatingPointError as e:
                            row.update(status="unstable", held=False, error=str(e),
                                       us_per_step_median=float(np.median(stats["W"]) * 1e6) if stats["W"] else None,
                                       cap_share=float(np.mean(np.array(stats["NI"]) >= plant.m.opt.iterations)) if stats["NI"] else None)
                            for w_ in plant.d.warning:
                                w_.number = 0
                        except Exception as e:
                            row.update(status="failed", error=repr(e), tb=traceback.format_exc()[-1500:])
                        row.update(script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"))
                        append(out, row)
                        print(f"icem {pid} E {E_MPa:g} ir {ir:g} tr {tr:g} s{sd}: {row.get('status')} held {row.get('held')} "
                              f"turn {row.get('turn_end_deg', float('nan')):+.1f} (logged {logged['turn_end_deg']:+.1f}) cap "
                              f"{row.get('cap_share', float('nan')):.3f} {row.get('us_per_step_median', float('nan')):.0f} us "
                              f"{time.time() - t0:.1f} s", flush=True)
                    del plant
                    gc.collect()


# ------------------------------------------------------------------------------------------------ mjw

SERVO_RUN = ROOT / "results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip_tpu2.7pads1"
RUNS = {"pads": SERVO_RUN, "skin": ROOT / "results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip_tpu2.7skin"}
T_LIFT, T_END = (0.48, 0.64), 2.0          # s: the palm rises 0.1 m over T_LIFT; the run ends at T_END
T_CREEP = 1.0                               # s: the tool's slip against the fingertips is read from here to T_END


def rl_scene(E_MPa, ir, min_d0=0.05, skin_k=None):
    """The servo plant's pad scene of the RL runs at the trainer's numerics, its pads rescaled from 10 MPa to E_MPa
    (each pad's K = 1 / (tc^2 (1 - d0) A), A the tip's and the tool's inverse weights; d0 re-solved at the shortest
    time constant of TRS / 2 that keeps d0 >= 0.05)."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(SERVO_RUN / "frozen_scene.xml"))
    set_rl(m, ir)
    tool = m.body("screwdriver_medium").id
    pads = [i for i in range(m.ngeom) if "_pad" in m.geom(i).name]
    tc_used = float(m.geom_solref[pads[0], 0])
    if abs(E_MPa - 10.0) > 1e-9:
        A = {f: float(m.body_invweight0[m.body(f"{f}_tip").id, 0] + m.body_invweight0[tool, 0]) for f in FINGERS}
        K = np.array([1.0 / (m.geom_solref[i, 0] ** 2 * (1.0 - m.geom_solimp[i, 0]) * A[m.geom(i).name.split("_")[0]])
                      for i in pads]) * (E_MPa / 10.0)
        Af = np.array([A[m.geom(i).name.split("_")[0]] for i in pads])
        for tc in (0.01, 0.0125, 0.015, 0.0175, 0.02):
            d0 = 1.0 - 1.0 / (tc ** 2 * K * Af)
            if d0.min() >= min_d0:
                break
        else:
            raise ValueError(f"pads at {E_MPa} MPa: d0 {d0.min():.3f} below 0.05 at tc {tc} s")
        for i, v in zip(pads, d0):
            m.geom_solref[i] = [tc, 1.0]
            m.geom_solimp[i, 0] = m.geom_solimp[i, 1] = v
        tc_used = tc
    if skin_k:                                   # the skin's springs re-sized, critical damping on mass plus armature
        for j in range(m.njnt):
            nm = m.joint(j).name
            if "_skin_" not in nm:
                continue
            dof = m.jnt_dofadr[j]
            k = skin_k["kth_N_m_rad"] if nm.endswith("_t") else skin_k["kt_N_m"]
            b = m.body(m.jnt_bodyid[j])
            inert = float(b.inertia[0]) if nm.endswith("_t") else float(b.mass[0])
            m.jnt_stiffness[j] = k
            m.dof_damping[dof] = 2.0 * math.sqrt(k * (inert + float(m.dof_armature[dof])))
    return m, dict(n_pads=len(pads), tc_s=tc_used, d0=[float(m.geom_solimp[pads, 0].min()), float(m.geom_solimp[pads, 0].max())],
                   skin_k=skin_k)


def rl_scene_run(run, ir, skin_k=None):
    """Another run's frozen scene at the trainer's numerics (its pads at their own modulus), its skin springs optionally
    re-sized (skin_k: ur5_shear_skin.sizing)."""
    global SERVO_RUN
    saved = SERVO_RUN
    SERVO_RUN = RUNS[run]
    try:
        return rl_scene(10.0, ir, 0.05, skin_k)
    finally:
        SERVO_RUN = saved


def grasp_setup(m, run=SERVO_RUN):
    import mujoco
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("open_ik").id)
    mujoco.mj_forward(m, d)
    bfc = np.load(run / "best_rollout.npz")["best_finger_ctrl"]
    fa = [m.actuator(f"a_{f}_{j}").id for f in FINGERS for j in ("yaw", "mcp", "pip")]
    return d, d.ctrl.copy(), bfc, fa, m.actuator("a_palm_pz").id, m.body("screwdriver_medium").id, m.body("palm_pose").id


def ctrl_at(t, c0, bfc, fa, pz):
    c = c0.copy()
    a = min(1.0, t / 0.16)
    e = 1 - (1 - a) ** 2
    c[fa] = c0[fa] + e * (bfc - c0[fa])
    c[pz] = c0[pz] + 0.1 * min(1.0, max(0.0, (t - T_LIFT[0]) / (T_LIFT[1] - T_LIFT[0])))
    return c


def placements(m, d, n):
    """The keyframe's state with the tool moved by RB.jitter(seed) for seeds 0..n-1 (sd 2 mm and 2 deg; seed 0 none)."""
    import reorient_backends as RB
    tb = m.body("screwdriver_medium").id
    qa = m.jnt_qposadr[m.body_jntadr[tb]]
    Q = np.tile(d.qpos, (n, 1))
    for s in range(n):
        dx, dy, dyaw = RB.jitter(s)
        Q[s, qa:qa + 3] += [dx, dy, 0.0]
        Q[s, qa + 3:qa + 7] = RB._yaw_quat(d.qpos[qa + 3:qa + 7], dyaw)
    return Q


def tool_in_tips(xpos, xmat, tool, tips):
    """The tool's centre in each fingertip body's frame, (worlds, fingers, 3) m."""
    out = np.zeros((xpos.shape[0], len(tips), 3))
    for k, b in enumerate(tips):
        R = xmat[:, b].reshape(-1, 3, 3)
        out[:, k] = np.einsum("wji,wj->wi", R, xpos[:, tool] - xpos[:, b])
    return out


def fingers_on_tool(geom_pairs, worlds, nw, tool_geoms, finger_of):
    """Number of fingers with a contact on the tool, per world, from contact geom pairs and their world ids."""
    on = np.zeros((nw, len(FINGERS)), bool)
    for (g1, g2), w in zip(geom_pairs, worlds):
        if (g1 in tool_geoms) == (g2 in tool_geoms):
            continue
        f = finger_of.get(g2 if g1 in tool_geoms else g1)
        if f is not None:
            on[w, FINGERS.index(f)] = True
    return on.sum(1)


def grasp_row(sim, E_MPa, ir, info, nw, n_pl, z0, ze, ra, rb_, fin, nf, run="pads"):
    """Held: the tool rose with the palm (within 20 mm) and at least two fingers touch it at the end. Slip: the median
    over the three fingertips of the tool centre's speed in the fingertip's frame from T_CREEP to T_END, um/s."""
    held = fin & (ze - z0 > 0.08) & (nf >= 2)
    slip = np.median(np.linalg.norm(rb_ - ra, axis=2), axis=1) / (T_END - T_CREEP) * 1e6
    sh = slip[held]
    return dict(sim=sim, run=run, E_MPa=E_MPa, ir=ir, scene=str(RUNS[run].relative_to(ROOT)), **info, opt=RL_OPT, n=nw,
                n_placements=n_pl, held=int(held.sum()), held_placements=int(held[:n_pl].sum()), nonfinite=int((~fin).sum()),
                slip_um_s=dict(median=float(np.median(sh)) if len(sh) else None,
                               p90=float(np.percentile(sh, 90)) if len(sh) else None),
                dz_mm=dict(median=float(np.nanmedian((ze - z0)[fin]) * 1e3) if fin.any() else None),
                per_world=dict(held=held[:n_pl].tolist(),
                               slip_um_s=[None if not np.isfinite(x) else round(float(x), 2) for x in slip[:n_pl]],
                               dz_mm=[None if not np.isfinite(x) else round(float(x) * 1e3, 2) for x in (ze - z0)[:n_pl]]),
                script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"))


def cmd_mjw(a):
    import mujoco
    out = OUT / "20261026-impratio_mjw.jsonl"
    for E_MPa in (a.E if a.run == "pads" else [10.0]):
        for ir in a.ir:
            if a.run == "skin" and a.skin_from_slides:
                import ur5_shear_skin as SK
                skin_k = SK.sizing(a.skin_from_slides)
            else:
                skin_k = None
            m, info = rl_scene(E_MPa, ir, a.min_d0, skin_k) if a.run == "pads" else rl_scene_run(a.run, ir, skin_k)
            d, c0, bfc, fa, pz, tool, palm = grasp_setup(m, RUNS[a.run])
            tips = [m.body(f"{f}_tip").id for f in FINGERS]
            tool_geoms = {i for i in range(m.ngeom) if m.geom_bodyid[i] == tool}
            finger_of = {i: f for i in range(m.ngeom) for f in FINGERS if m.body(m.geom_bodyid[i]).name.startswith(f + "_")}
            Q = placements(m, d, a.n)
            n_steps = int(round(T_END / m.opt.timestep))
            k_creep = int(round(T_CREEP / m.opt.timestep))
            if "cpu" in a.sims:
                n = a.n_cpu
                z0, ze, ra, rb_, fin, nf, W, NI, NE = (np.zeros(n), np.full(n, np.nan), np.full((n, 3, 3), np.nan),
                                                     np.full((n, 3, 3), np.nan), np.ones(n, bool), np.zeros(n, int), [], [], [])
                for s in range(n):
                    dd = mujoco.MjData(m)
                    dd.qpos[:] = Q[s]
                    dd.ctrl[:] = c0
                    mujoco.mj_forward(m, dd)
                    z0[s] = dd.xpos[tool, 2]
                    for i in range(n_steps):
                        dd.ctrl[:] = ctrl_at(i * m.opt.timestep, c0, bfc, fa, pz)
                        w0 = time.perf_counter()
                        mujoco.mj_step(m, dd)
                        W.append(time.perf_counter() - w0)
                        NI.append(niter(dd))
                        NE.append(dd.nefc)
                        if diverged(dd):
                            fin[s] = False
                            break
                        if i + 1 == k_creep:
                            ra[s] = tool_in_tips(dd.xpos[None], dd.xmat[None], tool, tips)[0]
                        if not np.all(np.isfinite(dd.qpos)):
                            fin[s] = False
                            break
                    if fin[s]:
                        ze[s] = dd.xpos[tool, 2]
                        rb_[s] = tool_in_tips(dd.xpos[None], dd.xmat[None], tool, tips)[0]
                        nf[s] = fingers_on_tool([tuple(c.geom) for c in dd.contact[:dd.ncon]], [0] * dd.ncon, 1,
                                                tool_geoms, finger_of)[0]
                NI = np.array(NI)
                row = grasp_row("cpu", E_MPa, ir, info, n, n, z0, ze, ra, rb_, fin, nf, a.run)
                row.update(cap_share=float(np.mean(NI >= m.opt.iterations)), niter_mean=float(NI.mean()),
                           nefc_max=int(max(NE)), us_per_step_median=float(np.median(W) * 1e6))
                append(out, row)
                print(f"cpu {a.run} E {E_MPa:g} ir {ir:g} tc {info['tc_s']:g}: held {row['held']}/{n}, nonfinite {row['nonfinite']}, slip "
                      f"{row['slip_um_s']['median']} um/s, dz {row['dz_mm']['median']} mm, cap {row['cap_share']:.3f}, nefc max "
                      f"{row['nefc_max']}, {row['us_per_step_median']:.0f} us/step", flush=True)
            if "mjw" in a.sims:
                import mujoco_warp as mjw
                import warp as wp
                nw = a.n_gpu
                njmax = a.njmax if abs(E_MPa - 10.0) < 1e-9 else a.njmax_soft
                wm = mjw.put_model(m)
                wd = mjw.make_data(m, nworld=nw, nconmax=a.nconmax, njmax=njmax)
                Qg = np.tile(Q, (int(math.ceil(nw / len(Q))), 1))[:nw]
                wd.qpos.assign(Qg.astype(np.float32))
                wd.qvel.zero_()
                if hasattr(wd, "qacc_warmstart"):
                    wd.qacc_warmstart.zero_()
                wd.ctrl.assign(np.tile(c0, (nw, 1)).astype(np.float32))
                mjw.forward(wm, wd)
                wp.synchronize()
                z0 = wd.xpos.numpy()[:, tool, 2].astype(float).copy()
                t_run, ra, NIg, NEg = 0.0, None, [], []
                for i in range(n_steps):
                    wd.ctrl.assign(np.tile(ctrl_at(i * m.opt.timestep, c0, bfc, fa, pz), (nw, 1)).astype(np.float32))
                    wp.synchronize()
                    w0 = time.perf_counter()
                    mjw.step(wm, wd)
                    wp.synchronize()
                    t_run += time.perf_counter() - w0
                    if i % 10 == 0:
                        NIg.append(wd.solver_niter.numpy().copy())
                        NEg.append(wd.nefc.numpy().copy())
                    if i + 1 == k_creep:
                        ra = tool_in_tips(wd.xpos.numpy().astype(float), wd.xmat.numpy().astype(float), tool, tips)
                xp, xm = wd.xpos.numpy().astype(float), wd.xmat.numpy().astype(float)
                fin = np.all(np.isfinite(wd.qpos.numpy()), axis=1)
                rb_ = tool_in_tips(xp, xm, tool, tips)
                nc = int(wd.nacon.numpy()[0])
                nf = fingers_on_tool([tuple(g) for g in wd.contact.geom.numpy()[:nc]], wd.contact.worldid.numpy()[:nc], nw,
                                     tool_geoms, finger_of)
                NIg, NEg = np.array(NIg), np.array(NEg)
                row = grasp_row("mjw", E_MPa, ir, info, nw, len(Q), z0, xp[:, tool, 2], ra, rb_, fin, nf, a.run)
                row.update(cap_share=float(np.mean(NIg >= m.opt.iterations)), nefc_max=int(NEg.max()),
                           nefc_over_njmax_share=float(np.mean(NEg > njmax)), us_per_world_step=float(t_run / n_steps / nw * 1e6),
                           nconmax=a.nconmax, njmax=njmax, nacon_end=nc)
                append(out, row)
                print(f"mjw {a.run} E {E_MPa:g} ir {ir:g} njmax {njmax}: held {row['held']}/{nw} (placements {row['held_placements']}/"
                      f"{len(Q)}), nonfinite {row['nonfinite']}, slip {row['slip_um_s']['median']} um/s, dz "
                      f"{row['dz_mm']['median']} mm, cap {row['cap_share']:.3f}, nefc max {row['nefc_max']}, "
                      f"{row['us_per_world_step']:.2f} us/world-step", flush=True)
                del wm, wd
                gc.collect()


# ------------------------------------------------------------------------------------------------ creep

UR5 = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
CREEP_SESSIONS = {"sphere2": (UR5 / "data/20261022-210852-sphere2", UR5 / "lab/20261022-210852-sphere2/plan.json"),
                  "rod1": (UR5 / "data/20261023-104526-rod1-joint", UR5 / "lab/20261023-104526-rod1/plan.json")}


def cmd_creep(a):
    """ur5_tip_calibration_pads.py --shear at both numerics: per press hold the indenter starts at the pads' law depth
    under the measured normal force and takes the measured force vector for 1 s; its tangential speed over the last
    0.5 s. Also the share of steps at the iteration cap and us per step."""
    import mujoco
    import ur5_tip_calibration_pads as UP
    out = OUT / "20261026-impratio_creep.jsonl"
    have = {(r["session"], r["E_MPa"], r["numerics"], r["ir"]) for r in rows_of(out)}
    pads = UP.Pads()
    for name in a.sessions:
        sd, pp = CREEP_SESSIONS[name]
        H = np.load(sd / "holds.npz", allow_pickle=False)
        plan = json.loads(pp.read_text())
        ind = plan["geometry"]["indenter"]
        R, kind = float(ind["radius"]), ind.get("kind", "sphere")
        E_fit = json.loads((sd / "pads.json").read_text())["E_meas_MPa"]
        idx = np.nonzero((H["kind"] == "press") & np.isfinite(H["f_n"]) & (H["f_n"] > 0.2))[0][::a.every]
        AX = [H["axis_P"][i] if kind == "rod" else None for i in idx]
        for E in (E_fit, 10.0):
            def delta_law(j, i):
                c, n, f_m = H["c_P"][i], H["n"][i], H["f_n"][i]
                lo, hi = -1.5, float(H["depth"][i]) + 0.5
                if E * float(pads.law(c + lo * n, R, AX[j])[0] @ n) < f_m:
                    return None
                for _ in range(40):
                    mid = 0.5 * (lo + hi)
                    if E * float(pads.law(c + mid * n, R, AX[j])[0] @ n) > f_m:
                        lo = mid
                    else:
                        hi = mid
                return 0.5 * (lo + hi)
            dl = [delta_law(j, i) for j, i in enumerate(idx)]
            for numerics in ("bed", "rl"):
                for ir in a.ir:
                    if (name, round(E, 4), numerics, ir) in have:
                        continue
                    rig = UP.Rig(pads, R, kind, mu=1.0, shear=True, impratio=ir)
                    rig.set_E(E)
                    if numerics == "rl":
                        set_rl(rig.m, ir)
                    m, d = rig.m, rig.d
                    v, W, NI, bad = [], [], [], 0
                    t0 = time.time()
                    for j, i in enumerate(idx):
                        if dl[j] is None:
                            continue
                        c = H["c_P"][i] + dl[j] * H["n"][i]
                        m.body_pos[rig.b] = np.asarray(c) * 1e-3
                        if AX[j] is not None:
                            q = np.zeros(4)
                            mujoco.mju_quatZ2Vec(q, np.asarray(AX[j], float) / np.linalg.norm(AX[j]))
                            m.geom_quat[rig.g] = q
                        mujoco.mj_resetData(m, d)
                        d.qfrc_applied[:3] = H["f"][i]
                        nT = int(round(1.0 / m.opt.timestep))
                        for k in range(nT):
                            w0 = time.perf_counter()
                            mujoco.mj_step(m, d)
                            W.append(time.perf_counter() - w0)
                            NI.append(niter(d))
                            if k == nT // 2 - 1:
                                q_half = d.qpos[:3].copy()
                        if not np.all(np.isfinite(d.qpos)):
                            bad += 1
                            continue
                        u = np.asarray(H["n"][i], float)
                        dq = (d.qpos[:3] - q_half) * 1e3
                        v.append(float(np.linalg.norm(dq - (dq @ u) * u)) / 0.5)
                    v, NI = np.array(v), np.array(NI)
                    row = dict(session=name, indenter=dict(kind=kind, R_mm=R), E_MPa=round(E, 4), numerics=numerics, ir=ir,
                               timestep_s=float(m.opt.timestep), iterations=int(m.opt.iterations), n=int(len(v)),
                               nonfinite=bad, creep_mm_s=dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)),
                                                              max=float(v.max())),
                               cap_share=float(np.mean(NI >= m.opt.iterations)), niter_mean=float(NI.mean()),
                               us_per_step_median=float(np.median(W) * 1e6), every=a.every,
                               script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"))
                    append(out, row)
                    print(f"creep {name} E {E:.3g} {numerics} ir {ir:g}: {np.median(v) * 1e3:.1f} um/s median over {len(v)}, "
                          f"non-finite {bad}, cap {row['cap_share']:.3f}, {row['us_per_step_median']:.0f} us "
                          f"({time.time() - t0:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replays")
    r.add_argument("--E", nargs="+", type=float, default=[10.0, 2.0], help="pad modulus, MPa")
    r.add_argument("--ir", nargs="+", type=float, default=list(IRS))
    r.add_argument("--sets", nargs="+", default=["d1d8", "icem"])
    r.add_argument("--hands", nargs="+", default=[f"D{k}" for k in range(1, 9)])
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    g = sub.add_parser("mjw")
    g.add_argument("--E", nargs="+", type=float, default=[10.0, 2.0])
    g.add_argument("--ir", nargs="+", type=float, default=[10.0, 100.0])
    g.add_argument("--sims", nargs="+", default=["cpu", "mjw"])
    g.add_argument("--n", type=int, default=64, help="placements (RB.jitter seeds)")
    g.add_argument("--n-cpu", type=int, default=16)
    g.add_argument("--n-gpu", type=int, default=256)
    g.add_argument("--nconmax", type=int, default=640, help="per world, the trainer's for pad scenes")
    g.add_argument("--njmax", type=int, default=3072, help="per world at 10 MPa, the trainer's for pad scenes")
    g.add_argument("--njmax-soft", type=int, default=8192, help="per world at a softer modulus")
    g.add_argument("--run", choices=list(RUNS), default="pads", help="the RL run whose frozen scene is gripped")
    g.add_argument("--min-d0", type=float, default=0.05, help="the softened pads' smallest d0 (picks the time constant)")
    g.add_argument("--skin-from-slides", type=float, default=None,
                   help="re-size the skin's springs from the UR5 slides at this fingertip force, N (ur5_shear_skin.sizing)")
    c = sub.add_parser("creep")
    c.add_argument("--sessions", nargs="+", default=list(CREEP_SESSIONS))
    c.add_argument("--ir", nargs="+", type=float, default=list(IRS))
    c.add_argument("--every", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "replays":
        cmd_replays(a)
    elif a.cmd == "mjw":
        cmd_mjw(a)
    else:
        cmd_creep(a)


if __name__ == "__main__":
    main()
