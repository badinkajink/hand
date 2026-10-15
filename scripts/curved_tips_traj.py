#!/usr/bin/env python3
"""Contact dynamics compared on trajectories (docs/handoff/20261015-curved_tips.md, step 4; owner, 2026-10-15: "held" is
too coarse to show that two contact models give the same dynamics).

Each run replays one deployed hand's bench plan (D1-D8, servo-limited plant kp0.72_kv0_fr0.18_fl0_dp0.0144, the
tool placed with reorient_backends.jitter(seed)) open loop, so every arm receives the same servo commands, and records
every 20 ms the tool's position and orientation and, per finger, the normal and friction force it applies to the tool:
the finger's contact forces on the tool summed as one vector F_f, its normal part N_f = |F_f . n_f| along n_f, the
normal-force-weighted mean contact normal, and its friction part T_f = |F_f - (F_f . n_f) n_f|.

Arms:
  drake   Drake 1.57 hydroelastic: the tip's compliant convex mesh (E 10 MPa, relaxation 0.01 s), SAP, 1 ms (the reference)
  plan    MuJoCo 3.6 with the tip's 1 mm pads (builder v2) at the planner's and evaluation numerics: 1 ms, impratio 100,
          Newton with 100 iterations at tolerance 1e-10
  rl      the same pads at the RL trainer's numerics: 2 ms, impratio 10, 10 iterations, tolerance 1e-8
  floor   'plan' with friction 1.001 in place of 1.0: the divergence that a 0.1 % change of one parameter produces
  dfloor  'drake' with friction 1.001

Rows: docs/experiments/20261015-curved_tips/traj/traces.jsonl, one fsynced row per (arm, tip, hand, seed), resumable.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/curved_tips_traj.py --tips tcd_cap6_u0.1 --arms plan rl floor drake --hands D1 D2 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curved_tips_common as C  # noqa: E402
import pad_usability_backends as PUB  # noqa: E402
import reorient_backends as RB  # noqa: E402

SCRIPT = "scripts/curved_tips_traj.py"
OUT = C.D / "traj" / "traces.jsonl"
PUB.SCN = C.LOGS / "scenes"
TRACE_DT = 0.02
ARMS = {"drake": dict(sim="drake", mu="1.0"), "dfloor": dict(sim="drake", mu="1.001"),
        "plan": dict(sim="mujoco", numerics="bed", ir=100.0, mu="1.0"),
        "rl": dict(sim="mujoco", numerics="rl", ir=10.0, mu="1.0"),
        "floor": dict(sim="mujoco", numerics="bed", ir=100.0, mu="1.001")}


def finger_forces(acc):
    """{finger: (F vector, normal-force vector, contacts)} -> {finger: (N, T, n)}."""
    out = {}
    for f, (Fv, Nw, n) in acc.items():
        ln = np.linalg.norm(Nw)
        if n == 0 or ln < 1e-9:
            out[f] = (0.0, 0.0, n)
            continue
        nb = Nw / ln
        fn = float(Fv @ nb)
        out[f] = (abs(fn), float(np.linalg.norm(Fv - fn * nb)), n)
    return out


def sample(t, p, q, ff):
    return {"t": round(float(t), 4), "p": [round(float(v) * 1e3, 4) for v in p], "q": [round(float(v), 6) for v in q],
            "N": {f: round(ff[f][0], 4) for f in RB.FINGERS}, "T": {f: round(ff[f][1], 4) for f in RB.FINGERS},
            "n": {f: ff[f][2] for f in RB.FINGERS}}


def run_mujoco(scene, meta, plan, traj, seed):
    """RB.run_mujoco with the extended 20 ms sample."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(scene))
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
    trace, ext, W = [], [], []

    def snap():
        acc = {f: [np.zeros(3), np.zeros(3), 0] for f in RB.FINGERS}
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
            fr = c.frame.reshape(3, 3)
            s = 1.0 if g2 in tool_geoms else -1.0
            acc[f][0] += s * (fr.T @ f6[:3])
            acc[f][1] += s * f6[0] * fr[0]
            acc[f][2] += 1
            F[f] += float(f6[0])
            cnt[f] += 1
        R9 = np.zeros(9)
        mujoco.mju_quat2Mat(R9, d.qpos[qa + 3:qa + 7])
        p = d.qpos[qa:qa + 3]
        trace.append({"t": round(float(d.time), 4), "cos": float(R9[8]), "x": float(p[0]), "y": float(p[1]),
                      "z": float(p[2]), "F": {k: round(v, 4) for k, v in F.items()}, "n": cnt})
        ext.append(sample(d.time, p, d.qpos[qa + 3:qa + 7], finger_forces(acc)))

    segs = RB.schedule(plan, traj)
    dt = m.opt.timestep
    k_trace = max(1, int(round(TRACE_DT / dt)))
    step, bad = 0, False
    for t_end, tgt, _ in segs:
        for n_, a in act.items():
            d.ctrl[a] = tgt[n_]
        while d.time < t_end - 1e-9:
            w0 = time.perf_counter()
            mujoco.mj_step(m, d)
            W.append(time.perf_counter() - w0)
            step += 1
            if step % k_trace == 0:
                snap()
                if not np.all(np.isfinite(d.qpos)) or abs(d.qpos[qa + 2]) > 2.0:
                    bad = True
                    break
        if bad:
            break
    res = RB.score(trace, segs) if not bad else {}
    res.update(status="ejected" if bad else "complete", us_per_step_median=float(np.median(W) * 1e6), n_steps=step)
    return res, ext


class Bench(RB.DrakeBench):
    """RB.DrakeBench with the extended sample."""

    def snap_ext(self):
        p = self.plant
        X = p.EvalBodyPoseInWorld(self.pc, self.tool)
        acc = {f: [np.zeros(3), np.zeros(3), 0] for f in RB.FINGERS}
        cr = p.get_contact_results_output_port().Eval(self.pc)
        tool_bodies = {self.tool.index()}
        for i in range(cr.num_hydroelastic_contacts()):
            info = cr.hydroelastic_contact_info(i)
            srf = info.contact_surface()
            ids = (srf.id_M(), srf.id_N())
            if not (set(ids) & self.tool_gids):
                continue
            f = self.finger_gid.get(ids[0]) or self.finger_gid.get(ids[1])
            if f is None or srf.num_faces() == 0:
                continue
            Fv = np.array(info.F_Ac_W().translational())        # on the body of geometry M
            if srf.id_M() not in self.tool_gids:
                Fv = -Fv
            nb = np.zeros(3)
            for k in range(srf.num_faces()):
                nb += srf.area(k) * np.asarray(srf.face_normal(k))
            ln = np.linalg.norm(nb)
            if ln < 1e-12 or np.linalg.norm(Fv) < 1e-6:
                continue
            nb /= ln
            fn = float(Fv @ nb)
            if fn < 0:
                nb, fn = -nb, -fn
            acc[f][0] += Fv
            acc[f][1] += fn * nb
            acc[f][2] += 1
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            pp = info.point_pair()
            if not ({pp.id_A, pp.id_B} & self.tool_gids):
                continue
            f = self.finger_gid.get(pp.id_A) or self.finger_gid.get(pp.id_B)
            if f is None:
                continue
            Fv = np.array(info.contact_force())                  # on body B
            nb = np.array(pp.nhat_BA_W)
            if pp.id_B not in self.tool_gids:
                Fv = -Fv
            fn = float(Fv @ nb)
            if fn < 0:
                nb, fn = -nb, -fn
            if fn > 1e-6:
                acc[f][0] += Fv
                acc[f][1] += fn * nb
                acc[f][2] += 1
        del tool_bodies
        q = X.rotation().ToQuaternion()
        return sample(self.ctx.get_time(), X.translation(), [q.w(), q.x(), q.y(), q.z()], finger_forces(acc))


def run_drake(db, meta, plan, traj, seed):
    """RB.run_drake with the extended sample."""
    db.reset(meta["q0"], meta["tool7"], seed)
    segs = RB.schedule(plan, traj)
    trace, ext, W = [], [], []
    bad, t = False, 0.0
    for t_end, tgt, _ in segs:
        db.set_targets(tgt)
        while t < t_end - 1e-9:
            t_next = min(t_end, (math.floor(t / TRACE_DT + 1e-6) + 1) * TRACE_DT)
            w0 = time.perf_counter()
            db.sim.AdvanceTo(t_next)
            W.append((time.perf_counter() - w0) / max(1, round((t_next - t) / RB.DT)))
            t = t_next
            if abs((t / TRACE_DT) - round(t / TRACE_DT)) < 1e-6:
                s = db.snap()
                trace.append(s)
                ext.append(db.snap_ext())
                if not np.isfinite(s["z"]) or abs(s["z"]) > 2.0:
                    bad = True
                    break
        if bad:
            break
    res = RB.score(trace, segs) if not bad else {}
    res.update(status="ejected" if bad else "complete", us_per_step_median=float(np.median(W) * 1e6),
               n_steps=int(round(t / RB.DT)))
    return res, ext


def done(path):
    have = set()
    if Path(path).exists():
        for line in open(path):
            try:
                r = json.loads(line)
                if r.get("status") in ("complete", "ejected"):
                    have.add((r["arm"], r["tip"], r["hand"], int(r["seed"])))
            except Exception:
                pass
    return have


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tips", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["plan", "rl", "floor", "drake"], choices=list(ARMS))
    ap.add_argument("--hands", nargs="+", default=list(RB.HANDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    have = done(a.out)
    for tip in a.tips:
        for hand in a.hands:
            plan, traj = RB.load_plan(hand)
            for arm in a.arms:
                spec = ARMS[arm]
                todo = [s for s in a.seeds if (arm, tip, hand, s) not in have]
                if not todo:
                    continue
                gc.collect()
                db = None
                try:
                    if spec["sim"] == "mujoco":
                        scene, meta = PUB.build(hand, tip, "pads", C.SERVO, spec["numerics"], spec["ir"], spec["mu"])
                    else:
                        scene, meta = PUB.drake_scene(hand, tip, C.SERVO, spec["mu"])
                        RB.DT = 0.001
                        db = Bench(scene, meta)
                except Exception as e:
                    for s in todo:
                        PUB.append_row(a.out, {"arm": arm, "tip": tip, "hand": hand, "seed": s, "status": "failed",
                                               "error": repr(e), "traceback": traceback.format_exc()[-1500:],
                                               "script": SCRIPT, "when": time.strftime("%Y-%m-%d %H:%M")})
                    print(f"{arm} {tip} {hand}: scene failed: {e!r}", flush=True)
                    continue
                for s in todo:
                    t0 = time.time()
                    row = {"arm": arm, **spec, "tip": tip, "hand": hand, "seed": s, "plant": C.SERVO,
                           "jitter": RB.jitter(s), "scene": str(Path(scene).relative_to(ROOT))}
                    try:
                        res, ext = (run_mujoco(scene, meta, plan, traj, s) if db is None else
                                    run_drake(db, meta, plan, traj, s))
                        row.update(res)
                        row["trace"] = ext
                        row.update(n_pads=meta.get("n_pads"), pad_builder=meta.get("pad_builder"),
                                   pad_d0=meta.get("pad_d0"))
                    except Exception as e:
                        row.update(status="failed", error=repr(e), traceback=traceback.format_exc()[-1500:])
                    row.update(script=SCRIPT, git_rev=RB.git_rev(), wall_s=round(time.time() - t0, 2),
                               when=time.strftime("%Y-%m-%d %H:%M"))
                    PUB.append_row(a.out, row)
                    print(f"{arm:6s} {tip:28s} {hand} s{s}: {row.get('status')} turn {row.get('turn_end_deg', float('nan')):+.1f} "
                          f"held {row.get('held_end')} {row.get('us_per_step_median', 0):.0f} us/step, {row['wall_s']:.1f} s",
                          flush=True)
                del db


if __name__ == "__main__":
    main()
