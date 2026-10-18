#!/usr/bin/env python3
"""Simulator runs of the finite-element calibration (phases C, D1 and D2 of
docs/experiments/20261018-fe_contact_calibration/plan.json): the calibrated moduli of fit/fits.json in Drake 1.57 and in
the 1 mm pads, on the contact bed, on the CAD hand's pinned-iCEM commands and on the D1-D8 bench plans.

Calibrations (fit/fits.json, model values): U the readiness moduli (the same E for the pads and Drake); M1 the modulus
fitted to the finite elements' approach under the task cylinders; M2 the modulus fitted to their twist arm r_bar; M3 M2's
modulus behind a series spring k_s (the spring check below). Each simulator model has its own M1 and M2.

bed     The contact bed of scripts/sim2real_fingertip_bed.py (two force-controlled rails squeeze the 25 mm, 24.5 g tool
        between two copies of the tip, friction 1.0): static approach at 0.5, 1 and 3 N, twist onset (T2) at 1 and 3 N,
        spin at 1 rad/s and 1 N, pull (T1) at 1 N. Arms: pads (builder v2, 1 ms, impratio 100), pads05 (0.5 mm pads), rl
        (pads at 2 ms, impratio 10), drake. Rows bed/<case>.jsonl.
icem    D1 (a): each logged pinned-iCEM run of the readiness cells (cells/icem.jsonl of 2026-10-16, the 30 % tips, cells
        cylinder:16:0.625 and cylinder:25:0.125) replayed open loop: reset at the seed's placement, the candidate's
        grip targets for 0.8 s, then the logged servo commands (the 20 ms log, interpolated linearly to each 1 ms
        step), the post's collisions off at 0.8 s, to 6.3 s. Plants: MuJoCo pads (scale-study cell scene with the
        tip's pads at E) and Drake (curved_tips_cell_drake.CellDrake at E with the tip's hull). Every 20 ms: tool pose,
        each finger's normal force and friction-force vector on the tool, the nine joint angles. Rows replays/icem.jsonl.
plan    D1 (b): pinned iCEM (hand_object_scale_planners icem_pin) re-planned on a curved tip's M2 pads on
        cylinder:16:0.625, seed 0; rows replays/plans.jsonl (the 20 ms command log), replayed by `icem --own`.
d1d8    D2: the D1-D8 bench plans replayed open loop as in scripts/sim2real_fingertip_replays.py with E per calibration.
        Arms: plan (pads, 1 ms, impratio 100), rl (2 ms, impratio 10), drake, ir1000 (pads at impratio 1000), pads05
        (0.5 mm pads), spring (M3: pads at M2's E behind a slide joint along the palmar normal held by a joint
        equality of stiffness k_s, solref 0.01 s). Rows replays/d1d8.jsonl.

    PY=logs/20261001-hom_contact/venv/bin/python
    ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- taskset -c 4 $PY scripts/fe_contact_calibration_sim.py bed \\
        --tips block4_g30 cap_g30 ell16_g30 ell11_g30 --calib M1 M2
"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import math
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("SCALE_HAND", "cad_v1")
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

D = ROOT / "docs/experiments/20261018-fe_contact_calibration"
READY = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
ASSETS = ROOT / "assets/mjcf/experimental/20261016-sim2real_fingertip_readiness"
LOGS = ROOT / "logs/20261018-fe_contact_calibration"
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
SCRIPT = "scripts/fe_contact_calibration_sim.py"
FINGERS = ("thumb", "index", "middle")
TIPS = ["block4_g30", "cap_g30", "ell16_g30", "ell11_g30"]
ICEM_CELLS = ("cylinder:16:0.625", "cylinder:25:0.125")


def append(path, row):
    import pad_usability_backends as PUB
    PUB.append_row(path, row)


def rows(path):
    out = []
    if Path(path).exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def spec_of(tip):
    return json.loads((ASSETS / tip / "README.json").read_text())


def fits():
    return json.loads((D / "fit" / "fits.json").read_text())["tips"]


def calib_E(tip, calib, model):
    """(E Pa, pad spacing m, k_s N/m or None) of a calibration for model 'pads' (the pad model of rule R4), 'pads1'
    (builder-v2 pads at 1 mm whatever R4 says) or 'drake' (model values)."""
    if calib == "U":
        return float(spec_of(tip)["pad_E_Pa"]), 0.001, None
    t = fits()[tip]
    c = "M2" if calib == "M3" else calib
    if model == "drake":
        E = t["models"]["drake"][t.get("calib_of", {}).get(c, c)]["E_MPa"] * 1e6
        s = None
    else:
        pm = "v2" if model == "pads1" else t.get("pad_model", {}).get(c, "v2")
        E = t["models"][pm][t.get("calib_of", {}).get(c, c)]["E_MPa"] * 1e6
        s = 0.0005 if pm == "v2h" else 0.001
    ks = t["k_s_N_per_mm"] * 1e3 if calib == "M3" else None
    return E, s, ks


# ------------------------------------------------------------------------------------------ bed (phase C)

def bed_model(key, arm, E, s=0.001):
    import tip_codesign_bed as B
    import tip_codesign_geometry as TG
    if arm == "drake":
        name, v = B.tcd_model(key, f"mres0.00025:E{E:.6g}", drake=True)
        name = f"drake_{key}_E{E:.4g}"
        B.register({name: v})
        return name, 1.0
    r_mm = TG.shape(key).a * 1e3
    t = f"tpu{r_mm:g}:tcd={key}"
    ir, dt = ("ir10", 2.0) if arm == "rl" else ("ir100", 1.0)
    ss = 0.0005 if arm == "pads05" else s
    sp = f"s{ss * 1e3:g}:rs{0.75 * ss * 1e3:g}"
    ex = f"hloc:Keach:mres0.00025:E{E:.6g}"
    name = f"mj_{key}_{arm}_s{ss * 1e3:g}_E{E:.4g}"
    B.register({name: (f"mj:spheres:{sp}:{ir}:tr0.02:{ex}:{t}", f"mj:spheres:{sp}:tr0.02:{ex}:{t}")})
    return name, dt


def cmd_bed(a):
    import contact_bed_common as CB
    import curved_tips_bed as CTB
    import sim2real_fingertip_bed as SFB
    import tip_codesign_bed as B
    CTB.T.MESH_DIR = LOGS / "meshes"
    NS = {"static": (0.5, 1.0, 3.0), "T2": (1.0, 3.0), "spin": (1.0,), "T1": (1.0,)}
    for tip in a.tips:
        key = spec_of(tip)["key"]
        for calib in a.calib:
            for arm in a.arms:
                if arm == "rl" and calib not in a.rl_calib:
                    continue
                Ep, s, _ = calib_E(tip, calib, "drake" if arm == "drake" else "pads")
                if arm == "pads05" and s == 0.0005:
                    continue
                name, dt = bed_model(key, arm, Ep, s)
                for case in a.cases:
                    out = D / "bed" / f"{case}.jsonl"
                    have = {(r["tip"], r["calib"], r["arm"], r["case"], r.get("N")) for r in rows(out)
                            if r.get("status") == "complete"}
                    for N in NS[case]:
                        if (tip, calib, arm, case, N) in have:
                            continue
                        t0 = time.time()
                        try:
                            if case == "static":
                                r = CTB.case_static(name, N, dt)
                                r.pop("trace", None)
                            elif case == "T2":
                                r = B.case_T2(name, N, dt, SFB.law_onset(key)[N])
                            elif case == "spin":
                                r = B.case_spin(name, N, 1.0, dt, decomp=arm != "drake")
                            else:
                                r = CTB.case_T1(name, N, dt)
                            r.setdefault("status", "complete")
                        except Exception as e:
                            r = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()[-1500:]}
                        if case != "static":
                            r.pop("trace", None)
                        r.update(tip=tip, key=key, calib=calib, arm=arm, case=case, model=name, E_Pa=Ep, N=N,
                                 omega=1.0 if case == "spin" else None, dt_ms=dt, rig_spec=CB.MODELS[name][0],
                                 script=SCRIPT, wall_s=round(time.time() - t0, 2), when=time.strftime("%Y-%m-%d %H:%M"))
                        append(out, r)
                        keys = ("approach_um", "touching_per_side", "rbar_onset_mm", "creep_deg_s", "rbar_spin_mm",
                                "mu_eff", "creep_mm_s")
                        print(case, tip, calib, arm, N, r.get("status"),
                              {q: round(r[q], 4) for q in keys if isinstance(r.get(q), (int, float))}, r.get("error", ""),
                              f"{time.time() - t0:.1f} s", flush=True)


# ------------------------------------------------------------------------------------------ cell plants (D1)

def cell_env(tip, E, s=0.001, ir=100.0):
    """hand_object_scale_sim pointed at the readiness variant's CAD fingers with the tip's pads at E (spacing s,
    impratio ir), its cell scenes cached per (tip, E, s, ir) under logs/."""
    import hand_object_scale_sim as S
    import reorient_backends as RB
    import tip_codesign_rig as R
    spec = spec_of(tip)
    ns = argparse.Namespace(E=E, pf=None, tr=RB.PAD_TR, tag=None, h="local", K="each")
    R.install(ns, force_tip=spec["key"])
    if s != 0.001:
        RB.replace_tips = functools.partial(RB.replace_tips, rs=0.75 * s)
        RB.PAD_S = s
    else:
        RB.PAD_S = 0.001
    S.CAD_DIR = ASSETS / tip / "cad_v1"
    S.GEN = ROOT / "logs/20261016-sim2real_fingertip_readiness/gen" / tip
    S.SCENES = LOGS / "cells" / (f"{tip}_E{E / 1e6:.5g}_s{s * 1e3:g}" + (f"_ir{ir:g}" if ir != 100.0 else ""))
    S.MU, S.PLANT, S.BUILDER, S.IMPRATIO = 1.0, SERVO, "v2", ir
    return S, spec


def cell_of(S, cs):
    shape, dd, t = cs.split(":")
    dd, t = float(dd), float(t)
    cell = next(c for c, sh, d_ in S.cell_list("diag") if sh == shape and d_ == dd and c.get("t") == t)
    return cell, shape, dd


def plants(tip, cs, ci, E, model, s=0.001, mesh_eps=1e-6, ir=100.0):
    """(plant, meta) of a cell's candidate: 'pads' (hom_turn3.MjPlant on the pads scene) or 'drake' (CellDrake, the tip
    hull's terrace outlines simplified to mesh_eps m)."""
    s = s or 0.001
    import mujoco
    import curved_tips_cell_drake as CD
    import hom_turn3 as H
    import tip_codesign_geometry as TG
    S, spec = cell_env(tip, E, s, ir)
    cell, shape, dd = cell_of(S, cs)
    kp = S.RB.plant_args(SERVO)["kp"]
    cand = S.candidates(cell, shape, dd)[ci]
    sc = S.cell_scenes(cell, shape, dd, models=("pads", "pt"))
    m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
    gg = S.grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
    gp = S.grip_poses(m_pt, gg, kp)
    if model == "drake":
        meta = S.candidate_meta(sc["pt"][1], gg, gp)
        sh = TG.shape(spec["key"])
        mp = sh.mesh_paths(LOGS / "meshes_cell" / sh.key, eps=mesh_eps)
        plant = CD.CellDrake(Path(sc["pt"][0]), meta, gg["palm"], E=E,
                             tip_files={"tpu_pos": str(mp[1.0]), "tpu_neg": str(mp[-1.0])})
    else:
        meta = S.candidate_meta(sc["pads"][1], gg, gp)
        plant = H.MjPlant(sc["pads"][0], meta)
        S.set_palm(plant.m, gg["palm"])
    return plant, meta, S


def mj_snap(plant):
    import mujoco
    m, d = plant.m, plant.d
    acc = {f: [0.0, np.zeros(3), 0] for f in FINGERS}
    f6 = np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        if (g1 in plant.tool_geoms) == (g2 in plant.tool_geoms):
            continue
        f = plant.finger_of.get(g2 if g1 in plant.tool_geoms else g1)
        if f is None:
            continue
        mujoco.mj_contactForce(m, d, i, f6)
        if f6[0] <= 1e-6:
            continue
        fr = c.frame.reshape(3, 3)
        on_tool = (fr.T @ f6[:3]) * (1.0 if g2 in plant.tool_geoms else -1.0)
        acc[f][0] += float(f6[0])
        acc[f][1] += on_tool - (on_tool @ fr[0]) * fr[0]
        acc[f][2] += 1
    return acc, d.qpos[plant.qa:plant.qa + 3].copy(), d.qpos[plant.qa + 3:plant.qa + 7].copy(), d.qpos[plant.qadr].copy()


def drake_snap(cd):
    p, pc = cd.plant, cd.pc
    X = p.EvalBodyPoseInWorld(pc, cd.tool)
    acc = {f: [0.0, np.zeros(3), 0] for f in FINGERS}
    cr = p.get_contact_results_output_port().Eval(pc)
    for i in range(cr.num_hydroelastic_contacts()):
        info = cr.hydroelastic_contact_info(i)
        srf = info.contact_surface()
        ids = (srf.id_M(), srf.id_N())
        if not (set(ids) & cd.tool_gids):
            continue
        f = cd.finger_gid.get(ids[0]) or cd.finger_gid.get(ids[1])
        if f is None or srf.num_faces() == 0:
            continue
        Fv = np.array(info.F_Ac_W().translational())
        if srf.id_M() not in cd.tool_gids:
            Fv = -Fv
        nb = np.zeros(3)
        for k in range(srf.num_faces()):
            nb += srf.area(k) * np.asarray(srf.face_normal(k))
        ln = np.linalg.norm(nb)
        if ln < 1e-12 or np.linalg.norm(Fv) < 1e-6:
            continue
        nb /= ln
        acc[f][0] += abs(float(Fv @ nb))
        acc[f][1] += Fv - (Fv @ nb) * nb
        acc[f][2] += 1
    for i in range(cr.num_point_pair_contacts()):
        info = cr.point_pair_contact_info(i)
        pp = info.point_pair()
        if not ({pp.id_A, pp.id_B} & cd.tool_gids):
            continue
        f = cd.finger_gid.get(pp.id_A) or cd.finger_gid.get(pp.id_B)
        if f is None:
            continue
        Fv = np.array(info.contact_force())
        if info.bodyB_index() != cd.tool.index():
            Fv = -Fv
        nb = np.asarray(pp.nhat_BA_W)
        fn = abs(float(Fv @ nb))
        if fn > 1e-6:
            acc[f][0] += fn
            acc[f][1] += Fv - (Fv @ nb) * nb
            acc[f][2] += 1
    q = X.rotation().ToQuaternion()
    qj = np.array([j.get_angle(pc) for j in cd.jidx])
    return acc, np.array(X.translation()), np.array([q.w(), q.x(), q.y(), q.z()]), qj


def quat_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


HOLD = 0.01        # s: the replays hold each servo command for 10 ms (the HOM controller's tick), the value of the logged
                   # command sequence (linear between its samples) at the middle of the interval, in both simulators;
                   # Drake's per-call overhead made 1 ms updates 10 times slower (100 s against 9 s per run)


def replay(plant, meta, log, seed, t_grip=0.8, t_post=0.8, t_end=6.3, drake=False, frame_cb=None, hold=HOLD):
    """Open-loop replay of a logged command sequence; returns (result, 20 ms trace)."""
    snap = (lambda: drake_snap(plant)) if drake else (lambda: mj_snap(plant))
    grip = np.array([meta["grip"][f][k] for f in FINGERS for k in range(3)])
    if drake:
        plant.m.geom_contype[0] = plant.m.geom_conaffinity[0] = 1
    else:
        post = plant.m.geom("post").id
        if not hasattr(plant, "post0"):
            plant.post0 = (int(plant.m.geom_contype[post]), int(plant.m.geom_conaffinity[post]))
        plant.m.geom_contype[post], plant.m.geom_conaffinity[post] = plant.post0
    plant.reset(seed)
    plant.set_targets(grip)
    plant.advance(t_grip)
    s0 = plant.state()
    u0 = s0["R"][:, 2] / np.linalg.norm(s0["R"][:, 2])
    a0 = np.cross(u0, [0.0, 0.0, 1.0])
    a0 /= np.linalg.norm(a0)
    h0 = np.cross(a0, u0)
    theta = lambda R: math.atan2(R[:, 2] @ h0, R[:, 2] @ u0)  # noqa: E731
    z0 = float(s0["p"][2])
    weight = meta["mass_kg"] * 9.81
    ts = np.array([t_grip] + [x["t"] - 0.001 for x in log])
    U = np.array([grip] + [x["u"] for x in log])
    dt = hold
    per_sample = int(round(0.02 / hold))
    trace = []
    t = t_grip
    k = 0
    bad = False
    w0 = time.perf_counter()
    while t < t_end - 1e-9:
        if t >= t_post - 1e-9:
            if drake:
                plant.m.geom_contype[0] = plant.m.geom_conaffinity[0] = 0
            else:
                post = plant.m.geom("post").id
                plant.m.geom_contype[post] = plant.m.geom_conaffinity[post] = 0
        u = np.array([np.interp(t + 0.5 * dt, ts, U[:, j]) for j in range(U.shape[1])])
        plant.set_targets(u)
        t = round(t + dt, 6)
        plant.advance(t)
        k += 1
        if k % per_sample == 0:
            acc, p, q, qj = snap()
            trace.append({"t": round(t, 3), "p": [round(float(v) * 1e3, 4) for v in p], "q": [round(float(v), 6) for v in q],
                          "N": {f: round(float(acc[f][0]), 4) for f in FINGERS},
                          "T": {f: [round(float(x), 4) for x in acc[f][1]] for f in FINGERS},
                          "n": {f: int(acc[f][2]) for f in FINGERS}, "qj": [round(float(v), 5) for v in qj],
                          "th": round(math.degrees(theta(quat_R(q))), 3)})
            if frame_cb is not None:
                frame_cb(plant, t, trace[-1])
            if not np.all(np.isfinite(p)) or p[2] < z0 - 0.03:
                bad = not np.all(np.isfinite(p))
                break
    se = plant.state()
    loaded = sum(1 for f in FINGERS if se["n"][f] > 0 and se["F"][f] >= weight)
    dropped = bool(se["p"][2] < z0 - 0.020)
    th_end = theta(se["R"])
    res = {"held": (not dropped) and loaded >= 2 and not bad, "turn_end_deg": round(math.degrees(th_end), 2),
           "fingers_loaded_end": loaded, "dropped": dropped, "unstable": bad,
           "z_drop_mm": round(1e3 * float(z0 - se["p"][2]), 2), "t_sim_end": round(float(t), 3),
           "wall_s": round(time.perf_counter() - w0, 2)}
    return res, trace


def icem_plans(own=False):
    """The logged iCEM runs to replay: (plan id, tip, cell, cand, log, logged result)."""
    out = []
    if own:
        for r in rows(D / "replays" / "plans.jsonl"):
            if r.get("status") == "ok":
                out.append((f"own_{r['variant']}_{r['cell']}", r["variant"], r["cell"], r["cand"], r["log"],
                            {"held": r["held"], "turn_end_deg": r["turn_end_deg"]}))
        return out
    for r in rows(READY / "cells" / "icem.jsonl"):
        if r.get("status") != "ok" or r["variant"] not in TIPS:
            continue
        cs = f"{r['shape']}:{r['d_mm']:g}:{r['t']:g}"
        if cs not in ICEM_CELLS:
            continue
        out.append((f"{r['variant']}_{cs}", r["variant"], cs, r["cand"], r["log"],
                    {"held": r["held"], "turn_end_deg": r["turn_end_deg"]}))
    return out


def cmd_icem(a):
    out = D / "replays" / "icem.jsonl"
    have = {(r["plan"], r["calib"], r["plant"], r["seed"]) for r in rows(out) if r.get("status") == "ok"}
    for pid, tip, cs, ci, log, logged in icem_plans(a.own):
        if a.tips and tip not in a.tips:
            continue
        if a.plans and not any(x in pid for x in a.plans):
            continue
        for calib in a.calib:
            for model in a.plants:
                seeds = [s for s in a.seeds if (pid, calib, model, s) not in have]
                if not seeds:
                    continue
                base, ir = (model.split("_ir")[0], float(model.split("_ir")[1])) if "_ir" in model else (model, 100.0)
                E, s, _ = calib_E(tip, calib, base)
                gc.collect()
                t0 = time.time()
                try:
                    plant, meta, S = plants(tip, cs, ci, E, "drake" if base == "drake" else "pads", s, ir=ir)
                except Exception as e:
                    print(f"{pid} {calib} {model}: build failed {e!r}", flush=True)
                    traceback.print_exc()
                    continue
                build_s = time.time() - t0
                eps = 1e-6
                for seed in seeds:
                    row = {"plan": pid, "tip": tip, "cell": cs, "cand": ci, "calib": calib, "plant": model, "E_Pa": E,
                           "pad_spacing_mm": s * 1e3 if model != "drake" else None, "seed": seed, "own": a.own,
                           "logged": logged, "servo": SERVO, "build_s": round(build_s, 1), "hold_ms": HOLD * 1e3}
                    for attempt in (0, 1):
                        try:
                            res, tr = replay(plant, meta, log, seed, drake=model == "drake")
                            row.update(res, trace=tr, status="ok", mesh_eps_m=eps if model == "drake" else None)
                            break
                        except Exception as e:
                            row.update(status="error", error=repr(e), tb=traceback.format_exc()[-1500:])
                            if model == "drake" and "Cannot instantiate plane" in repr(e) and attempt == 0:
                                # a degenerate contact polygon of the tip hull: rebuild with the terraces simplified
                                # to 10 um (curved_tips_cell_drake's mesh_eps) and run the seed again
                                eps = 1e-5
                                del plant
                                plant, meta, S = plants(tip, cs, ci, E, "drake", s, mesh_eps=eps)
                                continue
                            break
                    row.update(script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"))
                    append(out, row)
                    print(f"{pid:32s} {calib} {model:5s} s{seed}: " + (
                        f"held {row['held']} turn {row['turn_end_deg']:+.1f} (logged {logged['turn_end_deg']:+.1f}) "
                        f"{row['wall_s']} s" if row["status"] == "ok" else row["error"]), flush=True)
                del plant


def cmd_plan(a):
    """Pinned iCEM on a tip's M2 pads (D1 b); the row keeps the 20 ms command log for the replays."""
    import hand_object_scale_mpc as M
    import hand_object_scale_planners as HP
    out = D / "replays" / "plans.jsonl"
    have = {(r["variant"], r["cell"]) for r in rows(out) if r.get("status") == "ok"}
    ready = {(r["variant"], f"{r['shape']}:{r['d_mm']:g}:{r['t']:g}"): r["cand"] for r in rows(READY / "cells" / "icem.jsonl")
             if r.get("status") == "ok"}
    for tip in a.tips:
        cs = a.cell
        if (tip, cs) in have:
            continue
        E, s, _ = calib_E(tip, a.calib, a.model)
        S, spec = cell_env(tip, E, s)
        M.S = S
        cell, shape, dd = cell_of(S, cs)
        ci = ready[(tip, cs)]
        prm = HP.prm_of("icem_pin", ["log_every=1"])
        # every 1 ms command in the row (the readiness rows kept one command per 20 ms window, which a replay can only
        # interpolate): compact_log at every sample
        M.compact_log = lambda log, theta, every=1, _f=M.compact_log: _f(log, theta, every=1)
        plant, meta, geo, _, _ = M.setup(cell, shape, dd, ci)
        row = {"variant": tip, "tip": spec["key"], "cell": cs, "cand": ci, "calib": a.calib, "E_Pa": E,
               "pad_spacing_mm": s * 1e3, "seed": 0, "config": "icem_pin", **{f"prm_{k}": v for k, v in prm.items()},
               "when": time.strftime("%Y-%m-%d %H:%M")}
        try:
            res = M.mpc_turn(plant, meta, 0, prm, nthread=a.nthread)
            row.update(res, status="ok")
        except Exception as e:
            row.update(status="error", error=repr(e), tb=traceback.format_exc()[-1500:])
        row["script"] = SCRIPT
        append(out, row)
        print(f"{tip} {cs} c{ci}: " + (f"held {row['held']} turn {row['turn_end_deg']} {row['wall_s']} s"
                                       if row["status"] == "ok" else row["error"]), flush=True)


# ------------------------------------------------------------------------------------------ D1-D8 (D2)

ARMS = {"plan": dict(sim="mujoco", numerics="bed", ir=100.0, mu="1.0"),
        "rl": dict(sim="mujoco", numerics="rl", ir=10.0, mu="1.0"),
        "ir1000": dict(sim="mujoco", numerics="bed", ir=1000.0, mu="1.0"),
        "pads05": dict(sim="mujoco", numerics="bed", ir=100.0, mu="1.0", s=0.0005),
        "spring": dict(sim="mujoco", numerics="bed", ir=100.0, mu="1.0", spring=True),
        "drake": dict(sim="drake", mu="1.0")}


def add_spring(path, k_s, tc=0.01):
    """A slide joint between each finger's DIP link and its tip body along the palmar normal, held at zero by a joint
    equality whose soft-constraint stiffness is k_s (N/m): solref (tc, 1), solimp d0 the root near 1 of
    d (1 - d) = 1 / (tc^2 k_s w), w the slide dof's inverse weight."""
    import mujoco
    root = ET.parse(path).getroot()
    eq = root.find("equality")
    if eq is None:
        eq = ET.SubElement(root, "equality")
    for f in FINGERS:
        tipb = root.find(f".//body[@name='{f}_tip']")
        vis = tipb.find(f"geom[@name='{f}_tipvis']")
        sgn = -1.0 if vis is not None and vis.get("mesh", "").endswith("neg") else 1.0
        j = ET.Element("joint", name=f"{f}_tipslide", type="slide", axis=f"{sgn:g} 0 0", range="-0.003 0.003",
                       limited="true", armature="0", damping="0")
        tipb.insert(1, j)
    m0 = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    info = {}
    for f in FINGERS:
        w = float(m0.dof_invweight0[m0.jnt_dofadr[m0.joint(f"{f}_tipslide").id]])
        x = 1.0 / (tc ** 2 * k_s * w)
        if x > 0.25:
            raise ValueError(f"{f}: spring {k_s} N/m below the soft constraint's range (tc {tc} s, w {w:.3g})")
        dd = 0.5 * (1 + math.sqrt(1 - 4 * x))
        ET.SubElement(eq, "joint", joint1=f"{f}_tipslide", polycoef="0 1 0 0 0", solref=f"{tc:g} 1",
                      solimp=f"{dd:.8g} {dd:.8g} 0.001 0.5 2")
        info[f] = dict(invweight=w, d0=dd)
    out = Path(str(path).replace(".xml", f"_spring{k_s / 1e3:.4g}.xml"))
    tmp = out.with_name(out.name + f".{os.getpid()}.tmp")
    ET.ElementTree(root).write(tmp)
    os.replace(tmp, out)
    return out, info


def d1d8_scene(tip, key, arm, E, s, ks, hand):
    import pad_usability_backends as PUB
    import reorient_backends as RB
    import tip_codesign_rig as R
    sp = ARMS[arm]
    R.install(argparse.Namespace(E=E, pf=None, tr=RB.PAD_TR, tag=None, h="local", K="each"))
    ss = sp.get("s", s)
    PUB.SCN = LOGS / "replay_scenes" / f"{tip}_E{E / 1e6:.5g}_s{ss * 1e3:g}"
    if sp["sim"] == "drake":
        return PUB.drake_scene(hand, key, SERVO, sp["mu"])
    if ss != 0.001:
        saved = RB.replace_tips
        RB.replace_tips = functools.partial(saved, rs=0.75 * ss)
        try:
            out_dir = PUB.SCN / f"pads_{sp['numerics']}_ir{sp['ir']:g}"
            path, meta = PUB._RB_BUILD(hand, key, "pads", SERVO, sp["numerics"], sp["ir"], sp["mu"], out_dir=out_dir,
                                       pad_s=ss)
        finally:
            RB.replace_tips = saved
        meta.update(pad_rs_m=0.75 * ss)
    else:
        path, meta = PUB.build(hand, key, "pads", SERVO, sp["numerics"], sp["ir"], sp["mu"],
                               prefix=f"ir{sp['ir']:g}_" if sp["ir"] not in (100.0, 10.0) else "")
    if sp.get("spring"):
        path, info = add_spring(path, ks)
        meta = dict(meta, spring=info, k_s_N_per_m=ks)
    return path, meta


def cmd_d1d8(a):
    import curved_tips_traj as CT
    import reorient_backends as RB
    out = D / "replays" / "d1d8.jsonl"
    have = {(r["tip"], r["calib"], r["arm"], r["hand"], int(r["seed"])) for r in rows(out)
            if r.get("status") in ("complete", "ejected")}
    for tip in a.tips:
        key = spec_of(tip)["key"]
        for calib in a.calib:
            for arm in a.arms:
                sp = ARMS[arm]
                E, s, ks = calib_E(tip, "M3" if arm == "spring" else calib, "drake" if arm == "drake" else "pads")
                for hand in a.hands:
                    todo = [sd for sd in a.seeds if (tip, calib, arm, hand, sd) not in have]
                    if not todo:
                        continue
                    plan, traj = RB.load_plan(hand)
                    gc.collect()
                    db = None
                    try:
                        scene, meta = d1d8_scene(tip, key, arm, E, s, ks, hand)
                        if sp["sim"] == "drake":
                            RB.DT = 0.001
                            db = CT.Bench(scene, meta, E=E)
                    except Exception as e:
                        print(f"{tip} {calib} {arm} {hand}: scene failed {e!r}", flush=True)
                        traceback.print_exc()
                        continue
                    for sd in todo:
                        t0 = time.time()
                        row = {"tip": tip, "key": key, "calib": calib, "arm": arm, **{k: v for k, v in sp.items()},
                               "E_Pa": E, "k_s_N_per_m": ks if arm == "spring" else None, "hand": hand, "seed": sd,
                               "plant": SERVO, "jitter": RB.jitter(sd), "scene": str(Path(scene).relative_to(ROOT))}
                        try:
                            res, ext = (CT.run_mujoco(scene, meta, plan, traj, sd) if db is None else
                                        CT.run_drake(db, meta, plan, traj, sd))
                            row.update(res)
                            row["trace"] = ext
                        except Exception as e:
                            row.update(status="failed", error=repr(e), traceback=traceback.format_exc()[-1500:])
                        row.update(script=SCRIPT, wall_s=round(time.time() - t0, 2), when=time.strftime("%Y-%m-%d %H:%M"))
                        append(out, row)
                        print(f"{tip:10s} {calib} {arm:6s} {hand} s{sd}: {row.get('status')} turn "
                              f"{row.get('turn_end_deg', float('nan')):+.1f} held {row.get('held_end')} {row['wall_s']:.1f} s",
                              flush=True)
                    del db


# ------------------------------------------------------------------------------------------ film (phase E)

def cmd_film(a):
    """A replay with the pads (simulated, pads touching the tool red) beside the Drake run of the same commands (its
    recorded joint angles and tool pose set kinematically in the same MuJoCo scene, every 20 ms), 640 x 480 panels under
    label bars; the clip at 50 frames per second of simulated time, so it plays at real time."""
    import mujoco
    import contact_bed_common as CB
    import hand_object_scale_films as SF
    import reorient_backends_films as RBF
    import tip_codesign_film as TF
    from PIL import Image
    R = {(r["plan"], r["calib"], r["plant"], int(r["seed"])): r for r in rows(D / "replays" / "icem.jsonl")
         if r.get("status") == "ok"}
    plan = next(p for p in icem_plans(a.own) if p[0] == a.plan)
    pid, tip, cs, ci, log, logged = plan
    E, s, _ = calib_E(tip, a.calib, a.pads)
    plant, meta, S = plants(tip, cs, ci, E, "pads", s)
    dk = R[(pid, a.calib, "drake", a.seed)]
    pk = R.get((pid, a.calib, a.pads, a.seed))
    import hom_turn3 as H
    sc = S.cell_scenes(*cell_of(S, cs), models=("pads", "pt"))
    dd = float(cs.split(":")[1])
    scene = SF.film_scene(Path(sc["pads"][0]), dd / 2000.0)
    pl = H.MjPlant(scene, meta)
    cell, shape, _ = cell_of(S, cs)
    cand = S.candidates(cell, shape, dd)[ci]
    gg = S.grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
    S.set_palm(pl.m, gg["palm"])
    kin = H.MjPlant(scene, meta)                            # the Drake run's kinematic replay
    S.set_palm(kin.m, gg["palm"])
    pads = [i for i in range(pl.m.ngeom) if "_pad" in (pl.m.geom(i).name or "")]
    for m_ in (pl.m, kin.m):
        for i in pads:
            m_.geom_rgba[i] = (0.75, 0.4, 0.25, 0.0)
        m_.vis.global_.offwidth, m_.vis.global_.offheight = 640, 480
    RBF.W, RBF.H = TF.PW, TF.PH
    lookat = np.asarray(meta["tool7"], float)[:3]
    r1, cam1 = RBF.renderer_for(pl.m, lookat)
    r2, cam2 = RBF.renderer_for(kin.m, lookat)
    for cam in (cam1, cam2):
        cam.distance = 0.24 + 1.4 * dd / 1000.0
        cam.elevation += a.elev
        cam.azimuth += a.azim
    names = {"block4_g30": "Block", "cap_g30": "Capsule r 8.5 mm", "ell16_g30": "Half-ellipsoid 16/32 mm",
             "ell11_g30": "Half-ellipsoid 11/22 mm"}
    Ed, _, _ = calib_E(tip, a.calib, "drake")
    frames = []
    dtr = {round(e["t"], 3): e for e in dk["trace"]}

    def cb(p_, t, e):
        m, d = pl.m, pl.d
        for i in pads:
            m.geom_rgba[i, 3] = 0.0
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom[0]), int(d.contact[i].geom[1])
            for g, o in ((g1, g2), (g2, g1)):
                if o in pl.tool_geoms and "_pad" in (m.geom(g).name or ""):
                    m.geom_rgba[g] = (0.9, 0.1, 0.1, 1.0)
        r1.update_scene(d, cam1)
        left = TF.label_bar(r1.render().copy(), [f"Pads {s * 1e3:g} mm, {E / 1e6:.0f} MPa ({a.calib})",
                                                 f"t {t:4.2f} s   turn {e['th']:+5.1f} deg"])
        de = dtr.get(round(t, 3))
        if de is not None:
            kin.d.qpos[kin.qadr] = de["qj"]
            kin.d.qpos[kin.qa:kin.qa + 3] = np.array(de["p"]) * 1e-3
            kin.d.qpos[kin.qa + 3:kin.qa + 7] = de["q"]
            mujoco.mj_forward(kin.m, kin.d)
            label = f"t {t:4.2f} s   turn {de['th']:+5.1f} deg"
        else:
            label = f"t {t:4.2f} s   (Drake run ended)"
        r2.update_scene(kin.d, cam2)
        right = TF.label_bar(r2.render().copy(), [f"Drake, {Ed / 1e6:.0f} MPa ({a.calib})", label])
        frames.append(np.concatenate([left, right], 1))

    res, tr = replay(pl, meta, log, a.seed, frame_cb=cb)
    frames += [frames[-1]] * 25
    r1.close()
    r2.close()
    media = D / "media"
    media.mkdir(parents=True, exist_ok=True)
    stem = f"film_{pid.replace(':', '_')}_{a.calib}_s{a.seed}"
    CB.write_h264(frames, media / f"{stem}.mp4", crf=20, fps=50)
    for name, frac in (("grip", 0.02), ("mid", 0.5), ("end", 0.97)):
        Image.fromarray(frames[min(len(frames) - 1, int(frac * (len(frames) - 25)))]).save(media / f"{stem}_{name}.png")
    row = {"film": stem, "plan": pid, "tip": tip, "calib": a.calib, "seed": a.seed, "E_pads_Pa": E, "E_drake_Pa": Ed,
           "pads": {"held": res["held"], "turn_end_deg": res["turn_end_deg"]},
           "pads_row": {"held": pk["held"], "turn_end_deg": pk["turn_end_deg"]} if pk else None,
           "drake": {"held": dk["held"], "turn_end_deg": dk["turn_end_deg"]}, "frames": len(frames),
           "when": time.strftime("%Y-%m-%d %H:%M")}
    append(media / "films.jsonl", row)
    print(json.dumps(row), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bed")
    b.add_argument("--tips", nargs="+", default=TIPS)
    b.add_argument("--calib", nargs="+", default=["M1", "M2"])
    b.add_argument("--arms", nargs="+", default=["pads", "drake", "rl"])
    b.add_argument("--rl-calib", nargs="+", default=["M2"])
    b.add_argument("--cases", nargs="+", default=["static", "T2", "spin", "T1"])
    i = sub.add_parser("icem")
    i.add_argument("--tips", nargs="*", default=[])
    i.add_argument("--calib", nargs="+", default=["U", "M1", "M2"])
    i.add_argument("--plants", nargs="+", default=["pads", "drake"])
    i.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    i.add_argument("--own", action="store_true", help="replay the plans of `plan` (D1 b)")
    i.add_argument("--plans", nargs="*", default=[], help="only plan ids containing one of these strings")
    p = sub.add_parser("plan")
    p.add_argument("--tips", nargs="+", default=["cap_g30", "ell16_g30", "ell11_g30"])
    p.add_argument("--calib", default="M2")
    p.add_argument("--cell", default="cylinder:16:0.625")
    p.add_argument("--nthread", type=int, default=7)
    p.add_argument("--model", default="pads1", choices=["pads", "pads1"],
                   help="planner model: pads1 = builder-v2 pads at 1 mm (the pipeline's), pads = rule R4's pad model")
    d = sub.add_parser("d1d8")
    d.add_argument("--tips", nargs="+", default=TIPS)
    d.add_argument("--calib", nargs="+", default=["M1", "M2"])
    d.add_argument("--arms", nargs="+", default=["plan"])
    d.add_argument("--hands", nargs="+", default=[f"D{k}" for k in range(1, 9)])
    d.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    f = sub.add_parser("film")
    f.add_argument("--plan", required=True)
    f.add_argument("--own", action="store_true")
    f.add_argument("--calib", default="M2")
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--pads", default="pads", choices=["pads", "pads1"])
    f.add_argument("--elev", type=float, default=10.0)
    f.add_argument("--azim", type=float, default=0.0)
    a = ap.parse_args()
    {"bed": cmd_bed, "icem": cmd_icem, "plan": cmd_plan, "d1d8": cmd_d1d8, "film": cmd_film}[a.cmd](a)


if __name__ == "__main__":
    main()
