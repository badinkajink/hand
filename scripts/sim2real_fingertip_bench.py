#!/usr/bin/env python3
"""Bench procedure and first-bench package of the fingertip deployment-readiness study (docs/handoff/
20261016-sim2real_fingertip_readiness.md, steps 8 and 9).

  procedure   the assembly and bench measurements that the models need, in order, each with the model's prediction, the
              file it comes from and its status (every one pending: no printed 85A tip has been measured):
              docs/experiments/20261016-sim2real_fingertip_readiness/bench/procedure.json

Indentation predictions come from the 85A finite elements (solid modulus 12 MPa, NinjaFlex's) and are inverted for
Siraya's modulus: in linear elastic contact the response at force F and solid modulus E equals the 12 MPa response at
F x 12 MPa / E, so a measured approach d at force F gives E = 12 MPa x F / F12(d), with F12(d) the 12 MPa model's force at
approach d (log-log interpolation over 0.5, 1 and 3 N).

    python3 scripts/sim2real_fingertip_bench.py procedure
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
D = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
OUT = D / "bench"
E0 = 12.0
DATE_TAG = "20261016"


def fe_rows():
    return {(r["layout"], r["recipe"]): r for r in (json.loads(l) for l in open(D / "fe/fem.jsonl"))}


def _loglog(x, xs, ys):
    """Piecewise power law through (xs, ys), extended beyond the ends with the nearest segment's exponent."""
    lx, lxs, lys = math.log(x), np.log(xs), np.log(ys)
    k = int(np.clip(np.searchsorted(lxs, lx) - 1, 0, len(xs) - 2))
    s = (lys[k + 1] - lys[k]) / (lxs[k + 1] - lxs[k])
    return float(math.exp(lys[k] + s * (lx - lxs[k])))


def approach_at(row, F, E_s):
    """Approach (um) under the 25 mm tool at force F (N) for solid modulus E_s (MPa), from the 12 MPa rows."""
    Fs = np.array([0.5, 1.0, 3.0])
    ds = np.array([row["per_force"][f"{f:g}"]["delta_um"] for f in Fs])
    return _loglog(F * E0 / E_s, Fs, ds)


def modulus_from(row, F, d_um):
    """Siraya's solid modulus (MPa) from a measured approach d at force F, by the scaling of the module docstring."""
    Fs = np.array([0.5, 1.0, 3.0])
    ds = np.array([row["per_force"][f"{f:g}"]["delta_um"] for f in Fs])
    return E0 * F / _loglog(d_um, ds, Fs)


def cmd_procedure(a):
    fe = fe_rows()
    S = json.loads((D / "summary.json").read_text()) if (D / "summary.json").exists() else {}
    sens = json.loads((D / "sensing/sensing.json").read_text())["rows"]
    bed = S.get("bed", {})
    var = S.get("variants", {})

    def sense(layout, recipe, plane="plate"):
        return next(r for r in sens if r["layout"] == layout and r["recipe"] == recipe and r["plane"] == plane)
    indent = {}
    for lay, lab in (("four_d3x2_d5", "block, four magnets"), ("five_d3x2_d5", "block, five magnets")):
        for rec in ("w85_30", "w85_50"):
            r = fe[(lay, rec)]
            indent[f"{lab}, {rec[-2:]} % gyroid"] = {
                f"{F:g} N": {f"E_s {E:g} MPa": round(approach_at(r, F, E), 1) for E in (6, 9, 12, 18, 24)} for F in (0.5, 1.0, 3.0)}
    s5, s4 = sense("five_d3x2_d5", "w85_30"), sense("four_d3x2_d5", "w85_30")
    steps = [
        dict(id="seat", what="Seat coupon in 85A (print/20261016-siraya85a_seat_coupon_*): press a magnet into each 2.7, 2.8 and "
                             "2.9 mm round seat and slot; keep the narrowest seat that takes the magnet to its floor without "
                             "splitting the wall and holds it when the coupon is turned over and tapped",
             predicts="no model; sets the production seat (2.7 mm in every print file now)", status="pending"),
        dict(id="retention", what="Magnet retention: read the board's zero-load field (step 'board') before and after 50 presses "
                                  "of the contact point at 3 N; a magnet that moved in its seat shifts the field at its nearest chip",
             predicts="field change under 2 % between the two zero-load readings", status="pending"),
        dict(id="mass", what="Weigh each printed tip with its magnets, and one assembled finger",
             predicts={v: var[v]["tip_g"] for v in var if v != "block_e10"} | {"finger_model_g": 85.2, "owner_estimate_g": 120},
             status="pending"),
        dict(id="indentation", what="Indentation of the printed block under a 25 mm diameter rod at the contact point (the "
                                    "bench tool's radius), at 0.5, 1 and 3 N on the load cell, tip on a rigid back; three "
                                    "repeats; the approach from first touch",
             predicts=indent, inverts="E_s = 12 MPa x F / F12(d): modulus_from() in this script", status="pending"),
        dict(id="pull", what="Friction pull: two printed tips pinch the 25 mm rod at 0.5, 1 and 2 N per tip; pull along the "
                             "rod at 2 N/s (scripts/tpu_tip_calibrate.py fit --pull); onset force over twice the pinch force "
                             "is the friction coefficient",
             predicts=dict(model_mu=1.0, note="every simulation uses mu 1.0 (studies vary 0.5-1.0); the pads' onset/(2N) is "
                                              f"{(bed.get('block4_g30|v2') or {}).get('pull_mu_eff')} at mu 1"),
             status="pending"),
        dict(id="twist", what="Twist onset: the pinched rod (1 N per tip) under a slow torque ramp about the pinch axis; the "
                              "torque at which it starts to turn over (2 mu N) is the onset arm",
             predicts={k: (bed.get(f"{k}|v2") or {}).get("onset_arm_mm") for k in ("block4_g30", "block4_g50", "cap_g30",
                                                                                  "ell16_g30", "ell11_g30")} |
             {"drake_block4_g30": (bed.get("block4_g30|drake") or {}).get("onset_arm_mm")}, status="pending"),
        dict(id="servo_step", what="Servo free-air step: each finger servo commanded 10 deg at the station's 50 Hz with the "
                                   "finger in air; position readback at 111 Hz; then a stall against the load cell at the "
                                   "thumb MCP (servo identification protocol, 20261013-servo_identification)",
             predicts=dict(time_constant_s=0.02, kp_Nm_per_rad=0.72, torque_limit_Nm=0.18, datasheet_stall_Nm=0.23),
             status="pending"),
        dict(id="board", what="Board zero-load field: the five chips at GAIN_SEL 7, RES 0, OSR 0, DIG_FILT 5 (141 Hz), tip "
                              "unloaded, board on the dorsal face of the DIP plate, chips toward the plate; compare each "
                              "chip's field with the prediction; a reversed magnet shows its signature",
             predicts=dict(five_magnets_plate_mT=s5["B_chips_mT"], four_magnets_plate_mT=s4["B_chips_mT"],
                           gain_sel=dict(five=s5["gain_sel"], four=s4["gain_sel"]),
                           one_magnet_reversed_dB_uT=dict(five=s5["polarity_flip_dB_uT"], four=s4["polarity_flip_dB_uT"]),
                           force_bound_N=dict(five=s5["centre_force_N"], four=s4["centre_force_N"])),
             status="pending"),
        dict(id="calibration", what="Force calibration of the board: press the contact point at 0.25-3 N on the load cell "
                                    "with the rod, 15 positions over the face; fit the field change per newton per chip",
             predicts=dict(travel_um_per_N=dict(five=s5["magnet_travel_1N_um"], four=s4["magnet_travel_1N_um"]),
                           signal_U1_uT_per_N=dict(five=s5["signal_U1_uT_per_N"], four=s4["signal_U1_uT_per_N"])),
             status="pending"),
        dict(id="tags", what="AprilTag tracker: scripts/real_v1_tag_tracker.py --probe with the station's BENCH_DEFAULTS; for "
                             "the screwdriver remove --symmetric-object from BENCH_DEFAULTS (docs/hardware_control_station.md)",
             predicts="static tag id6 latched, object tag id0 sampled before any motion", status="pending"),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "procedure.json").write_text(json.dumps(dict(steps=steps, E0_MPa=E0, script="scripts/sim2real_fingertip_bench.py",
                                                        when=time.strftime("%Y-%m-%d %H:%M")), indent=1, default=float))
    print(f"{(OUT / 'procedure.json').relative_to(ROOT)}: {len(steps)} steps, all pending")
    r = fe[("four_d3x2_d5", "w85_30")]
    for d in (40.0, 60.8, 120.0):
        print(f"block four 30 %: approach {d} um at 1 N -> E_s {modulus_from(r, 1.0, d):.1f} MPa")


# ------------------------------------------------------------------------------------------------- the first-bench package

HOST = ROOT / "src/morphohand/driver/manta/host"
NAMES = [f"{f}_{j}" for f in ("thumb", "index", "middle") for j in ("yaw", "mcp", "pip")]
T_OPEN, T_GRIP_RAMP, T_GRIP_HOLD, T_HOLD, T_RELEASE, T_AFTER = 1.0, 0.1, 0.7, 1.0, 0.5, 0.5


def pick_source(variant, cell, seed):
    """The pinned-iCEM row of the variant on the cell (cells/icem.jsonl)."""
    best = None
    for l in open(D / "cells/icem.jsonl"):
        r = json.loads(l)
        if r.get("status") != "ok" or r["variant"] != variant:
            continue
        if f"{r['shape']}:{r['d_mm']:g}:{r['t']:g}" != cell or (seed is not None and r["seed"] != seed):
            continue
        key = (r["held"] and r["fingers_loaded_end"] == 3, r["held_turn_deg"])
        if best is None or key > best[0]:
            best = (key, r)
    if best is None:
        raise SystemExit(f"no pinned-iCEM row for {variant} on {cell}")
    return best[1]


def poses_of(meta, log, release_pip_open=0.0):
    """The station's set-points (sim degrees): open, grip, one per 20 ms command of the run, hold, release (the open pose
    with every PIP release_pip_open degrees more open, clipped at -15 deg; the operator's Release is also the abort)."""
    deg = lambda q: {f: {j: round(math.degrees(q[3 * i + k]), 3) for k, j in enumerate(("yaw", "mcp", "pip"))}  # noqa: E731
                     for i, f in enumerate(("thumb", "index", "middle"))}
    q_open = [meta["q0"][n] for n in NAMES]
    q_grip = [meta["grip"][f][k] for f in ("thumb", "index", "middle") for k in range(3)]
    P = [dict(name="open", ramp_s=0.0, hold_s=T_OPEN, joints=deg(q_open)),
         dict(name="grip", ramp_s=T_GRIP_RAMP, hold_s=T_GRIP_HOLD, joints=deg(q_grip))]
    for k, smp in enumerate(log):
        P.append(dict(name=f"turn_{k:03d}", ramp_s=0.02, hold_s=0.0, joints=deg(smp["u"])))
    P.append(dict(name="hold", ramp_s=0.0, hold_s=T_HOLD, joints=dict(P[-1]["joints"])))
    rel = {f: dict(j, pip=round(max(j["pip"] - release_pip_open, -15.0), 3)) for f, j in deg(q_open).items()}
    P.append(dict(name="release", ramp_s=T_RELEASE, hold_s=T_AFTER, joints=rel))
    return P


def schedule(poses, rate=50.0):
    """(t, q rad) at rate Hz, as HandPlan.run_trajectory interpolates: the first pose held, then each ramp linear."""
    out, t = [], 0.0
    q = np.array([math.radians(poses[0]["joints"][f][j]) for f, j in (n.split("_") for n in NAMES)])
    n_hold = int(round(poses[0]["hold_s"] * rate))
    for _ in range(max(1, n_hold)):
        out.append((t, q.copy()))
        t += 1.0 / rate
    for p in poses[1:]:
        qn = np.array([math.radians(p["joints"][f][j]) for f, j in (n.split("_") for n in NAMES)])
        n = max(1, int(round(p["ramp_s"] * rate)))
        for i in range(1, n + 1):
            out.append((t, q + (qn - q) * i / n))
            t += 1.0 / rate
        for _ in range(int(round(p["hold_s"] * rate))):
            out.append((t, qn.copy()))
            t += 1.0 / rate
        q = qn
    return out


def replay(plant, meta, sched, seed, post_off_at, cond):
    """Open-loop replay of the schedule on the cell's plant with one condition {kp, tau, mu}; per 20 ms the turn, the
    fingers' normal forces, actuator forces; the post's collisions end at post_off_at (None: never)."""
    m, d = plant.m, plant.d
    a = plant.act
    m.actuator_gainprm[a, 0] = cond["kp"]
    m.actuator_biasprm[a, 1] = -cond["kp"]
    m.actuator_forcerange[a] = np.array([-cond["tau"], cond["tau"]])
    m.geom_friction[:, 0] = cond["mu"]
    post = m.geom("post").id
    if not hasattr(plant, "post0"):
        plant.post0 = (int(m.geom_contype[post]), int(m.geom_conaffinity[post]))
    m.geom_contype[post], m.geom_conaffinity[post] = plant.post0
    plant.reset(seed)
    s0 = None
    out, i_turn, qs = [], None, []
    at_lim = np.zeros(9)
    run_lim = np.zeros(9)
    weight = meta["mass_kg"] * 9.81
    for k, (t, q) in enumerate(sched):
        if post_off_at is not None and t >= post_off_at - 1e-9:
            m.geom_contype[post] = m.geom_conaffinity[post] = 0
        plant.set_targets(q)
        plant.advance(t + 0.02)
        s = plant.state()
        if s0 is None and t >= T_OPEN + T_GRIP_RAMP + T_GRIP_HOLD - 1e-9:
            s0 = s
            u0 = s0["R"][:, 2] / np.linalg.norm(s0["R"][:, 2])
            a0 = np.cross(u0, [0.0, 0.0, 1.0])
            a0 /= max(1e-9, np.linalg.norm(a0))
            h0 = np.cross(a0, u0)
        tau = np.abs(d.actuator_force[a])
        lim = tau >= 0.98 * cond["tau"]
        run_lim = np.where(lim, run_lim + 0.02, 0.0)
        at_lim = np.maximum(at_lim, run_lim)
        th = math.degrees(math.atan2(s["R"][:, 2] @ h0, s["R"][:, 2] @ u0)) if s0 is not None else 0.0
        nf = sum(1 for f in ("thumb", "index", "middle") if s["F"][f] >= weight)
        out.append((round(t, 3), round(th, 2), [round(s["F"][f], 3) for f in ("thumb", "index", "middle")],
                    round(float(tau.max()), 4), nf, round(float(s["p"][2]) * 1e3, 2)))
        qs.append((t, s["q"].copy()))
    return out, at_lim, qs


def clearance(m_path, sched_q, palm):
    """Least distance between geoms of different fingers along the schedule (joint angles set kinematically), from
    MuJoCo's collision pass with a 10 mm margin on every finger geom; the pads excluded."""
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(m_path))
    m.body_pos[m.body("palm_pose").id] = palm
    d = mujoco.MjData(m)
    fing = np.full(m.ngeom, -1)
    for g in range(m.ngeom):
        bn = m.body(m.geom_bodyid[g]).name
        for k, f in enumerate(("thumb", "index", "middle")):
            if bn.startswith(f + "_"):
                fing[g] = k
        if "_pad" in (m.geom(g).name or ""):
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
    m.geom_margin[fing >= 0] = 0.010
    qadr = [m.jnt_qposadr[m.joint(n).id] for n in NAMES]
    best = (1.0, None, None)
    for t, q in sched_q:
        d.qpos[qadr] = q
        mujoco.mj_forward(m, d)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if fing[g1] >= 0 and fing[g2] >= 0 and fing[g1] != fing[g2] and c.dist < best[0]:
                best = (float(c.dist), round(t, 2), (m.geom(g1).name or m.body(m.geom_bodyid[g1]).name,
                                                     m.geom(g2).name or m.body(m.geom_bodyid[g2]).name))
    return dict(min_mm=round(best[0] * 1e3, 2), at_s=best[1], pair=best[2])


def capped(poses, meta, sc, gg, tau_cap, kp):
    """Poses whose commands ask each servo for at most tau_cap: every set-point after 'open' moved to within tau_cap / kp
    of the joint angle the nominal replay of the uncapped plan reaches at that time (post present, placement 0, the
    nominal plant), so the position servo's demand kp (q_cmd - q) stays at or below tau_cap where the fingers follow."""
    import hom_turn3 as H
    import sim2real_fingertip_cells as CE
    import hand_object_scale_sim as S
    plant = H.MjPlant(sc["pads"][0], meta)
    S.set_palm(plant.m, gg["palm"])
    sched = schedule(poses)
    _, _, qa = replay(plant, meta, sched, 0, None, dict(kp=kp, tau=0.18, mu=1.0))
    tq = np.array([t for t, _ in qa])
    Q = np.array([q for _, q in qa])
    out, t = [dict(poses[0])], poses[0]["hold_s"]
    names = [(f, j) for f in ("thumb", "index", "middle") for j in ("yaw", "mcp", "pip")]
    for pz in poses[1:]:
        t += pz["ramp_s"]
        k = int(np.clip(np.searchsorted(tq, t), 0, len(tq) - 1))
        q_now = Q[k]
        nj = {f: dict(v) for f, v in pz["joints"].items()}
        for i, (f, j) in enumerate(names):
            c = math.radians(nj[f][j])
            lo, hi = q_now[i] - tau_cap / kp, q_now[i] + tau_cap / kp
            nj[f][j] = round(math.degrees(min(max(c, lo), hi)), 3) if pz["name"] != "release" else nj[f][j]
        out.append(dict(pz, joints=nj))
        t += pz["hold_s"]
    return out


def cmd_export(a):
    """The first-bench package of one variant on one cell, from its pinned-iCEM run (module docstring)."""
    import os
    os.environ["SCALE_HAND"] = "cad_v1"
    sys.path.insert(0, str(HOST))
    import hom_turn3 as H
    import mujoco
    import sim2real_fingertip_cells as CE
    from manta_hand.plan import HandPlan
    src = pick_source(a.variant, a.cell, a.seed)
    S, spec = CE.setup(a.variant, "v2", src["mu"])
    cell, shape, dd = CE.cell_of(S, a.cell)
    cand = S.candidates(cell, shape, dd)[src["cand"]]
    kp = S.RB.plant_args(CE.SERVO)["kp"]
    sc = S.cell_scenes(cell, shape, dd, models=("pads", "pt"))
    m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
    gg = S.grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
    gp = S.grip_poses(m_pt, gg, kp)
    meta = S.candidate_meta(sc["pads"][1], gg, gp)
    name = f"{DATE_TAG}-s2r_{a.variant}_{shape[:3]}{dd:g}_t{cell['t']:g}_c{src['cand']}".replace(".", "p").replace("-0p", "-m0p")
    poses = poses_of(meta, src["log"], a.release_pip_open)
    if a.tau_cap:
        name += f"_cap{a.tau_cap:g}".replace(".", "p")
        poses = capped(poses, meta, sc, gg, a.tau_cap, kp)
    if a.release_pip_open:
        name += f"_rel{a.release_pip_open:g}"
    mounts = {f: [round(float(v) * 1e3, 2) for v in gg["mounts"][f][:2]] for f in ("thumb", "index", "middle")}
    obj_palm = (np.array([0.0, 0.0, S.Z_OBJ]) - gg["palm"]) * 1e3
    plan = dict(design=name, mounts_palm_mm=mounts, poses=poses, meta=dict(
        variant=a.variant, tip=spec["key"], pad_E_MPa=round(spec["pad_E_Pa"] / 1e6, 3), cad_model="assets/cad_v1 "
        "(MCP link pip_mount_MCP_link.stl, MCP-PIP 30.80 mm)", scene=f"assets/mjcf/experimental/20261016-sim2real_fingertip_readiness/"
        f"{a.variant}/cad_v1/hand.xml", cell=a.cell, cell_tag=cell["tag"], x_sep_mm=cell["x_sep_mm"], y_sep_mm=cell["y_sep_mm"],
        grasp=src["cand"], palm_height_mm=round(cand["h"] * 1e3, 2), straddle_mm=round(cand["spread"] * 1e3, 2),
        object="cylinder", object_d_mm=dd, object_mass_g=S.MASS * 1e3, object_centre_palm_mm=np.round(obj_palm, 2).tolist(),
        object_axis="palm y (along the index-middle pair)", source_row=dict(file="cells/icem.jsonl", variant=src["variant"],
        cell=a.cell, cand=src["cand"], seed=src["seed"], held=src["held"], held_turn_deg=src["held_turn_deg"],
        fingers_loaded_end=src["fingers_loaded_end"], cmd_rate_rms=src.get("cmd_rate_rms")),
        controller="pinned iCEM (hand_object_scale_planners icem_pin) on the servo plant, its executed servo targets replayed "
                   "open loop", plant=CE.SERVO, rate_hz=50, timing_s=dict(open=T_OPEN, grip_ramp=T_GRIP_RAMP,
                   grip_hold=T_GRIP_HOLD, turn=round(0.02 * len(src["log"]), 2), hold=T_HOLD, release=T_RELEASE, after=T_AFTER),
        release_pip_open_deg=a.release_pip_open, source="scripts/sim2real_fingertip_bench.py export"))
    OUT.mkdir(parents=True, exist_ok=True)
    pj = OUT / f"{name}_plan.json"
    pj.write_text(json.dumps(plan, indent=1))
    hp = HandPlan.from_json(pj)
    sched = schedule(poses)
    with open(OUT / f"{name}_traj.csv", "w") as fh:
        fh.write("t_s," + ",".join(f"{n}_deg" for n in NAMES) + "\n")
        for t, q in sched:
            fh.write(f"{t:.2f}," + ",".join(f"{math.degrees(v):.3f}" for v in q) + "\n")
    qs = np.array([q for _, q in sched])
    speed = np.degrees(np.abs(np.diff(qs, axis=0)).max(axis=0)) * 50.0
    station = dict(schema_errors=hp.schema_errors(), violations=[str(v) for v in hp.validate()],
                   describe=hp.describe().splitlines())
    # simulation of the plan: nominal and the plant, friction and post conditions
    t_post = T_OPEN + T_GRIP_RAMP + T_GRIP_HOLD
    conds = [dict(kp=kp_, tau=tau, mu=mu) for kp_ in (0.36, 0.72, 1.08) for tau in (0.12, 0.18, 0.23) for mu in (0.5, 0.7, 1.0)]
    rows = []
    plant = H.MjPlant(sc["pads"][0], meta)
    S.set_palm(plant.m, gg["palm"])
    nominal = None
    for post in ("removed", "present"):
        for c in conds:
            for seed in a.seeds:
                tr, lim, qa = replay(plant, meta, sched, seed, t_post if post == "removed" else None, c)
                i_hold = int(round((t_post + 0.02 * len(src["log"]) + T_HOLD) / 0.02)) - 1
                th_h, F_h, z_h = tr[i_hold][1], tr[i_hold][2], tr[i_hold][5]
                held = sum(1 for x in F_h if x >= S.MASS * 9.81) >= 2 and z_h > tr[int(t_post / 0.02)][5] - 20
                row = dict(post=post, **c, seed=seed, held_before_release=bool(held), turn_deg=th_h,
                           three_pads=sum(1 for x in F_h if x >= S.MASS * 9.81) == 3, F_hold=F_h,
                           peak_tau_Nm=max(x[3] for x in tr), longest_at_limit_s=round(float(lim.max()), 2))
                rows.append(row)
                if post == "removed" and c == dict(kp=0.72, tau=0.18, mu=1.0) and seed == 0:
                    nominal, q_nominal = tr, qa
    by = lambda f: [r for r in rows if f(r)]  # noqa: E731
    summ = {}
    for post in ("removed", "present"):
        R = by(lambda r: r["post"] == post)
        summ[post] = dict(held=f"{sum(r['held_before_release'] for r in R)}/{len(R)}",
                          three_pads=f"{sum(r['three_pads'] for r in R)}/{len(R)}",
                          turn_median_deg=round(float(np.median([r["turn_deg"] for r in R if r["held_before_release"]] or [0])), 1),
                          nominal=[r for r in R if r["kp"] == 0.72 and r["tau"] == 0.18 and r["mu"] == 1.0])
    clr = dict(commanded=clearance(sc["pads"][0], sched, gg["palm"]),
               achieved_nominal=clearance(sc["pads"][0], q_nominal, gg["palm"]))
    if clr["achieved_nominal"]["at_s"] is not None:
        k = int(np.argmin([abs(t - clr["achieved_nominal"]["at_s"]) for t, _ in q_nominal]))
        clr["achieved_nominal"]["cad_mesh_at_that_instant"] = mesh_clearance(sc["pads"][0], q_nominal[k][1], gg["palm"])
    # expected chip signals: U1's field change per newton of the variant's readout row times the finger's normal force
    sens = json.loads((D / "sensing/sensing.json").read_text())["rows"]
    lay = "five_d3x2_d5" if "block5" in a.variant else ("four_d3x2_d5_w1" if "w1" in a.variant else "four_d3x2_d5")
    rec = f"w85_{a.variant.rsplit('_g', 1)[1]}" if "_g" in a.variant else "w85_30"
    srow = next(r for r in sens if r["layout"] == lay and r["recipe"] == rec and r["plane"] == "plate")
    sig = [(t, [round(f * srow["signal_U1_uT_per_N"], 1) for f in F]) for t, _, F, _, _, _ in (nominal or [])][::5]
    ver = dict(plan=pj.name, station=station, peak_joint_speed_deg_s=dict(zip(NAMES, np.round(speed, 1).tolist())),
               servo_speed_setting=80, conditions=len(conds) * len(a.seeds), summary=summ, rows=rows,
               finger_clearance_cad=clr, expected_U1_field_change_uT=dict(per_N=srow["signal_U1_uT_per_N"],
               readout_row=f"{lay} {rec} plate", trace_every_0p1s=sig),
               nominal_trace=nominal, script="scripts/sim2real_fingertip_bench.py export",
               when=time.strftime("%Y-%m-%d %H:%M"))
    (OUT / f"{name}_verify.json").write_text(json.dumps(ver, indent=1, default=float))
    print(f"{pj.relative_to(ROOT)}: {len(poses)} poses; station errors {station['schema_errors']} violations "
          f"{station['violations']}; post removed {summ['removed']['held']} held, {summ['removed']['three_pads']} three pads, "
          f"median {summ['removed']['turn_median_deg']} deg; post present {summ['present']['held']} held, median "
          f"{summ['present']['turn_median_deg']} deg; finger clearance {clr}", flush=True)


def mesh_clearance(m_path, q, palm, n_per_mm2=4.0):
    """Least distance between the CAD meshes (visual geoms: links, servos, DIP link) and boards of different fingers at
    one configuration, from area-weighted surface samples and k-d trees; mm, with the closest pair of parts."""
    import mujoco
    from scipy.spatial import cKDTree
    import sim2real_fingertip_scenes as SC
    m = mujoco.MjModel.from_xml_path(str(m_path))
    m.body_pos[m.body("palm_pose").id] = palm
    d = mujoco.MjData(m)
    d.qpos[[m.jnt_qposadr[m.joint(n).id] for n in NAMES]] = q
    mujoco.mj_kinematics(m, d)
    pts = {}
    for g in range(m.ngeom):
        nm = m.geom(g).name or ""
        f = next((x for x in ("thumb", "index", "middle") if nm.startswith(x + "_")), None)
        if f is None:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and nm.endswith("_visual"):
            P = SC.sample_mesh_points(m, g, n_per_mm2=n_per_mm2)
        elif "board" in nm:
            h = m.geom_size[g]
            u = np.stack(np.meshgrid(*[np.linspace(-1, 1, 9)] * 3, indexing="ij"), -1).reshape(-1, 3)
            P = u * h
        else:
            continue
        W = P @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        pts.setdefault(f, []).append((nm, W))
    best = (1.0, None)
    fs = list(pts)
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            for na, A in pts[fs[i]]:
                tree = cKDTree(A)
                for nb, B in pts[fs[j]]:
                    dist = float(tree.query(B, k=1)[0].min())
                    if dist < best[0]:
                        best = (dist, (na, nb))
    return dict(min_mm=round(best[0] * 1e3, 2), pair=best[1])


def cmd_abort(a):
    """The abort path of an exported plan: the plan replayed to a time inside the turn, then every servo ramped to the
    plan's 'release' pose over 0.5 s (the operator's Stop motion and Release; --pip-open opens every PIP further); per abort
    time the least finger-to-finger clearance on the achieved joint angles, the largest actuator force, the time until no
    pad presses the cylinder and where the cylinder ends."""
    import os
    os.environ["SCALE_HAND"] = "cad_v1"
    import hom_turn3 as H
    import mujoco
    import sim2real_fingertip_cells as CE
    plan = json.loads((OUT / f"{a.plan}_plan.json").read_text())
    m = plan["meta"]
    S, spec = CE.setup(m["variant"], "v2", 1.0)
    cell, shape, dd = CE.cell_of(S, m["cell"])
    cand = S.candidates(cell, shape, dd)[m["grasp"]]
    kp = S.RB.plant_args(CE.SERVO)["kp"]
    sc = S.cell_scenes(cell, shape, dd, models=("pads", "pt"))
    m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
    gg = S.grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
    gp = S.grip_poses(m_pt, gg, kp)
    meta = S.candidate_meta(sc["pads"][1], gg, gp)
    plant = H.MjPlant(sc["pads"][0], meta)
    S.set_palm(plant.m, gg["palm"])
    poses = plan["poses"]
    t_turn = T_OPEN + T_GRIP_RAMP + T_GRIP_HOLD
    rows = []
    for dt_abort in a.at:
        t_ab = t_turn + dt_abort
        cut, t = [], 0.0
        for pz in poses:
            if pz["name"] in ("hold", "release"):
                break
            t += pz["ramp_s"] + pz["hold_s"]
            cut.append(pz)
            if t >= t_ab:
                break
        rel = next(p for p in poses if p["name"] == "release")["joints"]
        ab_pose = {f: dict(j, pip=max(j["pip"] - a.pip_open, -15.0)) for f, j in rel.items()}
        cut.append(dict(name="abort_open", ramp_s=0.5, hold_s=1.0, joints=ab_pose))
        sched = schedule(cut)
        for post in ("present", "removed"):
            tr, lim, qa = replay(plant, meta, sched, 0, t_turn if post == "removed" else None, dict(kp=kp, tau=0.18, mu=1.0))
            clr = clearance(sc["pads"][0], qa, gg["palm"])
            if clr["at_s"] is not None:
                k = int(np.argmin([abs(t - clr["at_s"]) for t, _ in qa]))
                clr["cad_mesh_at_that_instant"] = mesh_clearance(sc["pads"][0], qa[k][1], gg["palm"])
            free = [x[0] for x in tr if x[0] >= t_ab and max(x[2]) < 0.05]
            rows.append(dict(abort_after_s=dt_abort, post=post, clearance_achieved=clr, peak_tau_Nm=max(x[3] for x in tr),
                             longest_at_limit_s=round(float(lim.max()), 2), z_end_mm=tr[-1][5], z_grip_mm=tr[int(t_turn / 0.02)][5],
                             fingers_loaded_end=tr[-1][4], F_end_N=tr[-1][2],
                             released_after_s=round(free[0] - t_ab, 2) if free else None, abort_pose=ab_pose))
            print(f"abort after {dt_abort} s, post {post}: clearance {clr}, peak {rows[-1]['peak_tau_Nm']} N m, cylinder "
                  f"{tr[-1][5]:.1f} mm (grip {rows[-1]['z_grip_mm']:.1f})", flush=True)
    tag = "" if not a.pip_open else f"_pip{a.pip_open:g}"
    (OUT / f"{a.plan}_abort{tag}.json").write_text(json.dumps(dict(plan=a.plan, rows=rows, open_ramp_s=0.5, pip_open_deg=a.pip_open,
                                                              script="scripts/sim2real_fingertip_bench.py abort",
                                                              when=time.strftime("%Y-%m-%d %H:%M")), indent=1, default=float))


def cmd_film(a):
    """The plan's open-loop replay rendered (960 x 720, pads touching the cylinder red, the board green), nominal plant,
    one placement, the post present or removed; stills at the end of the grip, mid-turn, the hold and after the release."""
    import os
    os.environ["SCALE_HAND"] = "cad_v1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    import hom_turn3 as H
    import mujoco
    import contact_bed_common as CB
    import hand_object_scale_films as SF
    import reorient_backends_films as RBF
    import sim2real_fingertip_cells as CE
    import tip_codesign_film as TF
    from PIL import Image
    plan = json.loads((OUT / f"{a.plan}_plan.json").read_text())
    m_ = plan["meta"]
    S, spec = CE.setup(m_["variant"], "v2", 1.0)
    cell, shape, dd = CE.cell_of(S, m_["cell"])
    cand = S.candidates(cell, shape, dd)[m_["grasp"]]
    kp = S.RB.plant_args(CE.SERVO)["kp"]
    sc = S.cell_scenes(cell, shape, dd, models=("pads", "pt"))
    m_pt = mujoco.MjModel.from_xml_path(str(sc["pt"][0]))
    gg = S.grasp_geometry(cell, shape, dd, cand["h"], cand["spread"])
    gp = S.grip_poses(m_pt, gg, kp)
    meta = S.candidate_meta(sc["pads"][1], gg, gp)
    plant = H.MjPlant(SF.film_scene(Path(sc["pads"][0]), dd / 2000.0), meta)
    S.set_palm(plant.m, gg["palm"])
    m, d = plant.m, plant.d
    pads = [i for i in range(m.ngeom) if "_pad" in (m.geom(i).name or "")]
    boards = [i for i in range(m.ngeom) if "board" in (m.geom(i).name or "")]
    m.vis.global_.offwidth, m.vis.global_.offheight = 960, 720
    RBF.W, RBF.H = 960, 720
    rend, cam = RBF.renderer_for(m, np.asarray(meta["tool7"], float)[:3])
    cam.distance = 0.24 + 1.4 * dd / 1000.0
    cam.elevation += 10.0
    sched = schedule(plan["poses"])
    t_turn = T_OPEN + T_GRIP_RAMP + T_GRIP_HOLD
    t_hold = t_turn + 0.02 * (len(plan["poses"]) - 4)
    post = m.geom("post").id
    plant.reset(a.seed)
    frames, stills = [], {}
    title = f"Open-loop bench plan, {dd:g} mm cylinder, post {a.post}"  # one line of the 960 px bar
    tip = CE.plain(m_["variant"])
    for k, (t, q) in enumerate(sched):
        if a.post == "removed" and t >= t_turn - 1e-9:
            m.geom_contype[post] = m.geom_conaffinity[post] = 0
        plant.set_targets(q)
        plant.advance(t + 0.02)
        if k % 2:
            continue
        for i in pads:
            m.geom_rgba[i] = (0.75, 0.4, 0.25, 0.0)
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom[0]), int(d.contact[i].geom[1])
            for g, o in ((g1, g2), (g2, g1)):
                if o in plant.tool_geoms and g in pads:
                    m.geom_rgba[g] = (0.9, 0.1, 0.1, 1.0)
        rend.update_scene(d, cam)
        phase = ("open" if t < T_OPEN else "grip" if t < t_turn else "turn" if t < t_hold else
                 "hold" if t < t_hold + T_HOLD else "release")
        fr = TF.label_bar(rend.render().copy(), [title, f"{tip}   t {t:4.2f} s   {phase}"])
        frames.append(fr)
        for nm, tw in (("grip", t_turn - 0.1), ("turn", t_turn + 0.5 * (t_hold - t_turn)), ("hold", t_hold + 0.5),
                       ("release", t_hold + T_HOLD + T_RELEASE + 0.4)):
            if nm not in stills and t >= tw:
                stills[nm] = fr
    rend.close()
    media = D / "media"
    stem = f"bench_{a.plan.split('s2r_')[1]}_post{a.post}_s{a.seed}"
    CB.write_h264(frames, media / f"{stem}.mp4", crf=20)
    for nm, fr in stills.items():
        Image.fromarray(fr).save(media / f"{stem}_{nm}.png")
    print(media / f"{stem}.mp4", len(frames), sorted(stills), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("procedure")
    e = sub.add_parser("export")
    e.add_argument("--variant", required=True)
    e.add_argument("--cell", required=True)
    e.add_argument("--seed", type=int, default=None, help="the iCEM row's seed (default: the best row of the cell)")
    e.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="placements of the verification replays")
    e.add_argument("--tau-cap", type=float, default=None,
                   help="cap every commanded offset at this torque over the gain, N m (the servo's overload guard)")
    e.add_argument("--release-pip-open", type=float, default=0.0,
                   help="the release pose (the station's Release, also the abort target) this many degrees more open at "
                        "every PIP than the open pose, clipped at -15 deg; the plan name gains _rel<deg>")
    b = sub.add_parser("abort")
    b.add_argument("--plan", required=True, help="plan name under bench/ (without _plan.json)")
    b.add_argument("--at", nargs="+", type=float, default=[1.0, 3.0, 5.0], help="abort times after the grip, s")
    b.add_argument("--pip-open", type=float, default=0.0, help="the abort pose's PIP angles this many degrees more open "
                   "than the plan's release pose (clipped at -15 deg); rows in <plan>_abort_pip<deg>.json")
    fm = sub.add_parser("film")
    fm.add_argument("--plan", required=True)
    fm.add_argument("--post", default="present", choices=["present", "removed"])
    fm.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"procedure": cmd_procedure, "export": cmd_export, "abort": cmd_abort, "film": cmd_film}[a.cmd](a)


if __name__ == "__main__":
    main()
