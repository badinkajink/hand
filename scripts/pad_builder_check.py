#!/usr/bin/env python3
"""Pad builder v1 against v2 and Drake hydroelastic at trajectory level on the real_v1 hand (2026-10-15, before builder
v2 became the default of reorient_backends.replace_tips). Owner, 2026-10-15: "held" is too coarse to show that two
contact models give the same dynamics; compare the tool's pose and the fingertip normal and friction forces over time,
with seed spread.

Two tasks on the servo-limited plant (kp 0.72 N m/rad, 0.18 N m, friction 1.0, 1 ms step, impratio 100): the deployed
open-loop plans of D1-D8 replayed as on the bench (reorient_backends.schedule, seeds 0-4: tool placement jitter of 2 mm
and 2 deg), and the HOM closed-loop turn (hom_turn3, kp_model 0.72, seeds 0-2). Three contact models per tip: 1 mm
sphere pads with builder v1 (one stiffness per finger, h = 8.5 mm) and v2 (per-pad E a_i / h_i, h_i the local depth),
and Drake's compliant hydroelastic tip (E 10 MPa, SAP). Two tips: tpu2.7 (fingertip_geometry's block, the default) and
tcd_rs3_re3_p0.2 (the as-built insert as printed, tip_codesign_geometry). Every 20 ms a rollout records the tool's
position and quaternion and, per finger, the summed normal force and the vector sum of the friction forces on the tool.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/pad_builder_check.py run --what replay --tips tpu2.7 --conds v1 v2 drake --hands D1 D5 --seeds 0 1 2 3 4
    $PY scripts/pad_builder_check.py report
    .venv/bin/python scripts/pad_builder_check.py figure     # media/fig_builder_default.png of the tip study

Rows: docs/experiments/20261014-tip_codesign/builder_default/{replay,hom}.jsonl (one fsynced line per rollout, resumable);
report: builder_default/summary.json.
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
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hom_turn3 as HT  # noqa: E402
import reorient_backends as RB  # noqa: E402

SCRIPT = "scripts/pad_builder_check.py"
D = ROOT / "docs/experiments/20261014-tip_codesign/builder_default"
SCN = ROOT / "logs/20261015-tip_followups/scenes"
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
FINGERS, NAMES = RB.FINGERS, HT.NAMES
TRACE_DT = 0.02
PRM = dict(HT.DEFAULTS, kp_model=0.72)          # as the 2026-10-14 HOM rows on this plant
PAIRS = (("v1", "drake"), ("v2", "drake"), ("v1", "v2"))


# ---------------------------------------------------------------------------------- plants with friction forces

class MjPlantF(HT.MjPlant):
    def forces(self):
        """Per finger: summed normal force (N), friction force on the tool (world vector, N), loaded contacts."""
        import mujoco
        m, d = self.m, self.d
        N = {f: 0.0 for f in FINGERS}
        T = {f: np.zeros(3) for f in FINGERS}
        n = {f: 0 for f in FINGERS}
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if (g1 in self.tool_geoms) == (g2 in self.tool_geoms):
                continue
            f = self.finger_of.get(g2 if g1 in self.tool_geoms else g1)
            if f is None:
                continue
            mujoco.mj_contactForce(m, d, i, self.f6)
            if self.f6[0] <= 1e-6:
                continue
            fr = c.frame.reshape(3, 3)                    # rows: normal (geom1 to geom2), tangents; world frame
            on_tool = fr.T @ self.f6[:3] * (1.0 if g2 in self.tool_geoms else -1.0)
            N[f] += float(self.f6[0])
            T[f] += on_tool - (on_tool @ fr[0]) * fr[0]
            n[f] += 1
        return N, T, n

    def pose(self):
        d = self.d
        return d.qpos[self.qa:self.qa + 3].copy(), d.qpos[self.qa + 3:self.qa + 7].copy()


class DrakePlantF(HT.DrakePlant):
    def forces(self):
        db = self.db
        N = {f: 0.0 for f in FINGERS}
        T = {f: np.zeros(3) for f in FINGERS}
        n = {f: 0 for f in FINGERS}
        cr = db.plant.get_contact_results_output_port().Eval(db.pc)
        for i in range(cr.num_hydroelastic_contacts()):
            info = cr.hydroelastic_contact_info(i)
            srf = info.contact_surface()
            if not ({srf.id_M(), srf.id_N()} & db.tool_gids):
                continue
            f = db.finger_gid.get(srf.id_M()) or db.finger_gid.get(srf.id_N())
            if f is None or srf.num_faces() == 0:
                continue
            F = np.array(info.F_Ac_W().translational())          # on body A, the body of geometry M
            on_tool = F if srf.id_M() in db.tool_gids else -F
            nrm = np.zeros(3)
            for k in range(srf.num_faces()):
                nrm += srf.area(k) * np.asarray(srf.face_normal(k))
            nrm /= max(np.linalg.norm(nrm), 1e-30)
            fn = abs(float(on_tool @ nrm))
            if fn > 1e-6:
                N[f] += fn
                T[f] += on_tool - (on_tool @ nrm) * nrm
                n[f] += 1
        tool = db.tool.index()
        for i in range(cr.num_point_pair_contacts()):
            info = cr.point_pair_contact_info(i)
            pp = info.point_pair()
            if not ({pp.id_A, pp.id_B} & db.tool_gids):
                continue
            f = db.finger_gid.get(pp.id_A) or db.finger_gid.get(pp.id_B)
            if f is None:
                continue
            fB = np.array(info.contact_force())                   # on body B
            on_tool = fB if info.bodyB_index() == tool else -fB
            nh = np.asarray(pp.nhat_BA_W)
            fn = abs(float(on_tool @ nh))
            if fn > 1e-6:
                N[f] += fn
                T[f] += on_tool - (on_tool @ nh) * nh
                n[f] += 1
        return N, T, n

    def pose(self):
        db = self.db
        X = db.plant.EvalBodyPoseInWorld(db.pc, db.tool)
        return np.array(X.translation()), np.array(X.rotation().ToQuaternion().wxyz())


def snap(plant, t):
    p, q = plant.pose()
    N, T, n = plant.forces()
    R9 = np.zeros(9)
    import mujoco
    mujoco.mju_quat2Mat(R9, q / np.linalg.norm(q))
    return {"t": round(float(t), 4), "p": [round(float(x), 7) for x in p], "q": [round(float(x), 7) for x in q],
            "cos": float(R9[8]), "x": float(p[0]), "y": float(p[1]), "z": float(p[2]),
            "F": {f: round(N[f], 4) for f in FINGERS}, "T": {f: [round(float(x), 4) for x in T[f]] for f in FINGERS},
            "n": n}


# ---------------------------------------------------------------------------------- scenes and rollouts

def scene(hand, tip, cond, worker):
    """(pads or Drake scene, its meta, the mesh scene for HOM's mirror, its meta) in this worker's own folder."""
    out = SCN / f"w{worker}" / tip
    p_mesh, meta_mesh = RB.build_scene(hand, tip, "pt", SERVO, "bed", 100.0, "1.0", out_dir=out / "pt")
    if cond == "drake":
        return p_mesh, meta_mesh, p_mesh, meta_mesh
    p, meta = RB.build_scene(hand, tip, "pads", SERVO, "bed", 100.0, "1.0", out_dir=out / cond, builder=cond)
    return p, meta, p_mesh, meta_mesh


def replay(plant, hand, seed):
    plan, traj = RB.load_plan(hand)
    segs = RB.schedule(plan, traj)
    plant.reset(seed)
    rec, t, bad = [], 0.0, False
    w0 = time.perf_counter()
    for t_end, tgt, _ in segs:
        plant.set_targets(np.array([tgt[n] for n in NAMES]))
        while t < t_end - 1e-9:
            t_next = min(t_end, (math.floor(t / TRACE_DT + 1e-6) + 1) * TRACE_DT)
            plant.advance(t_next)
            t = t_next
            if abs(t / TRACE_DT - round(t / TRACE_DT)) < 1e-6:
                s = snap(plant, t)
                rec.append(s)
                if not np.isfinite(s["z"]) or abs(s["z"]) > 2.0:
                    bad = True
                    break
        if bad:
            break
    res = RB.score(rec, segs) if not bad else {}
    res.update(status="ejected" if bad else "complete", wall_s=round(time.perf_counter() - w0, 2))
    return res, rec


def hom(plant, mirror, meta, seed):
    rec = []

    def cb(pl, t, info):
        if not rec or t - rec[-1]["t"] >= TRACE_DT - 1e-9:
            rec.append(snap(pl, t))
    res, _ = HT.rollout(plant, mirror, meta, seed, PRM, frame_cb=cb)
    return res, rec


def compact(rec):
    """Columns of a trace: t (s), p (m), q (wxyz), F (N, per finger), T (N, per finger, world xyz), n."""
    return {"t": [s["t"] for s in rec], "p": [s["p"] for s in rec], "q": [s["q"] for s in rec],
            "F": [[s["F"][f] for f in FINGERS] for s in rec], "T": [[s["T"][f] for f in FINGERS] for s in rec],
            "n": [[s["n"][f] for f in FINGERS] for s in rec]}


def append_row(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, default=float) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def cmd_run(a):
    out = a.out or D / f"{a.what}.jsonl"
    have = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            have.add((r["tip"], r["cond"], r["hand"], int(r["seed"])))
    for tip in a.tips:
        for cond in a.conds:
            for hand in a.hands:
                todo = [s for s in a.seeds if (tip, cond, hand, s) not in have]
                if not todo:
                    continue
                p, meta, p_mesh, meta_mesh = scene(hand, tip, cond, a.worker)
                plant = DrakePlantF(p, meta) if cond == "drake" else MjPlantF(p, meta)
                mirror = HT.Mirror(p_mesh) if a.what == "hom" else None
                for seed in todo:
                    row = {"case_id": f"{a.what}|{tip}|{cond}|{hand}|{seed}", "what": a.what, "tip": tip, "cond": cond,
                           "hand": hand, "seed": seed, "plant": SERVO, "mu": 1.0, "impratio": 100.0,
                           "builder": None if cond == "drake" else cond, "jitter": RB.jitter(seed),
                           "n_pads": meta.get("n_pads"), "pad_d0": meta.get("pad_d0"), "pad_K": meta.get("pad_K"),
                           "scene": str(Path(p).relative_to(ROOT))}
                    t0 = time.time()
                    try:
                        res, rec = replay(plant, hand, seed) if a.what == "replay" else hom(plant, mirror, meta, seed)
                        row.update(res, trace=compact(rec))
                        row.setdefault("status", "ok")
                    except Exception as e:
                        row.update(status="error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-1500:])
                    row.update(script=SCRIPT, git_rev=RB.git_rev(), wall_total_s=round(time.time() - t0, 2),
                               when=time.strftime("%Y-%m-%d %H:%M"))
                    append_row(out, row)
                    print(f"{a.what} {tip} {cond:5s} {hand} s{seed}: {row.get('status')} turn "
                          f"{row.get('turn_end_deg', float('nan')):+.1f} held {row.get('held_end', row.get('held'))} "
                          f"{row['wall_total_s']} s", flush=True)
                del plant, mirror
                gc.collect()


# ---------------------------------------------------------------------------------- report

def _arr(tr):
    t = np.asarray(tr["t"])
    return t, np.asarray(tr["p"]), np.asarray(tr["q"]), np.asarray(tr["F"]), np.asarray(tr["T"])


def divergence(a, b):
    """Over the common 20 ms samples of two traces: RMS and final tool-orientation difference (deg, geodesic), RMS and
    final position difference (mm), RMS of the per-finger normal-force difference and of the per-finger friction-force
    vector difference (N), each over time and the three fingers."""
    ta, pa, qa, Fa, Ta = _arr(a)
    tb, pb, qb, Fb, Tb = _arr(b)
    ka = np.round(ta / TRACE_DT).astype(int)
    kb = np.round(tb / TRACE_DT).astype(int)
    common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
    if len(common) < 10:
        return None
    qa_, qb_ = qa[ia] / np.linalg.norm(qa[ia], axis=1, keepdims=True), qb[ib] / np.linalg.norm(qb[ib], axis=1, keepdims=True)
    ang = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(qa_ * qb_, axis=1)), 0, 1)))
    dp = 1e3 * np.linalg.norm(pa[ia] - pb[ib], axis=1)
    dF = Fa[ia] - Fb[ib]
    dT = np.linalg.norm(Ta[ia] - Tb[ib], axis=2)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))  # noqa: E731
    za, zb = _zaxis(qa_), _zaxis(qb_)                      # the tool's axis (body z), whose tilt the end turn measures
    ax = np.degrees(np.arccos(np.clip(np.sum(za * zb, axis=1), -1, 1)))
    return {"ang_rms_deg": rms(ang), "ang_end_deg": float(ang[-1]), "pos_rms_mm": rms(dp), "pos_end_mm": float(dp[-1]),
            "F_rms_N": rms(dF), "T_rms_N": rms(dT), "axis_rms_deg": rms(ax), "axis_end_deg": float(ax[-1]),
            "t_common_s": float(common[-1] * TRACE_DT)}


def _zaxis(q):
    """Body z axis in the world of unit quaternions (w, x, y, z), one per row."""
    w, x, y, z = q.T
    return np.column_stack([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


def _q(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    if not v:
        return None
    return {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
            "max": float(np.max(v)), "n": len(v)}


def cmd_report(a):
    out = {}
    for what in ("replay", "hom"):
        path = D / f"{what}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if r.get("status") in ("complete", "ok") and r.get("trace")]
        by = {}
        for r in rows:
            by.setdefault(r["tip"], {}).setdefault((r["hand"], int(r["seed"])), {})[r["cond"]] = r
        for tip, plc in by.items():
            res = {"pairs": {}, "seed_spread": {}, "per_placement": {}}
            for c1, c2 in PAIRS:
                divs = []
                for (hand, seed), cs in sorted(plc.items()):
                    if c1 in cs and c2 in cs:
                        dv = divergence(cs[c1]["trace"], cs[c2]["trace"])
                        if dv:
                            dv.update(hand=hand, seed=seed)
                            divs.append(dv)
                            res["per_placement"].setdefault(f"{hand} s{seed}", {})[f"{c1}-{c2}"] = {
                                k: round(v, 4) for k, v in dv.items() if isinstance(v, float)}
                res["pairs"][f"{c1}-{c2}"] = {k: _q([d[k] for d in divs]) for k in
                                              ("ang_rms_deg", "ang_end_deg", "pos_rms_mm", "F_rms_N", "T_rms_N",
                                               "axis_rms_deg", "axis_end_deg")}
            # seed spread: the same model's rollouts of one hand at two placements (every pair of seeds)
            for cond in ("drake", "v1", "v2"):
                sp = []
                hands = sorted({h for h, _ in plc})
                for hand in hands:
                    tr = [cs[cond]["trace"] for (h, s), cs in sorted(plc.items()) if h == hand and cond in cs]
                    for x, y in combinations(tr, 2):
                        dv = divergence(x, y)
                        if dv:
                            sp.append(dv)
                res["seed_spread"][cond] = {k: _q([d[k] for d in sp]) for k in
                                            ("ang_rms_deg", "ang_end_deg", "pos_rms_mm", "F_rms_N", "T_rms_N")}
            # end-state tallies of the earlier protocol, for the same rollouts
            tal = {}
            for cond in ("v1", "v2", "drake"):
                rs = [cs[cond] for cs in plc.values() if cond in cs]
                held = [r.get("held_end", r.get("held")) for r in rs]
                tal[cond] = {"n": len(rs), "held": int(sum(bool(h) for h in held)),
                             "turn_end_deg": {f"{r['hand']} s{r['seed']}": r.get("turn_end_deg") for r in rs}}
            for c in ("v1", "v2"):
                w = 0
                m_ = 0
                for cs in plc.values():
                    if c in cs and "drake" in cs:
                        m_ += 1
                        a1, a2 = cs[c].get("turn_end_deg"), cs["drake"].get("turn_end_deg")
                        h1, h2 = cs[c].get("held_end", cs[c].get("held")), cs["drake"].get("held_end", cs["drake"].get("held"))
                        if a1 is not None and a2 is not None and abs(a1 - a2) <= (3.0 if what == "replay" else 5.0) and h1 == h2:
                            w += 1
                tal[c]["within_of_drake"] = w
                tal[c]["n_vs_drake"] = m_
            res["tally"] = tal
            # end states that agree in turn (the tilt of the tool's axis) while the axis points elsewhere
            tol = 3.0 if what == "replay" else 5.0
            hid = {}
            for c in ("v1", "v2"):
                lst = []
                for (hand, seed), cs in sorted(plc.items()):
                    k = f"{hand} s{seed}"
                    dv = res["per_placement"].get(k, {}).get(f"{c}-drake")
                    if not dv or c not in cs or "drake" not in cs:
                        continue
                    a1, a2 = cs[c].get("turn_end_deg"), cs["drake"].get("turn_end_deg")
                    if a1 is not None and a2 is not None and abs(a1 - a2) <= tol and dv["axis_end_deg"] > 10.0:
                        lst.append([k, round(dv["axis_end_deg"], 1)])
                hid[c] = lst
            res["turn_agrees_axis_differs"] = hid
            out.setdefault(what, {})[tip] = res
    out.update(script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"), trace_dt_s=TRACE_DT, plant=SERVO, prm=PRM,
               note="within_of_drake: end turn within 3 deg (replay) or 5 deg (HOM) of Drake with the same held outcome")
    D.mkdir(parents=True, exist_ok=True)
    (D / "summary.json").write_text(json.dumps(out, indent=1))
    for what in ("replay", "hom"):
        for tip, res in out.get(what, {}).items():
            print(f"== {what} {tip}")
            for pair, v in res["pairs"].items():
                if v["ang_rms_deg"]:
                    print(f"  {pair:9s} angle RMS {v['ang_rms_deg']['median']:.2f} deg [{v['ang_rms_deg']['q25']:.2f}, "
                          f"{v['ang_rms_deg']['q75']:.2f}], end {v['ang_end_deg']['median']:.2f}; normal {v['F_rms_N']['median']:.3f} N, "
                          f"friction {v['T_rms_N']['median']:.3f} N (n {v['ang_rms_deg']['n']})")
            for cond, v in res["seed_spread"].items():
                if v["ang_rms_deg"]:
                    print(f"  seeds {cond:5s} angle RMS {v['ang_rms_deg']['median']:.2f} deg, normal {v['F_rms_N']['median']:.3f} N, "
                          f"friction {v['T_rms_N']['median']:.3f} N")
            print("  tally", {c: (t["held"], t["n"], t.get("within_of_drake"), t.get("n_vs_drake")) for c, t in res["tally"].items()})
            print("  turn agrees, axis > 10 deg off", res["turn_agrees_axis_differs"])


COL = {"v1": "#e08214", "v2": "#b2182b", "drake": "#2166ac"}
LAB = {"v1": "Pads, builder v1", "v2": "Pads, builder v2", "drake": "Drake hydroelastic"}
TIPS = {"tpu2.7": "Default block (tpu2.7)", "tcd_rs3_re3_p0.2": "As-built tip, printed"}


def _rot_from(tr, t0):
    t, _, q, _, _ = _arr(tr)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    k = int(np.argmin(np.abs(t - t0)))
    return t, np.degrees(2 * np.arccos(np.clip(np.abs(q @ q[k]), 0, 1)))


def cmd_figure(a):
    """media/fig_builder_default.png of the tip study: one replay's traces under the three models, then every
    placement's RMS difference from Drake (builder v1 and v2) against Drake's own spread between placements."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter
    plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans", "axes.edgecolor": "#888888", "axes.grid": True,
                         "grid.color": "#d9d9d9", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.04})
    S = json.loads((D / "summary.json").read_text())
    rows = {}
    for what in ("replay", "hom"):
        for line in open(D / f"{what}.jsonl"):
            r = json.loads(line)
            if r.get("trace"):
                rows[(what, r["tip"], r["cond"], r["hand"], int(r["seed"]))] = r
    pp = S["replay"][a.tip]["per_placement"]
    gain = {k: v["v1-drake"]["ang_rms_deg"] - v["v2-drake"]["ang_rms_deg"] for k, v in pp.items()
            if "v1-drake" in v and "v2-drake" in v}
    pick = a.placement or max(gain, key=gain.get)
    hand, seed = pick.split()[0], int(pick.split()[1][1:])
    fig = plt.figure(figsize=(7.2, 5.9), dpi=150)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.55, wspace=0.45)
    ax = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for cond in ("v1", "v2", "drake"):
        r = rows.get(("replay", a.tip, cond, hand, seed))
        if r is None:
            continue
        t, rot = _rot_from(r["trace"], RB.T_GRIP)
        F = np.asarray(r["trace"]["F"]).sum(1)
        T = np.linalg.norm(np.asarray(r["trace"]["T"]), axis=2).sum(1)
        for x, y in zip(ax, (rot, F, T)):
            x.plot(t, y, color=COL[cond], lw=1.3)
    for x, yl in zip(ax, ("Tool rotation from 0.8 s (°)", "Normal force, 3 fingers (N)", "Friction force, 3 fingers (N)")):
        x.set_xlabel("Time (s)")
        x.set_ylabel(yl)
    ax[1].set_title(f"{TIPS[a.tip]}, plan replay {hand}, seed {seed}", fontsize=9.5)
    groups = [(w, tip) for tip in ("tpu2.7", "tcd_rs3_re3_p0.2") for w in ("replay", "hom") if tip in S.get(w, {})]
    bx = [fig.add_subplot(gs[1, i]) for i in range(3)]
    rng = np.random.default_rng(0)
    for x, key, yl in zip(bx, ("ang_rms_deg", "F_rms_N", "T_rms_N"),
                          ("Tool orientation, RMS (°)", "Normal force, RMS (N)", "Friction force, RMS (N)")):
        for gi, (w, tip) in enumerate(groups):
            res = S[w][tip]
            for ci, cond in enumerate(("v1", "v2")):
                v = [d[f"{cond}-drake"][key] for d in res["per_placement"].values() if f"{cond}-drake" in d]
                xc = gi + (ci - 0.5) * 0.36
                x.scatter(xc + rng.uniform(-0.08, 0.08, len(v)), v, s=7, color=COL[cond], alpha=0.75, lw=0)
                if v:
                    x.plot([xc - 0.13, xc + 0.13], [np.median(v)] * 2, color="#111111", lw=1.6)
            sp = res["seed_spread"]["drake"][key]
            if sp:
                x.plot([gi - 0.42, gi + 0.42], [sp["median"]] * 2, color=COL["drake"], lw=1.2, ls="--")
        x.set_yscale("log")
        x.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        x.set_xticks(range(len(groups)))
        x.set_xticklabels([f"{'Block' if tip == 'tpu2.7' else 'Printed'} {'replay' if w == 'replay' else 'HOM'}"
                           for w, tip in groups], fontsize=8.5, rotation=35, ha="right", rotation_mode="anchor")
        x.set_xlim(-0.6, len(groups) - 0.4)
        x.set_ylabel(yl)
    hdl = [Line2D([], [], color=COL["v1"], lw=1.6, marker="o", ms=4, label=LAB["v1"]),
           Line2D([], [], color=COL["v2"], lw=1.6, marker="o", ms=4, label=LAB["v2"]),
           Line2D([], [], color=COL["drake"], lw=1.6, label=LAB["drake"]),
           Line2D([], [], color=COL["drake"], lw=1.2, ls="--", label="Drake between placements"),
           Line2D([], [], color="#111111", lw=1.6, label="Median")]
    fig.legend(handles=hdl, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.03), fontsize=8.5, handlelength=1.6,
               columnspacing=1.0)
    out = ROOT / "docs/experiments/20261014-tip_codesign/media/fig_builder_default.png"
    fig.savefig(out)
    print(f"wrote {out} (example {pick})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--what", choices=["replay", "hom"], required=True)
    r.add_argument("--tips", nargs="+", default=["tpu2.7", "tcd_rs3_re3_p0.2"])
    r.add_argument("--conds", nargs="+", default=["v1", "v2", "drake"], choices=["v1", "v2", "drake"])
    r.add_argument("--hands", nargs="+", default=list(RB.HANDS))
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    r.add_argument("--worker", type=int, default=0, help="scene folder of this process")
    r.add_argument("--out", type=Path, default=None)
    sub.add_parser("report")
    fg = sub.add_parser("figure")
    fg.add_argument("--tip", default="tpu2.7")
    fg.add_argument("--placement", default=None, help="'D3 s2'; default the replay where v2 gains most on v1")
    a = ap.parse_args()
    {"run": cmd_run, "report": cmd_report, "figure": cmd_figure}[a.cmd](a)


if __name__ == "__main__":
    main()
