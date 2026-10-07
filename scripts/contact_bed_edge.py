#!/usr/bin/env python3
"""Contact-model comparison bed, edge test (2026-10-07): the pinch on a sharp feature.

The bed rig of docs/experiments/20261005-contact_bed/PROTOCOL.md with the screwdriver cylinder replaced by a square bar
(side 20 mm, length 100 mm, 24.5 g, axis along world y) turned 45 deg about its axis, so that each pad presses on an
edge whose two faces are inclined 45 deg to the pinch axis (`hom_contact_rig` spec field `bar20`; the pads start at the
first touch, R_PAD + 20 mm / sqrt 2 from the axis). Reference: the hydroelastic pressure law p = E (1 - rho / R)
integrated over the two bar faces inside the fingertip sphere (scripts/hydroelastic_arm_integral.py bar_law), which
gives the pad approach, the pressure-weighted distance of the patch from the pinch axis, the patch half-length along
the edge and the sliding friction arm about the pinch axis.

  static   gravity off, pinch N per pad held 1 s: pad approach (mm), contact count, force-weighted distance from the
           pinch axis, patch half-length along the edge, and the -x pad's contacts (position, normal, force) for the
           page's figure.
  twist    task 2 of the bed (contact_bed_twist.run_case: ramp to spin onset, hold at half onset, driven 1 rad/s spin)
           with the bar's transverse inertia.

Models: CPU (rig venv, MuJoCo 3.6 + Drake 1.57) mj_point3_bar, mj_point4s_bar (condim 4 with the cylinder's fitted
torsion schedule), mj_pads1_bar, drake_hydro_bar; GPU (Newton venv) mjw_pads1_bar (MuJoCo-Warp, one world) and
newton_hydro_mc_bar (Newton hydroelastic, kh / m_eff, 0.5 mm voxels), both from scripts/contact_bed_newton.py.

    PY=logs/20261001-hom_contact/venv/bin/python                     # CPU models
    $PY scripts/contact_bed_edge.py static --models mj_pads1_bar drake_hydro_bar
    PYG=logs/20261004-contact-transfer/venv/bin/python               # GPU models
    WARP_CACHE_PATH=$(mktemp -d) MUJOCO_GL=egl $PYG scripts/contact_bed_edge.py twist --models newton_hydro_mc_bar
One fsynced JSON line per case in docs/experiments/20261007-newton_hydro_tests/edge_{static,twist}.jsonl; cases already
written are skipped.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contact_bed_common as B  # noqa: E402
import hydroelastic_arm_integral as A  # noqa: E402

H = B.H
OUT = B.ROOT / "docs/experiments/20261007-newton_hydro_tests"
BAR = 0.020
SCRIPT = "scripts/contact_bed_edge.py"
CPU = {"mj_point3_bar": "mj:point3:ir100:bar20",
       "mj_point4s_bar": "mj:point4s:fit0.000996044x0.2498:ir100:bar20",
       "mj_pads1_bar": "mj:spheres:s1:rs0.75:ir100:tr0.02:bar20",
       "drake_hydro_bar": "drake:hydro:E1e7:r1:rt0.01:bar20"}
GPU = ("mjw_pads1_bar", "newton_hydro_mc_bar", "newton_hydro_unreduced_mc_bar", "newton_hydro_unreduced_mc_bar_vox025",
       "newton_hydro_unreduced_mc_bar_vox1")
LAW = {}


def law(N):
    if N not in LAW:
        LAW[N] = A.bar_law(N, BAR)
    return LAW[N]


def ref_fields(N):
    b = law(N)
    return {"law_delta_mm": b["delta"] * 1e3, "law_arm_mm": b["arm"] * 1e3, "law_rbar_mm": b["rbar"] * 1e3,
            "law_half_y_mm": b["half_y"] * 1e3, "law_area_mm2": b["area"] * 1e6, "law_p_max_MPa": b["p_max"] / 1e6}


def new_rig(model, dt):
    if model in CPU:
        return B.new_rig(CPU[model], dt)
    import contact_bed_newton as CBN
    return CBN.bed_rig(model, dt)


def spec_of(model):
    if model in CPU:
        return CPU[model]
    import contact_bed_newton as CBN
    return CBN.rig_spec(model)


# ------------------------------------------------------------------------------------------ contacts

def mj_contacts(rig):
    """CPU MuJoCo: position, normal (pad toward tool), normal force and side of every pad contact."""
    out = []
    f6 = np.zeros(6)
    for i in range(rig.d.ncon):
        c = rig.d.contact[i]
        g1, g2 = int(c.geom[0]), int(c.geom[1])
        s = rig.geom_side.get(g1) or rig.geom_side.get(g2)
        if s is None:
            continue
        rig.mj.mj_contactForce(rig.m, rig.d, i, f6)
        n = np.array(c.frame[:3])
        if g2 in rig.geom_side:             # MuJoCo's normal points from geom 1 to geom 2
            n = -n
        out.append((s, np.array(c.pos), n, float(f6[0])))
    return out


def mjw_contacts(rig):
    """MuJoCo-Warp (one world): the same from the device arrays."""
    out = []
    n = int(rig.wd.nacon.numpy()[0])
    if n == 0:
        return out
    geom = rig.wd.contact.geom.numpy()[:n]
    adr = rig.wd.contact.efc_address.numpy()[:n, 0]
    pos = rig.wd.contact.pos.numpy()[:n].astype(float)
    frame = rig.wd.contact.frame.numpy()[:n].astype(float)
    force = rig.wd.efc.force.numpy()[0]
    for (g0, g1), a, p, fr in zip(geom, adr, pos, frame):
        s = rig.geom_side.get(int(g0)) or rig.geom_side.get(int(g1))
        if s is None or a < 0:
            continue
        nrm = fr.reshape(3, 3)[0]
        if int(g1) in rig.geom_side:
            nrm = -nrm
        out.append((s, p, nrm, float(force[a])))
    return out


def newton_contacts(rig):
    pos, nrm, fn, side = rig.mj_contacts()
    out = []
    mjm = rig.solver.mj_model
    geom = rig.solver.mjw_data.contact.geom.numpy()[:len(fn)]
    for p, n, f, s, (g0, g1) in zip(pos, nrm, fn, side, geom):
        if s is None:
            continue
        n = np.array(n, float)
        if int(g1) in rig.geom_side:
            n = -n
        out.append((s, np.array(p, float), n, float(f)))
    return out


def drake_faces(rig):
    """Drake hydroelastic: per pad the contact-surface faces (centroid, unit normal, area, pressure) and the solver's
    resultant force on the pad."""
    cr = rig.plant.get_contact_results_output_port().Eval(rig.pc)
    out = {}
    for i in range(cr.num_hydroelastic_contacts()):
        info = cr.hydroelastic_contact_info(i)
        srf = info.contact_surface()
        s = rig.geom_side.get(srf.id_M()) or rig.geom_side.get(srf.id_N())
        if s is None:
            continue
        tri = srf.is_triangle()
        mesh = srf.tri_mesh_W() if tri else srf.poly_mesh_W()
        field = srf.tri_e_MN() if tri else srf.poly_e_MN()
        nf = mesh.num_elements()
        A_ = np.array([mesh.area(f) for f in range(nf)])
        C = np.array([mesh.element_centroid(f) for f in range(nf)])
        P = np.array([field.EvaluateCartesian(f, C[f]) for f in range(nf)])
        Nn = np.array([mesh.face_normal(f) for f in range(nf)])
        F = np.array(info.F_Ac_W().translational())
        out[s] = dict(A=A_, C=C, P=P, n=Nn, F=F)
    return out


def patch_stats(pts, fn):
    """Force-weighted distance from the pinch axis, patch half-length along the edge (y) over contacts carrying more
    than 1 % of the largest, count."""
    if len(fn) == 0 or fn.sum() <= 0:
        return dict(n=0, rbar_mm=None, half_y_mm=None)
    rho = np.hypot(pts[:, 1], pts[:, 2])
    keep = fn > 0.01 * fn.max()
    return dict(n=int((fn > 1e-7).sum()), rbar_mm=float((fn * rho).sum() / fn.sum()) * 1e3,
                half_y_mm=float(np.abs(pts[keep, 1]).max()) * 1e3)


# ------------------------------------------------------------------------------------------ static

def static_case(model, N, dt_ms, T=1.0):
    dt = dt_ms * 1e-3
    t0w = time.time()
    rig = new_rig(model, dt)
    x0 = getattr(rig, "x0", H.tool_touch({"bar": BAR}))
    rig.set_pad_force(N)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    trace, W = [], []
    status = "complete"
    for _ in range(int(round(T / 0.1))):
        w0 = time.perf_counter()
        rig.step(0.1)
        W.append((time.perf_counter() - w0) / max(1, int(round(0.1 / dt))))
        px = rig.pad_x()
        st = rig.tool_state()
        trace.append([round(rig.t, 4), (x0 + px["L"]) * 1e3, (x0 - px["R"]) * 1e3, float(np.linalg.norm(st["pos"])) * 1e3])
        if not np.isfinite(trace[-1][1]) or abs(trace[-1][1]) > 3 or trace[-1][3] > 5:
            status = "ejected"
            break
    row = dict(task="edge_static", model=model, rig_spec=spec_of(model), N=N, dt_ms=dt_ms, bar_mm=BAR * 1e3,
               settle_s=T, trace_cols=["t", "pen_L_mm", "pen_R_mm", "tool_disp_mm"], trace=trace)
    row.update(ref_fields(N))
    if status == "complete":
        row["pen_mm"] = 0.5 * (trace[-1][1] + trace[-1][2])
        row["pen_change_last_0p1s_mm"] = trace[-1][1] - trace[-2][1] if len(trace) > 1 else None
        if rig.sim == "drake":
            faces = drake_faces(rig)
            for s in "LR":
                f = faces.get(s)
                if f is None:
                    row[f"n_{s}"] = 0
                    continue
                w = f["P"] * f["A"]
                st_ = patch_stats(f["C"], w)
                row.update({f"N_{s}": float(abs(f["F"][0])), f"N_pressure_x_{s}": float((w * np.abs(f["n"][:, 0])).sum()),
                            f"n_{s}": int(len(w)), f"rbar_{s}_mm": st_["rbar_mm"], f"half_y_{s}_mm": st_["half_y_mm"],
                            f"area_{s}_mm2": float(f["A"].sum()) * 1e6, f"p_max_{s}_MPa": float(f["P"].max()) / 1e6})
            fL = faces.get("L")
            if fL is not None:
                row["contacts_L"] = {"kind": "faces", "cols": ["x_mm", "y_mm", "z_mm", "nx", "ny", "nz", "area_mm2", "p_MPa"],
                                     "rows": [[round(c * 1e3, 4) for c in C] + [round(v, 4) for v in n] +
                                              [round(a * 1e6, 5), round(p / 1e6, 5)]
                                              for C, n, a, p in zip(fL["C"], fL["n"], fL["A"], fL["P"])]}
        else:
            cs = (mj_contacts(rig) if rig.sim == "mujoco" and not hasattr(rig, "wd") else
                  mjw_contacts(rig) if hasattr(rig, "wd") else newton_contacts(rig))
            for s in "LR":
                sel = [c for c in cs if c[0] == s]
                pts = np.array([c[1] for c in sel]).reshape(-1, 3)
                nrm = np.array([c[2] for c in sel]).reshape(-1, 3)
                fn = np.array([c[3] for c in sel])
                st_ = patch_stats(pts, fn)
                row.update({f"N_{s}": float((fn[:, None] * nrm).sum(0)[0]) * (1 if s == "L" else -1) if len(fn) else 0.0,
                            f"N_normal_sum_{s}": float(fn.sum()), f"n_{s}": st_["n"], f"rbar_{s}_mm": st_["rbar_mm"],
                            f"half_y_{s}_mm": st_["half_y_mm"]})
                if s == "L":
                    row["contacts_L"] = {"kind": "points", "cols": ["x_mm", "y_mm", "z_mm", "nx", "ny", "nz", "f_N"],
                                         "rows": [[round(v * 1e3, 4) for v in p] + [round(v, 4) for v in n] + [round(f, 6)]
                                                  for p, n, f in zip(pts, nrm, fn) if f > 1e-9]}
            if getattr(rig, "sim", "") == "newton":
                nc = rig.newton_contacts()
                if nc.get("n"):
                    mL = np.array([x == "L" for x in nc["side"]], bool)
                    pen = np.clip(-nc["d"][mL], 0, None)
                    fl = nc["k"][mL] * pen
                    st_ = patch_stats(nc["p"][mL], fl)
                    row.update(n_newton_L=int(mL.sum()), rbar_newton_L_mm=st_["rbar_mm"], half_y_newton_L_mm=st_["half_y_mm"],
                               area_eq_newton_L_mm2=float(nc["k"][mL].sum() / rig.kh_eff) * 1e6)
                    row["newton_contacts_L"] = {"cols": ["x_mm", "y_mm", "z_mm", "k_N_m", "pen_um"],
                                                "rows": [[round(v * 1e3, 4) for v in p] + [round(float(k), 3), round(float(d) * 1e6, 3)]
                                                         for p, k, d in zip(nc["p"][mL], nc["k"][mL], pen)]}
        for k in ("rbar", "half_y"):
            vals = [row.get(f"{k}_{s}_mm") for s in "LR"]
            row[f"{k}_mm"] = float(np.mean(vals)) if all(v is not None for v in vals) else None
        row["n_mean"] = 0.5 * (row.get("n_L", 0) + row.get("n_R", 0))
    row["status"] = status
    row["us_per_step_median"] = float(np.median(W) * 1e6) if W else None
    row["wall_s"] = time.time() - t0w
    return finish(row)


def finish(row):
    import contact_bed_newton as CBN
    last = getattr(CBN.NewtonRig, "last", None)
    if row["model"] in GPU and last is not None and getattr(last, "model_name", None) == row["model"]:
        row.setdefault("kh_scale", last.kh_scale)
        row.setdefault("inv_weight0", last.inv_w)
        if getattr(last, "pad_d0", None):
            row.setdefault("pad_d0", last.pad_d0)
    row.update(script=SCRIPT, git_rev=B.git_rev(), when=time.strftime("%Y-%m-%d %H:%M"))
    return row


# ------------------------------------------------------------------------------------------ kinematic

def kinematic_case(model, N):
    """Force-approach of the -x pad at the law's approach for pinch N, with no solver dynamics and no friction: the pad
    placed at approach delta on its rail, the +x pad 20 mm clear, and the force each model's contact law gives there.
    Pads: K_s x sphere depth along each contact normal (the spring every sphere contact settles to); Drake: its contact
    surface's pressure x area; Newton: its own contact set, k_i x overlap_i / kh_scale (the force SolverMuJoCo realises)."""
    t0w = time.time()
    b = law(N)
    delta = b["delta"]
    row = dict(task="edge_kinematic", model=model, rig_spec=spec_of(model), N=N, dt_ms=None, bar_mm=BAR * 1e3,
               delta_mm=delta * 1e3, **ref_fields(N))
    if model in CPU and CPU[model].startswith("mj:"):
        rig = B.new_rig(CPU[model], 1e-3)
        m, d = rig.m, rig.d
        d.qpos[m.joint("railL").qposadr[0]] = delta
        d.qpos[m.joint("railR").qposadr[0]] = 0.02
        rig.mj.mj_forward(m, d)
        K = rig.info.get("K_sphere")
        pts, nrm, f = [], [], []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom[0]), int(c.geom[1])
            if rig.geom_side.get(g1, rig.geom_side.get(g2)) != "L" or c.dist >= 0:
                continue
            n = np.array(c.frame[:3]) * (-1 if g2 in rig.geom_side else 1)
            pts.append(np.array(c.pos)), nrm.append(n), f.append((K or 0.0) * -float(c.dist))
        pts, nrm, f = np.array(pts).reshape(-1, 3), np.array(nrm).reshape(-1, 3), np.array(f)
        row.update(K_sphere=K, depth_max_mm=float(max([-x for x in [d.contact[i].dist for i in range(d.ncon)]] or [0])) * 1e3)
    elif model in CPU:
        from pydrake.all import HydroelasticContactRepresentation
        rig = H.DrakeRig(H.parse_spec(CPU[model]), 0.0, False)
        rig.joints["L"].set_translation(rig.pc, delta)
        rig.joints["R"].set_translation(rig.pc, -0.02)
        q = rig.sg.get_query_output_port().Eval(rig.sgc)
        pts, nrm, f = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        for srf in q.ComputeContactSurfaces(HydroelasticContactRepresentation.kPolygon):
            if (rig.geom_side.get(srf.id_M()) or rig.geom_side.get(srf.id_N())) != "L":
                continue
            mesh, field = srf.poly_mesh_W(), srf.poly_e_MN()
            nf = mesh.num_elements()
            A_ = np.array([mesh.area(i) for i in range(nf)])
            C = np.array([mesh.element_centroid(i) for i in range(nf)])
            P = np.array([field.EvaluateCartesian(i, C[i]) for i in range(nf)])
            nn = np.array([mesh.face_normal(i) for i in range(nf)])
            nn = nn * np.sign(nn[:, :1] + 1e-12)          # toward the tool (+x) for the -x pad
            pts, nrm, f = C, nn, P * A_
            row.update(area_mm2=float(A_.sum()) * 1e6, p_max_MPa=float(P.max()) / 1e6, faces=int(nf))
    else:
        import contact_bed_newton as CBN
        rig = CBN.NewtonRig(model, 1e-3)
        jl = [x.split("/")[-1] for x in rig.model.joint_label]
        qs = rig.model.joint_q_start.numpy()
        q = rig.s0.joint_q.numpy()
        q[qs[jl.index("railL")]] = delta
        q[qs[jl.index("railR")]] = 0.02
        rig.s0.joint_q.assign(q)
        rig.newton.eval_fk(rig.model, rig.s0.joint_q, rig.s0.joint_qd, rig.s0)
        rig.pipe.collide(rig.s0, rig.cc)
        nc = rig.newton_contacts()
        n = nc.get("n", 0)
        pts, nrm, f = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
        if n:
            nr = rig.cc.rigid_contact_normal.numpy()[:n].astype(float)
            s0 = rig.cc.rigid_contact_shape0.numpy()[:n]
            sgn = np.array([1.0 if int(a) in rig.shape_side else -1.0 for a in s0])
            mL = np.array([x == "L" for x in nc["side"]], bool)
            pen = np.clip(-nc["d"], 0, None)
            pts, nrm, f = nc["p"][mL], (nr * sgn[:, None])[mL], (nc["k"] * pen / rig.kh_scale)[mL]
            row.update(area_eq_mm2=float(nc["k"][mL].sum() / rig.kh_eff) * 1e6, overlap_max_mm=float(pen[mL].max()) * 1e3)
        row.update(kh_scale=rig.kh_scale)
    st_ = patch_stats(pts, f)
    row.update(F_x=float((f[:, None] * nrm).sum(0)[0]) if len(f) else 0.0, F_normal_sum=float(f.sum()), n=st_["n"],
               rbar_mm=st_["rbar_mm"], half_y_mm=st_["half_y_mm"], status="complete", wall_s=time.time() - t0w)
    row["F_x_over_law"] = row["F_x"] / N
    row["contacts_L"] = {"cols": ["x_mm", "y_mm", "z_mm", "nx", "ny", "nz", "f_N"],
                         "rows": [[round(v * 1e3, 4) for v in p_] + [round(v, 4) for v in n_] + [round(float(x), 7)]
                                  for p_, n_, x in zip(pts, nrm, f) if x > 1e-9]}
    return finish(row)


# ------------------------------------------------------------------------------------------ twist

def twist_case(model, N, dt_ms):
    import contact_bed_twist as TW
    I_t, _ = H.bar_inertia(BAR)
    keep = H.I_TOOL_T
    H.I_TOOL_T = I_t                     # the twist logic's frictionless spin-up and sliding-torque correction
    try:
        if model in CPU:
            B.MODELS[model] = (CPU[model], CPU[model])
            row = TW.run_case(model, N, dt_ms, film=False)
        else:
            import contact_bed_newton as CBN
            CBN.OUTDIR = OUT
            row = CBN.run_twist(model, N, dt_ms)
    finally:
        H.I_TOOL_T = keep
    row["task"] = "edge_twist"
    row["bar_mm"] = BAR * 1e3
    row["I_tool_kgm2"] = I_t
    row.update(ref_fields(N))
    for k in ("rbar_onset_mm", "rbar_slide_mm", "rbar_kin_mm"):
        if isinstance(row.get(k), float):
            row[k.replace("_mm", "_over_law")] = row[k] / row["law_arm_mm"]
    row["script"] = SCRIPT
    return finish(row)


# ------------------------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["static", "twist", "kinematic", "law"])
    ap.add_argument("--models", nargs="+", default=list(CPU))
    ap.add_argument("--N", nargs="+", type=float, default=[0.5, 1.0, 3.0])
    ap.add_argument("--dt", type=float, default=1.0)
    a = ap.parse_args()
    if a.task == "law":
        for N in a.N:
            print(N, {k: round(v, 4) for k, v in ref_fields(N).items()})
        return
    if any(m in GPU for m in a.models):
        import warp as wp
        wp.init()
    out = OUT / f"edge_{a.task}.jsonl"
    have = B.done(out, key=("model", "N"))
    for model in a.models:
        for N in a.N:
            if (model, N) in have:
                continue
            try:
                r = (static_case(model, N, a.dt) if a.task == "static" else kinematic_case(model, N) if a.task == "kinematic"
                     else twist_case(model, N, a.dt))
            except Exception as e:                      # a failed case is a row, not a lost run
                r = finish({"task": f"edge_{a.task}", "model": model, "N": N, "dt_ms": a.dt, "status": "failed",
                            "error": repr(e)[:500], "traceback": traceback.format_exc()[-2000:]})
            if a.task == "twist":
                import contact_bed_twist as TW
                TW.add_ratio(r, out)
                if r.get("N") == 3.0:
                    r["law_ratio_3_05"] = law(3.0)["arm"] / law(0.5)["arm"]
            H.append_row(out, r)
            keys = ("pen_mm", "law_delta_mm", "F_x", "n_mean", "n", "rbar_mm", "law_rbar_mm", "half_y_mm", "law_half_y_mm",
                    "rbar_onset_mm", "rbar_slide_mm", "rbar_kin_mm", "law_arm_mm", "rbar_ratio_3_05_onset", "creep_deg_s",
                    "us_per_step_median", "wall_s")
            print(model, N, r.get("status"), {k: round(r[k], 4) for k in keys if isinstance(r.get(k), float)},
                  r.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
