#!/usr/bin/env python3
"""Contact-model comparison bed, tasks 8 and 9 (2026-10-07): presliding compliance and load spread, against the elastic
references of scripts/contact_reference_laws.py (Mindlin, Hertz) and the pressure law.

  T8 cycle   gravity off; settle 0.4 s at N; an axial force on the tool cycled as a triangle 0 -> +F* (0.5 s) -> -F*
             (1 s) -> +F* (1 s) -> 0 (0.5 s), F* = mu N, half the slip force 2 mu N. Metrics: presliding displacement
             u(+F*) at the first peak (t 0.5 s); recovered displacement u(+F*) - u(F = 0) on the first unloading (t 1.0 s); loop area, the work
             of the force around the full cycle +F* -> -F* -> +F* (uJ); drift per cycle u(end of cycle) - u(start).
             Mindlin gives a closed loop (drift 0) with u, recovery and area from contact_reference_laws.mindlin_cycle;
             rigid Coulomb friction gives zeros; a model whose tool creeps at v = c F gives u = c * integral of F dt, no
             recovery while F > 0, area c * integral of F^2 dt and zero drift for a symmetric cycle.
  T9 sweep   gravity off; settle 1 s at N = 0.25, 0.5, 1, 2, 4 N per pad. Metrics: approach (mm); force-weighted RMS
             radius of the -x pad's contacts about their centroid in the plane normal to the pinch axis (mm); fraction
             of the normal load at contacts outside the overlap of the undeformed pad sphere and tool; log-log slopes of
             approach and RMS radius against N (pressure law 1/2 and 1/4, Hertz 2/3 and 1/3).

Models (contact_bed_common.MODELS): mj_point3, mj_pads1, drake_hydro by default; candidates register their own specs.
Rows: docs/experiments/20261007-native_compliance/{t8_cycle,t9_sweep}.jsonl, one fsynced line per case.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/contact_bed_compliance.py t8 --models mj_pads1 --N 1
    $PY scripts/contact_bed_compliance.py t9
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
import contact_reference_laws as L  # noqa: E402

H = B.H
OUT = B.ROOT / "docs/experiments/20261007-native_compliance"
SCRIPT = "scripts/contact_bed_compliance.py"
MODELS = ["mj_point3", "mj_pads1", "drake_hydro"]
T_SETTLE = 0.4
EXTRA = {}            # candidate model name -> rig factory (spec, dt) for models outside hom_contact_rig


def new_rig(model, dt):
    if model in EXTRA:
        return EXTRA[model](dt)
    return B.new_rig(B.MODELS[model][0], dt)


def spec_of(model):
    return getattr(EXTRA.get(model), "spec", None) or B.MODELS.get(model, (model,))[0]


def engine(rig):
    """MuJoCo version and integrator of a MuJoCo rig (the flex candidate needs 3.14's discrete integrator)."""
    if getattr(rig, "sim", None) != "mujoco":
        return {}
    return dict(mujoco=rig.mj.__version__, integrator=rig.mj.mjtIntegrator(int(rig.m.opt.integrator)).name)


# ------------------------------------------------------------------------------------------ T8

def f_cycle(t, Fs):
    """Triangle: 0 -> Fs over 0.5 s, -> -Fs over 1 s, -> Fs over 1 s, -> 0 over 0.5 s."""
    if t < 0.5:
        return Fs * t / 0.5
    if t < 1.5:
        return Fs - 2 * Fs * (t - 0.5)
    if t < 2.5:
        return -Fs + 2 * Fs * (t - 1.5)
    if t < 3.0:
        return Fs - Fs * (t - 2.5) / 0.5
    return 0.0


def run_t8(model, N, dt_ms):
    dt = dt_ms * 1e-3
    t0w = time.time()
    rig = new_rig(model, dt)
    rig.set_pad_force(N)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    rig.step(T_SETTLE)
    st0 = rig.tool_state()
    p0, a0 = st0["pos"].copy(), st0["axis"].copy()
    Fs = H.MU * N
    T, F, U, W = [], [], [], []
    for k in range(int(round(3.2 / dt))):
        t = (k + 1) * dt
        f = f_cycle(t - dt / 2, Fs)
        rig.set_tool_wrench(f * a0, np.zeros(3))
        w0 = time.perf_counter()
        rig.step(dt)
        W.append(time.perf_counter() - w0)
        u = float((rig.tool_state()["pos"] - p0) @ a0)
        T.append(t), F.append(f), U.append(u)
        if not math.isfinite(u) or abs(u) > 0.01:
            break
    T, F, U = np.array(T), np.array(F), np.array(U)
    status = "complete" if len(T) >= int(round(3.2 / dt)) - 1 else "ejected"
    row = dict(task="t8_cycle", model=model, rig_spec=spec_of(model), N=N, dt_ms=dt_ms, F_star=Fs, mu=H.MU, gravity=False)
    row.update(engine(rig))
    if status == "complete":
        at = lambda t: float(np.interp(t, T, U))  # noqa: E731
        m = (T >= 0.5) & (T <= 2.5)
        loop = float(np.sum(0.5 * (F[m][1:] + F[m][:-1]) * np.diff(U[m])))
        ref = L.mindlin_cycle(N)
        row.update(u_presliding_um=at(0.5) * 1e6, u_recovered_um=(at(0.5) - at(1.0)) * 1e6, loop_area_uJ=loop * 1e6,
                   drift_per_cycle_um=(at(2.5) - at(0.5)) * 1e6, u_end_um=at(3.2) * 1e6,
                   mindlin_u_presliding_um=ref["u_presliding"] * 1e6, mindlin_u_recovered_um=ref["u_recovered"] * 1e6,
                   mindlin_loop_area_uJ=ref["loop_area"] * 1e6)
    stride = max(1, int(round(0.005 / dt)))
    row.update(trace_cols=["t", "F_N", "u_um"], trace=[[round(float(a), 4), round(float(b), 5), round(float(c) * 1e6, 4)]
                                                     for a, b, c in zip(T[::stride], F[::stride], U[::stride])])
    row.update(status=status, us_per_step_median=float(np.median(W) * 1e6), wall_s=time.time() - t0w, script=SCRIPT,
               git_rev=B.git_rev(), when=time.strftime("%Y-%m-%d %H:%M"))
    return row


# ------------------------------------------------------------------------------------------ T9

def contacts_L(rig):
    """(points, force magnitudes) of the -x pad: MuJoCo contacts or Drake contact-surface faces (pressure x area)."""
    import contact_bed_edge as CE
    if rig.sim == "drake":
        f = CE.drake_faces(rig).get("L")
        return (f["C"], f["P"] * f["A"]) if f is not None else (np.zeros((0, 3)), np.zeros(0))
    cs = CE.mj_contacts(rig) if not hasattr(rig, "wd") else CE.mjw_contacts(rig)
    sel = [c for c in cs if c[0] == "L"]
    return np.array([c[1] for c in sel]).reshape(-1, 3), np.array([c[3] for c in sel])


def run_t9(model, N, dt_ms, T=1.0):
    dt = dt_ms * 1e-3
    t0w = time.time()
    rig = new_rig(model, dt)
    rig.set_pad_force(N)
    rig.set_tool_wrench(np.zeros(3), np.zeros(3))
    w0 = time.perf_counter()
    rig.step(T)
    wall = time.perf_counter() - w0
    x0 = getattr(rig, "x0", H.X0)
    px = rig.pad_x()
    delta = 0.5 * ((x0 + px["L"]) + (x0 - px["R"]))
    pts, f = contacts_L(rig)
    row = dict(task="t9_sweep", model=model, rig_spec=spec_of(model), N=N, dt_ms=dt_ms, settle_s=T, delta_mm=delta * 1e3)
    row.update(engine(rig))
    keep = f > 1e-9
    pts, f = pts[keep], f[keep]
    if len(f):
        w = f / f.sum()
        c = (w[:, None] * pts).sum(0)
        r2 = (w * ((pts[:, 1] - c[1]) ** 2 + (pts[:, 2] - c[2]) ** 2)).sum()
        cpad = np.array([px["L"], 0.0, 0.0])
        in_sphere = np.linalg.norm(pts - cpad, axis=1) <= H.R_PAD + 1e-6
        in_tool = np.hypot(pts[:, 0], pts[:, 2]) <= H.R_TOOL + 1e-6
        row.update(n=int(len(f)), r_rms_mm=float(math.sqrt(r2)) * 1e3, centroid_mm=[float(v) * 1e3 for v in c],
                   load_outside_overlap=float(f[~(in_sphere & in_tool)].sum() / f.sum()),
                   law_r_rms_at_delta_mm=L.law_patch(max(delta, 1e-7))["r_rms"] * 1e3 if delta > 0 else None)
        if row.get("law_r_rms_at_delta_mm"):
            row["r_rms_over_law_at_delta"] = row["r_rms_mm"] / row["law_r_rms_at_delta_mm"]
    hz = L.hertz(N)
    d_law, _, _ = H.winkler_law(N, 1e7)
    row.update(law_delta_mm=d_law * 1e3, law_r_rms_mm=L.law_patch(d_law)["r_rms"] * 1e3, hertz_delta_mm=hz["delta"] * 1e3,
               hertz_r_rms_mm=hz["r_rms"] * 1e3, status="complete" if len(f) else "no_contact",
               us_per_step_median=wall / max(1, int(round(T / dt))) * 1e6, wall_s=time.time() - t0w, script=SCRIPT,
               git_rev=B.git_rev(), when=time.strftime("%Y-%m-%d %H:%M"))
    return row


def add_slopes(path, model, dt_ms):
    """Log-log slopes of approach and RMS radius against N over a model's T9 rows (returned, not written)."""
    rows = [r for r in B.read_rows(path) if r.get("model") == model and r.get("dt_ms") == dt_ms and r.get("status") == "complete"]
    if len(rows) < 3:
        return {}
    lN = np.log([r["N"] for r in rows])
    out = {"delta_exp": float(np.polyfit(lN, np.log([r["delta_mm"] for r in rows]), 1)[0])}
    if all(r.get("r_rms_mm") for r in rows):
        out["r_rms_exp"] = float(np.polyfit(lN, np.log([r["r_rms_mm"] for r in rows]), 1)[0])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["t8", "t9"])
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--N", nargs="+", type=float)
    ap.add_argument("--dt", type=float, default=1.0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / ("t8_cycle.jsonl" if a.task == "t8" else "t9_sweep.jsonl")
    Ns = a.N or ([0.5, 1.0, 3.0] if a.task == "t8" else [0.25, 0.5, 1.0, 2.0, 4.0])
    have = B.done(out)
    for model in a.models:
        for N in Ns:
            if (model, N, a.dt) in have:
                continue
            try:
                r = run_t8(model, N, a.dt) if a.task == "t8" else run_t9(model, N, a.dt)
            except Exception as e:
                r = {"task": a.task, "model": model, "N": N, "dt_ms": a.dt, "status": "failed", "error": repr(e)[:500],
                     "traceback": traceback.format_exc()[-2000:], "script": SCRIPT}
            H.append_row(out, r)
            keys = ("u_presliding_um", "mindlin_u_presliding_um", "u_recovered_um", "mindlin_u_recovered_um", "loop_area_uJ",
                    "mindlin_loop_area_uJ", "drift_per_cycle_um", "delta_mm", "law_delta_mm", "hertz_delta_mm", "r_rms_mm",
                    "law_r_rms_mm", "hertz_r_rms_mm", "r_rms_over_law_at_delta", "load_outside_overlap", "us_per_step_median")
            print(model, N, r.get("status"), {k: round(r[k], 4) for k in keys if isinstance(r.get(k), float)}, r.get("error", ""),
                  flush=True)
        if a.task == "t9":
            print(model, "slopes", add_slopes(out, model, a.dt), flush=True)


if __name__ == "__main__":
    main()
