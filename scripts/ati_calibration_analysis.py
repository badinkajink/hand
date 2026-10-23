#!/usr/bin/env python3
"""The ATI Mini45's calibration from a session of scripts/ati_calibration_session.py (or its --sim): each rest's mean
gage values, gravity in the ATI's frame from the robot's joints, the drift from the reference station's rests, the mass
on minus the mass off at each station, and the 6 x 6 matrix C that turns gage values into the wrench at the ATI's
origin (N, N m, the ATI's frame of drawing 9230-05-1094 as placed by the stack: scripts/ati_stack_mount.py).

The reference wrench of a loaded rest at station s on eye e: the mass m (weighed with its holder) under local gravity
g_A, along the line from the eye: w = [m g_A, p_l x m g_A], p_l a point of the line's centre (line_point). The line's
stopper knot sits behind the eye's wall (1.6 mm); the line runs straight from the knot while it clears the hole's outer
edge, and past atan((R - d/2) / 1.6) from the eye's axis it bends over that edge and hangs d/2 off it (R the hole's
radius, the plan's 0.6 mm unless the session's --eye-hole-mm or --rho; d the line's diameter, the session's --line-mm
or --line-mm here). The tool's own weight and the gages' zero are the same with and without the mass at one station;
their difference leaves the mass's wrench alone.

Models (scripts/ati_calibration_fit.py): "full", the 6 x 6 matrix by ridge regression; "rows6" and "rows12", the row
shapes of FT31565 with six output gains, and with six gage gains as well. Scored by leaving one eye out at a time
(every station of that eye), per axis. With --truth (a simulated session's sim_truth.json, found by itself) the fitted
C is also scored on 2000 wrenches of the pad's range: forces to 5 N at points within 150 mm of the ATI's origin.

    .venv/bin/python scripts/ati_calibration_analysis.py <session folder> [--rho 0.6] [--line-mm 0.3] [--trim 0.5]
Writes <session>/analysis/pairs.npz, ati_calibration.json and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ati_calibration_fit as FIT  # noqa: E402
import ur5_tip_calibration_sim as URSIM  # noqa: E402

AXES = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")


def load_bin(path, cols):
    a = np.fromfile(path, "<f8")
    return a[: len(a) // cols * cols].reshape(-1, cols)


def events(folder):
    starts, ends, mass = {}, {}, []
    for line in open(Path(folder) / "events.jsonl"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "start":
            starts[e["id"]] = e
        elif e.get("event") == "end" and e.get("status") == "ok" and "rec" in e:
            ends[e["id"]] = e
        elif e.get("event") == "mass":
            mass.append(e)
    return starts, ends, mass


def rests(folder, plan, trim):
    """One row per rest: block, eye, station, role, mass on, window, mean and std of the gages (from ati.bin, the
    window's first `trim` s dropped), the mean joints."""
    starts, ends, _ = events(folder)
    A = load_bin(Path(folder) / "ati.bin", 10)
    t = A[:, 0]
    out = []
    for sid, e in ends.items():
        if sid not in starts or sid == "preflight":
            continue
        s = starts[sid]
        t0, t1 = e["rec"]["t"]
        m = (t >= t0 + trim) & (t <= t1)
        if m.sum() < 20:
            continue
        g = A[m, 4:10]
        out.append(dict(id=sid, block=s["block"], eye=s["eye"], station=s["station"], role=s.get("role"),
                        mass_on=bool(e["rec"]["mass_on"]), t=0.5 * (t0 + t1), g=g.mean(axis=0), g_sd=g.std(axis=0),
                        n=int(m.sum()), q=np.array(e["rec"]["q"], float), hexe=e["rec"].get("hexe")))
    out.sort(key=lambda r: r["t"])
    return out


def gravity_A(q, T_EA):
    """Unit gravity in the ATI's frame at joints q (CB3 kinematics)."""
    T = URSIM.fk(q) @ T_EA
    return T[:3, :3].T @ np.array([0.0, 0.0, -1.0])


def line_point(p, u, g_hat, R, t, d=0.0):
    """A point of the centre line of a line that hangs along g_hat from an eye: a hole of radius R through a wall t thick,
    its exit point p on the outer face and its axis u outward, the line's stopper knot against the wall's inner face. The
    line runs straight from the knot (p - t u) while it clears the hole's outer edge. Past atan((R - d/2) / t) from the
    axis it bends over the edge at p + R n, n the direction of g_hat across the axis, and hangs with its centre d/2 off
    the edge, on the outer side of the bend (d the line's diameter). mm in, mm out."""
    gu = float(g_hat @ u)
    gp = g_hat - gu * u
    s = float(np.linalg.norm(gp))
    if s * t <= (R - d / 2) * gu:
        return p - t * u
    m = u - gu * g_hat
    return p + R * gp / s + 0.5 * d * m / np.linalg.norm(m)


def mass_wrench(g_hat, eye, m_kg, g_local, line):
    p = line_point(np.array(eye["p_A"], float), np.array(eye["u_A"], float), g_hat, line["R"], line["t"], line["d"]) * 1e-3
    f = m_kg * g_local * g_hat
    return np.r_[f, np.cross(p, f)]


def drift_model(R):
    """Gage drift from the reference station's rests (mass off, one pose): piecewise linear in time, zero at the first."""
    ref = [r for r in R if r["role"] == "reference" and not r["mass_on"]]
    if len(ref) < 2:
        return lambda t: np.zeros(6), ref
    tt = np.array([r["t"] for r in ref])
    gg = np.array([r["g"] for r in ref]) - ref[0]["g"]
    return (lambda t: np.array([np.interp(t, tt, gg[:, k]) for k in range(6)])), ref


def pairs(R, plan, m_kg, line, drift):
    """Mass on minus mass off at each station of each block, drift removed: G (n, 6), W (n, 6), eye and station."""
    T_EA = np.array(plan["geometry"]["T_EA"], float)
    eyes = {e["name"]: e for e in plan["geometry"]["eyes"]}
    G, W, grp, eye_of, gA = [], [], [], [], []
    for b in sorted({r["block"] for r in R}):
        rb = [r for r in R if r["block"] == b and r["eye"]]
        for st in sorted({r["station"] for r in rb}):
            on = [r for r in rb if r["station"] == st and r["mass_on"]]
            off = [r for r in rb if r["station"] == st and not r["mass_on"] and r["role"] != "reference"]
            if not on or not off:
                continue
            g_off = np.mean([r["g"] - drift(r["t"]) for r in off], axis=0)
            for r in on:
                gh = gravity_A(r["q"], T_EA)
                G.append(r["g"] - drift(r["t"]) - g_off)
                W.append(mass_wrench(gh, eyes[r["eye"]], m_kg, plan["g_local"], line))
                grp.append(f"{r['eye']}:{st}")
                eye_of.append(r["eye"])
                gA.append(gh)
    return np.array(G), np.array(W), np.array(grp), np.array(eye_of), np.array(gA)


def leave_eye_out(G, W, eye_of, R):
    out = {m: [] for m in ("full", "rows6", "rows12")}
    for e in np.unique(eye_of):
        te = eye_of == e
        for m, fn in (("full", lambda g, w: FIT.fit_full(g, w)), ("rows6", lambda g, w: FIT.fit_rows6(g, w, R)),
                      ("rows12", lambda g, w: FIT.fit_rows12(g, w, R))):
            C = fn(G[~te], W[~te])
            out[m].append(np.abs(G[te] @ C.T - W[te]))
    return {m: dict(median=np.median(np.concatenate(v), 0).tolist(), p90=np.percentile(np.concatenate(v), 90, 0).tolist(),
                    max=np.concatenate(v).max(0).tolist()) for m, v in out.items()}


def score_truth(C, truth, n=2000, seed=1):
    """C applied to the true sensor's gages for wrenches of the pad's range, against the wrench in the nominal ATI frame."""
    rng = np.random.default_rng(seed)
    D = np.array(truth["D"], float)
    c, s_ = math.cos(math.radians(truth["clock_err_deg"])), math.sin(math.radians(truth["clock_err_deg"]))
    Rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
    off = np.array(truth["origin_err_mm"], float) * 1e-3
    E = []
    for _ in range(n):
        f = rng.normal(size=3)
        f *= rng.uniform(0.2, 5.0) / np.linalg.norm(f)
        p = rng.normal(size=3)
        p *= rng.uniform(0.0, 0.15) / np.linalg.norm(p)
        w_true = np.r_[f, np.cross(p, f)]                              # at the true origin, true axes
        f_nom = Rz @ f
        w_nom = np.r_[f_nom, Rz @ w_true[3:] + np.cross(off, f_nom)]
        E.append(np.abs(C @ (D @ w_true) - w_nom))
    E = np.array(E)
    return dict(median=np.median(E, 0).tolist(), p90=np.percentile(E, 90, 0).tolist(), max=E.max(0).tolist())


def hexe_gains(R, plan, m_kg):
    """The HEX-E's force gains from the same pairs: a 3 x 3 K with f_hex(on) - f_hex(off) = K (m g in the flange frame)."""
    X, Y = [], []
    for b in sorted({r["block"] for r in R}):
        rb = [r for r in R if r["block"] == b and r["eye"] and r["hexe"] is not None]
        for st in sorted({r["station"] for r in rb}):
            on = [r for r in rb if r["station"] == st and r["mass_on"]]
            off = [r for r in rb if r["station"] == st and not r["mass_on"] and r["role"] != "reference"]
            if not on or not off:
                continue
            f_off = np.mean([np.array(r["hexe"][:3]) for r in off], axis=0)
            for r in on:
                g_E = URSIM.fk(r["q"])[:3, :3].T @ np.array([0.0, 0.0, -plan["g_local"] * m_kg])
                X.append(g_E)
                Y.append(np.array(r["hexe"][:3]) - f_off)
    if len(X) < 6:
        return None
    X, Y = np.array(X), np.array(Y)
    K = np.linalg.lstsq(X, Y, rcond=None)[0].T
    U, S, Vt = np.linalg.svd(K)
    return dict(K=K.tolist(), column_gains=np.linalg.norm(K, axis=0).tolist(), singular=S.tolist(), n=len(X),
                residual_rms_N=float(np.sqrt(np.mean((X @ K.T - Y) ** 2))))


def line_of(plan, ses, rho=None, line_mm=None):
    """The eye's hole radius, wall and the line's diameter (mm): --rho and --line-mm here, else the session's record, else
    the plan's hole and a line of no thickness."""
    sa = ses.get("args", {})
    R = rho if rho is not None else (sa.get("eye_hole_mm") or plan["geometry"]["eye"]["hole_d"]) / 2
    d = line_mm if line_mm is not None else sa.get("line_mm")
    if d is None:
        print("the line's diameter is not recorded: taken as 0 (give --line-mm)")
        d = 0.0
    return dict(R=float(R), t=float(plan["geometry"]["eye"].get("wall", 1.6)), d=float(d))


def main(folder, rho=None, trim=0.5, line_mm=None):
    folder = Path(folder)
    plan = json.loads((folder / "plan.json").read_text())
    ses = json.loads((folder / "session.json").read_text())
    m_kg = ses["mass_g"] * 1e-3
    line = line_of(plan, ses, rho, line_mm)
    R = rests(folder, plan, trim)
    drift, ref = drift_model(R)
    G, W, grp, eye_of, gA = pairs(R, plan, m_kg, line, drift)
    out_dir = folder / "analysis"
    out_dir.mkdir(exist_ok=True)
    np.savez(out_dir / "pairs.npz", G=G, W=W, group=grp, eye=eye_of, g_A=gA)
    Rs = FIT.R_21031 if plan.get("sensor") == "Mini40" else FIT.R_31565       # the shape prior of the same model
    C_full = FIT.fit_full(G, W)
    C_r12 = FIT.fit_rows12(G, W, Rs)
    res_in = np.abs(G @ C_full.T - W)
    cv = leave_eye_out(G, W, eye_of, Rs)
    rec = dict(session=str(folder), mass_kg=m_kg, g_local=plan["g_local"], line_mm=line, trim_s=trim, rests=len(R), pairs=len(G),
               eyes=sorted(set(eye_of.tolist())), design_singular_values=FIT.design(W),
               drift_counts=dict(reference_rests=len(ref), span=(np.ptp([r["g"] for r in ref], axis=0).tolist() if ref else None)),
               C_full=C_full.tolist(), C_rows12=C_r12.tolist(),
               fit_residual_median=np.median(res_in, 0).tolist(), leave_eye_out=cv,
               frame="the ATI's sensing frame (drawing 9230-05-1094) as placed by the stack's clock; origin at the tool face's "
                     "centre; C turns gage values (raw counts minus their zero) into N and N m",
               settle_s=dict(median=float(np.median([r.get("settle_s", 0) for r in R])) if R else None),
               gage_sd_median=np.median([r["g_sd"] for r in R], axis=0).tolist())
    hg = hexe_gains(R, plan, m_kg)
    if hg:
        rec["hexe"] = hg
    tp = folder / "sim_truth.json"
    if tp.exists():
        truth = json.loads(tp.read_text())
        rec["truth_score"] = {m: score_truth(C, truth) for m, C in (("full", C_full), ("rows12", C_r12))}
    (out_dir / "ati_calibration.json").write_text(json.dumps(rec, indent=1, default=lambda o: o.tolist()) + "\n")
    f3 = lambda v: ", ".join(f"{x:.3f}" for x in v[:3])                       # noqa: E731
    t1 = lambda v: ", ".join(f"{x * 1e3:.1f}" for x in v[3:])                  # noqa: E731
    print(f"{len(R)} rests, {len(G)} pairs on eyes {rec['eyes']}; design singular values {rec['design_singular_values']}")
    for m in ("full", "rows6", "rows12"):
        print(f"leave one eye out, {m:6s}: median F {f3(cv[m]['median'])} N, T {t1(cv[m]['median'])} N mm; "
              f"90th F {f3(cv[m]['p90'])} N, T {t1(cv[m]['p90'])} N mm")
    if "truth_score" in rec:
        for m, s in rec["truth_score"].items():
            print(f"against the truth, {m:6s}: median F {f3(s['median'])} N, T {t1(s['median'])} N mm; "
                  f"90th F {f3(s['p90'])} N, T {t1(s['p90'])} N mm")
    if hg:
        print(f"HEX-E force gains (column norms) {np.round(hg['column_gains'], 3)}, residual {hg['residual_rms_N']:.3f} N")
    print("->", out_dir / "ati_calibration.json")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--rho", type=float, default=None, help="the eyes' hole radius, mm (default: the session's or the plan's)")
    ap.add_argument("--line-mm", type=float, default=None, help="the line's diameter, mm (default: the session's)")
    ap.add_argument("--trim", type=float, default=0.5)
    a = ap.parse_args()
    main(a.folder, a.rho, a.trim, a.line_mm)
