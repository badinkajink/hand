#!/usr/bin/env python3
"""Stiffness maps of four fingertips and how well each contact model reproduces them (ledger step 16 of the curved-tip
study, docs/handoff/20261015-curved_tips.md; owner, 2026-10-15 16:20).

Tips (design surfaces): the as-built block tcd_rs3_re3, the capsule tcd_cap8.5, the half-ellipsoid
tcd_ell7.4x8.5x16x32 and the quarter-ellipsoid tcd_qell7.4x8.5x11x22.

Probes, rigid and frictionless, moved along -x (the finger's palmar normal, the pinch direction) from the first touch of
the design surface; positions in the tip frame (y across the width, z toward the mount, the contact point at 0):
  sphere   radius 2 mm at the apex (0, 0); across the width at y 2.6, 4.1 and 5.2 mm; along the finger 4 mm proximal
           (z +4) and 4 and 7 mm distal (z -4, -7)
  cyl      the bed's tool, radius 12.5 mm with its axis along y, at z 0, +4 and -4 mm
Loads 0.5, 1 and 3 N. Readings: the approach (probe travel from first touch, um) and the local stiffness dF/d(delta)
(N/mm, from the approach at 0.95 and 1.05 of the load).

Models:
  fe     finite elements of the printed tip (scripts/tpu_tip_magnet.py's model: walls, skins and the homogenized gyroid
         core of recipe w95_20 in the print orientation, the 4.76 x 1.59 mm magnet in its pocket, the mount face and the
         back face, or a quarter-ellipsoid's back plane, fixed) on a grid of 0.25-0.32 mm over the probed region, with
         frictionless contact along x on the compliance of the palmar face nodes; the reference
  drake  Drake's compliant-convex field p = E (1 - g), g the gauge of the 0.25 mm hull about its volume centroid,
         integrated over the probe surface (curved_tips_law.DrakeFieldWide)
  v1     1 mm pads, one stiffness per finger (median K, foundation depth 8.5 mm)
  v2     1 mm pads, K_i = E a_i / h_i with h_i the depth below the volume centroid (a qell key: the depth to its back
         plane, (p_x - x_c) / max(n_x, 0.2))
  v2c    a qell key's pads with the centroid depth (tcd_qellc)
  fit    the v2 pads with K_i = c k_i: k_i the finite-element stiffness of the pad's patch (a unit force spread over the
         palmar face nodes within the pad's 1 mm cell, by tributary area, divided into their mean displacement), c one
         factor per tip that matches the sphere probe's approach at the apex at 1 N
E is 10 MPa in drake, v1, v2 and v2c (the pipeline's modulus); the summary also gives each model's error with its
modulus scaled to the finite elements' apex approach (sphere, 1 N), which measures the map's shape alone.

    FEM=~/miniconda3/envs/femtip/bin/python
    ~/.claude/bin/resguard.sh run --mem 5G --cpu 400 -- taskset -c 16-19 env OMP_NUM_THREADS=4 $FEM \\
        scripts/curved_tips_stiffness.py fe --tip tcd_cap8.5
    .venv/bin/python scripts/curved_tips_stiffness.py models --tip tcd_cap8.5
    .venv/bin/python scripts/curved_tips_stiffness.py summary
Rows: docs/experiments/20261015-curved_tips/stiffness/{fe,models}.jsonl; summary.json; patch stiffness per pad in
stiffness/patch_<tip>.npz.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tip_codesign_geometry as TG  # noqa: E402

OUT = ROOT / "docs/experiments/20261015-curved_tips/stiffness"
SCRIPT = "scripts/curved_tips_stiffness.py"
TIPS = ["tcd_rs3_re3", "tcd_cap8.5", "tcd_ell7.4x8.5x16x32", "tcd_qell7.4x8.5x11x22"]
R_SPH, R_CYL = 2.0, 12.5                                     # mm
SPHERE_PTS = [("apex", 0.0, 0.0), ("y2.6", 2.6, 0.0), ("y4.1", 4.1, 0.0), ("y5.2", 5.2, 0.0),
              ("z+4", 0.0, 4.0), ("z-4", 0.0, -4.0), ("z-7", 0.0, -7.0)]
CYL_PTS = [("cyl0", 0.0), ("cyl+4", 4.0), ("cyl-4", -4.0)]
FORCES = (0.5, 1.0, 3.0)
E_MODEL = 1e7
X_OFF = TG.REACH - 0.0085                                    # tip-frame x of the insert frame's x = 0, m


def probes():
    """(name, kind, y_tip_mm, z_tip_mm) of every probe position."""
    return [(n, "sphere", y, z) for n, y, z in SPHERE_PTS] + [(n, "cyl", 0.0, z) for n, z in CYL_PTS]


def build_of(key):
    """Print orientation of the tip as the study prints it: blocks palmar face up (layers across x), capsules and
    half-ellipsoids upright (layers across z), quarter-ellipsoids on their back face (layers across x)."""
    sh = TG.shape(key)
    return "z" if getattr(sh, "family", "") in ("cap", "ell") else "x"


def surface_ins(sh, y, z):
    """Palmar surface height (mm, insert frame) at insert-frame (y, z) in mm."""
    return sh.fe_surface_x(y, z)


def first_touch(sh, kind, y0, z0_ins):
    """Probe reference height (insert frame, mm) at first touch: for the sphere its centre's x, for the cylinder its
    axis's x, from the analytic surface on a 0.02 mm grid."""
    if kind == "sphere":
        g = np.arange(-R_SPH, R_SPH + 1e-9, 0.02)
        Y, Z = np.meshgrid(y0 + g, z0_ins + g, indexing="ij")
        rho2 = (Y - y0) ** 2 + (Z - z0_ins) ** 2
        ok = rho2 < R_SPH ** 2
        xs = surface_ins(sh, Y[ok], Z[ok])
        return float(np.nanmax(xs + np.sqrt(R_SPH ** 2 - rho2[ok])))
    zz = z0_ins + np.arange(-5.0, 5.0 + 1e-9, 0.02)
    yy = np.arange(-7.4, 7.4 + 1e-9, 0.1)
    Y, Z = np.meshgrid(yy, zz, indexing="ij")
    xs = surface_ins(sh, Y, Z)
    return float(np.nanmax(xs + np.sqrt(R_CYL ** 2 - (Z - z0_ins) ** 2)))


def probe_surface_ins(kind, xc, y0, z0_ins, Y, Z):
    """x (insert frame, mm) of the probe's lower surface above (Y, Z), +inf outside its footprint."""
    if kind == "sphere":
        r2 = (Y - y0) ** 2 + (Z - z0_ins) ** 2
        out = np.full(np.shape(Y), np.inf)
        ok = r2 < R_SPH ** 2
        out[ok] = xc - np.sqrt(R_SPH ** 2 - r2[ok])
        return out
    dz = Z - z0_ins
    out = np.full(np.shape(Y), np.inf)
    ok = np.abs(dz) < R_CYL
    out[ok] = xc - np.sqrt(R_CYL ** 2 - dz[ok] ** 2)
    return out


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, default=float) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def rows(path):
    out = []
    if Path(path).exists():
        for line in open(path):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


# ------------------------------------------------------------------------------------------ finite elements

def fe_grid(xb):
    """Zones per axis (insert frame, mm): 0.25 mm in x over the outer 3.1 mm, 0.32 mm in y over [-1, 6] and 0.3 mm in z
    over [6, 18.5], coarser elsewhere; the x zones start at the fixed back (face or plane), so it is a node plane."""
    if xb < -1.0:
        xz = [(xb, xb + 0.9, 0.45), (xb + 0.9, 1.0, 1.1), (1.0, 5.4, 0.55), (5.4, 8.5, 0.25)]
    elif xb < 5.0:
        xz = [(xb, 5.4, 0.55), (5.4, 8.5, 0.25)]
    else:
        xz = [(xb, 8.5, 0.25)]
    return dict(x=xz, y=[(-7.4, -1.0, 0.7), (-1.0, 6.0, 0.32), (6.0, 7.4, 0.35)],
                z=[(0.0, 6.0, 0.8), (6.0, 18.5, 0.3), (18.5, 21.41, 0.4)])


def cmd_fe(a):
    import curved_tips_fe as CF
    import tpu_tip_fem_tip as T
    import tpu_tip_magnet as M
    key = a.tip
    sh = TG.shape(key)
    xb = CF.install(sh)
    if getattr(sh, "family", "") != "qell":
        xb = T.X_BACK
    depth = CF.depth_of(sh)
    recipe = dict(M.RECIPES[a.recipe], build=build_of(key))
    M.GRID = fe_grid(xb)
    t0 = time.time()
    model, _ = M.build(recipe, "d4.76x1.59", depth)
    nfix_back = int(((np.abs(model.X[:, 0] - xb) < 1e-6) & model.fixed_nodes).sum())
    nodes = model.face_nodes(zlim=(3.5, 21.41))
    X = model.X[nodes].copy()
    xt = surface_ins(sh, X[:, 1], X[:, 2])
    xs = model.g.xs
    k = np.clip(np.searchsorted(xs, X[:, 0] - 1e-9), 1, len(xs) - 1)
    use = np.isfinite(xt) & (np.abs(xt - X[:, 0]) <= 0.5 * (xs[k] - xs[k - 1]) + 1e-6)
    X[:, 0] = np.where(use, xt, X[:, 0])
    tc = time.time()
    W = model.compliance(nodes)
    tc = time.time() - tc
    A = T.tributary_areas(model, nodes)
    print(f"{key}: build {recipe['build']}, back x {xb:.2f} mm ({nfix_back} fixed back nodes), magnet depth {depth} mm, "
          f"{len(nodes)} face nodes, compliance {tc:.0f} s", flush=True)
    base = dict(tip=key, recipe=a.recipe, build=recipe["build"], magnet="d4.76x1.59", depth_mm=depth, back_x_mm=xb,
                n_fixed_back_nodes=nfix_back, n_dof=len(model.free), n_el=model.n_el, n_face_nodes=len(nodes),
                t_factor_s=model.t_factor, t_compliance_s=tc, grid=M.GRID, script=SCRIPT)
    for name, kind, y_t, z_t in probes():
        z_ins = TG.Z_MOUNT * 1e3 - z_t
        xc = first_touch(sh, kind, y_t, z_ins)
        surf = probe_surface_ins(kind, xc, y_t, z_ins, X[:, 1], X[:, 2])
        g0 = surf - X[:, 0]
        cand = np.where(g0 < 1.5)[0]
        Wc, gc = W[np.ix_(cand, cand)], g0[cand]
        res = {}
        for F in FORCES:
            out = {}
            for f in (0.95 * F, F, 1.05 * F):
                d, p, _ = T.force_controlled(Wc, gc, f)
                out[f] = (d, p)
            d, p = out[F]
            kk = 0.1 * F / (out[1.05 * F][0] - out[0.95 * F][0])
            act = p > 1e-12 * F
            c = (p[act, None] * X[cand][act, 1:]).sum(0) / p[act].sum()
            res[f"{F:g}"] = dict(approach_um=1e3 * d, k_N_per_mm=kk, n_contact=int(act.sum()),
                                 centre_yz_mm=c.tolist(),
                                 patch_r_rms_mm=float(np.sqrt((p[act] * ((X[cand][act, 1:] - c) ** 2).sum(1)).sum() / p[act].sum())))
        append(OUT / "fe.jsonl", dict(base, probe=name, kind=kind, y_mm=y_t, z_mm=z_t, x_first_touch_mm=xc,
                                      n_candidates=int(len(cand)), per_force=res, when=time.strftime("%Y-%m-%d %H:%M")))
        print(f"  {name:6s} {kind:6s} " + "  ".join(f"{F:g} N: {v['approach_um']:6.1f} um {v['k_N_per_mm']:6.2f} N/mm"
                                                  for F, v in zip(FORCES, res.values())), flush=True)
    # patch stiffness of each v2 pad's 1 mm cell (the fitted pads)
    C, Nn, Ap, _ = sh.pads(E_MODEL, h="local", K="each", full=True)
    Pt = C + 0.00075 * Nn
    py, pz = Pt[:, 1] * 1e3, TG.Z_MOUNT * 1e3 - Pt[:, 2] * 1e3
    kpatch = np.full(len(C), np.nan)
    npatch = np.zeros(len(C), int)
    for i in range(len(C)):
        sel = np.where((np.abs(X[:, 1] - py[i]) <= 0.5) & (np.abs(X[:, 2] - pz[i]) <= 0.5))[0]
        if len(sel) == 0 or Nn[i, 0] < 0.3:
            continue
        w = A[sel] / A[sel].sum()
        u = W[np.ix_(sel, sel)] @ w
        kpatch[i] = 1.0 / float((u * w).sum())                  # N/mm, the patch's area-weighted mean displacement
        npatch[i] = len(sel)
    np.savez(OUT / f"patch_{key}.npz", C=C, N=Nn, A=Ap, k_N_per_mm=kpatch, n_nodes=npatch)
    append(OUT / "fe.jsonl", dict(base, probe="patch", kind="patch", n_pads=int(len(C)),
                                  n_pads_with_k=int(np.isfinite(kpatch).sum()),
                                  k_patch_median_N_per_mm=float(np.nanmedian(kpatch)), t_total_s=time.time() - t0,
                                  when=time.strftime("%Y-%m-%d %H:%M")))
    print(f"  patches: {int(np.isfinite(kpatch).sum())} of {len(C)} pads, median {np.nanmedian(kpatch):.2f} N/mm; "
          f"{time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------ models

class Probe:
    """A probe's surface samples (tip frame, m) at first touch, and the field and pad forces along +x on it."""

    def __init__(self, sh, kind, y_t, z_t):
        self.kind = kind
        z_ins = TG.Z_MOUNT * 1e3 - z_t
        xc_ins = first_touch(sh, kind, y_t, z_ins)
        self.c0 = np.array([(xc_ins * 1e-3) + X_OFF, y_t * 1e-3, z_t * 1e-3])     # sphere centre or a point on the axis
        if kind == "sphere":
            n = 24000
            i = np.arange(n) + 0.5
            u = np.column_stack([-(1 - 2 * i / n), np.zeros(n), np.zeros(n)])
            r = np.sqrt(1 - u[:, 0] ** 2)
            th = math.pi * (3 - math.sqrt(5)) * i
            u[:, 1], u[:, 2] = r * np.cos(th), r * np.sin(th)
            keep = u[:, 0] < 0.35
            self.u = u[keep]
            self.X0 = self.c0 + R_SPH * 1e-3 * self.u
            self.dA = 4 * math.pi * (R_SPH * 1e-3) ** 2 / n
        else:
            ph = np.linspace(-0.45, 0.45, 241)
            yy = np.linspace(-0.0080, 0.0080, 161)
            PH, YY = np.meshgrid(ph, yy, indexing="ij")
            u = np.column_stack([-np.cos(PH).ravel(), np.zeros(PH.size), np.sin(PH).ravel()])
            self.u = u
            self.X0 = np.column_stack([self.c0[0] + R_CYL * 1e-3 * u[:, 0], YY.ravel(), self.c0[2] + R_CYL * 1e-3 * u[:, 2]])
            self.dA = (ph[1] - ph[0]) * R_CYL * 1e-3 * (yy[1] - yy[0])

    def field_force(self, field, delta, idx=None):
        X = self.X0 if idx is None else self.X0[idx]
        u = self.u if idx is None else self.u[idx]
        p = field.p(X - np.array([delta, 0.0, 0.0]))
        return float(np.sum(p * self.dA * (-u[:, 0])))

    def pad_force(self, C, K, rs, delta):
        c = self.c0 - np.array([delta, 0.0, 0.0])
        if self.kind == "sphere":
            v = c - C
            dist = np.linalg.norm(v, axis=1)
            d = rs + R_SPH * 1e-3 - dist
        else:
            v = np.column_stack([c[0] - C[:, 0], np.zeros(len(C)), c[2] - C[:, 2]])
            dist = np.linalg.norm(v, axis=1)
            d = rs + R_CYL * 1e-3 - dist
        m = d > 0
        return float(np.sum(K[m] * d[m] * v[m, 0] / dist[m]))


DGRID = np.geomspace(5e-6, 1.6e-3, 44)                       # approach grid of the models' force curves, m


class NearField:
    """Drake's compliant-convex field p = E (1 - g) of a tip (tip_codesign_law.DrakeField: the hull at 0.25 mm,
    degenerate facets dropped, the gauge g about the volume centroid) with the gauge taken over the facets whose centroid
    lies within `radius` of a probe's region. For a point within a few tenths of a millimetre of the surface the
    maximizing facet is the one its ray from the centroid leaves through, next to the point, so the restriction leaves
    the field unchanged there (checked against the whole hull in cmd_check)."""

    def __init__(self, sh, E=E_MODEL, res=0.00025):
        V, F = sh.hull(res)
        tri = V[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(n, axis=1)
        ok = ln > 1e-14
        tri, n = tri[ok], n[ok] / ln[ok, None]
        self.c, self.E = sh.centroid(), E
        self.n = n
        self.dk = np.sum(n * (tri[:, 0] - self.c), 1)
        self.fc = tri.mean(1)
        self.sel = np.arange(len(n))

    def restrict(self, kind, c0, radius):
        if kind == "sphere":
            d = np.linalg.norm(self.fc - c0, axis=1)
        else:
            d = np.abs(self.fc[:, 2] - c0[2])
        self.sel = np.where(d < radius)[0]
        return len(self.sel)

    def p(self, X):
        n, dk = self.n[self.sel], self.dk[self.sel]
        out = np.empty(len(X))
        ch = max(64, int(4e6 // max(len(n), 1)))
        for s in range(0, len(X), ch):
            g = np.max((X[s:s + ch] - self.c) @ n.T / dk[None, :], axis=1)
            out[s:s + ch] = self.E * np.clip(1 - g, 0, None)
        return out


def from_curve(dg, Fg, F):
    """Approach (m) at load F and the local stiffness (N/mm) there, from a monotone force curve (log-log
    interpolation)."""
    ok = Fg > 0
    lf, ld = np.log(Fg[ok]), np.log(dg[ok])
    keep = np.r_[True, np.diff(lf) > 0]
    lf, ld = lf[keep], ld[keep]
    x = math.log(F)
    lnd = float(np.interp(x, lf, ld))
    j = int(np.clip(np.searchsorted(lf, x), 1, len(lf) - 1))
    slope = (lf[j] - lf[j - 1]) / (ld[j] - ld[j - 1])           # dlnF / dln(delta)
    d = math.exp(lnd)
    return d, slope * F / d * 1e-3


def cmd_models(a):
    key = a.tip
    sh = TG.shape(key)
    t0 = time.time()
    models = {}
    field = NearField(sh)
    models["drake"] = ("field", field)
    rs = 0.00075
    C1, K1 = sh.pads(E_MODEL, h="fixed", K="median")
    C2, N2, A2, K2 = sh.pads(E_MODEL, h="local", K="each", full=True)
    models["v1"] = ("pads", (C1, K1))
    models["v2"] = ("pads", (C2, K2))
    if getattr(sh, "family", "") == "qell":
        shc = TG.shape(key.replace("tcd_qell", "tcd_qellc"))
        Cc, Kc = shc.pads(E_MODEL, h="local", K="each")
        models["v2c"] = ("pads", (Cc, Kc))
    patch = OUT / f"patch_{key}.npz"
    fe = {(r["probe"]): r for r in rows(OUT / "fe.jsonl") if r["tip"] == key and r["kind"] != "patch"}
    fit_info = {"error": "no finite-element patch stiffness yet"}
    if patch.exists() and "apex" in fe:
        z = np.load(patch)
        kp = z["k_N_per_mm"] * 1e3                              # N/m
        if np.allclose(z["C"], C2):
            ok = np.isfinite(kp)
            ratio = float(np.nanmedian(kp[ok] / K2[ok]))         # pads without a patch keep v2's K times the median ratio
            Kf0 = np.where(ok, kp, K2 * ratio)
            pr = Probe(sh, "sphere", 0.0, 0.0)
            Fg = np.array([pr.pad_force(C2, Kf0, rs, d) for d in DGRID])
            target = fe["apex"]["per_force"]["1"]["approach_um"] * 1e-6
            cfit = 1.0 / float(np.exp(np.interp(math.log(target), np.log(DGRID), np.log(np.maximum(Fg, 1e-30)))))
            models["fit"] = ("pads", (C2, cfit * Kf0))
            fit_info = dict(c=cfit, n_with_patch=int(ok.sum()), n_pads=int(len(C2)), ratio_fill=ratio,
                            recipe="K_i = c k_i, k_i the patch stiffness of the pad's 1 mm cell in the finite elements; c "
                                   "matches the apex sphere probe's approach at 1 N")
        else:
            fit_info = {"error": "pad set differs from the patch file's"}
    for name, kind, y_t, z_t in probes():
        pr = Probe(sh, kind, y_t, z_t)
        nsel = field.restrict(kind, pr.c0 - np.array([R_SPH * 1e-3 if kind == "sphere" else R_CYL * 1e-3, 0, 0]),
                              0.006 if kind == "sphere" else 0.008)
        pdeep = field.p(pr.X0 - np.array([DGRID[-1], 0.0, 0.0]))
        idx = np.where(pdeep > 0)[0]
        for mname, (mk, obj) in models.items():
            if mk == "field":
                Fg = np.array([pr.field_force(field, d, idx) for d in DGRID])
            else:
                C_, K_ = obj
                Fg = np.array([pr.pad_force(C_, K_, rs, d) for d in DGRID])
            res = {}
            for F in FORCES:
                d, kk = from_curve(DGRID, Fg, F)
                res[f"{F:g}"] = dict(approach_um=d * 1e6, k_N_per_mm=kk)
            row = dict(tip=key, model=mname, probe=name, kind=kind, y_mm=y_t, z_mm=z_t, per_force=res, E_Pa=E_MODEL,
                       curve=dict(delta_um=(DGRID * 1e6).round(4).tolist(), F_N=Fg.round(6).tolist()),
                       script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"))
            if mname == "fit":
                row["fit"] = fit_info
            append(OUT / "models.jsonl", row)
            print(f"{key} {mname:5s} {name:6s} " + "  ".join(f"{F:g} N: {v['approach_um']:6.1f} um {v['k_N_per_mm']:6.2f} N/mm"
                                                           for F, v in zip(FORCES, res.values())), flush=True)
    print(f"{key}: models {time.time() - t0:.0f} s; fit {fit_info}", flush=True)


# ------------------------------------------------------------------------------------------ summary

def cmd_summary(a):
    fe = [r for r in rows(OUT / "fe.jsonl") if r["kind"] != "patch"]
    mo = rows(OUT / "models.jsonl")
    out = {"tips": {}, "terms": {
        "approach_um": "probe travel along -x from first touch of the design surface at the load",
        "k_N_per_mm": "local stiffness dF/d(approach) at the load",
        "err_abs": "model approach over the finite elements' minus 1, every model at E 10 MPa (fit: its fitted K)",
        "err_shape": "the same with the model's modulus scaled so its sphere-probe approach at the apex at 1 N equals the "
                     "finite elements' (the map's variation over position and load alone)"}}
    for tip in sorted({r["tip"] for r in fe}):
        F_ = {r["probe"]: r for r in fe if r["tip"] == tip}
        M_ = {}
        for r in mo:
            if r["tip"] == tip:
                M_.setdefault(r["model"], {})[r["probe"]] = r          # the last row of a (model, probe) wins
        t = {"fe": {p: {F: v["approach_um"] for F, v in r["per_force"].items()} for p, r in F_.items()},
             "fe_k": {p: {F: v["k_N_per_mm"] for F, v in r["per_force"].items()} for p, r in F_.items()}, "models": {}}
        for m, pr in M_.items():
            if "apex" not in pr or "apex" not in F_:
                continue
            curve = {p: (np.array(r["curve"]["delta_um"]) * 1e-6, np.array(r["curve"]["F_N"])) for p, r in pr.items()}
            d_fe = F_["apex"]["per_force"]["1"]["approach_um"] * 1e-6
            dg, Fg = curve["apex"]
            c = 1.0 / float(np.exp(np.interp(math.log(d_fe), np.log(dg), np.log(np.maximum(Fg, 1e-30)))))
            pts, ea, es = {}, [], []
            for p in F_:
                if p not in pr:
                    continue
                for F in FORCES:
                    ref = F_[p]["per_force"][f"{F:g}"]["approach_um"]
                    e_abs = pr[p]["per_force"][f"{F:g}"]["approach_um"] / ref - 1
                    d_s, k_s = from_curve(*curve[p], F / c)
                    e_sh = d_s * 1e6 / ref - 1
                    k_err = c * k_s / F_[p]["per_force"][f"{F:g}"]["k_N_per_mm"] - 1
                    pts.setdefault(p, {})[f"{F:g}"] = {"err_abs": e_abs, "err_shape": e_sh, "k_err_shape": k_err}
                    ea.append(e_abs)
                    es.append(e_sh)
            t["models"][m] = {"approach_um": {p: {F: v["approach_um"] for F, v in r["per_force"].items()} for p, r in pr.items()},
                              "k_N_per_mm": {p: {F: v["k_N_per_mm"] for F, v in r["per_force"].items()} for p, r in pr.items()},
                              "points": pts, "modulus_scale_to_fe": c,
                              "rms_err_abs": float(np.sqrt(np.mean(np.square(ea)))),
                              "rms_err_shape": float(np.sqrt(np.mean(np.square(es)))),
                              "max_err_shape": float(np.max(np.abs(es))), "fit": pr["apex"].get("fit")}
        out["tips"][tip] = t
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))
    for tip, t in out["tips"].items():
        print(tip)
        for m, v in t["models"].items():
            print(f"  {m:5s} rms abs {v['rms_err_abs']:.3f}  rms shape {v['rms_err_shape']:.3f}  max shape "
                  f"{v['max_err_shape']:.3f}  modulus scale {v['modulus_scale_to_fe']:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fe")
    f.add_argument("--tip", required=True)
    f.add_argument("--recipe", default="w95_20")
    m = sub.add_parser("models")
    m.add_argument("--tip", required=True)
    sub.add_parser("summary")
    a = ap.parse_args()
    {"fe": cmd_fe, "models": cmd_models, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    main()
