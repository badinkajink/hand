#!/usr/bin/env python3
"""Model curves of Drake's field and the pads under the finite-element probes, and the three calibrations (phase B of
docs/experiments/20261018-fe_contact_calibration/plan.json).

curves  Per tip (its simulator key, the as-printed surface) and probe of phase A: the normal force along the approach
        grid (44 points, 2 um to 1.2 mm) at the unit modulus E0 = 10 MPa, and the patch at every grid point: the
        pressure centre, the pressure-weighted RMS radius r_rms and mean radius r_bar (the twist arm), and the pads in
        contact. Models:
          drake   Drake's compliant-convex field p = E (1 - g), g the gauge of the 0.25 mm hull about its volume centroid
                  (curved_tips_stiffness.NearField), integrated over the probe surface
          v2      builder-v2 pads at 1 mm (radius 0.75 mm): K_i = E a_i / h_i, h_i the depth below the volume centroid
          v2h     builder-v2 pads at 0.5 mm (radius 0.375 mm)
          v1      builder-v1 pads at 1 mm (one stiffness per finger, h 8.5 mm)
          depth   (capsule only) the depth field p = E d / d_max, d the distance below the hull (k-d tree of surface
                  samples, queries in chunks of 4,096 points)
        Every force scales with E, so one curve per model gives every modulus.
fit     Per tip and model (drake, v2, v2h, depth):
          M1  one E by log-RMS of approach over the task cylinders (radius 5, 8, 12.5 mm across the finger at z 0, +-4 mm;
              8 and 12.5 mm along it) at 0.5, 1 and 3 N
          M2  one E by log-RMS of r_bar over the 8 and 12.5 mm cylinders across the finger at z 0, at 1 and 3 N
          M3  E_loc = M2's E and a series spring k_s by log-RMS of approach over the task cylinders and loads:
              approach = F / k_s + the local approach at E_loc; one k_s per tip for both simulators (their mean)
        with the criteria's residuals: approach (RMS and largest |model / FE - 1|), twist arm (model r_bar / FE r_bar),
        pads in contact at 1 N, and rule R3 (r_bar change over E x0.5 to x2 at the M1 modulus).

        fit --bench CSV takes bench indentation rows as the target in place of the finite elements (bench_target: one
        solid-modulus factor for the material, the bench approach where measured, the finite elements at that factor
        elsewhere) and writes fit/fits_bench.json, or --out PATH.

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 900 -- taskset -c 4-11 env OMP_NUM_THREADS=8 \\
        .venv/bin/python scripts/fe_contact_calibration_fit.py curves --tip cap_g30
    .venv/bin/python scripts/fe_contact_calibration_fit.py fit
    .venv/bin/python scripts/fe_contact_calibration_fit.py fit --bench <indentation.csv> --out <new study>/fit/fits.json
Rows: docs/experiments/20261018-fe_contact_calibration/fit/curves.jsonl, fit/fits.json (labelled model).
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

D = ROOT / "docs/experiments/20261018-fe_contact_calibration"
OUT = D / "fit"
SCRIPT = "scripts/fe_contact_calibration_fit.py"
SIM_KEY = {"block4_g30": "tcd_rs3_re3_p0.2", "cap_g30": "tcd_cap8.5_u0.1", "ell16_g30": "tcd_ell7.4x8.5x16x32_u0.1",
           "ell11_g30": "tcd_ell7.4x8.5x11x22_u0.1"}
U_E = {"block4_g30": 7.549794e6, "cap_g30": 13.446e6, "ell16_g30": 15.006e6, "ell11_g30": 14.226e6}
E0 = 1e7
X_OFF = TG.REACH - 0.0085                                   # tip-frame x of the insert frame's x = 0, m
DGRID = np.geomspace(2e-6, 1.0e-3, 40)                      # approach grid, m
TASK = ["cx5_z0", "cx5_z+4", "cx5_z-4", "cx8_z0", "cx8_z+4", "cx8_z-4", "cx12.5_z0", "cx12.5_z+4", "cx12.5_z-4",
        "cz8_y0", "cz12.5_y0"]
ARM = ["cx8_z0", "cx12.5_z0"]
FIT_F = (0.5, 1.0, 3.0)
ARM_F = (1.0, 3.0)
PADS = {"v2": (0.001, 0.00075, "local", "each"), "v2h": (0.0005, 0.000375, "local", "each"),
        "v1": (0.001, 0.00075, "fixed", "median")}


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


# ------------------------------------------------------------------------------------------ probes (tip frame, m)

class Probe:
    """A rigid probe's lower surface samples at first touch of the shape and the outward normals u (pointing out of
    the probe); kind 'sphere', 'cx' (cylinder, axis along y) or 'cz' (cylinder, axis along z)."""

    def __init__(self, sh, kind, R_mm, y_t, z_t):
        """sh: the design surface (analytic), which places the probe near first touch; each model's own first touch
        is then found exactly (field_touch, pad_touch)."""
        import fe_contact_calibration_fe as FE
        self.kind, self.R = kind, R_mm * 1e-3
        z_ins = TG.Z_MOUNT * 1e3 - z_t
        xc = FE.first_touch(sh, kind, R_mm, y_t, z_ins)
        self.c0 = np.array([xc * 1e-3 + X_OFF, y_t * 1e-3, z_t * 1e-3])
        R = self.R
        if kind == "sphere":
            n = 24000
            i = np.arange(n) + 0.5
            u = np.column_stack([-(1 - 2 * i / n), np.zeros(n), np.zeros(n)])
            r = np.sqrt(1 - u[:, 0] ** 2)
            th = math.pi * (3 - math.sqrt(5)) * i
            u[:, 1], u[:, 2] = r * np.cos(th), r * np.sin(th)
            keep = u[:, 0] < 0.35
            self.u = u[keep]
            self.X0 = self.c0 + R * self.u
            self.dA = np.full(len(self.u), 4 * math.pi * R ** 2 / n)
        elif kind == "cx":
            pm = min(1.0, 0.0056 / R)
            ph = np.linspace(-pm, pm, 241)
            yy = np.linspace(-0.0080, 0.0080, 161)
            PH, YY = np.meshgrid(ph, yy, indexing="ij")
            self.u = np.column_stack([-np.cos(PH).ravel(), np.zeros(PH.size), np.sin(PH).ravel()])
            self.X0 = np.column_stack([self.c0[0] + R * self.u[:, 0], YY.ravel(), self.c0[2] + R * self.u[:, 2]])
            self.dA = np.full(len(self.u), (ph[1] - ph[0]) * R * (yy[1] - yy[0]))
        else:
            pm = min(1.0, 0.0056 / R)
            ph = np.linspace(-pm, pm, 241)
            zz = np.linspace(TG.Z_END - 0.0002, TG.Z_MOUNT + 0.0002, 216)
            PH, ZZ = np.meshgrid(ph, zz, indexing="ij")
            self.u = np.column_stack([-np.cos(PH).ravel(), np.sin(PH).ravel(), np.zeros(PH.size)])
            self.X0 = np.column_stack([self.c0[0] + R * self.u[:, 0], self.c0[1] + R * self.u[:, 1], ZZ.ravel()])
            self.dA = np.full(len(self.u), (ph[1] - ph[0]) * R * (zz[1] - zz[0]))

    def field_touch(self, nf):
        """Shift along -x (m, negative: the probe starts inside) at which the probe's samples first reach the hull
        (planes n . (X - c) <= d_k of curved_tips_stiffness.NearField)."""
        n, dk, c = nf.n, nf.dk, nf.c
        up, dn = n[:, 0] > 1e-9, n[:, 0] < -1e-9
        flat = ~(up | dn)
        best = np.inf
        for s in range(0, len(self.X0), 256):
            Y = (self.X0[s:s + 256] - c) @ n.T - dk[None, :]           # > 0: outside that plane
            smin = np.max(np.where(up[None, :], Y / np.where(up, n[:, 0], 1.0)[None, :], -np.inf), axis=1)
            smax = np.min(np.where(dn[None, :], Y / np.where(dn, n[:, 0], 1.0)[None, :], np.inf), axis=1)
            ok = (smin <= smax) & np.all(np.where(flat[None, :], Y <= 1e-12, True), axis=1)
            if ok.any():
                best = min(best, float(smin[ok].min()))
        return best

    def pad_touch(self, C, rs):
        """Shift along -x (m) at which the probe first touches a pad sphere of radius rs."""
        Rr = self.R + rs
        if self.kind == "sphere":
            rho2 = (self.c0[1] - C[:, 1]) ** 2 + (self.c0[2] - C[:, 2]) ** 2
        elif self.kind == "cx":
            rho2 = (self.c0[2] - C[:, 2]) ** 2
        else:
            rho2 = (self.c0[1] - C[:, 1]) ** 2
        ok = rho2 < Rr ** 2
        return float(np.min(self.c0[0] - C[ok, 0] - np.sqrt(Rr ** 2 - rho2[ok])))

    def field(self, p_of, delta, idx):
        """Force along +x on the probe and the patch at approach delta for a pressure function p_of(X, idx)."""
        X = self.X0[idx] - np.array([delta, 0.0, 0.0])
        p = p_of(X, idx)
        w = p * self.dA[idx]                                # normal force per sample (the probe surface's normal)
        F = float(np.sum(w * (-self.u[idx, 0])))
        return F, patch(X[:, 1:], w)

    def pads(self, C, K, rs, delta):
        c = self.c0 - np.array([delta, 0.0, 0.0])
        if self.kind == "sphere":
            v = c - C
        elif self.kind == "cx":
            v = np.column_stack([c[0] - C[:, 0], np.zeros(len(C)), c[2] - C[:, 2]])
        else:
            v = np.column_stack([c[0] - C[:, 0], c[1] - C[:, 1], np.zeros(len(C))])
        dist = np.linalg.norm(v, axis=1)
        d = rs + self.R - dist
        m = d > 0
        if not m.any():
            return 0.0, patch(np.zeros((0, 2)), np.zeros(0))
        f = K[m] * d[m]
        P = C[m] + rs * v[m] / dist[m, None]
        return float(np.sum(f * v[m, 0] / dist[m])), patch(P[:, 1:], f)


def patch(yz, w):
    ok = w > 0
    if not ok.any():
        return dict(n=0, cy=float("nan"), cz=float("nan"), r_rms=float("nan"), r_bar=float("nan"))
    w, yz = w[ok], yz[ok]
    c = (w[:, None] * yz).sum(0) / w.sum()
    r = np.linalg.norm(yz - c, axis=1)
    return dict(n=int(ok.sum()), cy=float(c[0]), cz=float(c[1]), r_rms=float(np.sqrt((w * r * r).sum() / w.sum())),
                r_bar=float((w * r).sum() / w.sum()))


class Field:
    """Drake's compliant-convex field of the 0.25 mm hull at E0 (curved_tips_stiffness.NearField: p = E (1 - g), g the
    largest (X - c) . n_k / d_k over the facets k). The hull of an upright print has facets that run the length of a
    straight section, so a facet's centroid can lie far from the points it bounds; `prepare` finds, for each probe
    sample, the maximizing facet over all facets at six approaches along the grid, and `p` takes the maximum over
    those candidates (exact at the six approaches; `check` reports the force against the full maximum elsewhere)."""

    def __init__(self, sh):
        import curved_tips_stiffness as CS
        self.nf = CS.NearField(sh, E=E0)
        self.cand = None

    def full_g(self, X):
        nf = self.nf
        out = np.empty(len(X))
        arg = np.empty(len(X), int)
        for s in range(0, len(X), 512):
            G = (X[s:s + 512] - nf.c) @ nf.n.T / nf.dk[None, :]
            arg[s:s + 512] = np.argmax(G, axis=1)
            out[s:s + 512] = G[np.arange(len(G)), arg[s:s + 512]]
        return out, arg

    def prepare(self, X0, deltas):
        cols = [self.full_g(X0 - np.array([d, 0.0, 0.0]))[1] for d in deltas]
        self.cand = np.stack(cols, 1)
        self.X0_id = id(X0)

    def p_at(self, X, idx):
        nf = self.nf
        k = self.cand[idx]
        g = np.einsum("ij,ikj->ik", X - nf.c, nf.n[k]) / nf.dk[k]
        return E0 * np.clip(1 - g.max(1), 0, None)

    def p_full(self, X):
        return E0 * np.clip(1 - self.full_g(X)[0], 0, None)

class DepthField:
    """p = E0 d / d_max with d the depth below the 0.25 mm hull: the distance to the nearest surface sample (k-d tree of
    the shape's dense surface points, queries chunked at 4,096), d_max the largest depth (at the deepest interior
    sample); zero outside the hull."""

    def __init__(self, sh):
        from scipy.spatial import cKDTree
        P, _, _ = sh.dense
        self.tree = cKDTree(P)
        V, F = sh.hull(0.00025)
        self.n, self.d = TG.TipShape._planes(V, F)
        c = sh.centroid()
        g = np.stack(np.meshgrid(*(np.linspace(-0.006, 0.006, 25),) * 3, indexing="ij"), -1).reshape(-1, 3) + c
        g = g[self.inside(g)]
        self.dmax = float(self.tree.query(g)[0].max())

    def inside(self, X):
        out = np.empty(len(X), bool)
        for s in range(0, len(X), 4096):
            out[s:s + 4096] = np.all(X[s:s + 4096] @ self.n.T - self.d[None, :] <= 0, axis=1)
        return out

    def p(self, X):
        out = np.zeros(len(X))
        ins = self.inside(X)
        idx = np.where(ins)[0]
        for s in range(0, len(idx), 4096):
            k = idx[s:s + 4096]
            out[k] = E0 * self.tree.query(X[k])[0] / self.dmax
        return out


def cmd_curves(a):
    import fe_contact_calibration_fe as FE
    name = a.tip
    key = SIM_KEY[name]
    sh = TG.shape(key)
    shd = TG.shape(key.rsplit("_u", 1)[0].rsplit("_p", 1)[0])
    have = {(r["model"], r["probe"]) for r in rows(OUT / "curves.jsonl") if r["tip"] == name}
    models = [m for m in a.models if m != "depth" or name == "cap_g30"]
    t0 = time.time()
    field = Field(sh) if "drake" in models else None
    depth = DepthField(sh) if "depth" in models else None
    pads = {}
    for m in models:
        if m in PADS:
            s, rs, h, K = PADS[m]
            C, Kp = sh.pads(E0, s=s, rs=rs, h=h, K=K)
            pads[m] = (C, Kp, rs)
    print(f"{name} ({key}): models ready {time.time() - t0:.0f} s; pads " +
          ", ".join(f"{m} {len(v[0])}" for m, v in pads.items()), flush=True)
    for pname, kind, R, y_t, z_t in FE.probes():
        todo = [m for m in models if (m, pname) not in have]
        if not todo:
            continue
        t1 = time.time()
        pr = Probe(shd, kind, R, y_t, z_t)
        idx = None
        t_field = None
        if field is not None or depth is not None:
            nf = field.nf if field is not None else None
            if nf is None:
                import curved_tips_stiffness as CS
                nf = CS.NearField(sh, E=E0)
            t_field = pr.field_touch(nf)
            pr.X0 = pr.X0 - np.array([t_field, 0.0, 0.0])
        if field is not None:
            idx = np.where(field.p_full(pr.X0 - np.array([DGRID[-1], 0.0, 0.0])) > 0)[0]
            field.prepare(pr.X0[idx], DGRID[[0, 8, 16, 24, 32, 39]])
            pos = np.empty(len(pr.X0), int)
            pos[idx] = np.arange(len(idx))
            p_field = lambda X, ii: field.p_at(X, pos[ii])  # noqa: E731
            if pname in ("s_apex", "cx12.5_z0", "cz12.5_y0") and a.check:
                for dl in (DGRID[4], DGRID[12], DGRID[20], DGRID[28], DGRID[36]):
                    F1 = pr.field(p_field, dl, idx)[0]
                    F2 = pr.field(lambda X, ii: field.p_full(X), dl, idx)[0]
                    print(f"  field check {pname} delta {dl * 1e6:.1f} um: force {F1:.5g} N against the full maximum "
                          f"{F2:.5g} N ({100 * (F1 / F2 - 1) if F2 > 0 else 0:+.3f} %)", flush=True)
        t_pad = {m: pr.pad_touch(pads[m][0], pads[m][2]) for m in todo if m in pads}
        for m in todo:
            Fg, P = [], []
            for dl in DGRID:
                if m == "drake":
                    F, pt = pr.field(p_field, dl, idx)
                elif m == "depth":
                    F, pt = pr.field(lambda X, ii: depth.p(X), dl, idx if idx is not None else np.arange(len(pr.X0)))
                else:
                    C, Kp, rs = pads[m]
                    F, pt = pr.pads(C, Kp, rs, dl + t_pad[m])
                Fg.append(F)
                P.append(pt)
            row = dict(tip=name, key=key, model=m, probe=pname, kind=kind, R_mm=R, y_mm=y_t, z_mm=z_t, E0_Pa=E0,
                       delta_um=(DGRID * 1e6).round(4).tolist(), F_N=np.round(Fg, 7).tolist(),
                       r_rms_mm=[round(x["r_rms"] * 1e3, 5) for x in P], r_bar_mm=[round(x["r_bar"] * 1e3, 5) for x in P],
                       cy_mm=[round(x["cy"] * 1e3, 4) for x in P], cz_mm=[round(x["cz"] * 1e3, 4) for x in P],
                       n_contact=[x["n"] for x in P], script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"))
            row["first_touch_shift_um"] = (t_pad[m] if m in pads else t_field) * 1e6
            if m == "depth":
                row["d_max_mm"] = depth.dmax * 1e3
            append(OUT / "curves.jsonl", row)
        print(f"  {pname:10s} {', '.join(todo)} {time.time() - t1:.1f} s", flush=True)
    print(f"{name}: curves {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------------------------ fits

def curve_of(r):
    nz = lambda a: np.nan_to_num(np.array(a, float), nan=0.0)  # noqa: E731  (no contact: no patch)
    return dict(d=np.array(r["delta_um"]) * 1e-6, F=np.array(r["F_N"]), rbar=nz(r["r_bar_mm"]),
                rrms=nz(r["r_rms_mm"]), n=np.array(r["n_contact"]))


def at_load(c, F, E):
    """(approach m, r_bar mm, r_rms mm, contacts) of a curve at modulus E and load F (log-log interpolation)."""
    Fs = c["F"] * E / E0
    ok = Fs > 0
    lf, ld = np.log(Fs[ok]), np.log(c["d"][ok])
    keep = np.r_[True, np.diff(lf) > 0]
    lf, ld = lf[keep], ld[keep]
    x = math.log(F)
    d = float(np.exp(np.interp(x, lf, ld)))
    dd = c["d"][ok][keep]
    rb = float(np.interp(math.log(d), np.log(dd), c["rbar"][ok][keep]))
    rr = float(np.interp(math.log(d), np.log(dd), c["rrms"][ok][keep]))
    nn = float(np.interp(math.log(d), np.log(dd), c["n"][ok][keep]))
    return d, rb, rr, nn


def fe_rows():
    fe = {}
    for r in rows(D / "fe" / "fe.jsonl"):
        if r["grid"] == "base":
            fe[(r["tip"], r["probe"])] = r
    return fe


def evaluate(fe, cur, tip, E, ks=None, E_label=None):
    """Residuals of a model at modulus E (and series spring ks, N/m) against the finite elements."""
    app, arm, npad = {}, {}, {}
    for p in TASK:
        if (tip, p) not in fe or p not in cur:
            continue
        for F in FIT_F:
            d, rb, rr, nn = at_load(cur[p], F, E)
            d_tot = d + (F / ks if ks else 0.0)
            app[f"{p}@{F:g}"] = d_tot * 1e6 / fe[(tip, p)]["per_force"][f"{F:g}"]["approach_um"] - 1
    for p in ARM:
        if (tip, p) not in fe or p not in cur:
            continue
        for F in ARM_F:
            d, rb, rr, nn = at_load(cur[p], F, E)
            arm[f"{p}@{F:g}"] = rb / fe[(tip, p)]["per_force"][f"{F:g}"]["r_bar_mm"]
        npad[p] = at_load(cur[p], 1.0, E)[3]
    e = np.array(list(app.values()))
    ar = np.array(list(arm.values()))
    return dict(E_MPa=E / 1e6, k_s_N_per_mm=(ks / 1e3 if ks else None),
                approach_rms_pct=100 * float(np.sqrt(np.mean(e ** 2))) if len(e) else None,
                approach_max_pct=100 * float(np.max(np.abs(e))) if len(e) else None,
                approach_err_pct={k: round(100 * v, 2) for k, v in app.items()},
                arm_ratio={k: round(v, 4) for k, v in arm.items()},
                arm_min=float(ar.min()) if len(ar) else None, arm_max=float(ar.max()) if len(ar) else None,
                contacts_1N=npad,
                pass_approach=bool(len(e) and np.sqrt(np.mean(e ** 2)) <= 0.25 and np.max(np.abs(e)) <= 0.40),
                pass_arm=bool(len(ar) and ar.min() >= 0.70 and ar.max() <= 1.15))


def fit_E(fe, cur, tip, what, ks=None, E_loc=None):
    from scipy.optimize import minimize_scalar

    def loss(lnE):
        E = math.exp(lnE)
        out = []
        if what == "approach":
            for p in TASK:
                if (tip, p) in fe and p in cur:
                    for F in FIT_F:
                        d = at_load(cur[p], F, E)[0]
                        out.append(math.log(d * 1e6 / fe[(tip, p)]["per_force"][f"{F:g}"]["approach_um"]))
        else:
            for p in ARM:
                if (tip, p) in fe and p in cur:
                    for F in ARM_F:
                        rb = max(at_load(cur[p], F, E)[1], 1e-3)
                        out.append(math.log(rb / fe[(tip, p)]["per_force"][f"{F:g}"]["r_bar_mm"]))
        return float(np.mean(np.square(out)))
    grid = np.linspace(math.log(0.3e6), math.log(1e9), 121)
    vals = [loss(x) for x in grid]
    i = int(np.argmin(vals))
    lo, hi = grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]
    r = minimize_scalar(loss, bounds=(lo, hi), method="bounded", options=dict(xatol=1e-5))
    return math.exp(r.x), float(math.sqrt(r.fun))


def fit_ks(fe, cur, tip, E_loc, probes=None):
    from scipy.optimize import minimize_scalar

    def loss(lnk):
        k = math.exp(lnk)
        out = []
        for p in (probes or TASK):
            if (tip, p) in fe and p in cur:
                for F in FIT_F:
                    d = at_load(cur[p], F, E_loc)[0] + F / k
                    out.append(math.log(d * 1e6 / fe[(tip, p)]["per_force"][f"{F:g}"]["approach_um"]))
        return float(np.mean(np.square(out)))
    r = minimize_scalar(loss, bounds=(math.log(1e3), math.log(1e7)), method="bounded", options=dict(xatol=1e-6))
    return math.exp(r.x)


def _loglog(xs, ys, x):
    """Linear interpolation on log-log axes, extended linearly beyond the end points."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    u = math.log(x)
    if u <= lx[0]:
        return float(np.exp(ly[0] + (ly[1] - ly[0]) / (lx[1] - lx[0]) * (u - lx[0])))
    if u >= lx[-1]:
        return float(np.exp(ly[-1] + (ly[-1] - ly[-2]) / (lx[-1] - lx[-2]) * (u - lx[-1])))
    return float(np.exp(np.interp(u, lx, ly)))


def fe_at(row, F, lam=1.0):
    """(approach um, r_bar mm) of a finite-element row at load F for a solid modulus lam times the model's (12 MPa).
    Linear elasticity with frictionless contact: the displacement under load F at modulus lam E equals the one under F/lam
    at E, and the pressure scales with lam, so a(F; lam E) = a(F/lam; E) and r_bar(F; lam E) = r_bar(F/lam; E). The
    rows' loads (0.5, 1, 3, 6 N) are interpolated on log axes (r_bar on a log load axis), extended beyond the ends."""
    Fs = sorted(float(k) for k in row["per_force"])
    pf = {f: row["per_force"][f"{f:g}"] for f in Fs}
    x = F / lam
    a = _loglog(Fs, [pf[f]["approach_um"] for f in Fs], x)
    lF, rb = np.log(Fs), np.array([pf[f]["r_bar_mm"] for f in Fs], float)
    u = math.log(x)
    if u <= lF[0]:
        r = rb[0] + (rb[1] - rb[0]) / (lF[1] - lF[0]) * (u - lF[0])
    elif u >= lF[-1]:
        r = rb[-1] + (rb[-1] - rb[-2]) / (lF[-1] - lF[-2]) * (u - lF[-1])
    else:
        r = np.interp(u, lF, rb)
    return a, float(max(r, 1e-3))


def bench_target(fe, path):
    """The bench indentation rows in place of the finite elements as the fit target (plan.json, Future work).

    CSV columns: tip (block4_g30, cap_g30, ell16_g30, ell11_g30), probe (cx12.5_z0: the 25 mm rod across the finger at
    the contact point; s_apex: the 4 mm steel ball at the apex; any other probe name of fe/fe.jsonl), force_N, approach_um
    (from first touch), and optional repeat; repeats are averaged on a log scale. One solid-modulus factor lam for the
    material (Siraya 85A over the finite elements' 12 MPa) by log-RMS over every bench row; every tip's target is its
    finite elements at lam (fe_at), with the bench approach in place of the finite elements' where the bench measured it.
    Returns (target rows in the shape of fe_rows(), the target's description with lam per tip and the finite elements'
    bench residual after lam)."""
    import csv
    from scipy.optimize import minimize_scalar
    acc = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if not (r.get("tip") or "").strip():
                continue
            k = (r["tip"].strip(), r["probe"].strip(), float(r["force_N"]))
            if (k[0], k[1]) not in fe:
                raise SystemExit(f"{path}: no finite-element row for tip {k[0]!r}, probe {k[1]!r}")
            acc.setdefault(k, []).append(math.log(float(r["approach_um"])))
    if not acc:
        raise SystemExit(f"{path}: no bench rows")
    bench = {k: math.exp(float(np.mean(v))) for k, v in acc.items()}

    def lam_fit(keys):
        def loss(ll):
            return float(np.mean([(math.log(bench[k]) - math.log(fe_at(fe[k[:2]], k[2], math.exp(ll))[0])) ** 2
                                  for k in keys]))
        r = minimize_scalar(loss, bounds=(math.log(0.1), math.log(10.0)), method="bounded", options=dict(xatol=1e-6))
        return math.exp(r.x)
    lam = lam_fit(list(bench))
    desc = {"kind": "bench", "csv": str(path), "lambda": lam, "n_rows": sum(len(v) for v in acc.values()),
            "terms": {"lambda": "solid-modulus factor of Siraya 85A over the finite elements' 12 MPa, one per material, "
                                "by log-RMS of approach over every bench row",
                      "fe_residual_rms_pct": "RMS of (finite-element approach at lambda / bench approach - 1) over a tip's "
                                             "bench rows",
                      "exponent": "bench and finite-element force exponent n = ln(a(3 N) / a(0.5 N)) / ln 6 per probe"},
            "tips": {}}
    for tip in sorted({k[0] for k in bench}):
        keys = [k for k in bench if k[0] == tip]
        e = [bench[k] / fe_at(fe[k[:2]], k[2], lam)[0] - 1 for k in keys]
        ex = {}
        for p in sorted({k[1] for k in keys}):
            if (tip, p, 0.5) in bench and (tip, p, 3.0) in bench:
                ex[p] = {"bench": math.log(bench[(tip, p, 3.0)] / bench[(tip, p, 0.5)]) / math.log(6),
                         "fe_at_lambda": math.log(fe_at(fe[(tip, p)], 3.0, lam)[0] / fe_at(fe[(tip, p)], 0.5, lam)[0])
                         / math.log(6)}
        desc["tips"][tip] = {"lambda_own": lam_fit(keys), "fe_residual_rms_pct": 100 * float(np.sqrt(np.mean(np.square(e)))),
                             "exponent": ex, "rows": {f"{k[1]}@{k[2]:g}": bench[k] for k in keys}}
    out = {}
    for (tip, p), r in fe.items():
        pf = {}
        for F in sorted({*FIT_F, *ARM_F, 6.0}):
            a, rb = fe_at(r, F, lam)
            pf[f"{F:g}"] = {"approach_um": bench.get((tip, p, F), a), "r_bar_mm": rb}
        out[(tip, p)] = dict(r, per_force=pf)
    return out, desc


def cmd_fit(a):
    fe = fe_rows()
    target = {"kind": "finite elements", "rows": "fe/fe.jsonl (grid base)"}
    if getattr(a, "bench", None):
        fe, target = bench_target(fe, a.bench)
    cv = {}
    for r in rows(OUT / "curves.jsonl"):
        cv.setdefault((r["tip"], r["model"]), {})[r["probe"]] = curve_of(r)
    out = {"terms": {
        "E_MPa": "modulus of the model: Drake's hydroelastic modulus, or E in the pads' K_i = E a_i / h_i (model value)",
        "approach_rms_pct": "RMS of (model approach / finite-element approach - 1) over the task cylinders and 0.5, 1, 3 N",
        "approach_max_pct": "largest |model approach / finite-element approach - 1| over the same",
        "arm_ratio": "model r_bar over finite-element r_bar under the 8 and 12.5 mm cylinders across the finger at z 0, "
                     "1 and 3 N", "k_s_N_per_mm": "series spring between the DIP link and the tip, N/mm",
        "contacts_1N": "pads in contact (pad models) or field samples in contact at 1 N"},
        "criteria": {"approach": "RMS <= 25 %, largest <= 40 %", "twist_arm": "0.70 <= ratio <= 1.15"},
        "tips": {}, "script": SCRIPT}
    for tip in SIM_KEY:
        t = out["tips"].setdefault(tip, {"sim_key": SIM_KEY[tip], "U_E_MPa": U_E[tip] / 1e6, "models": {}})
        fe_arm = {p: {F: fe[(tip, p)]["per_force"][f"{F:g}"]["r_bar_mm"] for F in ARM_F} for p in ARM if (tip, p) in fe}
        fe_app = {p: {F: fe[(tip, p)]["per_force"][f"{F:g}"]["approach_um"] for F in FIT_F} for p in TASK if (tip, p) in fe}
        t["fe"] = {"r_bar_mm": fe_arm, "approach_um": fe_app}
        if (tip, "s_apex") in fe and (tip, "cx12.5_z0") in fe:
            ap = fe[(tip, "s_apex")]["per_force"]
            rd = fe[(tip, "cx12.5_z0")]["per_force"]
            t["fe"].update(exponent_sphere_apex=math.log(ap["3"]["approach_um"] / ap["0.5"]["approach_um"]) / math.log(6),
                           exponent_rod_z0=math.log(rd["3"]["approach_um"] / rd["0.5"]["approach_um"]) / math.log(6),
                           rod_over_sphere_1N=rd["1"]["approach_um"] / ap["1"]["approach_um"],
                           exponent_task={p: math.log(fe[(tip, p)]["per_force"]["3"]["approach_um"] /
                                                      fe[(tip, p)]["per_force"]["0.5"]["approach_um"]) / math.log(6)
                                          for p in TASK if (tip, p) in fe})
        for m in ("drake", "v2", "v2h", "v1", "depth"):
            cur = cv.get((tip, m))
            if not cur:
                continue
            res = {"U": evaluate(fe, cur, tip, U_E[tip])}
            E1, l1 = fit_E(fe, cur, tip, "approach")
            res["M1"] = dict(evaluate(fe, cur, tip, E1), loss=l1)
            E2, l2 = fit_E(fe, cur, tip, "arm")
            res["M2"] = dict(evaluate(fe, cur, tip, E2), loss=l2)
            ks = fit_ks(fe, cur, tip, E2)
            res["M3_own"] = dict(evaluate(fe, cur, tip, E2, ks))
            # outside the criteria: the spring fitted over the rods across the finger only (the bench tool's
            # orientation), evaluated over every task cylinder
            ksx = fit_ks(fe, cur, tip, E2, [p for p in TASK if p.startswith("cx")])
            res["M3x"] = dict(evaluate(fe, cur, tip, E2, ksx))
            ex = res["M3x"]["approach_err_pct"]
            cxe = np.array([v for k, v in ex.items() if k.startswith("cx")]) / 100
            cze = np.array([v for k, v in ex.items() if k.startswith("cz")]) / 100
            res["M3x"].update(cx_rms_pct=100 * float(np.sqrt(np.mean(cxe ** 2))), cx_max_pct=100 * float(np.max(np.abs(cxe))),
                              cz_range_pct=[100 * float(cze.min()), 100 * float(cze.max())] if len(cze) else None)
            # rule R3: r_bar change over E x0.5 to x2 about M1's modulus, under the 12.5 mm rod at 1 N
            if "cx12.5_z0" in cur:
                rb = [at_load(cur["cx12.5_z0"], 1.0, E1 * s)[1] for s in (0.5, 2.0)]
                res["R3_rbar_change"] = abs(rb[0] / rb[1] - 1)
            if "cx12.5_z0" in cur and "s_apex" in cur:
                ap = [at_load(cur["s_apex"], F, E1)[0] for F in (0.5, 3.0)]
                rd = [at_load(cur["cx12.5_z0"], F, E1)[0] for F in (0.5, 1.0, 3.0)]
                res["exponents_at_M1"] = dict(sphere_apex=math.log(ap[1] / ap[0]) / math.log(6),
                                              rod_z0=math.log(rd[2] / rd[0]) / math.log(6),
                                              rod_over_sphere_1N=rd[1] / at_load(cur["s_apex"], 1.0, E1)[0])
            t["models"][m] = res
        # one k_s per tip for both simulators: the mean of Drake's and the pads' (v2) fits
        ks_list = [t["models"][m]["M3_own"]["k_s_N_per_mm"] for m in ("drake", "v2") if m in t["models"]]
        if ks_list:
            ks = float(np.mean(ks_list)) * 1e3
            t["k_s_N_per_mm"] = ks / 1e3
            for m in ("drake", "v2", "v2h"):
                if m in t["models"]:
                    cur = cv[(tip, m)]
                    t["models"][m]["M3"] = evaluate(fe, cur, tip, t["models"][m]["M2"]["E_MPa"] * 1e6, ks)
        # rule R3: M2 equals M1 where the arm cannot set E (r_bar changes by less than 5 % over E x0.5 to x2); M3 then
        # puts M1's modulus behind the spring
        r3 = [t["models"][m].get("R3_rbar_change") for m in ("drake", "v2") if m in t["models"]]
        t["R3_applies"] = bool(r3 and all(x is not None and x < 0.05 for x in r3))
        t["calib_of"] = {"M2": "M1"} if t["R3_applies"] else {}
        if t["R3_applies"]:
            own = {}
            for m in ("drake", "v2", "v2h"):
                if m in t["models"]:
                    E1 = t["models"][m]["M1"]["E_MPa"] * 1e6
                    own[m] = fit_ks(fe, cv[(tip, m)], tip, E1)
            ks = float(np.mean([own[m] for m in ("drake", "v2") if m in own]))
            t["k_s_N_per_mm"] = ks / 1e3
            for m in own:
                t["models"][m]["M3"] = dict(evaluate(fe, cv[(tip, m)], tip, t["models"][m]["M1"]["E_MPa"] * 1e6, ks),
                                            E_loc_from="M1 (rule R3)")
        # rule R4: 0.5 mm pads where the 1 mm pads have fewer than 6 pads in contact at 1 N under the 8 or 12.5 mm
        # cylinder, or miss twist_arm where the 0.5 mm pads pass
        t["pad_model"] = {}
        for c in ("M1", "M2", "M3"):
            cc = t["calib_of"].get(c, c) if c != "M3" else c
            v2, vh = t["models"].get("v2", {}).get(cc), t["models"].get("v2h", {}).get(cc)
            if not v2:
                continue
            few = any(n < 6 for n in v2["contacts_1N"].values())
            t["pad_model"][c] = "v2h" if (vh and (few or (not v2["pass_arm"] and vh["pass_arm"]))) else "v2"
    out["when"] = time.strftime("%Y-%m-%d %H:%M")
    if target["kind"] != "finite elements":
        out["target"] = target
    dest = Path(a.out) if getattr(a, "out", None) else OUT / ("fits_bench.json" if target["kind"] == "bench" else "fits.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, dest)
    if target["kind"] == "bench":
        print(f"bench target {target['csv']}: lambda {target['lambda']:.4f} over {target['n_rows']} rows; "
              + "; ".join(f"{t}: own lambda {v['lambda_own']:.4f}, finite elements {v['fe_residual_rms_pct']:.1f} % RMS "
                          f"after lambda, exponents {json.dumps({p: {k: round(x, 3) for k, x in e.items()} for p, e in v['exponent'].items()})}"
                          for t, v in target["tips"].items()))
    print(f"wrote {dest}")
    for tip, t in out["tips"].items():
        print(tip, "FE", {k: (round(v, 3) if isinstance(v, float) else None) for k, v in t["fe"].items()
                          if k.startswith("exp") and not isinstance(v, dict) or k == "rod_over_sphere_1N"})
        for m, res in t["models"].items():
            for c in ("U", "M1", "M2", "M3"):
                if c in res:
                    v = res[c]
                    print(f"  {m:6s} {c}: E {v['E_MPa']:8.2f} MPa" + (f" k_s {v['k_s_N_per_mm']:.2f} N/mm" if v.get("k_s_N_per_mm") else "")
                          + f"  approach rms {v['approach_rms_pct']:.1f} % max {v['approach_max_pct']:.1f} %  arm "
                          f"{v['arm_min']:.2f}-{v['arm_max']:.2f}  contacts {v['contacts_1N']}  "
                          f"{'PASS' if v['pass_approach'] else 'fail'}/{'PASS' if v['pass_arm'] else 'fail'}")
            if "R3_rbar_change" in res:
                print(f"  {m:6s} R3 r_bar change over E x0.5-x2: {100 * res['R3_rbar_change']:.1f} %")
        print(f"  R3 {t.get('R3_applies')}  pad model per calibration {t.get('pad_model')}  k_s {t.get('k_s_N_per_mm')}")


def cmd_patchmaps(a):
    """The contact patch under the 12.5 mm rod across the finger at z 0 at 1 N for the figure: the Drake field's pressure
    on the probe samples and the pads' forces at the M1 and M2 moduli (fit/fits.json); fit/patch_<tip>.npz."""
    import fe_contact_calibration_fe as FE
    fits = json.loads((OUT / "fits.json").read_text())["tips"]
    for name in a.tips:
        key = SIM_KEY[name]
        sh = TG.shape(key)
        shd = TG.shape(key.rsplit("_u", 1)[0].rsplit("_p", 1)[0])
        field = Field(sh)
        _, kind, R, y_t, z_t = next(p for p in FE.probes() if p[0] == "cx12.5_z0")
        pr = Probe(shd, kind, R, y_t, z_t)
        pr.X0 = pr.X0 - np.array([pr.field_touch(field.nf), 0.0, 0.0])
        idx = np.where(field.p_full(pr.X0 - np.array([DGRID[-1], 0.0, 0.0])) > 0)[0]
        out = {}
        cv = {r["model"]: curve_of(r) for r in rows(OUT / "curves.jsonl") if r["tip"] == name and r["probe"] == "cx12.5_z0"}
        t = fits[name]
        for c in ("M1", "M2"):
            E = t["models"]["drake"][c]["E_MPa"] * 1e6
            d = at_load(cv["drake"], 1.0, E)[0]
            X = pr.X0[idx] - np.array([d, 0.0, 0.0])
            p = field.p_full(X) * E / E0
            ok = p > 0
            out[f"drake_{c}_yz"] = X[ok][:, 1:] * 1e3
            out[f"drake_{c}_p"] = p[ok]
            for pm in ("v2", "v2h"):
                Ep = t["models"][pm][c]["E_MPa"] * 1e6
                s_, rs, h, K = PADS[pm]
                C, Kp = sh.pads(Ep, s=s_, rs=rs, h=h, K=K)
                tp = pr.pad_touch(C, rs)
                dp = at_load(cv[pm], 1.0, Ep)[0]
                cc = pr.c0 - np.array([dp + tp, 0.0, 0.0])
                v = np.column_stack([cc[0] - C[:, 0], np.zeros(len(C)), cc[2] - C[:, 2]])
                dist = np.linalg.norm(v, axis=1)
                dd = rs + pr.R - dist
                m = dd > 0
                P = C[m] + rs * v[m] / dist[m, None]
                out[f"{pm}_{c}_yz"] = P[:, 1:] * 1e3
                out[f"{pm}_{c}_f"] = Kp[m] * dd[m]
        np.savez(OUT / f"patch_{name}.npz", **out)
        print(name, {k: v.shape for k, v in out.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curves")
    c.add_argument("--tip", required=True, choices=list(SIM_KEY))
    c.add_argument("--models", nargs="+", default=["drake", "v2", "v2h", "v1", "depth"])
    c.add_argument("--check", action="store_true", help="compare the k-nearest-facet gauge with the full maximum")
    ft = sub.add_parser("fit")
    ft.add_argument("--bench", default=None,
                    help="CSV of bench indentation rows (tip, probe, force_N, approach_um[, repeat]) that replace the finite "
                         "elements as the target where measured (bench_target); the other probes take the finite elements "
                         "at the fitted solid-modulus factor")
    ft.add_argument("--out", default=None, help="fits JSON to write (default fit/fits.json, or fit/fits_bench.json with "
                                                "--bench); a new study folder's fit/fits.json for fe_contact_calibration_sim.py "
                                                "--study")
    pm = sub.add_parser("patchmaps")
    pm.add_argument("--tips", nargs="+", default=list(SIM_KEY))
    a = ap.parse_args()
    {"curves": cmd_curves, "fit": cmd_fit, "patchmaps": cmd_patchmaps}[a.cmd](a)


if __name__ == "__main__":
    main()
