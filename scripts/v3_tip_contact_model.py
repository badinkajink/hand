#!/usr/bin/env python3
"""One modulus for the contact models of the four-magnet v3 fingertips, fitted to the forces of the UR5 presses (owner,
2026-10-30: "check if our calibration results on the v3 four n52 magnet fingertips enable a more consistent FE /
sphere-packed / hydroelastic contact model in simulation"; "stick to one hydroelastic modulus, and accept errors at the
extrema"; forces matter more than the indentation; "recover the surface area of contact/contact pose").

Holds: every press of the v3 sessions over 0.2 N (logs/<session>/holds.npz of scripts/ur5_tip_calibration_analysis.py):
v3 ID 1 and v3 ID 2 under the low 5 mm sphere, the tall 5 mm sphere and the 12.5 mm rod, and for comparison the old tip's
sphere2 and rod1 sessions of 2026-10-22/23, which the contact overview quoted until 2026-10-30. Each hold gives the
indenter's registered centre c_P in the pad frame (the HEX-E's deflection removed), the contact normal n, the registered
depth and the HEX-E's force vector.

Models, all at the indenter's registered position:
  pads   builder-v2 1 mm sphere pads on the FE surface points (ur5_tip_calibration_pads.Pads): K_i = E a_i / h_i, h_i the
         depth of the pad's volume centroid below surface point i
  field  Drake's compliant-convex pressure field p = E (1 - g), g the gauge of the pad's convex surface about the same
         centroid (the largest (X - c) . n_k / d_k over the surface triangles and the base), integrated over the part of
         the indenter's surface inside the pad (samples about 0.1 mm apart)
  FE     the four-magnet FE pad (b_05R4 surface basis; walls and skins at 12 MPa around the gyroid core), frictionless,
         with its compliance scaled by s; its modulus is quoted as 12 MPa / s
Every pads and field force scales with E, so one computation per hold gives every modulus (G, in N per MPa).

Fits and outputs (docs/experiments/20261030-v3_tip_contact_model/data/20261030-contact_model_fit.json):
  - E of the pads and of the field, and s of the FE, by the median log ratio of measured over model force, per session,
    per indenter, per tip and level, and one value over the low sphere and the rod of both tips (the "joint" modulus);
  - the force ratio measured / model at the joint modulus by level and region, and the depth residual: how far the
    indenter would move along n from its registered position for the model's force to equal the measured one;
  - the same with a series compliance a + c F added to the depth (modulus and spring fitted together), for comparison;
  - the patch of each model: pressure-weighted centre, area and mean radius r_bar about the centre (the friction arm);
  - the twists: torsion about the normal as half the out-and-back difference at each angle, against mu F r_bar of each
    model at the twist's force (mu from the shear page: 0.37 at 1 N, 0.44 from 2 N);
  - the angle between the measured force and each model's force, and the distance from the model's pressure centre to
    the registered contact point;
  - force-against-depth curves at the face targets of each session (median over targets) for the figures.

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 500 -- taskset -c 0-3 .venv/bin/python scripts/v3_tip_contact_model.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration_fe as FE  # noqa: E402
import ur5_tip_calibration_pads as PD  # noqa: E402

D = ROOT / "docs/experiments/20261030-v3_tip_contact_model"
OUT = D / "data/20261030-contact_model_fit.json"
BASIS4 = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261028-b_05R4_surface_basis.npz"
BASIS5 = FE.BASIS
LAB = ROOT / "docs/experiments/20261022-ur5_tip_calibration/lab"
OLD = ROOT / "docs/experiments/20261022-ur5_tip_calibration/data"
# (key, tip, indenter, holds folder, plan folder, FE basis)
SESSIONS = [
    ("v3id1_low", "v3 ID 1", "low", ROOT / "logs/20261028-spherev3id1_twists", LAB / "20261027-225213-spherev3id1", BASIS4),
    ("v3id2_low", "v3 ID 2", "low", ROOT / "logs/20261029-spherev3id2", LAB / "20261028-184415-spherev3id2", BASIS4),
    ("v3id1_tall", "v3 ID 1", "tall", ROOT / "logs/20261029-tallv3id1", LAB / "20261028-114908-tallv3id1", BASIS4),
    ("v3id2_tall", "v3 ID 2", "tall", ROOT / "logs/20261029-tallv3id2", LAB / "20261028-200508-tallv3id2", BASIS4),
    ("v3id1_rod", "v3 ID 1", "rod", ROOT / "logs/20261028-rodv3id1", LAB / "20261028-102701-rodv3id1", BASIS4),
    ("v3id2_rod", "v3 ID 2", "rod", ROOT / "logs/20261029-rodv3id2", LAB / "20261029-031559-rodv3id2", BASIS4),
    ("old_sphere2", "tip 0", "low", OLD / "20261022-210852-sphere2", LAB / "20261022-210852-sphere2", BASIS5),
    ("old_rod1", "tip 0", "rod", OLD / "20261023-104526-rod1-joint", LAB / "20261023-104526-rod1", BASIS5),
]
JOINT = ("v3id1_low", "v3id2_low", "v3id1_rod", "v3id2_rod")     # the one modulus: low sphere and rod, both v3 tips
MIN_N = 0.2
E_FE_MPA = 12.0                                                 # walls and skins of the FE (NinjaFlex 85A data sheet)
S_GRID = np.array([0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0])
MU = {1.0: 0.37, 2.5: 0.44, 5.0: 0.44}                          # shear page (cb2t slides): 0.37 at 1 N, 0.44 at 2 N
DEPTH_GRID = np.round(np.arange(0.0, 2.81, 0.04), 3)
SPACING = dict(sphere=0.1, rod=0.15)                            # field samples on the indenter, mm


# ------------------------------------------------------------------------------------------------------------ models
class Field:
    """Drake's compliant-convex field of the pad: p / E = 1 - g, g = max over the surface triangles' planes and the base
    plane of (X - c) . n_k / d_k, c the pad solid's node centroid (the pads' c_V)."""

    def __init__(self, basis):
        B = np.load(basis)
        P, N, T = B["pts"].astype(float), B["normals"].astype(float), B["tris"]
        self.c = B["node_centroid"].astype(float)
        v0, v1, v2 = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        n *= np.sign(np.einsum("ij,ij->i", n, N[T].mean(1)))[:, None]
        nb = np.array([[-1.0, 0.0, 0.0]])                       # the base, x_P = x_cut
        self.n = np.r_[n, nb]
        self.d = np.r_[np.einsum("ij,ij->i", n, v0 - self.c), self.c[0] - float(B["x_cut"])]
        self.mid = np.r_[(v0 + v1 + v2) / 3.0, [[float(B["x_cut"]), self.c[1], self.c[2]]]]
        self.Nn = self.n / self.d[:, None]
        self.lo, self.hi = P.min(0) - 0.2, P.max(0) + 0.2         # the pad's bounding box

    def planes(self, p, r=9.0, kind="sphere"):
        """Indices of the planes whose triangle lies within r of the point p (the base always); all planes for a rod,
        whose line contact can reach farther."""
        if kind == "rod":
            return None
        k = np.nonzero(np.linalg.norm(self.mid - p, axis=1) < r)[0]
        return np.unique(np.r_[k, len(self.n) - 1])

    def p(self, X, k=None):
        """Pressure per MPa (N/mm^2 per MPa) at the points X (mm, pad frame), over the planes k (all when None)."""
        Nn = self.Nn if k is None else self.Nn[k]
        g = ((X - self.c) @ Nn.T).max(axis=1)
        return np.clip(1.0 - g, 0.0, None)


def fib_cap(R, cos_min, spacing):
    """Points on a sphere of radius R about the origin, within the cap z/R >= cos_min, about `spacing` apart: unit
    directions (outward normals) and the area of each point (mm^2)."""
    area = 2 * np.pi * R * R * (1 - cos_min)
    n = max(int(area / spacing ** 2), 200)
    k = np.arange(n) + 0.5
    z = 1 - (1 - cos_min) * k / n
    phi = np.pi * (1 + 5 ** 0.5) * k
    r = np.sqrt(1 - z * z)
    U = np.c_[r * np.cos(phi), r * np.sin(phi), z]
    return U, np.full(n, area / n)


def rot_z_to(v):
    """Rotation matrix taking +z to the unit vector v."""
    v = np.asarray(v, float) / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    a = np.cross(z, v)
    s = np.linalg.norm(a)
    a /= s
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def indenter_samples(c, n, R, axis, reach):
    """Surface samples of the indenter facing the pad (outward normal toward -n), reaching `reach` mm toward the pad
    from the indenter's nearest point: points X, outward normals U and areas A (mm^2)."""
    if axis is None:
        cos_min = max(1.0 - reach / R, -1.0)
        U0, A = fib_cap(R, cos_min, SPACING["sphere"])
        Rm = rot_z_to(-np.asarray(n, float))
        U = U0 @ Rm.T
        return c + R * U, U, A
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    e1 = -np.asarray(n, float) - float(-np.asarray(n, float) @ a) * a
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    phm = float(np.arccos(max(1.0 - reach / R, -1.0)))
    nph = max(int(2 * phm * R / SPACING["rod"]), 8)
    ph = (np.arange(nph) + 0.5) / nph * 2 * phm - phm
    L = 20.0
    nt = int(2 * L / SPACING["rod"])
    t = (np.arange(nt) + 0.5) / nt * 2 * L - L
    PH, TT = np.meshgrid(ph, t, indexing="ij")
    U = np.cos(PH).reshape(-1, 1) * e1 + np.sin(PH).reshape(-1, 1) * e2
    X = c + R * U + TT.reshape(-1, 1) * a
    A = np.full(len(X), (2 * phm * R / nph) * (2 * L / nt))
    return X, U, A


def field_contact(field, c, n, R, axis, reach, planes=None):
    """The field's force on the indenter per MPa (N, pad frame), its patch: pressure-weighted centre, area (mm^2) and the
    pressure-weighted mean distance from the centre in the tangent plane of n (r_bar, mm)."""
    X, U, A = indenter_samples(c, n, R, axis, reach)
    bb = np.all((X > field.lo) & (X < field.hi), axis=1)
    X, U, A = X[bb], U[bb], A[bb]
    ins = ((X - field.c) @ (field.Nn if planes is None else field.Nn[planes]).T).max(axis=1) < 1.0
    if not ins.any():
        return np.zeros(3), None, 0.0, np.nan
    X, U, A = X[ins], U[ins], A[ins]
    p = field.p(X, planes)
    w = p * A
    F = -(w[:, None] * U).sum(0)
    if w.sum() <= 0:
        return np.zeros(3), None, 0.0, np.nan
    cen = (w[:, None] * X).sum(0) / w.sum()
    return F, cen, float(A[p > 0].sum()), r_bar(X, w, cen, n)


def r_bar(X, w, cen, n):
    v = X - cen
    v = v - (v @ n)[:, None] * n[None]
    return float((w * np.linalg.norm(v, axis=1)).sum() / w.sum())


def pads_contact(pads, c, n, R, axis):
    """The pads' force on the indenter per MPa, their force-weighted centre, pads in contact and r_bar."""
    v = c - pads.C
    if axis is not None:
        a = np.asarray(axis, float) / np.linalg.norm(axis)
        v = v - (v @ a)[:, None] * a[None]
    d = np.linalg.norm(v, axis=1)
    o = PD.RS + R - d
    m = o > 0
    if not m.any():
        return np.zeros(3), None, 0, np.nan
    u = v[m] / d[m, None]
    f = pads.k[m] * o[m]
    P = pads.C[m] + PD.RS * u
    cen = (f[:, None] * P).sum(0) / f.sum()
    return (f[:, None] * u).sum(0), cen, int(m.sum()), r_bar(P, f, cen, n)


class FEPatch(FE.FEPad):
    def contact(self, c, s=1.0, R=5.0, axis=None):
        """FE.FEPad.force with the loaded nodes: force on the pad (N), centre, nodes loaded, r_bar (needs n: set .n)."""
        r = self.X - c
        if axis is not None:
            a = np.asarray(axis, float) / np.linalg.norm(axis)
            r = r - (r @ a)[:, None] * a[None]
        dist = np.linalg.norm(r, axis=1)
        gap = dist - R
        nh = r / dist[:, None]
        K = np.nonzero((gap < 0.6) & (np.einsum("ij,ij->i", nh, self.N) < -0.2))[0]
        if len(K) == 0 or gap[K].min() >= 0:
            return np.zeros(3), None, 0, np.nan
        idx = (3 * K[:, None] + np.arange(3)[None]).ravel()
        Ck = self.C[np.ix_(idx, idx)] * s
        Nk = np.zeros((3 * len(K), len(K)))
        for j, i in enumerate(K):
            Nk[3 * j:3 * j + 3, j] = nh[i]
        Mn = Nk.T @ Ck @ Nk
        Mn = 0.5 * (Mn + Mn.T) + 1e-9 * np.eye(len(K))
        Lc = np.linalg.cholesky(Mn)
        p, _ = nnls(Lc.T, -np.linalg.solve(Lc, gap[K]))
        F = (Nk @ p).reshape(-1, 3)
        if p.sum() <= 0:
            return np.zeros(3), None, 0, np.nan
        cen = (p[:, None] * self.X[K]).sum(0) / p.sum()
        return F.sum(axis=0), cen, int((p > 1e-9).sum()), r_bar(self.X[K], p, cen, self.n_hold)


# ------------------------------------------------------------------------------------------------------------ helpers
def lmed(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def ratio_stats(r):
    """Median and quartiles of a ratio, and the share within 25 % of 1."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r) & (r > 0)]
    if not len(r):
        return None
    return dict(n=int(len(r)), median=float(np.median(r)), p25=float(np.percentile(r, 25)), p75=float(np.percentile(r, 75)),
                within25=float(np.mean(np.abs(np.log(r)) < np.log(1.25))), mad_log=float(np.median(np.abs(np.log(r)))))


def bisect_depth(fun, f_target, lo, hi, it=36):
    """Offset t along n (outward positive) at which fun(t) (force, decreasing in t) equals f_target."""
    if fun(lo) < f_target:
        return np.nan
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if fun(mid) > f_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------------------------------------------------ main
def make_ctx(key):
    _, tip, ind, hdir, pdir, basis = next(s for s in SESSIONS if s[0] == key)
    H = np.load(hdir / "holds.npz", allow_pickle=False)
    g = json.loads((pdir / "plan.json").read_text())["geometry"]["indenter"]
    return dict(R=float(g["radius"]), kind=g.get("kind", "sphere"), H=H, pads=PD.Pads(basis=basis), field=Field(basis),
                fe=FEPatch(basis))


def session_rows(key):
    """Every press over MIN_N with the three models at its registered position (phase 1, one worker per session)."""
    t0 = time.time()
    ctx = make_ctx(key)
    H, R, kind, pads, field, fe = (ctx[k] for k in ("H", "R", "kind", "pads", "field", "fe"))
    press = np.nonzero((H["kind"] == "press") & np.isfinite(H["f_n"]) & (H["f_n"] > MIN_N))[0]
    rows = []
    for i in press:
        c, n = H["c_P"][i].astype(float), H["n"][i].astype(float)
        ax = H["axis_P"][i].astype(float) if kind == "rod" else None
        dep = float(H["depth"][i])
        pl = field.planes(H["p"][i], kind=kind)
        Fp, cp, npd, rp = pads_contact(pads, c, n, R, ax)
        Ff, cf, af, rf = field_contact(field, c, n, R, ax, max(dep, 0.0) + 0.6, pl)
        fe.n_hold = n
        fe_F = [float(-fe.contact(c, s, R, ax)[0] @ n) for s in S_GRID]
        rows.append(dict(i=int(i), id=str(H["id"][i]), region=str(H["region"][i]), level=float(H["level_N"][i]),
                         f_n=float(H["f_n"][i]), f=H["f"][i].astype(float).tolist(), depth=dep, c=c.tolist(), n=n.tolist(),
                         p=H["p"][i].astype(float).tolist(), axis=None if ax is None else ax.tolist(),
                         Gp=float(Fp @ n), Gp_vec=Fp.tolist(), cen_p=None if cp is None else cp.tolist(), n_pads=npd, rbar_p=rp,
                         Gf=float(Ff @ n), Gf_vec=Ff.tolist(), cen_f=None if cf is None else cf.tolist(), area_f=af, rbar_f=rf,
                         fe=fe_F))
    print(f"{key}: {len(rows)} presses over {MIN_N} N ({time.time() - t0:.0f} s)", flush=True)
    return key, rows


def fit_log(meas, G):
    ok = (G > 0) & np.isfinite(G)
    return float(np.exp(np.median(np.log(meas[ok] / G[ok])))), int((~ok).sum())


def fe_force_at(rows, s):
    """FE force of each hold at compliance scale s, by log interpolation over S_GRID (0 where the FE misses)."""
    out = []
    for r in rows:
        F = np.asarray(r["fe"], float)
        if (F <= 0).all():
            out.append(0.0)
            continue
        lF = np.log(np.maximum(F, 1e-9))
        out.append(float(np.exp(np.interp(np.log(s), np.log(S_GRID), lF))))
    return np.array(out)


def fit_fe(rows):
    meas = np.array([r["f_n"] for r in rows])
    grid = np.geomspace(S_GRID[0], S_GRID[-1], 120)
    best = None
    for s in grid:
        F = fe_force_at(rows, s)
        ok = F > 0
        cost = float(np.median(np.abs(np.log(meas[ok] / F[ok]))))
        mb = float(np.median(np.log(meas[ok] / F[ok])))
        if best is None or abs(mb) < abs(best[1]):
            best = (float(s), mb, cost)
    return best


def curves(rows, ctx, E_p, E_f, s_fe):
    """Force along n against depth at the face targets (first hold per target): per MPa for pads and field, the FE at
    s = 1 and at s_fe, and each target's measured holds; medians over targets."""
    seen, first = set(), []
    for r in rows:
        if r["region"] == "face" and r["id"] not in seen:
            seen.add(r["id"])
            first.append(r)
    R, ax_kind = ctx["R"], ctx["kind"]
    Gp = np.zeros((len(first), len(DEPTH_GRID)))
    Gf = np.zeros_like(Gp)
    F1 = np.zeros_like(Gp)
    Fs = np.zeros_like(Gp)
    rb = {m: np.full_like(Gp, np.nan) for m in ("pads", "field", "fe")}
    for j, r in enumerate(first):
        c0, n, d0 = np.array(r["c"]), np.array(r["n"]), r["depth"]
        ax = np.array(r["axis"]) if ax_kind == "rod" else None
        pl = ctx["field"].planes(np.array(r["p"]), kind=ctx["kind"])
        ctx["fe"].n_hold = n
        for k, dd in enumerate(DEPTH_GRID):
            c = c0 + (d0 - dd) * n
            Fp, _, _, rp = pads_contact(ctx["pads"], c, n, R, ax)
            Ff, _, _, rf = field_contact(ctx["field"], c, n, R, ax, dd + 0.6, pl)
            Gp[j, k], Gf[j, k] = max(float(Fp @ n), 0.0), max(float(Ff @ n), 0.0)
            rb["pads"][j, k], rb["field"][j, k] = rp, rf
            a, _, _, _ = ctx["fe"].contact(c, 1.0, R, ax)
            b, _, _, rfe = ctx["fe"].contact(c, s_fe, R, ax)
            F1[j, k], Fs[j, k] = max(float(-a @ n), 0.0), max(float(-b @ n), 0.0)
            rb["fe"][j, k] = rfe
    def med(A):
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                m = np.nanmedian(A, axis=0)
        return [None if not np.isfinite(x) else round(float(x), 6) for x in m]
    return dict(n_targets=len(first), depth_mm=DEPTH_GRID.tolist(), pads_N_per_MPa=med(Gp), field_N_per_MPa=med(Gf),
                fe_s1_N=med(F1), fe_fit_N=med(Fs), rbar_mm={m: med(v) for m, v in rb.items()},
                per_target=dict(pads=np.round(Gp, 6).tolist(), field=np.round(Gf, 6).tolist()))


def depth_residual(rows, ctx, E, model):
    """Offset along n (mm, positive = the print sinks deeper than the model) at which the model's force at modulus E
    equals the measured force, per hold."""
    R = ctx["R"]
    out = []
    for r in rows:
        c, n = np.array(r["c"]), np.array(r["n"])
        ax = np.array(r["axis"]) if r["axis"] is not None else None
        if model == "pads":
            fun = lambda t: E * float(pads_contact(ctx["pads"], c + t * n, n, R, ax)[0] @ n)  # noqa: E731
        else:
            pl = ctx["field"].planes(np.array(r["p"]), kind=ctx["kind"])
            fun = lambda t: E * float(field_contact(ctx["field"], c + t * n, n, R, ax,  # noqa: E731
                                                    max(r["depth"] - t, 0.0) + 0.6, pl)[0] @ n)
        out.append(bisect_depth(fun, r["f_n"], -1.5, max(r["depth"], 0.0) + 0.6, it=24))
    return np.array(out)


def residual_job(args):
    """Depth residual of the pads at modulus E for every hold of one session (one worker per session and modulus)."""
    key, rows, E = args
    return key, float(E), depth_residual(rows, make_ctx(key), float(E), "pads")


def spring_from(resid, rows_by, keys, grid):
    """Modulus E and a series compliance a + c F fitted together to the registered depth of the holds of `keys`: for each
    E on the grid the pads' depth residual of every hold, then a and c by least squares; the E with the smallest RMS.
    Returns E, a, c, RMS (mm) and the RMS of the best E alone (a = c = 0)."""
    F = np.concatenate([[r["f_n"] for r in rows_by[k]] for k in keys])
    best, best0, table = None, None, []
    for E in grid:
        dl = np.concatenate([resid[(k, float(E))] for k in keys])
        ok = np.isfinite(dl)
        X = np.c_[np.ones(ok.sum()), F[ok]]
        ab, *_ = np.linalg.lstsq(X, dl[ok], rcond=None)
        rms = float(np.sqrt(np.mean((X @ ab - dl[ok]) ** 2)))
        rms0 = float(np.sqrt(np.mean(dl[ok] ** 2)))
        table.append(dict(E_MPa=float(E), a_mm=float(ab[0]), c_mm_N=float(ab[1]), rms_mm=rms, rms_alone_mm=rms0, n=int(ok.sum())))
        if best is None or rms < best["rms_mm"]:
            best = table[-1]
        if best0 is None or rms0 < best0["rms_alone_mm"]:
            best0 = table[-1]
    return dict(E_MPa=best["E_MPa"], a_mm=best["a_mm"], c_mm_N=best["c_mm_N"], rms_mm=best["rms_mm"], n=best["n"],
                alone=dict(E_MPa=best0["E_MPa"], rms_mm=best0["rms_alone_mm"], n=best0["n"]), grid=table)


def twists(H, ctx, E_p, E_f, s_fe):
    """Torsion about the contact normal in the twists: half the difference of the out and back holds at the same angle,
    signed to resist the twist; with each model's friction arm r_bar at the hold's registered position."""
    m = np.nonzero(H["kind"] == "twist")[0]
    by = {}
    for i in m:
        by.setdefault(str(H["id"][i]), []).append(i)
    R = ctx["R"]
    out = []
    for aid, ii in by.items():
        ii = sorted(ii, key=lambda i: float(H["t"][i]))
        used = set()
        for a, i in enumerate(ii):
            ang = round(float(H["turn_deg"][i]))
            if ang == 0 or i in used:
                continue
            for j in ii[a + 1:]:
                if round(float(H["turn_deg"][j])) == ang and j not in used:
                    used.update((i, j))
                    M = 0.5 * (float(H["tn"][i]) - float(H["tn"][j])) * np.sign(ang)
                    F = 0.5 * (float(H["f_n"][i]) + float(H["f_n"][j]))
                    c, n = H["c_P"][i].astype(float), H["n"][i].astype(float)
                    ax = H["axis_P"][i].astype(float) if ctx["kind"] == "rod" else None
                    # each model at the depth where its force equals the hold's measured force
                    arms = {}
                    for model, E in (("pads", E_p), ("field", E_f)):
                        if model == "pads":
                            fun = lambda t: E * float(pads_contact(ctx["pads"], c + t * n, n, R, ax)[0] @ n)  # noqa: E731
                        else:
                            pl = ctx["field"].planes(H["p"][i], kind=ctx["kind"])
                            fun = lambda t: E * float(field_contact(ctx["field"], c + t * n, n, R, ax,  # noqa: E731
                                                                    max(float(H["depth"][i]) - t, 0.0) + 0.6, pl)[0] @ n)
                        t = bisect_depth(fun, F, -1.5, max(float(H["depth"][i]), 0.0) + 0.6, it=30)
                        if not np.isfinite(t):
                            arms[model] = None
                            continue
                        cc = c + t * n
                        if model == "pads":
                            arms[model] = pads_contact(ctx["pads"], cc, n, R, ax)[3]
                        else:
                            arms[model] = field_contact(ctx["field"], cc, n, R, ax, max(float(H["depth"][i]) - t, 0.0) + 0.6, pl)[3]
                    out.append(dict(id=aid, region=str(H["region"][i]), level=float(H["level_N"][i]), angle=ang, M_Nmm=M, f_n=F,
                                    arm_pads_mm=arms["pads"], arm_field_mm=arms["field"]))
                    break
    return out


def session_detail(args):
    """Phase 2 of one session at the joint modulus: force ratios, depth residuals, force direction, pressure centre,
    the curves at the face targets and the twists (curves and twists for the v3 sessions only)."""
    k, rows, Ep, Ef, sf, E_own = args
    t0 = time.time()
    ctx = make_ctx(k)
    meas = np.array([r["f_n"] for r in rows])
    Gp = np.array([r["Gp"] for r in rows])
    Gf = np.array([r["Gf"] for r in rows])
    Fe = fe_force_at(rows, sf)
    lv = np.array([r["level"] for r in rows])
    rg = np.array([r["region"] for r in rows])
    with np.errstate(divide="ignore", invalid="ignore"):
        rp = {m: meas / (E * G) for m, E, G in (("pads", Ep, Gp), ("field", Ef, Gf))}
        rp["fe"] = meas / np.where(Fe > 0, Fe, np.nan)
        rp["pads_own"] = meas / (E_own * Gp)
    d_p = depth_residual(rows, ctx, Ep, "pads")
    nn = np.array([r["n"] for r in rows])
    f_m = np.array([r["f"] for r in rows])
    um = f_m / np.linalg.norm(f_m, axis=1, keepdims=True)
    ang = lambda U: np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", um, U), -1, 1)))  # noqa: E731
    unit = lambda V: V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)  # noqa: E731
    a_geo = ang(-nn)
    a_p = ang(-unit(np.array([r["Gp_vec"] for r in rows])))
    a_f = ang(-unit(np.array([r["Gf_vec"] for r in rows])))
    pt = np.array([r["p"] for r in rows])
    def off(key):
        """Distance in the tangent plane of n from the model's pressure centre to the registered contact point (mm)."""
        out_ = []
        for r, q in zip(rows, pt):
            if r[key] is None:
                out_.append(np.nan)
                continue
            v = np.array(r[key]) - q
            v = v - (v @ np.array(r["n"])) * np.array(r["n"])
            out_.append(float(np.linalg.norm(v)))
        return np.array(out_)
    o_p, o_f = off("cen_p"), off("cen_f")
    tang = np.linalg.norm(f_m - np.einsum("ij,ij->i", f_m, nn)[:, None] * nn, axis=1) / meas
    regions = sorted(set(rg.tolist()))
    det = dict(
        by_level={f"{L:g}": {m: ratio_stats(v[lv == L]) for m, v in rp.items()} | dict(
            depth_mm=lmed([r["depth"] for r, x in zip(rows, lv) if x == L]), depth_resid_pads_um=lmed(d_p[lv == L] * 1e3))
            for L in sorted(set(lv.tolist()))},
        by_region={g_: {m: ratio_stats(v[rg == g_]) for m, v in rp.items()} for g_ in regions},
        all={m: ratio_stats(v) for m, v in rp.items()},
        depth_resid_mm=dict(pads=dict(median=lmed(d_p), rms=float(np.sqrt(np.nanmean(d_p ** 2))), n=int(np.isfinite(d_p).sum()))),
        direction_deg=dict(geometric=dict(median=lmed(a_geo), p90=float(np.nanpercentile(a_geo, 90))),
                           pads=dict(median=lmed(a_p), p90=float(np.nanpercentile(a_p, 90))),
                           field=dict(median=lmed(a_f), p90=float(np.nanpercentile(a_f, 90))),
                           tangential_over_normal=dict(median=lmed(tang), p90=float(np.nanpercentile(tang, 90))),
                           by_region={g_: dict(geometric=lmed(a_geo[rg == g_]), pads=lmed(a_p[rg == g_]), field=lmed(a_f[rg == g_]),
                                               tangential_over_normal=lmed(tang[rg == g_])) for g_ in regions}),
        centre_offset_um=dict(pads=lmed(o_p * 1e3), field=lmed(o_f * 1e3),
                              by_region={g_: dict(pads=lmed(o_p[rg == g_] * 1e3), field=lmed(o_f[rg == g_] * 1e3)) for g_ in regions}),
        holds=dict(id=[r["id"] for r in rows], region=rg.tolist(), level=lv.tolist(), f_n=np.round(meas, 4).tolist(),
                   depth=[round(r["depth"], 4) for r in rows], Gp=np.round(Gp, 6).tolist(), Gf=np.round(Gf, 6).tolist(),
                   fe_fit=np.round(Fe, 5).tolist(), fe_s1=[round(r["fe"][1], 5) for r in rows],
                   rbar_p=[None if not np.isfinite(r["rbar_p"]) else round(r["rbar_p"], 4) for r in rows],
                   rbar_f=[None if not np.isfinite(r["rbar_f"]) else round(r["rbar_f"], 4) for r in rows],
                   area_f=[round(r["area_f"], 3) for r in rows], n_pads=[r["n_pads"] for r in rows],
                   depth_resid_pads=[None if not np.isfinite(x) else round(float(x), 5) for x in d_p],
                   angle_geo=np.round(a_geo, 2).tolist(), angle_pads=np.round(a_p, 2).tolist(), angle_field=np.round(a_f, 2).tolist()))
    if k.startswith("v3"):
        det["curves"] = curves(rows, ctx, Ep, Ef, sf)
        det["twists"] = twists(ctx["H"], ctx, Ep, Ef, sf)
    a_ = det["all"]
    print(f"{k}: at the joint modulus measured/model median pads {a_['pads']['median']:.3f} (within 25 % "
          f"{a_['pads']['within25']:.2f}), field {a_['field']['median']:.3f}, FE {a_['fe']['median'] if a_['fe'] else None}; "
          f"depth residual pads median {det['depth_resid_mm']['pads']['median'] * 1e3:.0f} um, rms "
          f"{det['depth_resid_mm']['pads']['rms'] * 1e3:.0f}; direction geo/pads/field {det['direction_deg']['geometric']['median']:.1f}/"
          f"{det['direction_deg']['pads']['median']:.1f}/{det['direction_deg']['field']['median']:.1f} deg; centre offset pads "
          f"{det['centre_offset_um']['pads']:.0f} um ({time.time() - t0:.0f} s)", flush=True)
    return k, det


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None, help="session keys (default: all)")
    ap.add_argument("--no-spring", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    sess = [s for s in SESSIONS if a.only is None or s[0] in a.only]
    keys = [s[0] for s in sess]
    ex = ProcessPoolExecutor(max_workers=a.jobs, mp_context=mp.get_context("fork"))
    rows_by = dict(ex.map(session_rows, keys))
    meta = {}
    for key, tip, ind, hdir, pdir, basis in sess:
        g = json.loads((pdir / "plan.json").read_text())["geometry"]["indenter"]
        meta[key] = dict(tip=tip, indenter=ind, holds=str(hdir.relative_to(ROOT)), plan=str((pdir / "plan.json").relative_to(ROOT)),
                         basis=str(basis.relative_to(ROOT)), R_mm=float(g["radius"]), kind=g.get("kind", "sphere"),
                         n=len(rows_by[key]))
    out = dict(script="scripts/v3_tip_contact_model.py", made=time.strftime("%Y-%m-%d %H:%M"), min_N=MIN_N, E_FE_MPa=E_FE_MPA,
               s_grid=S_GRID.tolist(), mu={f"{k:g}": v for k, v in MU.items()}, joint=list(JOINT), sessions=meta, fits={})

    def fits_of(ks):
        meas = np.concatenate([[r["f_n"] for r in rows_by[k]] for k in ks])
        Gp = np.concatenate([[r["Gp"] for r in rows_by[k]] for k in ks])
        Gf = np.concatenate([[r["Gf"] for r in rows_by[k]] for k in ks])
        lv = np.concatenate([[r["level"] for r in rows_by[k]] for k in ks])
        Ep, mp_ = fit_log(meas, Gp)
        Ef, mf = fit_log(meas, Gf)
        s, mb, cost = fit_fe([r for k in ks for r in rows_by[k]])
        by_level = {}
        for L in sorted(set(lv.tolist())):
            m = lv == L
            if m.sum() >= 5:
                by_level[f"{L:g}"] = dict(n=int(m.sum()), E_pads_MPa=fit_log(meas[m], Gp[m])[0], E_field_MPa=fit_log(meas[m], Gf[m])[0])
        return dict(n=int(len(meas)), E_pads_MPa=Ep, miss_pads=mp_, E_field_MPa=Ef, miss_field=mf, s_fe=s,
                    E_fe_MPa=E_FE_MPA / s, fe_median_abs_log=cost, by_level=by_level)
    groups = {k: [k] for k in rows_by}
    for ind in ("low", "tall", "rod"):
        ks = [k for k in rows_by if k.startswith("v3") and meta[k]["indenter"] == ind]
        if ks:
            groups[f"v3_{ind}"] = ks
    for tip in ("v3 ID 1", "v3 ID 2"):
        ks = [k for k in JOINT if k in rows_by and meta[k]["tip"] == tip]
        if ks:
            groups[f"{tip.replace(' ', '').lower()}_low_rod"] = ks
    if all(k in rows_by for k in JOINT):
        groups["joint"] = list(JOINT)
    for g_, ks in groups.items():
        out["fits"][g_] = fits_of(ks)
        f = out["fits"][g_]
        print(f"fit {g_:16s} n {f['n']:4d}: pads {f['E_pads_MPa']:.3f} MPa (miss {f['miss_pads']}), field {f['E_field_MPa']:.3f} "
              f"MPa (miss {f['miss_field']}), FE s {f['s_fe']:.2f} = {f['E_fe_MPa']:.2f} MPa", flush=True)
    J = out["fits"].get("joint") or out["fits"][keys[0]]
    Ep, Ef, sf = J["E_pads_MPa"], J["E_field_MPa"], J["s_fe"]
    jobs = [(k, rows_by[k], Ep, Ef, sf, out["fits"][k]["E_pads_MPa"]) for k in keys]
    out["sessions_detail"] = dict(ex.map(session_detail, jobs))
    if not a.no_spring and all(k in rows_by for k in JOINT):
        grid = np.round(np.geomspace(0.8, 32.0, 25), 4)
        resid = {(k, E): d for k, E, d in ex.map(residual_job, [(k, rows_by[k], E) for k in JOINT for E in grid])}
        sp = dict(joint=spring_from(resid, rows_by, list(JOINT), grid))
        for ind in ("low", "rod"):
            sp[ind] = spring_from(resid, rows_by, [k for k in JOINT if meta[k]["indenter"] == ind], grid)
        for g_, v in sp.items():
            print(f"spring {g_}: E {v['E_MPa']:.2f} MPa + {v['a_mm']:.3f} mm + {v['c_mm_N']:.3f} mm/N: rms {v['rms_mm'] * 1e3:.0f} um; "
                  f"E alone {v['alone']['E_MPa']:.2f} MPa: rms {v['alone']['rms_mm'] * 1e3:.0f} um", flush=True)
        out["spring"] = sp
    ex.shutdown()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
