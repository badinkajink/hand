#!/usr/bin/env python3
"""Per-pad force laws of the sphere pads refitted to the UR5 presses of the v3 tips IDs 1-3 (owner, 2026-11-15: "the
sphere packed models turn into a polynomial force-distance response due to the finite size of the spheres, compared to
the pretty linear responses we observed ... not sure we applied this to the fresh v3 id1-3 fingertips").

The fit of 2026-10-30 (v3_tip_contact_model.py) scaled one modulus, 2.47 MPa, to the forces of 3,568 holds under the low
5 mm sphere and the rod; the curve's shape was not fitted. At that modulus the measured over model force is 1.54 at
0.25 N and 0.49 at 8 N: the pads' force grows about as depth^2 and the presses' as depth^1.1-1.2. This script refits the
pads' law at each hold's registered indenter position (the HEX-E's deflection removed; no other rig compliance), with
the pads of the scenes (builder v2 on the FE surface points of b_05R4: stiffness E k_i, k_i = a_i / h_i, radius 0.75 mm).

Laws for one pad at overlap o (mm) with the indenter:
  linear       f = E k o                                    E (the scenes' law)
  spring       linear, the indenter moved back along n by c F before the overlaps are taken      E, c (mm/N)
  saturating   f = E k o_s (1 - exp(-o / o_s))              E, o_s (mm): linear below o_s, about E k o_s above
  power        f = E k o_r (o / o_r)^m, o_r = 0.1 mm        E, m
  sat_spring   saturating with the series compliance        E, o_s, c

Each law is fitted by least squares on log(F_meas / F_model), the normal force along the hold's contact normal, on the
holds of IDs 1 and 2 (both indenters) and tested on ID 3's; and fitted on one indenter (IDs 1-3) and tested on the
other. Reported per fit: the parameters, the RMS of the log ratio on the fit and test holds, the median ratio by force
level and the share of test holds within 25 %.

    ~/.claude/bin/resguard.sh run --mem 2G -- .venv/bin/python scripts/fingertip/v3_pad_law_refit.py [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path

import ur5_tip_calibration_pads as PD  # noqa: E402
import v3_tip_contact_model as V3  # noqa: E402

KEYS = V3.JOINT
LEVELS = (0.25, 1.0, 2.0, 3.5, 5.0, 8.0)
O_REF = 0.1
LAWS = {"linear": ("E",), "spring": ("E", "c"), "saturating": ("E", "o_s"), "power": ("E", "m"),
        "sat_spring": ("E", "o_s", "c")}
X0 = {"E": 2.5, "c": 0.1, "o_s": 0.3, "m": 0.8}
BOUNDS = {"E": (0.05, 500.0), "c": (0.0, 1.0), "o_s": (0.005, 5.0), "m": (0.1, 2.0)}


def load():
    """Per hold: tip, indenter, level, measured normal force, and the pads in contact at the registered position (their
    stiffness per MPa k, the vector v from pad centre to the indenter's centre line and the hold's normal)."""
    pads = PD.Pads(basis=V3.BASIS4, centroid=V3.CENTROID)
    H_rows, P_rows = [], []
    for key in KEYS:
        _, tip, ind, hdir, pdir, _basis = next(s for s in V3.SESSIONS if s[0] == key)
        H = np.load(hdir / "holds.npz", allow_pickle=False)
        g = json.loads((pdir / "plan.json").read_text())["geometry"]["indenter"]
        R, kind = float(g["radius"]), g.get("kind", "sphere")
        press = np.nonzero((H["kind"] == "press") & np.isfinite(H["f_n"]) & (H["f_n"] > V3.MIN_N))[0]
        for i in press:
            c, n = H["c_P"][i].astype(float), H["n"][i].astype(float)
            v = c - pads.C
            nn = n.copy()
            if kind == "rod":
                a = H["axis_P"][i].astype(float)
                a /= np.linalg.norm(a)
                v = v - (v @ a)[:, None] * a[None]
                nn = n - (n @ a) * a
            d = np.linalg.norm(v, axis=1)
            m = (PD.RS + R - d) > -1e-9
            if not m.any():
                continue
            h = len(H_rows)
            H_rows.append(dict(key=key, tip=tip, ind="rod" if kind == "rod" else "sphere", level=float(H["level_N"][i]),
                               f_n=float(H["f_n"][i]), R=R, n=n, nn=nn))
            P_rows.append((np.full(m.sum(), h), pads.k[m], v[m]))
    hold = np.concatenate([p[0] for p in P_rows])
    k = np.concatenate([p[1] for p in P_rows])
    v = np.concatenate([p[2] for p in P_rows])
    return H_rows, dict(hold=hold, k=k, v=v)


class Model:
    def __init__(self, H, P, sel):
        self.H = [H[i] for i in sel]
        idx = {h: j for j, h in enumerate(sel)}
        m = np.isin(P["hold"], sel)
        self.hold = np.array([idx[h] for h in P["hold"][m]])
        self.k, self.v = P["k"][m], P["v"][m]
        self.R = np.array([r["R"] for r in self.H])[self.hold]
        self.n = np.array([r["n"] for r in self.H])[self.hold]
        self.nn = np.array([r["nn"] for r in self.H])[self.hold]
        self.meas = np.array([r["f_n"] for r in self.H])
        self.N = len(self.H)

    def force(self, law, p, s=None):
        """Normal force of each hold (N) with the indenter moved back by s (mm, per hold) along n."""
        v = self.v if s is None else self.v + s[self.hold, None] * self.nn
        d = np.linalg.norm(v, axis=1)
        o = np.clip(PD.RS + self.R - d, 0.0, None)
        if law in ("linear", "spring"):
            f = o
        elif law in ("saturating", "sat_spring"):
            f = p["o_s"] * -np.expm1(-o / p["o_s"])
        elif law == "power":
            f = O_REF * (o / O_REF) ** p["m"]
        un = np.einsum("ij,ij->i", v, self.n) / np.maximum(d, 1e-12)
        return p["E"] * np.bincount(self.hold, weights=self.k * f * un, minlength=self.N)

    def predict(self, law, p):
        if "c" not in p or p["c"] <= 0:
            return self.force(law, p)
        lo, hi = np.zeros(self.N), np.full(self.N, 3.0)            # s - c F(s) rises with s: bisect its root
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            g = mid - p["c"] * self.force(law, p, mid)
            lo, hi = np.where(g < 0, mid, lo), np.where(g < 0, hi, mid)
        return self.force(law, p, 0.5 * (lo + hi))


def unpack(law, x):
    return {name: float(np.clip(np.exp(xi), *BOUNDS[name])) for name, xi in zip(LAWS[law], x)}


def fit(model, law):
    def cost(x):
        F = model.predict(law, unpack(law, x))
        r = np.log(model.meas / np.maximum(F, 1e-9))
        return float(np.mean(r ** 2))
    x0 = np.log([X0[n] for n in LAWS[law]])
    if law == "linear":
        F1 = model.force("linear", {"E": 1.0})
        return {"E": float(np.exp(np.mean(np.log(model.meas / np.maximum(F1, 1e-9)))))}
    best = None
    for scale in (1.0, 3.0, 0.3):                                     # a few starts for the 2- and 3-parameter laws
        xs = x0.copy()
        xs[0] += np.log(scale)
        r = minimize(cost, xs, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-7, maxiter=4000))
        if best is None or r.fun < best.fun:
            best = r
    return unpack(law, best.x)


def stats(model, law, p):
    F = model.predict(law, p)
    lr = np.log(model.meas / np.maximum(F, 1e-9))
    lv = np.array([r["level"] for r in model.H])
    by = {}
    for L in LEVELS:
        m = np.isclose(lv, L, atol=0.05 * L + 0.02)
        if m.sum():
            by[str(L)] = round(float(np.exp(np.median(lr[m]))), 3)
    return {"n": model.N, "rms_log": round(float(np.sqrt(np.mean(lr ** 2))), 4),
            "within25": round(float(np.mean(np.abs(lr) < np.log(1.25))), 3), "ratio_by_level": by}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    t0 = time.time()
    H, P = load()
    tips = np.array([r["tip"] for r in H])
    inds = np.array([r["ind"] for r in H])
    print(f"{len(H)} holds, {len(P['k'])} pad contacts ({time.time() - t0:.0f} s)", flush=True)
    splits = {"IDs 1-2 -> ID 3": (np.nonzero(tips != "v3 ID 3")[0], np.nonzero(tips == "v3 ID 3")[0]),
              "sphere -> rod": (np.nonzero(inds == "sphere")[0], np.nonzero(inds == "rod")[0]),
              "rod -> sphere": (np.nonzero(inds == "rod")[0], np.nonzero(inds == "sphere")[0]),
              "all": (np.arange(len(H)), np.arange(len(H)))}
    out = {"holds": len(H), "pad_contacts": int(len(P["k"])), "keys": list(KEYS), "fits": {}}
    for name, (tr, te) in splits.items():
        Mtr, Mte = Model(H, P, tr), Model(H, P, te)
        out["fits"][name] = {}
        for law in LAWS:
            t1 = time.time()
            p = fit(Mtr, law)
            rec = {"params": {k: round(v, 5) for k, v in p.items()}, "fit": stats(Mtr, law, p), "test": stats(Mte, law, p),
                   "wall_s": round(time.time() - t1, 1)}
            out["fits"][name][law] = rec
            print(f"{name:16s} {law:11s} {json.dumps(rec['params']):48s} rms fit {rec['fit']['rms_log']:.3f} test "
                  f"{rec['test']['rms_log']:.3f} within25 {rec['test']['within25']:.2f} by level "
                  f"{json.dumps(rec['test']['ratio_by_level'])}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
