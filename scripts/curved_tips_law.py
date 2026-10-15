#!/usr/bin/env python3
"""Quasi-static contact laws of the curved fingertips against the bed's tool cylinder (R 12.5 mm): the references of
the curved-tip study (docs/handoff/20261015-curved_tips.md, step 1), computed with tip_codesign_law's fields.

Fields per tip (tip_codesign_law):
  drake      Drake's compliant-convex field p = E (1 - g), g the gauge of the tip's mesh about its volume centroid, on a
             tool-surface grid of +-12.5 mm along the tool and +-0.40 rad around it (the 2026-10-14 grid's +-0.22 rad
             clips the patch of a tip curved across the width, which is long around the tool); mesh resolutions 0.5
             and 0.25 mm (res in the row)
  winkler    p = E delta / 8.5 mm on 0.1 mm surface samples
  local      p = E delta / h with h = n . (x - c_V), Drake's field to first order
  pads_v1    the 1 mm pads with the finger's median K and h 8.5 mm (builder v1)
  pads_v2    the 1 mm pads with each pad's K = E a_i / h_i and its local h_i (builder v2)

Per field and pinch force N (0.5, 1, 2, 3 N) with the tool line at psi from the finger's width direction: the approach
delta at psi 0, the all-slip friction arm at psi 0, 23, 46 and 69 deg (the onset arm's law value is the psi-0 arm, the
spin arm's the mean over 23-69 deg), the patch area, rho90 and the loaded pads; the quasi-static brake force at 45 deg
(the bed's T5). One fsynced row per (tip, field, res).

    .venv/bin/python scripts/curved_tips_law.py --tips tcd_cap6_u0.1 tcd_sph10.55 --out docs/experiments/20261015-curved_tips/bed/law.jsonl
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
import tip_codesign_law as L  # noqa: E402

SCRIPT = "scripts/curved_tips_law.py"
PSI = (0, 23, 46, 69)
NS = (0.5, 1.0, 2.0, 3.0)


class DrakeFieldWide(L.DrakeField):
    """tip_codesign_law.DrakeField on a wider tool-surface grid. The gauge of a fine mesh is evaluated only where the
    gauge of the tip's 1 mm mesh is below 1.1: that mesh is inscribed in the fine one, so its gauge is the larger, by
    about the 1 mm mesh's sag over the foundation depth (under 1 %)."""

    PHIMAX, NPHI, NU = 0.40, 80, 240

    def __init__(self, shape, E=L.E0, res=None):
        super().__init__(shape, E, res)
        self.coarse = L.DrakeField(shape, E, res=0.001) if res is not None and res < 0.001 else None

    @staticmethod
    def _gauge(f, X):
        out = np.empty(len(X))
        ch = max(64, int(4e6 // len(f.n)))
        for s in range(0, len(X), ch):
            out[s:s + ch] = np.max((X[s:s + ch] - f.c) @ f.n.T / f.dk[None, :], axis=1)
        return out

    def p(self, X):
        if self.coarse is None:
            return super().p(X)
        out = np.zeros(len(X))
        cand = self._gauge(self.coarse, X) < 1.1
        if cand.any():
            out[cand] = self.E * np.clip(1 - self._gauge(self, X[cand]), 0, None)
        return out

    def forces(self, psi, delta):
        a0, t = L._axis(self.reach, psi, delta)
        e1 = np.array([-1.0, 0, 0])
        e2 = np.cross(t, e1)
        umax, phimax = 0.0125, self.PHIMAX
        u = (np.arange(self.NU) + 0.5) / self.NU * 2 * umax - umax
        ph = (np.arange(self.NPHI) + 0.5) / self.NPHI * 2 * phimax - phimax
        U, PH = np.meshgrid(u, ph, indexing="ij")
        nT = np.cos(PH)[..., None] * e1 + np.sin(PH)[..., None] * e2
        X = (a0 + U[..., None] * t + L.RC * nT).reshape(-1, 3)
        nT = nT.reshape(-1, 3)
        p = self.p(X)
        m = p > 0
        edge = float(np.abs(PH.reshape(-1)[m]).max()) if m.any() else 0.0
        self.edge_frac = edge / phimax                 # how close the patch comes to the grid's edge around the tool
        dA = (2 * umax / self.NU) * L.RC * (2 * phimax / self.NPHI)
        out = L._arm(X[m], p[m] * dA, -nT[m])
        return out + (float(m.sum() * dA),)


def fields(shape, E=L.E0, res=(0.0005, 0.00025)):
    out = {}
    for r in res:
        out[("drake", r)] = DrakeFieldWide(shape, E, res=r)
    out[("winkler", None)] = L.Winkler(shape, E)
    out[("local", None)] = L.Winkler(shape, E, local=True)
    out[("pads_v1", None)] = L.Pads(shape, E)
    out[("pads_v2", None)] = L.Pads(shape, E, each=True, h="local")
    return out


def rows(key, E=L.E0, have=()):
    shape = L.shape_of(key)
    out = []
    for (name, res), mdl in fields(shape, E).items():
        if (key, name, res) in have:
            continue
        t0 = time.time()
        tab, edge = {}, 0.0
        for N in NS:
            for ps in PSI:
                d, arm, rho90, n, area = L.solve(mdl, math.radians(ps), N)
                edge = max(edge, getattr(mdl, "edge_frac", 0.0))
                tab[f"{N:g}|{ps}"] = {"delta_um": d * 1e6, "arm_mm": arm * 1e3, "rho90_mm": rho90 * 1e3, "n": n,
                                      "area_mm2": area * 1e6}
        br = {f"N{p}": L.brake(mdl, p)[0] for p in (45,)}
        out.append({
            "tip": key, "field": name, "res_m": res, "E_Pa": E, "table": tab,
            "approach_um": {f"{N:g}": tab[f"{N:g}|0"]["delta_um"] for N in NS},
            "onset_arm_mm": {f"{N:g}": tab[f"{N:g}|0"]["arm_mm"] for N in NS},
            "spin_arm_mm": {f"{N:g}": float(np.mean([tab[f"{N:g}|{ps}"]["arm_mm"] for ps in (23, 46, 69)])) for N in NS},
            "brake_N": br, "grid_edge_frac": edge if name == "drake" else None,
            "n_pads": int(len(mdl.C)) if hasattr(mdl, "C") else None,
            "centroid_mm": (shape.centroid() * 1e3).round(4).tolist(),
            "script": SCRIPT, "wall_s": round(time.time() - t0, 2), "when": time.strftime("%Y-%m-%d %H:%M")})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tips", nargs="+", required=True)
    ap.add_argument("--E", type=float, default=L.E0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    have = set()
    if a.out.exists():
        have = {(r["tip"], r["field"], r.get("res_m")) for r in map(json.loads, open(a.out))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for key in a.tips:
        for r in rows(key, a.E, have):
            with open(a.out, "a") as fh:
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            t = r["table"]
            print(f"{key:28s} {r['field']:8s} {r['res_m'] or '':7} delta(1 N) {t['1|0']['delta_um']:6.1f} um  onset arm "
                  f"{t['1|0']['arm_mm']:.3f} mm  spin arm {r['spin_arm_mm']['1']:.3f} mm  N45 {r['brake_N']['N45']:.3f} N"
                  + (f"  grid edge {r['grid_edge_frac']:.2f}" if r["grid_edge_frac"] is not None else "")
                  + f"  ({r['wall_s']:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
