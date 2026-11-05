#!/usr/bin/env python3
"""Travel limits of the three gantries of file "hand" from their geometry, by translational sweeps (sweep.py).

X stage S1 (beam, its rail block and nut, the Y-axis hardware) against the fixed parts F along x_A; Y stage S2 (the
finger, its rail block and its nut) against S1 along y_A; S2 against F along x_A and y_A as maps over the other axis.
Sliding pairs (rail block on rail, nut on lead screw) are excluded. Writes travel.json beside this file.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python logs/20261104-cad_v3/gantry/travel.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classify  # noqa: E402
import sweep  # noqa: E402

REPO = HERE.parents[2]
H = 0.25          # grid spacing across the sweep direction, mm
TOL = 0.02        # contact tolerance, mm


def load_all():
    meshes = {n: trimesh.load(REPO / p["stl"], force="mesh") for n, p in classify.PARTS.items()}
    R = {}
    for k in (0, 1):
        t = time.time()
        R[k] = {n: sweep.rasterize(m, k, H) for n, m in meshes.items()}
        print("rasterized along", "xy"[k], "%.1fs" % (time.time() - t), "bad lines",
              {n: r["bad"] for n, r in R[k].items() if r["bad"]}, flush=True)
    return meshes, R


def pair_free(Rk, mov, fix, excl, shift=0, tol=TOL):
    """Free displacement range of the group `mov` against `fix` along +k, from blocked intervals of every pair of
    parts not in `excl`; with the limiting pairs, the contact points and penetrations at s = 0."""
    M = sweep.concat(Rk, mov)
    F = sweep.concat(Rk, fix)
    s_lo, s_hi, iM, iF = sweep.blocked(M, F, shift)
    wm, wf = M["who"][iM], F["who"][iF]
    ex = np.zeros(len(wm), bool)
    for a, b in excl:
        ex |= ((wm == a) & (wf == b)) | ((wm == b) & (wf == a))
    s_lo, s_hi, iM, iF, wm, wf = s_lo[~ex], s_hi[~ex], iM[~ex], iF[~ex], wm[~ex], wf[~ex]
    lo, up, il, iu, pen = sweep.free_range(s_lo, s_hi, tol)
    out = dict(lower=lo, upper=up, penetration_mm=pen)
    if pen > 0:
        p = (s_lo < -tol) & (s_hi > tol)
        d = np.minimum(-s_lo[p], s_hi[p])
        pairs = {}
        for a, b, dd in zip(wm[p], wf[p], d):
            pairs[(a, b)] = max(pairs.get((a, b), 0.0), float(dd))
        out["penetrating_pairs"] = sorted(([a, b, round(v, 3)] for (a, b), v in pairs.items()), key=lambda r: -r[2])
    for tag, i, val in (("upper", iu, up), ("lower", il, lo)):
        if i < 0:
            continue
        u, v = sweep.cell_uv(M["cell"][iM[i]] + shift, H)
        kpos = (M["hi"][iM[i]] + val) if tag == "upper" else (M["lo"][iM[i]] + val)
        out[tag + "_pair"] = [str(wm[i]), str(wf[i])]
        out[tag + "_contact_uvk"] = [round(float(u), 3), round(float(v), 3), round(float(kpos), 3)]
        # the next few distinct pairs, to see how close the runners-up are
        sel = (s_lo >= -tol) if tag == "upper" else (s_hi <= tol)
        vals = s_lo[sel] if tag == "upper" else -s_hi[sel]
        ws = list(zip(wm[sel], wf[sel], vals))
        best = {}
        for a, b, x in ws:
            if (a, b) not in best or x < best[(a, b)]:
                best[(a, b)] = float(x)
        rank = sorted(best.items(), key=lambda e: e[1])[:6]
        out[tag + "_next"] = [[a, b, round(x if tag == "upper" else -x, 3)] for (a, b), x in rank]
    return out


def blocked_set(Rk, mov, fix, excl, shift=0):
    M = sweep.concat(Rk, mov)
    F = sweep.concat(Rk, fix)
    s_lo, s_hi, iM, iF = sweep.blocked(M, F, shift)
    wm, wf = M["who"][iM], F["who"][iF]
    ex = np.zeros(len(wm), bool)
    for a, b in excl:
        ex |= ((wm == a) & (wf == b)) | ((wm == b) & (wf == a))
    return s_lo[~ex], s_hi[~ex], wm[~ex], wf[~ex]


def main():
    roles, finger_of = classify.classify()
    meshes, R = load_all()
    res = {"grid_mm": H, "tol_mm": TOL, "gantries": {}}
    for g in ("g1", "g2", "gt"):
        F, S1, S2, sliding, ynut = classify.groups(g, roles, finger_of[g])
        r = dict(finger=finger_of[g], F=F, S1=S1, S2=S2, sliding=sorted(map(list, sliding)), y_nut=ynut)
        t0 = time.time()
        r["x_S1_vs_F"] = pair_free(R[0], S1, F, sliding)
        r["x_S2_vs_F"] = pair_free(R[0], S2, F, sliding)
        r["y_S2_vs_S1"] = pair_free(R[1], S2, S1, sliding)
        r["y_S2_vs_F"] = pair_free(R[1], S2, F, sliding)
        # the stages against the other gantries' moving parts at their CAD poses (inter-gantry clearance)
        others_S1 = [n for gg in ("g1", "g2", "gt") if gg != g for n in classify.groups(gg, roles, finger_of[gg])[1]]
        others_S2 = [n for gg in ("g1", "g2", "gt") if gg != g for n in classify.groups(gg, roles, finger_of[gg])[2]]
        r["x_S1_vs_other_beams"] = pair_free(R[0], S1, others_S1, set())
        r["x_S2_vs_other_stages"] = pair_free(R[0], S2, others_S1 + others_S2, set())
        r["y_S2_vs_other_stages"] = pair_free(R[1], S2, others_S1 + others_S2, set())
        # maps: S2 against F along y as a function of the x displacement s, and along x as a function of t
        xr = r["x_S1_vs_F"]
        yr = r["y_S2_vs_S1"]
        smap, tmap = [], []
        for n in range(int(np.floor((xr["lower"] - 3) / 1.0)), int(np.ceil((xr["upper"] + 3) / 1.0)) + 1):
            sh = int(round(n / H)) * sweep.MUL          # k = y: grid (x, z), a-index along x
            s_lo, s_hi, wm, wf = blocked_set(R[1], S2, F, sliding, sh)
            smap.append(dict(s=float(n), t_blocked=_merge(s_lo, s_hi, yr["lower"] - 3, yr["upper"] + 3, wm, wf)))
        for n in range(int(np.floor((yr["lower"] - 3) / 1.0)), int(np.ceil((yr["upper"] + 3) / 1.0)) + 1):
            sh = int(round(n / H)) * sweep.MUL          # k = x: grid (y, z), a-index along y
            s_lo, s_hi, wm, wf = blocked_set(R[0], S2, F, sliding, sh)
            tmap.append(dict(t=float(n), s_blocked=_merge(s_lo, s_hi, xr["lower"] - 3, xr["upper"] + 3, wm, wf)))
        r["map_t_blocked_vs_s"] = smap
        r["map_s_blocked_vs_t"] = tmap
        print(g, "done %.1fs" % (time.time() - t0), flush=True)
        for key in ("x_S1_vs_F", "x_S2_vs_F", "y_S2_vs_S1", "y_S2_vs_F", "x_S1_vs_other_beams", "x_S2_vs_other_stages",
                    "y_S2_vs_other_stages"):
            q = r[key]
            print("  %-22s lower %8.3f upper %8.3f pen %.3f  low pair %s  up pair %s" % (
                key, q["lower"], q["upper"], q["penetration_mm"], q.get("lower_pair"), q.get("upper_pair")))
            if q.get("penetrating_pairs"):
                print("      penetrating:", q["penetrating_pairs"][:8])
        res["gantries"][g] = r
    (HERE / "travel.json").write_text(json.dumps(res, indent=1, default=float))
    print("wrote", HERE / "travel.json")


def _merge(s_lo, s_hi, a, b, wm, wf):
    """Blocked displacement intervals clipped to [a, b] and merged, with the part pairs that block them."""
    sel = (s_hi > a) & (s_lo < b) & (s_hi - s_lo > 1e-6)
    iv = sorted(zip(np.maximum(s_lo[sel], a), np.minimum(s_hi[sel], b), wm[sel], wf[sel]))
    out = []
    for lo, hi, m, f in iv:
        if out and lo <= out[-1][1] + 1e-9:
            out[-1][1] = max(out[-1][1], float(hi))
            out[-1][2].add((str(m), str(f)))
        else:
            out.append([float(lo), float(hi), {(str(m), str(f))}])
    return [[round(lo, 3), round(hi, 3), sorted(map(list, p))[:6]] for lo, hi, p in out]


if __name__ == "__main__":
    main()
