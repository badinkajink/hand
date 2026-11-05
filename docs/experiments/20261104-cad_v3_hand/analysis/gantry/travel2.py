#!/usr/bin/env python3
"""Gantry-intrinsic travel limits and cross-gantry clearances (second pass of travel.py).

Per gantry: the fixed group is its own frame and X hardware plus the top-level parts (base plate, robot mount,
adaptor, sensor brackets); the X stage S1 is the beam with its rail block, nut and Y hardware; the Y stage is split
into the rigid group S2r (rail block, carriage, yaw servo, nut) and the finger links at the CAD pose S2l.
Cross-gantry: the y_A gap between the thumb gantry's parts and the index/middle gantries' parts in every z_A row,
with the parts extruded along x_A (the stages pass each other in x_A), at the CAD position and with the thumb gantry
moved +2.5 mm in y_A (onto the base plate's holes). Writes travel2.json.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- .venv/bin/python logs/20261104-cad_v3/gantry/travel2.py
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
from travel import H, TOL, pair_free, blocked_set, _merge  # noqa: E402

REPO = HERE.parents[2]
THUMB_SHIFT_MM = 2.5


def finger_split(finger):
    parts = [p for p in classify.PARTS.values() if classify.sub(p) == finger]
    car = [p for p in parts if abs(p["volume"] - 23913) < 3][0]
    nut = [p for p in parts if classify.kind(p) == "nut"][0]
    lo, hi = classify.bbox(car)
    servo = [p for p in parts if abs(p["volume"] - 6673) < 3 and np.all(np.asarray(p["com"]) > lo)
             and np.all(np.asarray(p["com"]) < hi)][0]
    rigid = [car["name"], servo["name"], nut["name"]]
    links = [p["name"] for p in parts if p["name"] not in rigid]
    return rigid, links


TOP_FIXED = ["000", "001", "002", "004", "005"]   # sensor plate, robot mount, base plate, adaptor, bracket


def own_fixed(g, roles, top=True):
    fixed_roles = ("frame", "x_rail", "x_screw", "x_motor", "x_coupler", "x_shaft", "x_bearing", "x_mount")
    F = [n for r in fixed_roles for n in roles[g][r]]
    if top:
        F += TOP_FIXED
    return sorted(set(F))


def gantry_all(g, roles, finger):
    names = [n for v in roles[g].values() for n in v]
    names += [p["name"] for p in classify.PARTS.values() if classify.sub(p) == finger]
    return sorted(set(names))


def y_gap_rows(R0, A, B, shiftA_cells=0):
    """Gap along +y from group B (below) to group A (above) in every z row of the (y, z) grid of the x-sweep, with
    the parts extruded along x. Returns (min gap mm, z of that row, the parts that set it, number of rows compared)."""
    def rows(names, sh):
        cell = np.concatenate([R0[n]["cell"] for n in names]) + sh
        who = np.concatenate([np.full(len(R0[n]["cell"]), n, dtype=object) for n in names])
        a = cell // sweep.MUL - sweep.OFF_CELL
        b = cell % sweep.MUL - sweep.OFF_CELL
        return a, b, who
    aA, bA, wA = rows(A, shiftA_cells * sweep.MUL)
    aB, bB, wB = rows(B, 0)
    best = (np.inf, None, None, 0)
    zrows = np.intersect1d(np.unique(bA), np.unique(bB))
    for b in zrows:
        ia = np.where(bA == b)[0]
        ib = np.where(bB == b)[0]
        i = ia[np.argmin(aA[ia])]
        j = ib[np.argmax(aB[ib])]
        gap = (aA[i] - aB[j]) * H
        if gap < best[0]:
            best = (float(gap), float(0.04321 + b * H), [str(wA[i]), str(wB[j])], len(zrows))
    return best


def main():
    roles, finger_of = classify.classify()
    meshes = {n: trimesh.load(REPO / p["stl"], force="mesh") for n, p in classify.PARTS.items()}
    R = {k: {n: sweep.rasterize(m, k, H) for n, m in meshes.items()} for k in (0, 1)}
    out = {"grid_mm": H, "tol_mm": TOL, "thumb_shift_mm": THUMB_SHIFT_MM, "gantries": {}, "cross": {}}
    for g in ("g1", "g2", "gt"):
        _, S1, _, sliding, ynut = classify.groups(g, roles, finger_of[g])
        rigid, links = finger_split(finger_of[g])
        S2r = roles[g]["y_block"] + rigid
        F = own_fixed(g, roles)
        r = dict(finger=finger_of[g], F_own=F, S1=S1, S2_rigid=S2r, S2_links=links)
        r["x_S1_vs_F"] = pair_free(R[0], S1, F, sliding)
        r["x_S2r_vs_F"] = pair_free(R[0], S2r, F, sliding)
        r["x_S2l_vs_F"] = pair_free(R[0], links, F, sliding)
        r["y_S2r_vs_S1"] = pair_free(R[1], S2r, S1, sliding)
        r["y_S2l_vs_S1"] = pair_free(R[1], links, S1, sliding)
        r["y_S2r_vs_F"] = pair_free(R[1], S2r, F, sliding)
        r["y_S2l_vs_F"] = pair_free(R[1], links, F, sliding)
        xr, yr = r["x_S1_vs_F"], r["y_S2r_vs_S1"]
        # finger (rigid and links) against the fixed parts over the whole travel rectangle
        hits = []
        for grp, names in (("rigid", S2r), ("links", links)):
            for n in range(int(np.floor(xr["lower"])), int(np.ceil(xr["upper"])) + 1):
                s_lo, s_hi, wm, wf = blocked_set(R[1], names, F, sliding, int(round(n / H)) * sweep.MUL)
                for lo, hi, pr in _merge(s_lo, s_hi, yr["lower"], yr["upper"], wm, wf):
                    hits.append([grp, float(n), lo, hi, pr])
        r["finger_vs_fixed_in_travel"] = hits
        out["gantries"][g] = r
        print(g, flush=True)
        for key in [k for k in r if k.startswith(("x_", "y_"))]:
            q = r[key]
            print("  %-14s lower %9.3f upper %9.3f pen %.3f  lower %s  upper %s" % (
                key, q["lower"], q["upper"], q["penetration_mm"], q.get("lower_pair"), q.get("upper_pair")), flush=True)
            if q.get("penetrating_pairs"):
                print("      penetrating:", q["penetrating_pairs"][:6])
        print("  finger vs fixed parts inside the travel rectangle:", hits[:6], "n", len(hits), flush=True)
    # cross-gantry gaps in y_A, parts extruded along x_A
    thumb_all = gantry_all("gt", roles, finger_of["gt"])
    thumb_all = [n for n in thumb_all if n not in TOP_FIXED]
    for g in ("g1", "g2"):
        pair_all = [n for n in gantry_all(g, roles, finger_of[g]) if n not in TOP_FIXED]
        groups = {
            "thumb_S1_vs_pair_S1": (classify.groups("gt", roles, finger_of["gt"])[1],
                                    classify.groups(g, roles, finger_of[g])[1]),
            "thumb_S1_vs_pair_frame_and_X": (classify.groups("gt", roles, finger_of["gt"])[1], own_fixed(g, roles, top=False)),
            "thumb_frame_and_X_vs_pair_S1": (own_fixed("gt", roles, top=False), classify.groups(g, roles, finger_of[g])[1]),
            "thumb_all_vs_pair_all": (thumb_all, pair_all),
        }
        for key, (A, B) in groups.items():
            A = [n for n in A if n not in TOP_FIXED]
            B = [n for n in B if n not in TOP_FIXED]
            for tag, sh in (("cad", 0), ("thumb_on_base_plate_holes", int(round(THUMB_SHIFT_MM / H)))):
                gap, z, pr, nrows = y_gap_rows(R[0], A, B, sh)
                out["cross"].setdefault(g, {}).setdefault(key, {})[tag] = dict(min_gap_y_mm=gap, at_z_A=z, parts=pr,
                                                                                rows=nrows)
                print("  cross %s %-30s %-26s gap %7.3f mm at z_A %.2f parts %s" % (g, key, tag, gap, z, pr), flush=True)
    (HERE / "travel2.json").write_text(json.dumps(out, indent=1, default=float))
    print("wrote", HERE / "travel2.json")


if __name__ == "__main__":
    main()
