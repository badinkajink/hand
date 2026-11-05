#!/usr/bin/env python3
"""Identify the gantry parts of file "hand" by geometry: per gantry the fixed X-axis hardware, the X stage (beam)
with the Y-axis hardware it carries, and the Y stage (the finger on its rail block and nut)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ALL = json.loads((REPO / "logs/20261104-cad_v3/extract/parts.json").read_text())["parts"]
PARTS = {p["name"]: p for p in ALL if p["file"] == "hand"}

GANTRIES = {"g1": "gantry_v3 <1>", "g2": "gantry_v3 <2>", "gt": "gantry_v3_thumb <1>"}
FINGERS = ["Finger v3 <1>", "Finger v3 <2>", "Finger v3 <3>"]

# volume signatures (mm^3, rounded), from the part list
KIND = {53883: "frame", 77381: "frame", 50194: "beam", 51850: "beam", 3254: "rail", 4919: "rail", 1923: "block",
        2271: "screw", 8194: "screw", 1077: "nut", 11194: "motor", 6739: "coupler", 117: "shaft", 236: "bearing",
        295: "collar", 4927: "motor_mount", 4930: "motor_mount"}


def sub(p):
    return p["chain"][0] if len(p["chain"]) > 1 else "TOP"


def bbox(p):
    b = np.asarray(p["bbox"], float)
    return b[:3], b[3:]


def kind(p):
    v = int(round(p["volume"]))
    for k, nm in KIND.items():
        if abs(v - k) <= 3:
            return nm
    return None


def long_axis(p, length):
    lo, hi = bbox(p)
    ext = hi - lo
    return int(np.argmin(np.abs(ext - length)))


def axis_of(p):
    """Axis of a rotational hardware part from its cylindrical faces (the most common axis direction)."""
    ax = [np.abs(np.asarray(c["axis"])) for c in p["cylinders"]]
    if not ax:
        return None
    k = [int(np.argmax(a)) for a in ax]
    return max(set(k), key=k.count)


def classify():
    """Returns {gantry key: {role: [names]}} and {gantry key: finger instance}."""
    out = {}
    finger_of = {}
    tops = [p for p in PARTS.values() if sub(p) == "TOP"]
    for g, chain in GANTRIES.items():
        mine = [p for p in PARTS.values() if sub(p) == chain]
        frame = [p for p in mine if kind(p) == "frame"][0]
        flo, fhi = bbox(frame)
        roles = {r: [] for r in ("frame", "x_rail", "x_block", "x_screw", "x_nut", "x_motor", "x_coupler", "x_shaft",
                                 "x_bearing", "x_mount", "beam", "y_rail", "y_block", "y_screw", "y_motor", "y_coupler",
                                 "y_shaft", "y_bearing", "other")}
        # top-level hardware coaxial with this gantry's X screw belongs to it (gantry <2>'s X motor and coupler)
        xs = [p for p in mine if kind(p) == "screw"]
        xscrew = [p for p in xs if long_axis(p, 100.0 if g != "gt" else 150.0) == 0 and
                  (np.subtract(*bbox(p)[::-1])[0] > 90)][0]
        sc = 0.5 * (np.add(*bbox(xscrew)))
        frames = {gg: [q for q in PARTS.values() if sub(q) == ch and kind(q) == "frame"][0]
                  for gg, ch in GANTRIES.items()}
        for p in tops:
            k = kind(p)
            if k in ("motor", "coupler", "shaft") and axis_of(p) == 0 or k == "motor" and long_axis(p, 28.6) == 0:
                c = 0.5 * np.add(*bbox(p))
                near = min(frames, key=lambda gg: abs(0.5 * np.add(*bbox(frames[gg]))[0] - c[0])
                           + 1e3 * abs(0.5 * np.add(*bbox(frames[gg]))[1] - c[1] > 60.0))
                if abs(c[1] - sc[1]) < 2.0 and abs(c[2] - sc[2]) < 2.0 and near == g:
                    mine.append(p)
        for p in mine:
            k = kind(p)
            lo, hi = bbox(p)
            ext = hi - lo
            if k == "frame":
                roles["frame"].append(p["name"])
            elif k == "beam":
                roles["beam"].append(p["name"])
            elif k == "motor_mount":
                roles["x_mount"].append(p["name"])
            elif k == "rail":
                roles["x_rail" if ext[0] > ext[1] else "y_rail"].append(p["name"])
            elif k == "screw":
                roles["x_screw" if ext[0] > ext[1] else "y_screw"].append(p["name"])
            elif k == "block":
                roles["x_block" if ext[0] > ext[1] else "y_block"].append(p["name"])
            elif k == "nut":
                roles["x_nut"].append(p["name"])
            elif k == "motor":
                roles["x_motor" if long_axis(p, 28.6) == 0 else "y_motor"].append(p["name"])
            elif k in ("coupler", "shaft", "bearing", "collar"):
                a = axis_of(p)
                kk = {"coupler": "coupler", "shaft": "shaft", "bearing": "bearing", "collar": "bearing"}[k]
                roles[("x_" if a == 0 else "y_") + kk].append(p["name"])
            else:
                roles["other"].append(p["name"])
        out[g] = roles
        # the finger whose nut is coaxial with this gantry's Y screw
        ys = PARTS[roles["y_screw"][0]]
        yc = 0.5 * np.add(*bbox(ys))
        for f in FINGERS:
            nuts = [p for p in PARTS.values() if sub(p) == f and kind(p) == "nut"]
            c = 0.5 * np.add(*bbox(nuts[0]))
            if abs(c[0] - yc[0]) < 2.0 and abs(c[2] - yc[2]) < 2.0:
                finger_of[g] = f
    return out, finger_of


def groups(g, roles, finger):
    """Fixed parts F (everything fixed to the base), the X stage S1 and the Y stage S2 of gantry g."""
    fixed_roles = ("frame", "x_rail", "x_screw", "x_motor", "x_coupler", "x_shaft", "x_bearing", "x_mount")
    s1_roles = ("beam", "x_block", "x_nut", "y_rail", "y_screw", "y_motor", "y_coupler", "y_shaft", "y_bearing")
    S1 = [n for r in s1_roles for n in roles[g][r]]
    S2 = roles[g]["y_block"] + [p["name"] for p in PARTS.values() if sub(p) == finger]
    # fixed: this gantry's X hardware, every other gantry's frame and X hardware, and the top-level parts
    F = []
    for gg in roles:
        F += [n for r in fixed_roles for n in roles[gg][r]]
    moving_all = set()
    for gg in roles:
        moving_all |= {n for r in s1_roles for n in roles[gg][r]} | set(roles[gg]["y_block"])
    finger_parts = {p["name"] for p in PARTS.values() if sub(p).startswith("Finger")}
    F += [p["name"] for p in PARTS.values() if sub(p) == "TOP" and p["name"] not in F and p["name"] not in moving_all
          and p["name"] not in finger_parts]
    F = sorted(set(F))
    sliding = {(roles[g]["x_block"][0], roles[g]["x_rail"][0]), (roles[g]["x_nut"][0], roles[g]["x_screw"][0]),
               (roles[g]["y_block"][0], roles[g]["y_rail"][0])}
    ynut = [p["name"] for p in PARTS.values() if sub(p) == finger and kind(p) == "nut"][0]
    sliding.add((ynut, roles[g]["y_screw"][0]))
    return F, S1, S2, sliding, ynut


if __name__ == "__main__":
    roles, finger_of = classify()
    for g, r in roles.items():
        print("==", g, GANTRIES[g], "finger:", finger_of.get(g))
        for k, v in r.items():
            if v:
                print("   %-10s" % k, ", ".join("%s(%s)" % (n, PARTS[n]["label"]) for n in v))
    for g in roles:
        F, S1, S2, sl, ynut = groups(g, roles, finger_of[g])
        print(g, "F", len(F), "S1", len(S1), "S2", len(S2), "sliding", sorted(sl))
    used = set()
    for g in roles:
        for v in roles[g].values():
            used |= set(v)
    rest = [n for n, p in PARTS.items() if n not in used and not sub(p).startswith("Finger")]
    print("not in any gantry role:", [(n, PARTS[n]["label"], sub(PARTS[n]), int(PARTS[n]["volume"])) for n in rest])
