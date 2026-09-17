#!/usr/bin/env python3
"""Collect the 2026-09-16 pinch-and-swing study into one JSON for the page builder.

    python3 scripts/swing_reorient_data.py
    python3 scripts/swing_reorient_page.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "docs/experiments"
P = E / "20260916-turn_probe"
OUT = E / "20260916-swing_reorient/swing_reorient.json"
D = {"sv1_w6689_b060": "D1", "sv1_w2360_b075": "D2", "sv1_u1364_b080": "D3", "g12_b095": "D4",
     "sv1_u0060_b75": "D5", "sv1_u0308_b050": "D6", "rv05_manual_b85": "D7", "sv1_w0099_b100": "D8"}
HW = {"D1": "6/7", "D2": "10/10", "D3": "4/10", "D4": "9/10", "D5": "3/10", "D6": "2/10", "D7": "10/10", "D8": "4/10"}
W = 0.240


def seam(r, n):
    for s in r.get("seams", []):
        if s.get("phase") == n:
            return s
    return {}


def held(s):
    return bool(s and s.get("pad_contacts", 0) >= 2 and s.get("pad_force_N", 0) >= W)


def probe(name):
    r = json.load(open(P / f"{name}.json"))
    a = r["analysis"]
    T = r["turn"]
    sm = {s["phase"]: s for s in r["seams"]}
    return {"name": name, "analysis": a, "carry_ik_mm": round(T["carry_ik_mm"], 1),
            "end_minus_q0_deg": {j: round(float(np.degrees(T["end"][j] - T["q0"][j])), 1) for j in T["end"]},
            "q0_deg": {j: round(float(np.degrees(v)), 1) for j, v in T["q0"].items()},
            "seams": {k: {kk: sm[k].get(kk) for kk in ("cos", "pad_contacts", "pad_force_N", "slide_mm", "z")}
                      for k in ("lifted", "turned", "reoriented", "regripped", "staged") if k in sm},
            "drop": r["result"]["drop_stage"]}


def probe_trace(name, every=4):
    r = json.load(open(P / f"{name}.json"))
    T = r["turn"]
    st = [s for s in r["steps"] if T["step0"] - 20 <= s["step"] <= T["step0"] + T["turn_steps"] + 300]
    out = []
    for i, s in enumerate(st):
        if i % every:
            continue
        out.append({"u": round((s["step"] - T["step0"]) / T["turn_steps"], 3), "cos": s["cos"], "z": s["z"],
                    "N": {f: s["pads"][f]["n"] for f in ("thumb", "index", "middle")},
                    "yaw_cmd": {f: round(float(np.degrees(s["q_cmd"][f + "_yaw"])), 1) for f in ("thumb", "index", "middle")},
                    "yaw": {f: round(float(np.degrees(s["q"][f + "_yaw"])), 1) for f in ("thumb", "index", "middle")},
                    "slip": {f: round(s["pads"][f]["slip"] * 1e3, 1) for f in ("thumb", "index", "middle")}})
    return out


def feasibility(name):
    r = json.load(open(P / f"{name}.json"))
    F = r["feasibility"]
    out = {}
    for c in F["cells"]:
        if c["angle"] >= 0:
            continue
        out.setdefault(c["pivot"], {}).setdefault(str(c["axis_k"]), {})[str(c["angle"])] = c["res_mm"]
    return {"span_mm": F["span_mm"], "cells": out}


def swing_rows(paths):
    rows = []
    for p in paths:
        j = json.load(open(p))
        for r in j["rows"]:
            if r.get("error"):
                continue
            tag, parent = r["tag"], r.get("parent", r["tag"])
            ta = re.search(r"_ta(\d+)", tag)
            sq = re.search(r"_sq(\d+)", r["arm"])
            L, T, R, G, S = (seam(r, k) for k in ("lifted", "turned", "reoriented", "regripped", "staged"))
            rows.append({"hand": D.get(parent, parent), "plant": r["plant"], "squeeze": int(sq.group(1)) if sq else 2,
                         "thumb": int(ta.group(1)) if ta else 10, "pinch": r.get("pinch"), "yaw": r.get("yaw_deg"),
                         "seed": r["seed"], "sweep": Path(p).parent.name,
                         "lift_held": held(L), "lift_N": L.get("pad_force_N"),
                         "turned_cos": T.get("cos"), "reor_cos": R.get("cos"), "reor_held": held(R),
                         "reor_N": R.get("pad_force_N"), "reor_pads": R.get("pad_contacts"),
                         "regrip_cos": G.get("cos"), "regrip_held": held(G),
                         "staged_cos": S.get("cos"), "staged_held": held(S),
                         "carried": r.get("drop_stage") is None, "drop": r.get("drop_stage"),
                         "final_tilt": r.get("final_tilt_deg"), "turns": r.get("turns")})
    return rows


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "probes": {k: probe(k) for k in (
            "sv1_u0308_b050__fast__sq10_k0.15_b0.5_a-90",
            "sv1_u0308_b050__fast__sq10_k0.15_b1.3_a-90",
            "sv1_u0308_b050__fast__sq10_k0.15_b0.5_a-90_elliptic_ir10",
            "sv1_u0308_b050__fast__sq10_k0.15_b1.3_a-90_elliptic_ir10",
            "d6_pinch_middle", "d6_pinch_index",
            "d6_pinch_middle_mu1.0", "d6_pinch_middle_mu0.6", "d6_turn_mu1.0", "d6_turn_mu0.6",
            "d7_cal_pinch_middle_y180")},
        "trace_default": probe_trace("sv1_u0308_b050__fast__sq10_k0.15_b0.5_a-90"),
        "trace_elliptic": probe_trace("sv1_u0308_b050__fast__sq10_k0.15_b0.5_a-90_elliptic_ir10"),
        "trace_d7_pinch": probe_trace("d7_cal_pinch_middle_y180", every=2),
        "feasibility_d6": feasibility("d6_feas"),
        "workspace_d6": json.load(open(E / "20260916-turn_workspace/sv1_u0308_b050_workspace.json")),
        "swing": swing_rows([E / f"20260916-{d}/chain_hands.json" for d in
                             ("swing_cal", "swing_cal25", "swing_pinch", "swing_axial", "swing_thumb", "swing_seeds")]),
        "sens": {},
        "hw": HW,
    }
    for p in sorted(glob.glob(str(P / "d8_sens_*.json"))):
        r = json.load(open(p))
        sm = {s["phase"]: s for s in r["seams"]}
        k = Path(p).stem.replace("d8_sens_", "")
        data["sens"][k] = {ph: {kk: sm[ph].get(kk) for kk in ("cos", "pad_contacts", "pad_force_N")}
                           for ph in ("lifted", "turned", "reoriented", "regripped", "staged") if ph in sm}
        data["sens"][k]["drop"] = r["result"]["drop_stage"]
    data["n_rollouts"] = {"swing": len(data["swing"]), "probes": len(data["probes"]) + len(data["sens"]) + 4}
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT}  swing rollouts {len(data['swing'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
