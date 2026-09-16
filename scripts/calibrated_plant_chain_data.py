#!/usr/bin/env python3
"""Collect the 2026-09-16 calibrated-plant chain sweeps into one JSON for the page.

    python3 scripts/calibrated_plant_chain_data.py

Reads docs/experiments/20260916-plant_{pivot,squeeze,budget,relief}/chain_hands.json and the
two instrumented D6 rollouts in 20260916-plant_probe/.
"""
from __future__ import annotations
import json, math, os, re, statistics as st
from collections import defaultdict

W = 0.240
DEP = {"sv1_w6689_b060": "D1", "sv1_w2360_b075": "D2", "sv1_u1364_b080": "D3", "g12_b095": "D4",
       "sv1_u0060_b75": "D5", "sv1_u0308_b050": "D6", "rv05_manual_b85": "D7",
       "sv1_w0099_b100": "D8"}
BENCH_HOLD = {"D1": "6/7", "D2": "10/10", "D3": "4/10", "D4": "9/10", "D5": "3/10",
              "D6": "2/10", "D7": "10/10", "D8": "4/10"}
PH = ["lifted", "turned", "reoriented", "regripped", "staged", "set_down", "pressed",
      "handover_thumb", "handover_index", "handover_middle", "handover_grip", "gait_grip",
      "gaited"]
ROOT = "docs/experiments"
OUT = f"{ROOT}/20260916-calibrated_plant_chain/calibrated_plant.json"


def seam(r, ph):
    return next((s for s in (r.get("seams") or []) if s.get("phase") == ph), {})


def held(s):
    return (s.get("pad_contacts") or 0) >= 2 and (s.get("pad_force_N") or 0) >= W \
        and (s.get("z") or 0) > 0.08


def carried(s):
    return (s.get("pad_contacts") or 0) >= 2 and (s.get("pad_force_N") or 0) >= W


def knob(r, key):
    m = re.search(rf"_{key}(-?[\d.]+)", r.get("arm", ""))
    return float(m.group(1)) if m else None


def deg(s):
    c = s.get("cos")
    return None if c is None else round(90.0 - math.degrees(math.acos(max(-1.0, min(1.0, c)))), 1)


def load(name):
    d = json.load(open(f"{ROOT}/20260916-plant_{name}/chain_hands.json"))
    rows = [r for r in d["rows"] if not r.get("error")]
    for r in rows:
        r["_id"] = DEP.get(r["tag"], r["tag"])
        r["_sq"] = knob(r, "sq") or 2.0
        r["_b"] = knob(r, "b") or 0.5
        r["_l"] = knob(r, "l") or 0.0
        r["_k"] = round(r.get("axis_k") or 0, 2)
    return rows


def main():
    piv, sq, bud, rel = load("pivot"), load("squeeze"), load("budget"), load("relief")
    out = {"n": {"pivot": len(piv), "squeeze": len(sq), "budget": len(bud), "relief": len(rel)}}

    # 1. shipped vs corrected at the chain's 2 mm squeeze
    cmp = []
    for h in sorted(set(r["_id"] for r in piv)):
        e = {"id": h, "bench_hold": BENCH_HOLD.get(h, "")}
        for p in ("shipped", "corrected"):
            rs = [r for r in piv if r["_id"] == h and r.get("plant") == p]
            L = [seam(r, "lifted") for r in rs if r.get("seed") == 0]
            e[p] = {"lift_N": round(st.median([s.get("pad_force_N", 0) for s in L]), 2) if L else None,
                    "lift_pads": int(st.median([s.get("pad_contacts", 0) for s in L])) if L else None,
                    "held_turn": sum(1 for r in rs if held(seam(r, "turned"))),
                    "n": len(rs),
                    "best_cos": round(max(seam(r, "turned").get("cos", -1) for r in rs), 3) if rs else None,
                    "chains": sum(1 for r in rs if r.get("ok"))}
        cmp.append(e)
    out["compare"] = cmp

    # 2. squeeze -> lift force on the corrected plant
    sqs = sorted(set(r["_sq"] for r in sq))
    out["squeeze"] = {"values": sqs, "rows": []}
    for h in sorted(set(r["_id"] for r in sq)):
        e = {"id": h, "lift_N": [], "lift_pads": [], "held_turn": [], "n": []}
        for v in sqs:
            rs = [r for r in sq if r["_id"] == h and r["_sq"] == v]
            L = [seam(r, "lifted") for r in rs]
            e["lift_N"].append(round(st.median([s.get("pad_force_N", 0) for s in L]), 2) if L else None)
            e["lift_pads"].append(int(st.median([s.get("pad_contacts", 0) for s in L])) if L else None)
            e["held_turn"].append(sum(1 for r in rs if held(seam(r, "turned"))))
            e["n"].append(len(rs))
        out["squeeze"]["rows"].append(e)

    # 3. budget and relief -> turn angle on the corrected plant, squeeze 10
    for name, rows, key in (("budget", bud, "_b"), ("relief", rel, "_l")):
        vals = sorted(set(r[key] for r in rows))
        block = {"values": vals, "rows": []}
        for h in sorted(set(r["_id"] for r in rows)):
            e = {"id": h, "turn_deg": [], "force_N": [], "slide_mm": [], "held": [], "n": []}
            for v in vals:
                rs = [r for r in rows if r["_id"] == h and r[key] == v]
                T = [seam(r, "turned") for r in rs]
                e["turn_deg"].append(round(st.median([deg(s) for s in T]), 1) if T else None)
                e["force_N"].append(round(st.median([s.get("pad_force_N", 0) for s in T]), 2) if T else None)
                e["slide_mm"].append(round(st.median([s.get("slide_mm", 0) for s in T]), 1) if T else None)
                e["held"].append(sum(1 for s in T if held(s)))
                e["n"].append(len(rs))
            block["rows"].append(e)
        out[name] = block

    # 4. the funnel on the corrected plant at squeeze 10 (budget sweep + relief-0 rows)
    fun_rows = bud + [r for r in rel if r["_l"] == 0.0]
    funnel = []
    for h in sorted(set(r["_id"] for r in fun_rows)):
        rs = [r for r in fun_rows if r["_id"] == h]
        stg = max(rs, key=lambda r: seam(r, "staged").get("cos", -1))
        g = [r for r in rs if carried(seam(r, "gaited"))]
        funnel.append({
            "id": h, "n": len(rs), "bench_hold": BENCH_HOLD.get(h, ""),
            "carried": [sum(1 for r in rs if carried(seam(r, p))) for p in PH],
            "staged_cos": round(seam(stg, "staged").get("cos", 0), 3),
            "staged_N": round(seam(stg, "staged").get("pad_force_N", 0), 2),
            "staged_pads": seam(stg, "staged").get("pad_contacts", 0),
            "final_tilt_med": round(st.median([r.get("final_tilt_deg") or 90 for r in g]), 1) if g else None,
            "turns_med": round(st.median([r.get("turns") or 0 for r in g]), 3) if g else None,
            "stood": sum(1 for r in rs if r.get("stood_ok")),
            "ok": sum(1 for r in rs if r.get("ok")),
        })
    out["funnel"] = {"phases": PH, "rows": funnel, "n": len(fun_rows)}

    # 5. the instrumented D6 rollouts
    probe = {}
    for p in ("shipped", "corrected"):
        r = json.load(open(f"{ROOT}/20260916-plant_probe/20260916-d6_sq10_{p}.json"))
        probe[p] = {"ok": r.get("ok"), "drop": r.get("drop_stage"), "cycles": r.get("cycles_run"),
                    "turns": r.get("turns"), "final_tilt": r.get("final_tilt_deg"),
                    "seams": [{"phase": s["phase"], "cos": s["cos"], "pads": s["pad_contacts"],
                               "N": s["pad_force_N"], "slide": s["slide_mm"],
                               "q_err": s.get("q_err_deg", {}), "sat": s.get("sat", {})}
                              for s in r["seams"]]}
    out["probe"] = probe

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"-> {OUT}  " + ", ".join(f"{k} {v}" for k, v in out["n"].items()))


if __name__ == "__main__":
    main()
