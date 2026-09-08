#!/usr/bin/env python3
"""Collect every chain sweep run on the real_v1 deployed hands into one JSON.

The journey page (scripts/reorientation_journey_page.py) reads only this file, so the
page cannot drift from the sweeps. Rerun after any new docs/experiments/*/chain_hands.json.
"""
from __future__ import annotations
import json, glob, os, statistics as st
from collections import defaultdict

WEIGHT_N = 0.240                      # the screwdriver's own weight
DEP = {"sv1_w6689_b060": "D1", "sv1_w2360_b075": "D2", "sv1_u1364_b080": "D3",
       "g12_b095": "D4", "sv1_u0060_b75": "D5", "sv1_u0308_b050": "D6",
       "rv05_manual_b85": "D7", "sv1_w0099_b100": "D8"}
PHASES = ["lifted", "turned", "reoriented", "regripped", "staged", "set_down", "pressed",
          "handover_thumb", "handover_index", "handover_middle", "handover_grip", "gaited"]
OUT = "docs/experiments/20260908-reorientation_journey/journey.json"


def seam(r, ph):
    for s in (r.get("seams") or []):
        if s.get("phase") == ph:
            return s
    return None


def held(s):
    return bool(s) and (s.get("pad_contacts") or 0) >= 2 \
        and (s.get("pad_force_N") or 0) >= WEIGHT_N and (s.get("z") or 0) > 0.08


def load_rows():
    rows = []
    for f in sorted(glob.glob("docs/experiments/2026090*/chain_hands.json")):
        d = json.load(open(f))
        for r in d["rows"]:
            r["_sweep"] = os.path.basename(os.path.dirname(f))
            r["_mode"] = d.get("stand", "?")
            rows.append(r)
    return rows


def main():
    rows = load_rows()
    air = [r for r in rows if r["_mode"] == "air"]
    out = {"weight_N": WEIGHT_N, "n_rollouts": len(rows), "n_air": len(air),
           "n_sweeps": len({r["_sweep"] for r in rows})}

    # per-sweep ledger
    ledger = []
    for sw in sorted({r["_sweep"] for r in rows}):
        sel = [r for r in rows if r["_sweep"] == sw]
        ledger.append({
            "sweep": sw, "mode": sel[0]["_mode"], "n": len(sel),
            "survive_turn": sum(1 for r in sel if (seam(r, "turned") or {}).get("pad_contacts", 0) >= 1),
            "held_turn": sum(1 for r in sel if held(seam(r, "turned"))
                             and (seam(r, "turned") or {}).get("cos", -1) > 0.7),
            "chain": sum(1 for r in sel if r.get("ok")),
        })
    out["ledger"] = ledger

    # attrition: the first seam with no pad and the tool on the floor
    lost = defaultdict(int)
    for r in air:
        where = "carried"
        for p in PHASES:
            s = seam(r, p)
            if s is None:
                break
            if (s.get("pad_contacts") or 0) == 0 and (s.get("z") or 0) < 0.05:
                where = p
                break
        lost[where] += 1
    out["attrition"] = [{"phase": p, "n": lost[p]} for p in PHASES + ["carried"] if lost.get(p)]

    # per hand
    hands = []
    for tag, hid in sorted(DEP.items(), key=lambda kv: kv[1]):
        sel = [r for r in air if r["run"] == tag]
        if not sel:
            continue
        surv = [r for r in sel if (seam(r, "turned") or {}).get("pad_contacts", 0) >= 1]
        ht = [r for r in sel if held(seam(r, "turned")) and (seam(r, "turned") or {}).get("cos", -1) > 0.7]
        best = max(sel, key=lambda r: ((seam(r, "turned") or {}).get("cos", -1)))
        hands.append({
            "id": hid, "tag": tag, "n": len(sel),
            "survive": len(surv), "held_turn": len(ht),
            "chain": sum(1 for r in sel if r.get("ok")),
            "best_cos": round((seam(best, "turned") or {}).get("cos", 0), 3),
            "best_axis_k": best.get("axis_k"),
        })
    out["hands"] = hands

    # pivot x hand, open loop only
    piv = defaultdict(dict)
    for r in air:
        if (r.get("force_target") or 0) > 0 or (r.get("load_target") or 0) > 0:
            continue
        hid = DEP.get(r["run"])
        s = seam(r, "turned")
        if not hid or not s:
            continue
        k = round(r.get("axis_k") or 0, 2)
        piv[hid].setdefault(k, []).append(s.get("cos") or -1.0)
    ks = sorted({k for v in piv.values() for k in v})
    out["pivot"] = {"axis_k": ks,
                    "rows": [{"id": h, "cos": [round(max(piv[h][k]), 3) if piv[h].get(k) else None
                                               for k in ks]} for h in sorted(piv)]}

    # the force loop as a control
    force = []
    for tgt in sorted({round(r.get("force_target") or 0, 1) for r in air}):
        sel = [r for r in air if abs((r.get("force_target") or 0) - tgt) < 1e-9]
        sl = [(seam(r, "turned") or {}).get("slide_mm", 0) for r in sel]
        force.append({"target_N": tgt, "n": len(sel),
                      "held_turn": sum(1 for r in sel if held(seam(r, "turned"))
                                       and (seam(r, "turned") or {}).get("cos", -1) > 0.7),
                      "median_slide_mm": round(st.median(sl), 1)})
    out["force"] = force

    # the pad-pose tracker as a control
    trk = defaultdict(list)
    for r in rows:
        if r["_sweep"] != "20260906-padtrack":
            continue
        g = next((p[1:] for p in (r.get("arm") or "").split("_") if p.startswith("x")), None)
        trk[(DEP.get(r["run"], r["run"]), g)].append((seam(r, "turned") or {}).get("slide_mm", 0))
    out["track"] = [{"id": k[0], "gain": float(k[1]), "median_slide_mm": round(st.median(v), 1)}
                    for k, v in sorted(trk.items())]

    # D5 at its own pivot, open loop against the force loop
    d5 = []
    for r in air:
        if r["run"] != "sv1_u0060_b75" or abs((r.get("axis_k") or 0) - 0.15) > 1e-9:
            continue
        if r["_sweep"] not in ("20260906-pivot", "20260906-pivot15", "20260906-forceloop"):
            continue
        s = seam(r, "turned") or {}
        d5.append({"force_target": r.get("force_target"), "seed": r.get("seed"),
                   "cos": round(s.get("cos", 0), 3), "slide_mm": round(s.get("slide_mm", 0), 1),
                   "pads": s.get("pad_contacts", 0), "force_N": round(s.get("pad_force_N", 0), 2)})
    # the same configuration appears in several sweeps; one row per distinct outcome
    seen, uniq = set(), []
    for r in sorted(d5, key=lambda r: (r["force_target"], r["seed"])):
        k = (r["force_target"], r["seed"], r["cos"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    out["d5_control"] = uniq

    # every completed air chain, and the D6 cell
    # one row per distinct (hand, pivot, seed); overlapping sweeps re-ran the same cells
    done = {}
    for r in air:
        if not r.get("ok"):
            continue
        k = (r["run"], round(r.get("axis_k") or 0, 2), r.get("seed"))
        e = done.setdefault(k, {"id": DEP.get(r["run"], "--"), "tag": r["run"],
                                "axis_k": r.get("axis_k"), "seed": r.get("seed"), "sweeps": [],
                                "staged_cos": round((seam(r, "staged") or {}).get("cos", 0), 3),
                                "gaited_cos": round((seam(r, "gaited") or {}).get("cos", 0), 3),
                                "slide_max_mm": r.get("slide_max_mm")})
        if r["_sweep"] not in e["sweeps"]:
            e["sweeps"].append(r["_sweep"])
    out["completed"] = [done[k] for k in sorted(done)]

    best = max((r for r in air if r.get("ok") and r["run"] == "sv1_u0308_b050"),
               key=lambda r: (seam(r, "staged") or {}).get("cos", -1))
    out["d6"] = {
        "tag": best["run"], "arm": best.get("arm"), "axis_k": best.get("axis_k"),
        "seed": best.get("seed"), "turn_steps": best.get("turn_steps"),
        "cycles": f"{best.get('cycles_run')}/{best.get('cycles_asked')}",
        "free_frac": best.get("free_frac"), "slide_max_mm": best.get("slide_max_mm"),
        "seams": [{k: s[k] for k in ("phase", "cos", "tilt_deg", "slide_mm",
                                     "pad_contacts", "pad_force_N", "z")}
                  for s in best["seams"]],
    }
    # D6 at pivot 0.15, open loop, grouped by finger-travel budget. The `_f0`/`_w`/`_v`
    # suffixes on the arm tag are force-loop settings that are inert at force_target 0, so
    # rollouts differing only in those are the same configuration and dedupe by seed.
    cell = [r for r in air if r["run"] == "sv1_u0308_b050"
            and abs((r.get("axis_k") or 0) - 0.15) < 1e-9 and (r.get("force_target") or 0) == 0]

    def budget_of(r):
        for p in (r.get("arm") or "").split("_"):
            if p.startswith("b") and p[1:].replace(".", "").isdigit():
                return float(p[1:])
        return None
    bud = defaultdict(dict)
    for r in cell:
        b = budget_of(r)
        if b is None or "k0.15" not in (r.get("arm") or ""):
            continue
        k = r.get("seed")
        bud[b][k] = bud[b].get(k, False) or bool(r.get("ok"))
    out["d6"]["budget"] = [{"budget_rad": b, "seeds": len(v), "chains": sum(v.values())}
                           for b, v in sorted(bud.items())]
    seeds = bud.get(0.5, {})
    out["d6"]["cell_seeds"] = len(seeds)
    out["d6"]["cell_ok"] = sum(seeds.values())
    out["d6"]["pivot15_n"] = len({(r.get("arm"), r.get("seed")) for r in cell})
    out["d6"]["pivot15_ok"] = len({(r.get("arm"), r.get("seed")) for r in cell if r.get("ok")})

    # the one-factor-at-a-time ablation off the working reference
    ab = json.load(open("docs/experiments/20260906-ablate/ablate.json"))
    arms = []
    for a in dict.fromkeys(r["arm"] for r in ab["rows"]):
        sel = [r for r in ab["rows"] if r["arm"] == a]
        drops = [r.get("drop_stage") for r in sel if r.get("drop_stage")]
        arms.append({"arm": a, "n": len(sel), "ok": sum(1 for r in sel if r.get("ok")),
                     "drop": (min(set(drops), key=drops.count) if drops else None),
                     "drops": sorted(set(drops))})
    out["ablation"] = arms

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"-> {OUT}  {out['n_rollouts']} rollouts, {out['n_sweeps']} sweeps, "
          f"{len(done)} completed chains")


if __name__ == "__main__":
    main()
