#!/usr/bin/env python3
"""Aggregate pinch-and-swing sweeps: per (hand, plant, squeeze, thumb station, pinch arm)."""
import json, sys, re, collections
D = {"sv1_w6689_b060": "D1", "sv1_w2360_b075": "D2", "sv1_u1364_b080": "D3", "g12_b095": "D4",
     "sv1_u0060_b75": "D5", "sv1_u0308_b050": "D6", "rv05_manual_b85": "D7", "sv1_w0099_b100": "D8"}
W = 0.240
def seam(r, n):
    for s in r.get("seams", []):
        if s.get("phase") == n: return s
    return {}
def held(s): return bool(s and s.get("pad_contacts", 0) >= 2 and s.get("pad_force_N", 0) >= W)
agg = collections.OrderedDict()
for p in sys.argv[1:]:
    j = json.load(open(p))
    for r in j["rows"]:
        if r.get("error"): continue
        tag = r["tag"]; parent = r.get("parent", tag)
        ta = re.search(r"_ta(\d+)", tag); ta = int(ta.group(1)) if ta else 10
        sq = re.search(r"_sq(\d+)", r["arm"]); sq = int(sq.group(1)) if sq else 2
        key = (D.get(parent, parent), r["plant"], sq, ta, r.get("pinch"), r.get("yaw_deg"))
        L, T, R, G, S = (seam(r, k) for k in ("lifted", "turned", "reoriented", "regripped", "staged"))
        a = agg.setdefault(key, {"n": 0, "lifted": 0, "swung": 0, "swung_held": 0, "regrip_up": 0, "staged_held": 0, "carried": 0, "cos_r": [], "N_r": []})
        a["n"] += 1
        a["lifted"] += held(L)
        a["swung"] += R.get("cos", 0) > 0.85
        a["swung_held"] += R.get("cos", 0) > 0.85 and held(R)
        a["regrip_up"] += G.get("cos", 0) > 0.85 and held(G)
        a["staged_held"] += S.get("cos", 0) > 0.85 and held(S)
        a["carried"] += r.get("drop_stage") is None
        if held(R): a["cos_r"].append(R["cos"]); a["N_r"].append(R["pad_force_N"])
print(f"{'hand':<4} {'plant':<6} {'sq':>3} {'thumb':>5} {'pinch':<12} {'yaw':>4} | {'n':>2} {'lift':>4} {'swung':>5} {'+held':>5} {'regrip>0.85':>11} {'staged':>6} {'carried':>7} | reoriented cos (held) / N")
for k, a in agg.items():
    cr = f"{min(a['cos_r']):+.2f}..{max(a['cos_r']):+.2f} / {min(a['N_r']):.1f}..{max(a['N_r']):.1f}" if a["cos_r"] else "-"
    print(f"{k[0]:<4} {k[1]:<6} {k[2]:>3} {k[3]:>5} {str(k[4]):<12} {k[5]:4.0f} | {a['n']:>2} {a['lifted']:>4} {a['swung']:>5} {a['swung_held']:>5} {a['regrip_up']:>11} {a['staged_held']:>6} {a['carried']:>7} | {cr}")
