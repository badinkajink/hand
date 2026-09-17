#!/usr/bin/env python3
"""Pinch-and-swing table for a chain_hands.json: per hand x released finger x heading."""
import json, sys
D = {"sv1_w6689_b060": "D1", "sv1_w2360_b075": "D2", "sv1_u1364_b080": "D3", "g12_b095": "D4",
     "sv1_u0060_b75": "D5", "sv1_u0308_b050": "D6", "rv05_manual_b85": "D7", "sv1_w0099_b100": "D8"}
W = 0.240
def seam(r, n):
    for s in r.get("seams", []):
        if s.get("phase") == n: return s
    return {}
def held(s):
    # a swung tool hangs 15-20 mm below the pinch, so no height test; two pads at the weight
    return bool(s and s.get("pad_contacts", 0) >= 2 and s.get("pad_force_N", 0) >= W)
rows = []
for p in sys.argv[1:]:
    j = json.load(open(p))
    for r in j["rows"]:
        if r.get("error"): continue
        L, T, R, G, S = (seam(r, k) for k in ("lifted", "turned", "reoriented", "regripped", "staged"))
        rows.append((D.get(r["tag"], r["tag"]), r["plant"], r.get("pinch"), r.get("yaw_deg"), r["seed"], L, T, R, G, S, r.get("drop_stage")))
print(f"{'hand':<4} {'plant':<6} {'pinch':<5} {'yaw':>4} sd | lift N/p | turned cos  p   N | reoriented cos  p   N  held  tipdown | regrip cos | staged cos p N | drop")
for hd, pl, pn, yw, sd, L, T, R, G, S, dr in sorted(rows, key=lambda x: (x[0], x[1], str(x[2]), x[3], x[4])):
    def g(s, k, f="{:+.2f}"):
        v = s.get(k); return "  -  " if v is None else f.format(v)
    print(f"{hd:<4} {pl:<6} {str(pn):<5} {yw:4.0f} {sd:>2} | {g(L,'pad_force_N','{:4.2f}')}/{g(L,'pad_contacts','{:d}')} | "
          f"{g(T,'cos')} {g(T,'pad_contacts','{:d}')} {g(T,'pad_force_N','{:4.2f}')} | "
          f"{g(R,'cos')} {g(R,'pad_contacts','{:d}')} {g(R,'pad_force_N','{:4.2f}')}  {'HELD' if held(R) else '----'}  "
          f"{'TIP' if held(R) and R.get('cos',0) > 0.9 else ('HND' if held(R) and R.get('cos',0) < -0.9 else '   ')} | "
          f"{g(G,'cos')} | {g(S,'cos')} {g(S,'pad_contacts','{:d}')} {g(S,'pad_force_N','{:4.2f}')} | {dr}")
