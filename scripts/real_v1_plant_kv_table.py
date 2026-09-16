#!/usr/bin/env python3
"""Seam table for a chain_hands.json, one row per rollout: lifted / turned / staged / gait."""
import json, sys
from pathlib import Path
D = {"sv1_w6689_b060": "D1", "sv1_w2360_b075": "D2", "sv1_u1364_b080": "D3", "g12_b095": "D4",
     "sv1_u0060_b75": "D5", "sv1_u0308_b050": "D6", "rv05_manual_b85": "D7", "sv1_w0099_b100": "D8"}
j = json.load(open(sys.argv[1]))
rows = j["rows"]
def seam(r, name):
    for s in r.get("seams", []):
        if s.get("phase") == name: return s
    return {}
def g(s, k, fmt):
    v = s.get(k)
    return "   -  " if v is None else fmt.format(v)
hdr = (f"{'hand':<4} {'plant':<9} {'k':>4} sd | lifted cos  pads  F_N  | turned cos  pads  F_N  slide  qerr(th/ix/md) | staged cos pads  F_N | turns  drop_stage")
print(hdr)
for r in sorted(rows, key=lambda r: (D.get(r['tag'], r['tag']), r['plant'], r['axis_k'], r['seed'])):
    L = seam(r, "lifted"); T = seam(r, "turned"); S = seam(r, "staged")
    qe = T.get("q_err_deg", {})
    qs = f"{qe.get('thumb', 0):4.0f}/{qe.get('index', 0):3.0f}/{qe.get('middle', 0):3.0f}" if qe else "     -     "
    print(f"{D.get(r['tag'], r['tag']):<4} {r['plant']:<9} {r['axis_k']:>4} {r['seed']:>2} | "
          f"{g(L,'cos','{:+.3f}')} {g(L,'pad_contacts','{:4d}')} {g(L,'pad_force_N','{:5.2f}')} | "
          f"{g(T,'cos','{:+.3f}')} {g(T,'pad_contacts','{:4d}')} {g(T,'pad_force_N','{:5.2f}')} {g(T,'slide_mm','{:5.1f}')}  {qs} | "
          f"{g(S,'cos','{:+.3f}')} {g(S,'pad_contacts','{:4d}')} {g(S,'pad_force_N','{:5.2f}')} | "
          f"{r.get('turns', 0):5.2f}  {r.get('drop_stage')}")
