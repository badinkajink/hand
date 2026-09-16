#!/usr/bin/env python3
"""Shipped plant against the bench-calibrated one, per hand and pivot, from a chain sweep.

    python3 scripts/real_v1_plant_compare.py docs/experiments/20260916-plant_pivot/chain_hands.json

Every cell is scored at the `turned` seam on the signed cosine (+1 tip down), the pad count,
the pad force against the tool's 0.240 N weight, floor clearance, and `slide_mm` -- the
tool's travel in the palm frame since closure, whose hardware ejection separator is 20 mm.
"""
from __future__ import annotations
import json, sys
from collections import defaultdict

W = 0.240
DEP = {"sv1_w6689_b060": "D1", "sv1_w2360_b075": "D2", "sv1_u1364_b080": "D3", "g12_b095": "D4",
       "sv1_u0060_b75": "D5", "sv1_u0308_b050": "D6", "rv05_manual_b85": "D7",
       "sv1_w0099_b100": "D8"}


def seam(r, ph):
    return next((s for s in (r.get("seams") or []) if s.get("phase") == ph), {})


def held(s):
    return (s.get("pad_contacts") or 0) >= 2 and (s.get("pad_force_N") or 0) >= W \
        and (s.get("z") or 0) > 0.08


def main(path: str) -> None:
    d = json.load(open(path))
    rows = d["rows"]
    plants = sorted({r.get("plant", "shipped") for r in rows})
    ks = sorted({round(r.get("axis_k") or 0, 2) for r in rows})
    cell = defaultdict(list)
    for r in rows:
        cell[(DEP.get(r["tag"], r["tag"]), r.get("plant", "shipped"), round(r.get("axis_k") or 0, 2))].append(r)

    print(f"{len(rows)} rollouts  plants {plants}  pivots {ks}\n")
    print("LIFT: pads / N at `lifted`, first seed")
    print(f"  {'hand':6s}" + "".join(f"{p:>22s}" for p in plants))
    for h in sorted({k[0] for k in cell}):
        line = f"  {h:6s}"
        for p in plants:
            rs = [r for (hh, pp, kk), v in cell.items() if hh == h and pp == p for r in v]
            L = seam(rs[0], "lifted") if rs else {}
            line += f"{L.get('pad_contacts', 0):>10d}p {L.get('pad_force_N', 0):6.2f}N   "
        print(line)

    print("\nTURN: best signed cos at `turned` per (hand, plant, pivot); * = held (>=2 pads >=0.240 N, floor-free)")
    hdr = f"  {'hand':6s}{'plant':11s}" + "".join(f"{f'k{k:.2f}':>16s}" for k in ks) + "   held/n  slide<20"
    print(hdr)
    for h in sorted({k[0] for k in cell}):
        for p in plants:
            line = f"  {h:6s}{p:11s}"
            nh = n = ns = 0
            for k in ks:
                rs = cell.get((h, p, k), [])
                if not rs:
                    line += f"{'-':>16s}"
                    continue
                best = max(rs, key=lambda r: seam(r, "turned").get("cos", -1))
                t = seam(best, "turned")
                hs = sum(1 for r in rs if held(seam(r, "turned")))
                nh += hs
                n += len(rs)
                ns += sum(1 for r in rs if held(seam(r, "turned")) and seam(r, "turned").get("slide_mm", 99) < 20)
                mark = "*" if held(t) else " "
                line += f"{t.get('cos', 0):+7.3f}{mark} {t.get('pad_force_N', 0):5.1f}N "
            line += f"   {nh:2d}/{n:<3d}  {ns:2d}"
            print(line)

    print("\nCHAIN: completed (ok) per plant")
    for p in plants:
        oks = [r for r in rows if r.get("plant", "shipped") == p and r.get("ok")]
        print(f"  {p:11s} {len(oks)}  " + ", ".join(f"{DEP.get(r['tag'], r['tag'])}@k{r['axis_k']:.2f}s{r['seed']}" for r in oks))


if __name__ == "__main__":
    main(sys.argv[1])
