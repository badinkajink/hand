#!/usr/bin/env python3
"""Per-hand table of a chain sweep, keyed on whatever the arm tag varies.

    python3 scripts/real_v1_plant_table.py <chain_hands.json> --by b      # budget
    python3 scripts/real_v1_plant_table.py <chain_hands.json> --by sq     # squeeze

Scores the `turned` seam: signed cos (+1 tip down), pad force, slide since closure, and the
held test (>= 2 pads at >= the tool's 0.240 N, floor-free). Turn angle in degrees is
90 - tilt, read off the signed cosine so a handle-down tool reads negative.
"""
from __future__ import annotations
import argparse, json, math, re
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


def knob(r, key):
    m = re.search(rf"_{key}(-?[\d.]+)", r.get("arm", ""))
    return float(m.group(1)) if m else None


def deg(s):
    c = s.get("cos")
    return None if c is None else 90.0 - math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--by", default="b")
    ap.add_argument("--seam", default="turned")
    a = ap.parse_args()
    d = json.load(open(a.path))
    rows = [r for r in d["rows"] if not r.get("error")]
    errs = [r for r in d["rows"] if r.get("error")]
    if errs:
        print(f"{len(errs)} error rows: " + ", ".join(f"{DEP.get(r['tag'], r['tag'])}:{r['arm']}" for r in errs))
    vals = sorted({knob(r, a.by) for r in rows if knob(r, a.by) is not None})
    ks = sorted({round(r.get("axis_k") or 0, 2) for r in rows})
    print(f"{len(rows)} rows  plant {sorted({r.get('plant') for r in rows})}  {a.by} {vals}  pivots {ks}")
    print(f"\n`{a.seam}` seam: turn deg (+ tip down) / N / slide mm; * = held  |  held/n  held&slide<20  chains")
    print(f"  {'hand':5s}{a.by:5s}" + "".join(f"{f'k{k:.2f}':>24s}" for k in ks) + "   held/n <20 chain")
    for h in sorted({DEP.get(r["tag"], r["tag"]) for r in rows}):
        for v in vals:
            line = f"  {h:5s}{v:<5g}"
            nh = n = ns = ch = 0
            for k in ks:
                rs = [r for r in rows if DEP.get(r["tag"], r["tag"]) == h and knob(r, a.by) == v
                      and round(r.get("axis_k") or 0, 2) == k]
                if not rs:
                    line += f"{'-':>24s}"
                    continue
                b = max(rs, key=lambda r: seam(r, a.seam).get("cos", -1))
                t = seam(b, a.seam)
                hs = sum(1 for r in rs if held(seam(r, a.seam)))
                nh += hs
                n += len(rs)
                ns += sum(1 for r in rs if held(seam(r, a.seam)) and seam(r, a.seam).get("slide_mm", 99) < 20)
                ch += sum(1 for r in rs if r.get("ok"))
                line += (f"{deg(t):+6.1f}{'*' if held(t) else ' '} {t.get('pad_force_N', 0):4.1f}N "
                         f"{t.get('slide_mm', 0):4.0f}mm   ")
            line += f"  {nh:2d}/{n:<2d} {ns:3d}   {ch}"
            print(line)
        print()


if __name__ == "__main__":
    main()
