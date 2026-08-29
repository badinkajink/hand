#!/usr/bin/env python3
"""Read the deployment-envelope sweeps and say what to build tomorrow.

    uv run python scripts/real_v1_deploy_report.py --envelope <envelope.json> --cells <cells.json>

Three tables, in the order a bench decision needs them:

  1. PER DESIGN, the nominal repeat rate. What fraction of identical rollouts keep the shaft and
     stand it up, under nothing worse than the 0.5 mm spawn jitter the search itself used. A
     design that cannot repeat in simulation cannot repeat on a bench, and the search reported
     these at n=3.
  2. PER AXIS, where each design's cliff is. Read as `the largest perturbation still surviving`,
     which is the number that becomes a hardware tolerance.
  3. THE ENSEMBLE, everything wrong at once. This is the honest prediction for tomorrow.

`win` throughout = kept the shaft (contact, and no more than 20 mm of height lost) AND final
cos >= 0.7. Cosine alone scores a shaft lying on the table at 1.0 and a dropped one at whatever
it happened to land on, which is how a 400-step hold hid a release for a whole day.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

WIN_COS = 0.7


def win(r: dict) -> bool:
    return bool(r.get("ok")) and float(r.get("final_cos", 0.0)) >= WIN_COS


def group(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def rate(rs) -> float:
    return sum(1 for r in rs if win(r)) / max(1, len(rs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envelope", type=Path, required=True)
    ap.add_argument("--cells", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = json.loads(args.envelope.read_text())
    lines: list[str] = []

    def p(s=""):
        print(s)
        lines.append(s)

    designs = sorted({r["design"] for r in rows})
    base = group([r for r in rows if r["axis"] == "baseline"], "design")

    p("=" * 96)
    p("1. NOMINAL REPEATABILITY  (the as-simulated hand, 0.5 mm spawn jitter, nothing else wrong)")
    p("=" * 96)
    p(f"{'design':14} {'n':>3}  {'win rate':>9}  {'mean cos':>9}  {'sd':>6}  "
      f"{'kept':>6}  {'search said':>12}")
    order = sorted(designs, key=lambda t: -rate(base.get((t,), [])))
    for t in order:
        rs = base.get((t,), [])
        if not rs:
            continue
        cs = [r["final_cos"] for r in rs]
        p(f"{t:14} {len(rs):3}  {rate(rs)*100:8.0f}%  {np.mean(cs):+9.3f}  {np.std(cs):6.3f}  "
          f"{sum(1 for r in rs if r['ok'])/len(rs)*100:5.0f}%")

    p()
    p("=" * 96)
    p("2. ONE THING WRONG AT A TIME  (win rate, %, over the repeats at each point)")
    p("=" * 96)
    axes = [a for a in dict.fromkeys(r["axis"] for r in rows)
            if not a.startswith("ensemble") and a != "baseline"]
    for axis in axes:
        sub = [r for r in rows if r["axis"] == axis]
        labels = list(dict.fromkeys(r["label"] for r in sub))
        p(f"\n  {axis}")
        p(f"    {'design':14} " + " ".join(f"{lab:>10}" for lab in labels))
        for t in order:
            cells = [group(sub, "design", "label").get((t, lab), []) for lab in labels]
            if not any(cells):
                continue
            p(f"    {t:14} " + " ".join(
                f"{rate(c)*100:9.0f}%" if c else f"{'—':>10}" for c in cells))

    p()
    p("=" * 96)
    p("3. EVERYTHING WRONG AT ONCE  (independent draws of every axis; level = x plausible error)")
    p("=" * 96)
    ens = sorted({r["axis"] for r in rows if r["axis"].startswith("ensemble")})
    p(f"    {'design':14} " + " ".join(f"{e.replace('ensemble',''):>10}" for e in ens)
      + f"  {'nominal':>10}")
    g = group([r for r in rows if r["axis"].startswith("ensemble")], "design", "axis")
    for t in order:
        cells = [g.get((t, e), []) for e in ens]
        p(f"    {t:14} " + " ".join(f"{rate(c)*100:9.0f}%" if c else f"{'—':>10}" for c in cells)
          + f"  {rate(base.get((t,), []))*100:9.0f}%")

    p()
    p("  Which axis is doing the killing, pooled over designs: the win rate of the draws where")
    p("  that knob was in its worst third, against the draws where it was in its best third.")
    keys = ("mass", "friction", "kp", "torque", "solimp", "solref", "radius", "damping",
            "bias_deg", "mount_mm")
    e10 = [r for r in rows if r["axis"] == "ensemble1.0"]
    p(f"    {'knob':10} {'low third':>10} {'high third':>11}   {'spread':>7}")
    scored = []
    for k in keys:
        vals = [(r["spec"].get(k), r) for r in e10 if r["spec"].get(k) is not None]
        if len(vals) < 30:
            continue
        vals.sort(key=lambda x: x[0])
        n = len(vals) // 3
        lo, hi = rate([v[1] for v in vals[:n]]), rate([v[1] for v in vals[-n:]])
        scored.append((abs(hi - lo), k, lo, hi))
    for spread, k, lo, hi in sorted(scored, reverse=True):
        p(f"    {k:10} {lo*100:9.0f}% {hi*100:10.0f}%   {spread*100:6.0f}pp")
    # Placement is a signed offset, so it is scored on |error| rather than on the signed value.
    for k, lab, scale in (("place", "|place|", 1000.0), ("yaw", "|yaw|", 180.0 / np.pi)):
        vals = []
        for r in e10:
            v = r["spec"].get(k)
            if v is None:
                continue
            vals.append((abs(np.hypot(v[0], v[1]) if isinstance(v, list) else v) * scale, r))
        if len(vals) < 30:
            continue
        vals.sort(key=lambda x: x[0])
        n = len(vals) // 3
        lo, hi = rate([v[1] for v in vals[:n]]), rate([v[1] for v in vals[-n:]])
        p(f"    {lab:10} {lo*100:9.0f}% {hi*100:10.0f}%   {abs(hi-lo)*100:6.0f}pp"
          f"   (<{vals[n-1][0]:.1f} vs >{vals[-n][0]:.1f})")

    if args.cells and args.cells.exists():
        cells = [c for c in json.loads(args.cells.read_text()) if c.get("pose")]
        if cells:
            p()
            p("=" * 96)
            p("4. WHERE TO SET THE OPERATING POINT  (every cell scored under 24 wrong hands)")
            p("=" * 96)
            p("  A cell is only shippable if its NEIGHBOURS work too: axis_k is a commanded")
            p("  geometry and the built hand realises it with millimetres of error. `plateau` is")
            p("  the mean ensemble win rate of the cell and its two neighbours in pivot height.")
            byd = group(cells, "design")
            p(f"  {'design':14} {'best-nominal cell':>22} {'nom':>6} {'ens':>6}   "
              f"{'best-robust cell':>22} {'nom':>6} {'ens':>6} {'plateau':>8}")
            for t in sorted(byd):
                cs = byd[(t,)]
                bn = max(cs, key=lambda c: (c["nom_cos"], c["ens_win"]))
                order_k = defaultdict(list)
                for c in cs:
                    order_k[(c.get("straddle_mm"), c.get("thumb_axial_mm"),
                             c["angle_deg"])].append(c)
                plateau = {}
                for key, group_ in order_k.items():
                    group_.sort(key=lambda c: c["axis_k"])
                    for i, c in enumerate(group_):
                        nb = group_[max(0, i - 1):i + 2]
                        plateau[id(c)] = float(np.mean([x["ens_win"] for x in nb]))
                br = max(cs, key=lambda c: (plateau[id(c)], c["ens_win"]))
                f = lambda c: (f"k{c['axis_k']:.3f} a{c['angle_deg']:.0f} "  # noqa: E731
                               f"s{c.get('straddle_mm') or 0:.0f} t{c.get('thumb_axial_mm') or 0:.0f}")
                p(f"  {t:14} {f(bn):>22} {bn['nom_cos']:6.3f} {bn['ens_win']:6.2f}   "
                  f"{f(br):>22} {br['nom_cos']:6.3f} {br['ens_win']:6.2f} {plateau[id(br)]:8.2f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
