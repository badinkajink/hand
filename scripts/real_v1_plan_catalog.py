#!/usr/bin/env python3
"""Write the station's catalog.json: what simulation expects from each plan on the bench.

    python3 scripts/real_v1_plan_catalog.py \
        --bands docs/experiments/20260830-real_v1-budget-rescreen/deploy_plan_bands.json \
        --clearance docs/experiments/20260830-real_v1-budget-rescreen/deploy_clearance.txt \
        --deploy-dir docs/experiments/20260829-real_v1_deploy/deploy

The station keys this by PLAN FILE STEM, not by design: `sv1_u0060_b75` and `sv1_u0060_b100`
are one hand at two residual clips and they behave nothing alike -- one holds the shaft for the
full 9.6 s and the other drops it -- so a design-level entry would show the same expectation
under both.  Entries that already exist and are not regenerated here (the 200-draw robustness
numbers from DEPLOY.md) are preserved.

Everything written here is SIMULATION.  It is what to expect, and the point of writing it down
before a bench session is that the session can disagree with it.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CLEARANCE_ROW = re.compile(r"^\s{2}(\S+)\s+([-+]?\d+\.\d)\s+([-+]?\d+\.\d)\s+(.*)$")


def recommend(s: dict, budgets: list[float], step: float = 0.05,
              margin: float = 0.10) -> float | None:
    """The clip to ship: the best-aligned point in the band's INTERIOR that the servos allow.

    Not the lower edge.  On the six pilot plans, alignment fell monotonically across the band and
    the edge was the answer; on their own deployed cells it does not -- g12 dips to 0.48 at 0.80
    and comes back to 0.63 at 0.95, and rv04_mid peaks at 0.65 and is worthless by 0.90.  So the
    shape has to be read rather than assumed.  What survives from the edge rule is the MARGIN:
    the yaw joints arrive 4-6 deg (0.07-0.10 rad) short under load, so a clip within 0.10 rad of
    the lower edge is achieved outside the band.
    """
    band = s["band_rad"]
    if not band:
        return None
    short = s["servo_short_deg"]
    ok = [b for b in budgets
          if band[0] + margin - 1e-9 <= b <= band[1] - step + 1e-9
          and short.get(str(b), 0.0) == 0.0]
    return max(ok, key=lambda b: s["cos_by_budget"][str(b)]) if ok else None


def expectation(s: dict, clearance: float | None) -> str:
    kept, reps = s["own_kept"]
    band, own = s["band_rad"], s["budget_rad"]
    if s["servo_short_deg"].get(str(own), 0.0) > 0.0:
        return (f"WILL NOT LOAD: {s['servo_short_deg'][str(own)]:.2f} deg outside a servo's "
                f"range at this clip, and HandRuntime.load_plan refuses it")
    if kept == 0:
        where = ("below its band" if band and own < band[0] else
                 "above its band" if band and own > band[1] else
                 "and it has no band at any clip" if not band else "inside its band")
        return f"drops the shaft in every sim rollout at this clip -- {where}"
    cos = s["own_cos"]
    turn = ("stands the shaft up" if cos >= 0.85 else
            "most of the way up" if cos >= 0.7 else
            "a partial turn" if cos >= 0.4 else
            "holds but barely turns")
    edge = ""
    if band:
        edge = (f", {round((own - band[0]) * 100)} centirad above its band's lower edge"
                if own > band[0] else ", at its band's lower edge")
    tight = (" -- tightest clearance of any shipped plan" if clearance is not None
             and clearance < 8.0 else "")
    return f"holds {kept}/{reps}, {turn} (cos {cos:.2f}){edge}{tight}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bands", type=Path, required=True)
    ap.add_argument("--clearance", type=Path, required=True)
    ap.add_argument("--deploy-dir", type=Path, required=True)
    ap.add_argument("--source", default=None, help="provenance line for the new entries")
    a = ap.parse_args()

    bands = json.loads(a.bands.read_text())["summary"]
    clearance: dict[str, tuple[float, float]] = {}
    for line in a.clearance.read_text().splitlines():
        m = CLEARANCE_ROW.match(line)
        if m and ("run on" in m.group(4) or "NO SAFE PATH" in m.group(4)):
            clearance[m.group(1)] = (float(m.group(2)), float(m.group(3)))

    path = a.deploy_dir / "catalog.json"
    catalog = json.loads(path.read_text()) if path.exists() else {"schema_version": 1,
                                                                 "designs": {}}
    designs = catalog.setdefault("designs", {})

    # rank by simulated alignment among the plans that keep the tool at their own clip
    holders = sorted((t for t, s in bands.items()
                      if s["own_kept"][0] == s["own_kept"][1]
                      and s["servo_short_deg"].get(str(s["budget_rad"]), 0.0) == 0.0),
                     key=lambda t: -bands[t]["own_cos"])
    budgets = sorted(float(b) for b in next(iter(bands.values()))["cos_by_budget"])
    for tag, s in bands.items():
        c = clearance.get(tag)
        worst = min(c) if c else None
        entry = dict(designs.get(tag, {}))
        entry.update({
            "budget_rad": s["budget_rad"],
            "band_rad": s["band_rad"],
            "held_cos": s["own_cos"],
            "held_reps": s["own_kept"],
            "best_cos": s["best_cos"],
            "best_at_rad": s["best_at"],
            "clearance_mm": worst,
            "servo_short_deg": s["servo_short_deg"].get(str(s["budget_rad"]), 0.0),
            "max_commandable_rad": max((b for b in budgets
                                        if s["servo_short_deg"].get(str(b), 0.0) == 0.0),
                                       default=None),
            "recommended_rad": recommend(s, budgets),
            "sim_rank": holders.index(tag) + 1 if tag in holders else None,
            "expect": expectation(s, worst),
        })
        if worst is not None and worst < 5.0:
            entry["expect"] = ("DO NOT RUN: the modelled fingers interpenetrate along this "
                               "trajectory. " + entry["expect"])
        designs[tag] = entry

    catalog["source"] = a.source or (
        "per-plan residual-clip band scan (real_v1_plan_band.py, 33 clips x 4 reps, 9.6 s hold) "
        "and real_v1_trajectory_clearance.py --all; simulation only")
    path.write_text(json.dumps(catalog, indent=2) + "\n")

    print(f"{'plan':22s} {'clip':>5s} {'band':>11s} {'cos':>6s} {'kept':>6s} {'clear':>7s}  rank")
    for tag in sorted(bands, key=lambda t: (bands[t]["own_kept"][0] != bands[t]["own_kept"][1],
                                            -bands[t]["own_cos"])):
        s, c = bands[tag], clearance.get(tag)
        band = f"{s['band_rad'][0]:.2f}-{s['band_rad'][1]:.2f}" if s["band_rad"] else "none"
        print(f"{tag:22s} {s['budget_rad']:5.2f} {band:>11s} {s['own_cos']:6.3f} "
              f"{s['own_kept'][0]:3d}/{s['own_kept'][1]:<2d} "
              f"{(min(c) if c else float('nan')):+7.1f}  {designs[tag]['sim_rank'] or '-'}")
    print(f"\nwrote {path} ({len(bands)} plans)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
