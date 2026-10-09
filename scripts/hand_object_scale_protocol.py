#!/usr/bin/env python3
"""The bench protocol of the hand-object scale study, generated from the simulated landscape: which cells, at which
grasp, how many trials, in what order, and how many bench hours. Writes
docs/experiments/20261008-hand_object_scale/protocol.json. Nothing here touches the hardware.

Cells. The bench grid is every second layout of the simulated family (t = -1, -0.75, ..., 1). For each object it takes
the feasible bench layouts within one bench step (0.25 in t, 15 mm of thumb-pair span) of the simulated ridge of each
dynamic task (held turn, grasp robustness), so each ridge is bracketed by three bench layouts.

Trials. The turn is scored by the AprilTag; its trial count per cell is the n that puts the 95 % interval of the cell
mean within 3 deg, n = ceil((1.96 s / 3)^2) clamped to [8, 15], with s the larger of the simulated spread over
placements and 4.4 deg, the median spread of held turns per hand in the 2026-09 eight-hand bench record
(paper/figures/ranking.json). Grasp robustness on the bench is the hanging mass at slip: steel washers added in 5 g
steps to a hook under the object until it moves 3 mm (the tag) or drops, two series per cell; it measures the
-z threshold of the simulated ramps, the weakest of the six in most held simulated grips (the page counts them).

Time. 90 s per turn trial including the reset (the eight-hand session of the IROS paper ran 80 trials and eight
reconfigurations in about 2 h); 6 min per ballast series; 30 s per reconfiguration; 45 min per bench day for homing, the tracker check and the
reference cell; 5 h of trials per day.

  .venv/bin/python scripts/hand_object_scale_protocol.py
"""
from __future__ import annotations

import json
import math
import random

import numpy as np
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hand_object_scale_kin as K  # noqa: E402

OUT_DIR = K.OUT_DIR
BENCH_T = [round(-1 + 0.25 * i, 3) for i in range(9)]
SD_BENCH_DEG = 4.4          # median held-turn spread per hand, eight-hand bench record (ranking.json, bench_sd)
CI_DEG = 3.0
N_MIN, N_MAX = 8, 15
BALLAST_SERIES = 2
T_TRIAL_S, T_BALLAST_S, T_RECONF_S, T_DAY_OVERHEAD_S, T_DAY_S = 90.0, 360.0, 30.0, 2700.0, 5 * 3600.0
REF_CELL = ("x100.0_y110.0", "cylinder", 25.0)


def main() -> int:
    L = json.loads((OUT_DIR / "landscape.json").read_text())
    R = json.loads((OUT_DIR / "reach.json").read_text())
    O = {(o["shape"], float(o["d_mm"])): o for o in json.loads((OUT_DIR / "objects.json").read_text())["objects"]}
    F = L["families"]["diag"]
    lays = {round(l["t"], 3): l for l in F["layouts"]}
    feas = {(c["tag"], c["shape"], c["d_mm"]): c["feasible"] for c in R["cells"]}
    # the core set: per shape, four objects with interior ridges for both dynamic tasks, spread evenly in log d
    core = {}
    for shape, ds in K.OBJECT_SET.items():
        # a ridge counts as located when it is not pinned at the gantries' end of the family (|t| < 1); a sphere's
        # dynamic ridges sit at the edges of its feasible band, which is a measurement, not a censoring
        ok = [d for d in ds if all(abs(F["ridges"].get(f"{shape}:{d:g}", {}).get(t, {}).get("at", 1.0)) < 1.0 - 1e-6
                                   for t in ("turn", "hold"))]
        if len(ok) <= 4:
            core[shape] = ok
        else:
            want = np.exp(np.linspace(np.log(ok[0]), np.log(ok[-1]), 4))
            pick = []
            for w in want:
                c_ = min((d for d in ok if d not in pick), key=lambda d: abs(math.log(d / w)))
                pick.append(c_)
            core[shape] = sorted(pick)
    cells = []
    for shape, ds in K.OBJECT_SET.items():
        for d in ds:
            key = f"{shape}:{d:g}"
            col = F["objects"].get(key, {})
            rid = F["ridges"].get(key, {})
            pick = {}
            for task in ("turn", "hold"):
                if task not in rid:
                    continue
                for t in BENCH_T:
                    if abs(t - rid[task]["at"]) <= 0.25 + 1e-9 and feas.get((lays[t]["tag"], shape, float(d))):
                        pick.setdefault(t, set()).add(task)
            for t, why in sorted(pick.items()):
                lay = lays[t]
                c = col.get(lay["tag"], {})
                seeds = c.get("turn_seeds") or []
                sd = statistics.pstdev(seeds) if len(seeds) > 1 else 0.0
                s = max(sd, SD_BENCH_DEG)
                n = int(min(N_MAX, max(N_MIN, math.ceil((1.96 * s / CI_DEG) ** 2)))) if "turn" in why else 0
                cells.append({"tag": lay["tag"], "t": t, "x_sep_mm": lay["x_sep"], "y_sep_mm": lay["y_sep"],
                              "shape": shape, "d_mm": float(d), "near_ridge_of": sorted(why),
                              "set": "core" if d in core.get(shape, []) else "extended",
                              "turn_grasp": c.get("turn_grasp"), "hold_grasp": c.get("hold_grasp"),
                              "sim_turn_deg": c.get("turn"), "sim_turn_sd_deg": round(sd, 2),
                              "sim_hold_N": c.get("hold"), "n_turn": n,
                              "n_ballast": BALLAST_SERIES if "hold" in why else 0,
                              "mass_g": O[(shape, float(d))]["mass_g"]})
    # randomized order per day, with revisits and the reference cell
    rng = random.Random(20261008)
    core_c = [c for c in cells if c["set"] == "core"]
    ext_c = [c for c in cells if c["set"] == "extended"]
    rng.shuffle(core_c)
    rng.shuffle(ext_c)
    order = core_c + ext_c
    days, day, used = [], [], 0.0
    for c in order:
        cost = T_RECONF_S + c["n_turn"] * T_TRIAL_S + c["n_ballast"] * T_BALLAST_S
        if day and used + cost > T_DAY_S:
            days.append(day)
            day, used = [], 0.0
        day.append(c)
        used += cost
    if day:
        days.append(day)
    plan = []
    seen = []
    for i, dl in enumerate(days):
        entries = [{"cell": "reference", "tag": REF_CELL[0], "shape": REF_CELL[1], "d_mm": REF_CELL[2], "n_turn": 5}]
        entries += [{"cell": "study", **{k: c[k] for k in ("tag", "shape", "d_mm", "n_turn", "n_ballast")}} for c in dl]
        revisit = rng.sample(seen, min(2, len(seen))) if seen else []
        entries += [{"cell": "revisit", **{k: c[k] for k in ("tag", "shape", "d_mm")}, "n_turn": 5} for c in revisit]
        seen += dl
        plan.append({"day": i + 1, "entries": entries})
    hours = {
        "turn_trials": sum(c["n_turn"] for c in cells) * T_TRIAL_S / 3600,
        "ballast": sum(c["n_ballast"] for c in cells) * T_BALLAST_S / 3600,
        "reconfiguration": len(cells) * T_RECONF_S / 3600,
        "reference_and_revisits": sum(e["n_turn"] for d_ in plan for e in d_["entries"] if e["cell"] != "study")
        * T_TRIAL_S / 3600,
        "day_overhead": len(plan) * T_DAY_OVERHEAD_S / 3600,
    }
    hours["total"] = sum(hours.values())
    for st in ("core", "extended"):
        cs = [c for c in cells if c["set"] == st]
        hours[f"{st}_trials_only"] = (sum(c["n_turn"] for c in cs) * T_TRIAL_S + sum(c["n_ballast"] for c in cs)
                                      * T_BALLAST_S + len(cs) * T_RECONF_S) / 3600
    core_days = sum(1 for d_ in plan if any(e.get("cell") == "study" and next(
        (c["set"] for c in cells if c["tag"] == e["tag"] and c["shape"] == e["shape"] and c["d_mm"] == e["d_mm"]),
        "") == "core" for e in d_["entries"]))
    objects = sorted({(c["shape"], c["d_mm"]) for c in cells})
    out = {"meta": {"date": "2026-10-08", "bench_t": BENCH_T, "sd_bench_deg": SD_BENCH_DEG, "ci_deg": CI_DEG,
                    "n_min": N_MIN, "n_max": N_MAX, "t_trial_s": T_TRIAL_S, "t_ballast_s": T_BALLAST_S,
                    "t_reconf_s": T_RECONF_S, "t_day_overhead_s": T_DAY_OVERHEAD_S, "t_day_s": T_DAY_S,
                    "seed": 20261008, "reference_cell": REF_CELL},
           "core": core, "core_days": core_days,
           "cells": cells, "days": plan, "hours": {k: round(v, 2) for k, v in hours.items()},
           "objects": [{"shape": s, "d_mm": d, **{k: O[(s, d)][k] for k in ("mass_g", "recipe", "on_target")}}
                       for s, d in objects]}
    (OUT_DIR / "protocol.json").write_text(json.dumps(out, indent=1))
    print(f"{len(cells)} cells over {len(objects)} objects, {sum(c['n_turn'] for c in cells)} turn trials, "
          f"{sum(c['n_ballast'] for c in cells)} ballast series, {len(plan)} bench days, {hours['total']:.1f} h; "
          f"core {core}: {hours['core_trials_only']:.1f} h of trials over {core_days} days")
    return 0


if __name__ == "__main__":
    sys.exit(main())
