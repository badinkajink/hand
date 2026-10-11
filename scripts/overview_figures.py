#!/usr/bin/env python3
"""Interactive data figures of the topic overviews (scripts/topic_overview_page.py), drawn from the rows the detailed
pages were built from; no simulation runs here. A FIGURES entry of an overview text module names one of these
functions with "fn"; the function returns an interactive_figure.Figure whose caption the overview builder sets.

Figures the overviews keep as images, and why, are listed in scripts/topic_overview_page.py (STATIC_REASON).
"""
from __future__ import annotations

import json
import os

import interactive_figure as IF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E = os.path.join(ROOT, "docs", "experiments")


def gait_endurance(fid: str) -> IF.Figure:
    """Accumulated rotation of the standing tool over 40 gait cycles, three seeds per hand (gait_study.json, arm
    "endurance"); colour by finger layout, dashes for the second stored scene of a layout (_sp40)."""
    rows = [r for r in json.load(open(os.path.join(E, "20260902-real_v1_gait", "gait_study.json"))) if r["arm"] == "endurance"]
    layout = lambda d: d.split("_sp")[0] if "_sp" in d else d.rsplit("_", 1)[0]
    layouts = sorted({layout(r["design"]) for r in rows})
    col = {lay: IF.PALETTE[i % len(IF.PALETTE)] for i, lay in enumerate(layouts)}
    shown = {"rv05_manual_stored", "rv03_narrowy_sp30", "rv04_mid_sp30", "rv00_wide_sp30"}
    series = []
    for r in sorted(rows, key=lambda r: (r["design"], r["seed"])):
        d = r["design"]
        sp = "40" if d.endswith("_sp40") else "30"
        x = [0] + [c["cycle"] for c in r["cycles"]]
        y = [0.0] + [c["spin_deg"] / 360.0 for c in r["cycles"]]
        series.append(IF.Series(f"{d}-s{r['seed']}", f"{d}, seed {r['seed']}", col[layout(d)], {"turns": {"x": x, "med": y}},
                                dash="7 4" if sp == "40" else "", width=1.6,
                                facets={"design": d, "seed": str(r["seed"])}))
    designs = sorted({r["design"] for r in rows})
    legend = [(lay, col[lay], "", [s.id for s in series if layout(s.facets["design"]) == lay]) for lay in layouts]
    legend += [("Stored scene _sp30 (solid)", "#000", "", [s.id for s in series if not s.facets["design"].endswith("_sp40")]),
               ("Stored scene _sp40 (dashed)", "#000", "7 4", [s.id for s in series if s.facets["design"].endswith("_sp40")])]
    panel = IF.Panel("turns", "Accumulated rotation of the standing tool (turns)", (-1.0, 7.0), band=False, dec=3, height=260)
    return IF.Figure(fid, None, "Gait cycle", [panel], series, legend=legend, x_domain=(0, 40), x_dec=0,
                     facets=[{"key": "design", "label": "Hand", "values": [(d, d) for d in designs], "on": sorted(shown)},
                             {"key": "seed", "label": "Seed", "values": [("0", "0"), ("1", "1"), ("2", "2")]}])


def wrist_payload(fid: str) -> IF.Figure:
    """Seeds completing the chain's 8 gait cycles against the wrist stack, with the UR5e told its payload and without
    (bench_study.json, arms "payload" at a 0.30 mm carry squeeze and "stack", relay hand-over), as on the bench
    geometry page."""
    import real_v1_bench_page as B
    rows = json.load(open(os.path.join(E, "20260904-real_v1_bench", "bench_study.json")))["rows"]
    pct = lambda g: 100.0 * g["k"] / g["n"] if g["n"] else float("nan")
    decl = [(s, B.agg(B.sel(rows, arm="payload", stack_mm=float(s), carry_squeeze_mm=0.30, mode="relay"))) for s in (0, 50, 100, 150)]
    und = [(s, B.agg(B.sel(rows, arm="stack", stack_mm=float(s), mode="relay"))) for s in (0, 25, 50, 75, 100, 125, 150)]
    ser = [IF.Series("declared", "Payload declared", IF.PALETTE[1], {"pct": {"x": [s for s, _ in decl], "med": [pct(g) for _, g in decl]}},
                     mode="both", point_labels={"pct": [f"{g['k']} of {g['n']} seeds" for _, g in decl]}),
           IF.Series("undeclared", "Payload not declared", IF.PALETTE[0], {"pct": {"x": [s for s, _ in und], "med": [pct(g) for _, g in und]}},
                     mode="both", dash="7 4", point_labels={"pct": [f"{g['k']} of {g['n']} seeds" for _, g in und]})]
    panel = IF.Panel("pct", "Seeds completing 8 gait cycles (%)", (0.0, 100.0), band=False, dec=1, height=220,
                     vrefs=((100.0, "The bench: 100 mm between the tool flange and the palm plate"),))
    for x in ser:
        x.facets = {"arm": x.id}
    return IF.Figure(fid, None, "Hardware between the tool flange and the palm plate (mm)", [panel], ser,
                     x_domain=(0, 150), x_dec=0, facets=[{"key": "arm", "label": "Arm scene",
                                                         "values": [("declared", "Payload declared"), ("undeclared", "Payload not declared")]}])
