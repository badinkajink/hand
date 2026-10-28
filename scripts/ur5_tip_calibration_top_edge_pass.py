#!/usr/bin/env python3
"""The top-edge pass of v3 ID 1 on the tall indenter, run with the owner present (owner, 2026-10-27: "we can calibrate
the top edge finally. and also we might have to be a little more ambitious with the flank presses"; the plan's README,
docs/experiments/20261027-v3_tip_calibration_and_crosstalk, "The top edge and the flanks at the end").

Targets. The FE surface points (20261020-b_v2_surface_basis.npz) of the tip's end, the edges and the flanks with no
press target of v3 ID 1's plans within --gap mm (1.0): the sphere's stage 2 (as resumed) and morning plans, and the
tall's stage and morning plans. A point is reached as in scripts/ur5_tip_calibration_tall_pass.py (Cell.press_ok): some
turn about the vertical in -20 to +40 deg (steps of 5) keeps joint 5 within 34 to 98 deg and joint 6 within -62 deg and
--j6-max (58, the owner present) at the rest and the deep pose, by the simulator's IK on the session's registration, and
the cell's clearances, with the cap trimmed at x_F -0.10 and the four M2 button heads in the finger's group
(ur5_cb3_cell's defaults), are at least mount 3.0, finger 0.8 (1.5 on the flanks), HEX-E 6.0, pad 0.3 and robot links
8.0 mm, the first three raised by the press's depth at its largest force less 1 mm, and the elbow keeps 20 deg or more
from straight (|q3|). Of the turns that pass, the one with the larger of the mount's margin and a fifth of joint 5's
degrees over 30 is taken. The reached points are thinned
so that none lies within --spacing mm (1.0) of another: the tip's end first, then the edges, then the flanks, and in each
region the points farthest from a press first.

Actions, with ids that no earlier plan of v3 ID 1 uses (the program skips an id already done in a resumed folder):
  top-ref-a, -b    the reference press at the face's centre (index 788, 0.5 to 2 N) first and after the presses;
  press            each target at R29's levels by region (tip end and flanks 0.25 to 2 N, edges 0.25 to 3 N), ordered
                   by joint 6, so that the wrist turns once across the pass;
  slide            each target whose eight directions of turns (1.146 and 2.292 deg about the sphere's centre) pass the
                   press's checks, at 1 N;
  top-final-ref    the reference press again, last.
The plan is then checked as the lab checks it (ur5_tip_calibration_r24_pass3.clearance, mount v7), and actions under a
margin are dropped. Its expected robot time takes each kind's median duration in v2 ID 1's tall session (143529).

Travel. Pressing the tip's end points the finger down at the indenter, so the arm already stands stretched (|q3| 26 to
31 deg at the rests). The program's lifted travel between turned poses rises until the pad stands travel.lift_mm over
the indenter's top; at the plans' 60 mm it brought the elbow within 3.7 deg of straight between top-tip-516,
top-tip-457 and top-edge-571 on 143529's registration (scripts/ur5_tip_calibration_holder_paths.py). The plan sets
lift_mm to --lift-mm (20): no action flagged, the elbow at 26.3 deg or more and every clearance unchanged (at 30 mm,
22.9 deg).

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- .venv/bin/python scripts/ur5_tip_calibration_top_edge_pass.py \\
        --session lab/<v3 ID 1's tall pass 1 session> [--j6-max 58] [--out lab/20261028-plan_tall_lab_top_edge_v3id1.json]
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration_tall_pass as TP  # noqa: E402

P3 = TP.P3
LAB = TP.LAB
LEVELS = dict(tip=[0.25, 0.5, 1.0, 2.0], edge=TP.LEVELS["edge"], flank=TP.LEVELS["flank"])
COVER = ("20261027-plan_sphere_lab_stage2_resume_v3id1.json", "20261027-plan_sphere_lab_morning_v3id1.json",
         "20261027-plan_tall_lab_stage2_v3id1.json", "20261027-plan_tall_lab_morning_v3id1.json")
BASE = LAB / "20261027-plan_tall_lab_stage2_v3id1.json"
DURATIONS_FROM = "20261027-143529-tallv2id1"


def covered():
    """The press targets (P frame, mm) of v3 ID 1's plans."""
    P = []
    for f in COVER:
        P += [a["target"]["p"] for a in json.loads((LAB / f).read_text())["actions"] if a["kind"] == "press"]
    return np.array(P, float)


def durations():
    """Median duration (s) of each kind of action, presses by region, in v2 ID 1's tall session."""
    import ur5_tip_calibration_suite_value as SV
    recs, _ = SV.records(DURATIONS_FROM)
    d = collections.defaultdict(list)
    for s, e, act in recs:
        k = act.get("kind")
        r = (act.get("target") or {}).get("region")
        d[(k, r if k == "press" else None)].append(e["t"] - s["t"])
    return {k: float(np.median(v)) for k, v in d.items()}


def run(a):
    t0 = time.time()
    base = json.loads(Path(a.base_plan).read_text())
    cell = TP.Cell(base, a.session, a.j6_max, a.j6_min, a.finger_flank)
    b = np.load(P3.BASIS)
    pts, nrm = b["pts"], b["normals"]
    reg_of = [P3.region(v) or "" for v in nrm]
    have = covered()
    near = np.array([np.min(np.linalg.norm(have - p, axis=1)) for p in pts])
    cand = [i for i in range(len(pts)) if reg_of[i] in LEVELS and near[i] > a.gap]
    print(f"{len(cand)} points {a.gap} mm or more from v3 ID 1's {len(have)} press targets: "
          + ", ".join(f"{r} {sum(reg_of[i] == r for i in cand)}" for r in LEVELS), flush=True)
    reached = {}
    for k, i in enumerate(cand):
        r = reg_of[i]
        best = None
        for tw in TP.TURNS:
            res = cell.press_ok(pts[i], nrm[i], tw, LEVELS[r][-1], cell.finger_flank if r == "flank" else None)
            if res is None:
                continue
            worst, j = res
            qs = [TP.CELL.SIM.ik(T, cell.qh) for T in cell.deep(pts[i], nrm[i], tw)]
            if any(q is None or abs(np.degrees(np.arctan2(np.sin(q[2]), np.cos(q[2])))) < 20.0 for q in qs):
                continue
            s = min(worst["mount"] - TP.NEED["mount"] - TP.extra(LEVELS[r][-1]), (j[0] - 30.0) / 5.0)
            if best is None or s > best[0]:
                best = (s, tw, worst, j)
        if best:
            reached[i] = best
        if k % 50 == 0:
            print(f"  {k}/{len(cand)} points, {len(reached)} reached, {time.time() - t0:.0f} s", flush=True)
    order = sorted(reached, key=lambda i: (["tip", "edge", "flank"].index(reg_of[i]), -near[i]))
    keep = TP.fps_spacing(pts, a.spacing, order)
    presses, slides = [], []
    for i in keep:
        r = reg_of[i]
        _, tw, worst, j = reached[i]
        lv = LEVELS[r]
        fl = cell.finger_flank if r == "flank" else None
        x = TP.action("press", f"top-{r}-{i:03d}", pts[i], nrm[i], i, r, tw, levels=lv, stop_N=lv[-1],
                      max_depth_mm=2.5 if lv[-1] <= 2.0 else 3.0, hold_s=0.6)
        x["_j6"] = [round(j[2], 1), round(j[3], 1)]
        presses.append(x)
        if cell.slide_ok(pts[i], nrm[i], tw, finger=fl):
            y = TP.action("slide", f"top-slide-{r}-{i:03d}-1N", pts[i], nrm[i], i, r, tw, press_N=1.0,
                          moves=[dict(dir_deg=float(d), degs=TP.SLIDE_DEGS) for d in range(0, 360, 45)],
                          hold_s=0.4, turn_v=0.04)
            y["_j6"] = x["_j6"]
            slides.append(y)
    print(f"{len(reached)} reached, {len(keep)} kept at {a.spacing} mm, {len(slides)} slides, {time.time() - t0:.0f} s",
          flush=True)
    ref = TP.action("press", "top-ref-a", pts[788], nrm[788], 788, "face", 0.0, levels=[0.5, 1.0, 2.0], stop_N=2.0,
                    max_depth_mm=2.5, role="ref")
    by_j6 = lambda x: x["_j6"][0]  # noqa: E731
    acts = ([ref] + sorted(presses, key=by_j6) + [dict(copy.deepcopy(ref), id="top-ref-b")] + sorted(slides, key=by_j6)
            + [dict(copy.deepcopy(ref), id="top-final-ref", stage="final")])
    plan = {k: copy.deepcopy(v) for k, v in base.items() if k != "actions"}
    plan["created"] = a.created
    plan["travel"]["lift_mm"] = float(a.lift_mm)
    plan["actions"] = [{k: v for k, v in x.items() if not k.startswith("_")} for x in acts]
    per, under = P3.clearance(plan, a.session, "v7")
    kept, dropped = [], []
    for x in acts:
        m = per.get(x["id"])
        top = float(x.get("press_N") or x.get("stop_N") or 2.0)
        fl = a.finger_flank if x["target"]["region"] == "flank" else TP.NEED["finger"]
        need = dict(mount=TP.NEED["mount"], finger=fl, hex_e=TP.NEED["hex_e"])
        bad = x["id"] in under or m is None or any(g in m and m[g] < need[g] + TP.extra(top) for g in need)
        (dropped if bad else kept).append(x)
    dur = durations()
    exp_s = sum(dur.get((x["kind"], x["target"]["region"] if x["kind"] == "press" else None),
                        dur.get((x["kind"], "edge" if x["kind"] == "press" else None), 60.0)) for x in kept)
    n_by = {r: sum(x["kind"] == "press" and x["target"]["region"] == r for x in kept) for r in LEVELS}
    s_by = {r: sum(x["kind"] == "slide" and x["target"]["region"] == r for x in kept) for r in LEVELS}
    past = [x["id"] for x in kept if x.get("_j6") and x["_j6"][1] > 45.0]
    plan["actions"] = [{k: v for k, v in x.items() if not k.startswith("_")} for x in kept]
    plan["notes"] = list(base.get("notes") or []) + [
        f"{a.created}, workstation: v3 ID 1's top-edge pass (scripts/ur5_tip_calibration_top_edge_pass.py) on "
        f"{Path(a.session).name}'s registration, with the owner present: presses at {sum(n_by.values())} points with no "
        f"press target of v3 ID 1's plans within {a.gap} mm ({', '.join(f'{r} {v}' for r, v in n_by.items())}), "
        f"{len([x for x in kept if x['kind'] == 'slide'])} slides at 1 N, reference presses top-ref-a, -b and "
        f"top-final-ref; joint 5 >= {TP.J5[0]:.0f} deg, joint 6 within {a.j6_min:g} and {a.j6_max:g} deg ({len(past)} "
        f"actions past +45 deg); cap trimmed at x_F -0.10 with the M2 button heads; margins mount 3.0, finger 0.8 (flanks "
        f"{a.finger_flank:g}), HEX-E 6.0 mm raised by the depth at the largest force less 1 mm; the elbow 20 deg or more "
        f"from straight. travel.lift_mm {a.lift_mm:g} (60 in the other plans): the lifted travel at 60 mm brought the "
        f"elbow within 3.7 deg of straight. No gravity poses."]
    plan["suite"] = dict(stage="4 tall, top edge", tip="v3 ID 1", session=Path(a.session).name,
                         script="scripts/ur5_tip_calibration_top_edge_pass.py", expected_h=round(exp_s / 3600, 2),
                         presses=n_by, slides=s_by, j6_past_45=past)
    out = Path(a.out)
    out.write_text(json.dumps(plan, indent=1) + "\n")
    print(f"{out.name}: {len(kept)} actions, about {exp_s / 3600:.2f} h; presses {n_by}; slides {s_by}; past +45 deg "
          f"{len(past)}; dropped by the check {len(dropped)}: {[x['id'] for x in dropped][:12]}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", default=str(LAB / "20261027-143529-tallv2id1"))
    ap.add_argument("--base-plan", default=str(BASE))
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--j6-max", type=float, default=58.0)
    ap.add_argument("--j6-min", type=float, default=-62.0)
    ap.add_argument("--finger-flank", type=float, default=1.5)
    ap.add_argument("--lift-mm", type=float, default=20.0, help="the plan's travel.lift_mm (60 in the other plans)")
    ap.add_argument("--created", default=time.strftime("%Y-%m-%d %H:%M"))
    ap.add_argument("--out", default=str(LAB / "20261028-plan_tall_lab_top_edge_v3id1.json"))
    run(ap.parse_args())
