#!/usr/bin/env python3
"""Rod pass 2 of R27 on v1 ID 3 (rod3t, mount v7), with the owner present (owner, 2026-10-26 15:05: "i think it can't hurt
to get higher resolution + some data parity on the rod ... might as well design and tack it on. we can do rolls, twists,
shear, variable speeds/forces on it too. if it truly is redundant, we'll find out!"). Runs on rod pass 1's registration,
every ID starting with rod-:

  presses  R19's grid targets and up to 30 more by farthest-point sampling of the FE surface away from them, at 0.15 mm/s
           to 8 N on the face, 6 N on the edges and 4 N on the flanks, with the standard levels below those, ordered by
           force; a target whose margins fail at that force is pressed to the highest level they allow, 2 N at least;
           the rod's reference press before them, after them and at the end;
  rates    at six face targets (those nearest pass 5's rate points that pass), presses to 2 N at 0.045, 0.6 and
           2.0 mm/s, and to 1 and 3 N at about 4 mm/s;
  slides   around the rod (dir_deg 0 and 180 turn about its axis), 0.1 and 0.2 mm on its 12.5 mm radius, at eight face
           targets at 1, 2 and 4 N with turns at 0.69 deg/s, and at 2 N at 6.9 deg/s. A turn at dir_deg 90 would tilt the
           pad over the contact line;
  twists   about the vertical to +15 deg, at 2.5 and 5 N at 0.28 deg/s and at 5 N at 6.9 deg/s, at the face target where
           joint 5 stays highest at +15 deg. At the rod a turn about the vertical moves joint 5: at the reference point
           +15 deg takes it to 26 deg, and a negative turn swings the mount into the rod (-5.9 mm at -15 deg);
  rolls    about the contact line (dir_deg 0), 3 and 6 deg each way, at R19's two roll points, at 1.5 and 4 N at
           0.28 deg/s and at 1.5 N at 6.9 deg/s.

Speeds are actual at the slider's 0.30. Checks as in R26 (scripts/ur5_tip_calibration_hard_pass.py): the margins (mount
3.0 mm, finger 0.8 mm, HEX-E 6 mm) raised by each action's depth at its largest force less 1 mm, with the sphere's fit
as the bound, since the rod went in less; joint 5 at 34 deg or more; joint 6 within -65 deg and the day's cap (+45 deg
after cb3t's HEX-E dropouts), a turn's end and a slide's or roll's rest with 5 deg to spare. The clearance and the IK
use SESSION's registration: rod pass 1's, or a simulated rod session as its stand-in before pass 1 runs.

    /usr/bin/python3 scripts/ur5_tip_calibration_rod_pass2.py SESSION [--j6-max 45] [--out PLAN]
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration_hard_pass as H  # noqa: E402
import ur5_tip_calibration_r24_pass3 as P3  # noqa: E402

LAB = P3.EXP / "lab"
PASS1 = LAB / "20261026-plan_rod_lab_r27_pass1_rod3t.json"
R19 = P3.EXP / "plan/20261023-plan_rod_lab_reg.json"
OUT = LAB / "20261026-plan_rod_lab_r27_pass2_rod3t.json"
LEVELS = dict(face=[0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 6.5, 8.0], edge=[0.25, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0],
              flank=[0.25, 0.5, 1.0, 2.0, 3.0, 4.0])
N_MORE, MIN_GAP = 30, 1.0                               # new targets; mm from every other target
RATE_V = dict(v015=0.00015, v200=0.002, v667=0.0067)   # 0.045, 0.6 and 2.0 mm/s actual
SLIDE = [{"dir_deg": 0.0, "degs": [0.458, 0.917]}, {"dir_deg": 180.0, "degs": [0.458, 0.917]}]
ROLL = [{"dir_deg": 0.0, "degs": [3.0, 6.0, -3.0, -6.0]}]
TWIST = [{"degs": [5.0, 10.0, 15.0, 10.0, 5.0]}]
SLOW_TURN, SLIDE_TURN = 0.016, 0.04                     # 0.28 and 0.69 deg/s actual; H.FAST_TURN 6.9 deg/s


def run(a):
    p1, r19 = json.loads(PASS1.read_text()), json.loads(R19.read_text())
    ses = Path(a.session)
    geo = P3.C.Geometry(p1)
    reg = P3.C.Registration(geo).load(json.loads((ses / "registration.json").read_text()))
    by = {x["id"]: x for x in r19["actions"]}
    b = np.load(P3.BASIS)
    pts, nrm = b["pts"], b["normals"]
    J6 = (-65.0, a.j6_max)

    def press(src, new_id, region, target=None):
        c = P3.clean(src)
        lv = LEVELS[region]
        c.update(id=new_id, stage="grid", kind="press", levels=lv, stop_N=lv[-1], v=0.0005, max_depth_mm=4.0, hold_s=0.6)
        c.pop("role", None)
        if target is not None:
            c["target"] = target
        return c

    grid = [x for x in r19["actions"] if x["stage"] == "grid" and x.get("role") != "ref"]
    old = [press(x, f"rod-{x['id']}", x["target"]["region"]) for x in grid]
    have = np.array([x["target"]["p"] for x in grid] + [by["ref-00"]["target"]["p"]])
    cand = []
    for i, (p, n) in enumerate(zip(pts, nrm)):
        r = P3.region(n)
        if r in LEVELS and np.min(np.linalg.norm(have - p, axis=1)) >= MIN_GAP:
            t = dict(frame="P", p=[round(float(v), 4) for v in p], n=[round(float(v), 5) for v in n], index=i, region=r)
            cand.append(press(by["ref-00"], f"rod-more-{i:03d}", r, t))
    ref = dict(P3.clean(by["ref-00"]), id="rod-ref-a")
    rates = []
    face = [c for c in old if c["target"]["region"] == "face"]
    order, used = [], set()                             # each rate point's nearest face target first, then the next nearest
    for rank in range(len(face)):
        for q in P3.RATE_POINTS:
            near = sorted(face, key=lambda c: np.linalg.norm(np.array(c["target"]["p"]) - q))
            if near[rank]["id"] not in used:
                used.add(near[rank]["id"])
                order.append(near[rank])
    for g in order:
        k = g["id"][len("rod-grid-"):]
        for tag, v in RATE_V.items():
            rates.append(dict(copy.deepcopy(g), id=f"rod-rate-{k}-{tag}", v=v, levels=[0.25, 0.5, 1.0, 2.0], stop_N=2.0,
                              max_depth_mm=2.5))
        rates.append(dict(copy.deepcopy(g), id=f"rod-fast4-{k}", v=H.FAST_V, levels=[1.0, 3.0], stop_N=3.0, max_depth_mm=3.5))

    def turn(src, new_id, kind, stage, moves, F, tv):
        c = P3.clean(src)
        for k in ("levels", "stop_N", "v", "max_depth_mm", "role", "fit_N"):
            c.pop(k, None)
        c.update(id=new_id, kind=kind, stage=stage, moves=copy.deepcopy(moves), press_N=F, turn_v=tv, hold_s=0.4)
        return c

    slides = []
    for c in face:
        k = c["id"][len("rod-grid-"):]
        slides.append([turn(c, f"rod-slide-{k}-{F:g}N", "slide", "shear", SLIDE, F, SLIDE_TURN) for F in (1.0, 2.0, 4.0)]
                      + [turn(c, f"rod-slide-{k}-2N-tv400", "slide", "shear", SLIDE, 2.0, H.FAST_TURN)])
    tw_src = sorted(face, key=lambda c: -(P3.joints(geo, reg, c, 15.0) or (0.0,))[0])[:3]
    twists = [[turn(c, f"rod-twist-{c['id'][len('rod-grid-'):]}-{F:g}N" + ("-tv400" if tv == H.FAST_TURN else ""), "twist",
                    "twist", TWIST, F, tv) for F, tv in ((2.5, SLOW_TURN), (5.0, SLOW_TURN), (5.0, H.FAST_TURN))] for c in tw_src]
    rolls = []
    for src in ("roll-0", "roll-3"):
        rolls += [turn(by[src], f"rod-{src}-1.5N", "roll", "roll", ROLL, 1.5, SLOW_TURN),
                  turn(by[src], f"rod-{src}-4N", "roll", "roll", ROLL, 4.0, SLOW_TURN),
                  turn(by[src], f"rod-{src}-1.5N-tv400", "roll", "roll", ROLL, 1.5, H.FAST_TURN)]
    head = {k: copy.deepcopy(v) for k, v in p1.items() if k not in ("actions", "notes")}
    probe = dict(head, actions=[ref] + old + cand + rates + [s for g in slides for s in g] + [t for g in twists for t in g]
                 + rolls)
    per, under = P3.clearance(probe, ses, "v7")

    def ok(c, peak, j6_spare=0.0, end_deg=0.0):
        m = per.get(c["id"])
        if c["id"] in under or m is None:
            return False, "clearance"
        for g, base in H.BASE.items():
            if g in m and m[g] < base + H.extra(peak):
                return False, f"{g} {m[g]:.2f} mm"
        tw = float(c.get("twist", 0.0) or 0.0)
        for t in {tw, tw + end_deg}:
            j = P3.joints(geo, reg, c, t)
            if not j or j[0] < H.J5_MIN or j[1] > P3.J5[1] or j[2] < J6[0] or j[3] > J6[1] - j6_spare:
                return False, f"joints {None if not j else tuple(round(v, 1) for v in j)}"
        return True, ""

    dropped = []

    def keep(cs, peak_of, **kw):
        out = []
        for c in cs:
            good, why = ok(c, peak_of(c), **kw)
            (out.append(c) if good else dropped.append((c["id"], why)))
        return out

    top = lambda c: float(c.get("stop_N") or c.get("press_N"))   # noqa: E731

    def lower(cs):
        """Each press at the highest level of its region that its margins allow (2 N at least); the rest dropped."""
        out = []
        for c in cs:
            lv = LEVELS[c["target"]["region"]]
            why = ""
            for t in sorted((x for x in lv if x >= 2.0), reverse=True):
                good, why = ok(c, t)
                if good:
                    out.append(dict(c, levels=[x for x in lv if x <= t], stop_N=t))
                    break
                if not why.startswith(("mount", "finger", "hex_e")):
                    break                               # clearance at every force, or the joints: no lower level helps
            else:
                why = why or "no level"
            if not out or out[-1]["id"] != c["id"]:
                dropped.append((c["id"], why))
        return out

    old_ok = lower(old)
    n_drop = len(dropped)
    more_ok = lower(cand)
    why_more = collections.Counter(w.split()[0] for _, w in dropped[n_drop:])
    del dropped[n_drop:]
    picked, P = [], np.array([c["target"]["p"] for c in old_ok] + [by["ref-00"]["target"]["p"]])
    if more_ok:                                         # farthest from the targets kept so far, one at a time
        Q = np.array([c["target"]["p"] for c in more_ok])
        d = np.min(np.linalg.norm(Q[:, None] - P[None], axis=2), axis=1)
        while len(picked) < N_MORE and d.max() >= MIN_GAP:
            i = int(np.argmax(d))
            picked.append(more_ok[i])
            d = np.minimum(d, np.linalg.norm(Q - Q[i], axis=1))
    presses = []
    for r in ("flank", "edge", "face"):                 # by force: 4, 6, then 8 N
        presses += [c for c in old_ok + picked if c["target"]["region"] == r]
    rates_ok, n_rate = [], 0
    for k in range(0, len(rates), 4):                   # a point's four rate presses stay together
        grp = rates[k:k + 4]
        if n_rate < 6 and all(ok(c, 6.0 if c["v"] == H.FAST_V else 2.0)[0] for c in grp):   # 3 N + up to 3 N of overshoot
            rates_ok += grp
            n_rate += 1
    sl_ok = [g for g in slides if all(ok(c, top(c), 5.0)[0] for c in g)]
    pick = H.fps(np.array([g[0]["target"]["p"] for g in sl_ok]), 8) if sl_ok else []
    sl = [c for i in pick for c in sl_ok[i]]
    tw_ok = next((g for g in twists if all(ok(c, top(c), end_deg=15.0)[0] for c in g)), [])
    if not tw_ok:
        dropped.append(("rod-twist-*", "no face target passes"))
    ro_ok = keep(rolls, top, j6_spare=5.0)
    good, why = ok(ref, 2.0)
    if not good:
        raise SystemExit(f"the reference press fails: {why}")
    refs = {k: dict(ref, id=f"rod-ref-{k}") for k in "abc"}
    fin = dict(P3.clean(by["final-ref"]), id="rod-final-ref")
    acts = [refs["a"]] + presses + [refs["b"]] + rates_ok + [refs["c"]] + sl + ro_ok + tw_ok + [fin]
    ids = [c["id"] for c in acts]
    assert len(ids) == len(set(ids)) and all(i.startswith("rod-") for i in ids)
    plan = dict(copy.deepcopy(head), actions=acts)
    plan["notes"] = [f"2026-10-26, lab: R27 rod pass 2 (scripts/ur5_tip_calibration_rod_pass2.py) with {ses.name}'s "
                     f"registration: {len(presses)} presses ({len(old_ok)} of R19's {len(old)} grid targets, {len(picked)} "
                     f"more), {len(rates_ok)} rate presses, {len(sl)} slides, {len(ro_ok)} rolls, {len(tw_ok)} twists and four "
                     f"reference presses. Clearance on v7 with each action's margins raised by its depth at its largest force "
                     f"less 1 mm, joint 5 >= 34 deg, joint 6 within -65 and +{a.j6_max:g} deg."]
    out = Path(a.out)
    out.write_text(json.dumps(plan, indent=1) + "\n")
    print(f"{out.name}: {len(acts)} actions")
    print(f"  presses {len(presses)}: R19's {len(old_ok)} of {len(old)}, {len(picked)} more of {len(more_ok)} candidates that "
          f"pass ({len(cand)} on the surface); by region "
          + str({r: sum(c['target']['region'] == r for c in presses) for r in LEVELS}))
    print(f"  rates {len(rates_ok) // 4} points; slides {len(sl) // 4} points ({len(sl_ok)} pass); rolls {[c['id'] for c in ro_ok]}; "
          f"twists {[c['id'] for c in tw_ok]}")
    least = sorted(acts, key=lambda c: per.get(c["id"], {}).get("mount", 99))[:4]
    print("  least mount: " + ", ".join(f"{c['id']} {per[c['id']]['mount']:.2f}" for c in least if c["id"] in per))
    print(f"  the candidates failed on: {dict(why_more)}")
    print(f"  dropped {len(dropped)}: " + "; ".join(f"{i} ({w})" for i, w in dropped)[:1500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", help="a rod session with a registration (rod pass 1, or a simulated stand-in)")
    ap.add_argument("--j6-max", type=float, default=45.0, help="the day's cap on joint 6, deg")
    ap.add_argument("--out", default=str(OUT))
    run(ap.parse_args())
