#!/usr/bin/env python3
"""Hard and fast contacts on v1 ID 3 (cb3t, mount v7), request R26 (owner, 2026-10-26 13:5x: "maybe we can add harder
(much harder?) presses on id 3 ... much harder, much faster rotations/shears/normal presses are scary, but we can build up
to it gradually ... as long we stay within the same joint limit ranges it should be fine ... we have the 10N limit
anyway"). The viewer rejects presses past 3.5 to 4 N: the calibration so far stops at 5 N on the face, 3 N on the edges
and 2 N on the flanks.

Two plans, to run on cb3t's registration after pass 2a and before passes 2b to 6, with the owner present:

  hard1  the reference press (0.5, 1 and 2 N), one press at the reference point to 1, 2, 3.5, 5, 6, 7 and 8 N at 0.15 mm/s,
         and the reference press again. If the two reference presses disagree, the pad has moved in its seat.
  hard2  presses at 0.15 mm/s to 8 N on the face, 6 N on the edges and 4 N on the flanks, ordered by force; presses to 1
         and 3 N at up to 4 mm/s at the six points of the rate set; slides at 4 N, and at 2 N with turns at 6.9 deg/s;
         shear-0 at 4.5 N and shear-3 at 4 N; roll-0 at 4 N, and at 1.5 N at 6.9 deg/s; twist-0 to 15 deg and twist-3 to
         20 deg at 6 N; twist-0 at 2.5 and 5 N at 6.9 deg/s; the reference press after the presses and at the end.

Speeds are actual at the slider's 0.30. A press accelerates at 0.05 m/s^2 from 0.5 mm before the expected surface, which
limits its speed at contact to about 4 mm/s whatever v asks; faster presses need a stronger stop in the program first.

On cb2t a press went 0.461 F^0.782 mm past contact (1.69 mm at 5 N on the face; 2.34 mm at 8 N by the fit). The
clearance check looks 1 mm past contact, so each action's margins must exceed the usual ones (mount 3.0 mm, finger 0.8 mm)
by its depth at the largest force (plus 0.15 mm, the fit's 90th percentile) less 1 mm; the HEX-E keeps 6 mm plus the same. Joint 5 stays at 34 deg or more and
joint 6 within -65 and +50 deg (the cap after cb3t's HEX-E dropout at +54.5 deg on 10-26), by the simulator's IK on cb3t's
registration; a twist's end pose and a slide's or roll's rest pose with 5 deg to spare.

    .venv/bin/python scripts/ur5_tip_calibration_hard_pass.py
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration_r24_pass3 as P3  # noqa: E402

LAB = P3.EXP / "lab"
SES = LAB / "20261026-134522-cb3t"
LEVELS = dict(face=[2.0, 4.0, 5.0, 6.0, 7.0, 8.0], edge=[2.0, 3.0, 4.0, 5.0, 6.0], flank=[1.0, 2.0, 3.0, 4.0])
PICK = dict(flank=6, edge=10, face=12)
RATE_POINTS = ("grid-000", "grid-009", "grid-015", "grid-023", "grid-032", "grid-043")
BASE = dict(mount=3.0, finger=0.8, hex_e=6.0)
J5_MIN, J6 = 34.0, (-65.0, 50.0)
FAST_V, FAST_TURN = 0.0133, 0.4                         # 4 mm/s and 6.9 deg/s actual at the slider's 0.30


def extra(F):
    return max(0.0, 0.461 * F ** 0.782 + 0.15 - 1.0)


def plan(name):
    return json.loads((LAB / f"20261026-plan_sphere_lab_r25_{name}_cb3t.json").read_text())


def fps(P, k):
    """k indices of P by farthest-point sampling, from the point nearest the centroid."""
    if len(P) <= k:
        return list(range(len(P)))
    out = [int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))]
    d = np.linalg.norm(P - P[out[0]], axis=1)
    while len(out) < k:
        out.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(P - P[out[-1]], axis=1))
    return out


def run(a):
    p2b, p3, p5, p6 = (plan(n) for n in ("pass2b", "pass3", "pass5", "pass6"))
    geo = P3.C.Geometry(p2b)
    reg = P3.C.Registration(geo).load(json.loads((SES / "registration.json").read_text()))
    by = {x["id"]: x for p in (p2b, p3, p5, p6) for x in p["actions"]}

    def mk(src, new_id, **kw):
        c = P3.clean(by[src])
        c.update(id=new_id, **kw)
        return c

    def top(c):
        return float(c.get("press_N") or max(c.get("levels") or [c.get("stop_N", 2.0)]))

    presses = []
    for x in list(p2b["actions"]) + list(p6["actions"]):
        if x["kind"] == "press" and x["id"].startswith(("grid-", "dense-")):
            r = x["target"]["region"]
            lv = LEVELS[r]
            presses.append(mk(x["id"], f"hard-{x['id']}", levels=lv, stop_N=lv[-1], v=0.0005, max_depth_mm=4.0, hold_s=0.6))
    fast = [mk(g, f"fast4-{g}", levels=[1.0, 3.0], stop_N=3.0, v=FAST_V, max_depth_mm=3.5, hold_s=0.6) for g in RATE_POINTS]
    slides = sorted({x["id"].rsplit("-", 1)[0] for x in p5["actions"] if x["kind"] == "slide" and x["id"].startswith("slide-")})
    hs = [mk(f"{s}-1N", f"hard-{s}-4N", press_N=4.0) for s in slides]
    fs = [mk(f"{s}-1N", f"fast-{s}-2N", press_N=2.0, turn_v=FAST_TURN) for s in slides]
    turns = [mk("shear-0-3N", "hard-shear-0-4.5N", press_N=4.5), mk("shear-3-2N", "hard-shear-3-4N", press_N=4.0),
             mk("roll-0", "hard-roll-0-4N", press_N=4.0, turn_v=0.016),
             mk("roll-0-tv133", "fast-roll-0-1.5N-tv400", turn_v=FAST_TURN),
             mk("twist-0-5.0N", "hard-twist-0-6N", press_N=6.0, turn_v=0.016),
             mk("twist-3-5.0N-20deg", "hard-twist-3-6N-20deg", press_N=6.0),
             mk("twist-0-2.5N-tv133", "fast-twist-0-2.5N-tv400", turn_v=FAST_TURN),
             mk("twist-0-5.0N-tv133", "fast-twist-0-5N-tv400", turn_v=FAST_TURN)]
    ramp = mk("ref-00", "hard-ramp-ref", levels=[1.0, 2.0, 3.5, 5.0, 6.0, 7.0, 8.0], stop_N=8.0, v=0.0005, max_depth_mm=4.0)
    head = {k: copy.deepcopy(v) for k, v in p2b.items() if k != "actions"}
    probe = dict(head, actions=presses + fast + hs + fs + turns + [ramp])
    per, under = P3.clearance(probe, SES, "v7")

    def ok(c, peak, j6_spare=0.0, end_deg=0.0):
        m = per.get(c["id"])
        if c["id"] in under or m is None:
            return False, "clearance"
        for g, b in BASE.items():
            if g in m and m[g] < b + extra(peak):
                return False, f"{g} {m[g]:.2f} mm"
        tw = float(c.get("twist", 0.0) or 0.0)
        for t in {tw, tw + end_deg}:
            j = P3.joints(geo, reg, c, t)
            if not j or j[0] < J5_MIN or j[1] > P3.J5[1] or j[2] < J6[0] or j[3] > J6[1] - j6_spare:
                return False, f"joints {j}"
        return True, ""

    dropped = []
    keep = {}
    for c in presses:
        good, why = ok(c, top(c))
        if good:
            keep.setdefault(c["target"]["region"], []).append(c)
        else:
            dropped.append((c["id"], why))
    chosen = []
    for r in ("flank", "edge", "face"):                         # ordered by force: 4, 6, then 8 N
        cs = keep.get(r, [])
        idx = fps(np.array([c["target"]["p"] for c in cs]), PICK[r]) if cs else []
        chosen += [cs[i] for i in idx]
    fast_ok = [c for c in fast if ok(c, 6.0)[0]]                # 3 N plus up to 3 N of overshoot at 4 mm/s
    pairs = [(h, f) for h, f in zip(hs, fs) if h["target"]["region"] == "face" and ok(h, 4.0, 5.0)[0] and ok(f, 2.0, 5.0)[0]]
    pick = fps(np.array([h["target"]["p"] for h, _ in pairs]), 6) if pairs else []
    hard_sl, fast_sl = [pairs[i][0] for i in pick], [pairs[i][1] for i in pick]
    turn_ok = []
    for c in turns:
        end = max((max(m["degs"]) for m in c["moves"]), default=0.0) if c["kind"] == "twist" else 0.0
        good, why = ok(c, top(c), 0.0 if c["kind"] == "twist" else 5.0, end)
        (turn_ok if good else dropped).append(c if good else (c["id"], why))
    ramp_ok, why = ok(ramp, 8.0)
    if not ramp_ok:
        raise SystemExit(f"the ramp at the reference point fails: {why}")

    refs = {k: mk("ref-00", f"hard-ref-{k}") for k in "abcd"}
    fin = mk("final-ref", "hard-final-ref")
    note = (f"2026-10-26, workstation: R26 for cb3t (scripts/ur5_tip_calibration_hard_pass.py) on cb3t's registration, "
            f"clearance on v7 with each action's margins raised by its depth at its largest force less 1 mm, joint 5 >= 34 "
            f"deg, joint 6 within -65 and +50 deg. ")
    h1 = dict(copy.deepcopy(head), actions=[refs["a"], ramp, refs["b"]])
    h1["notes"] = list(head.get("notes") or []) + [note + "hard1: the reference press, a press there to 8 N, the reference "
                                                   "press again."]
    acts2 = [refs["c"]] + chosen + fast_ok + [refs["d"]] + [t for t in turn_ok if t["kind"] == "slide"] + hard_sl + fast_sl \
        + [t for t in turn_ok if t["kind"] != "slide"] + [fin]
    h2 = dict(copy.deepcopy(head), actions=acts2)
    h2["notes"] = list(head.get("notes") or []) + [note + f"hard2: {len(chosen)} hard presses, {len(fast_ok)} fast presses, "
                                                   f"{len(hard_sl)} slides at 4 N and {len(fast_sl)} at 2 N fast, "
                                                   f"{len(turn_ok)} shear, roll and twist actions."]
    for n, p in (("hard1", h1), ("hard2", h2)):
        out = LAB / f"20261026-plan_sphere_lab_r26_{n}_cb3t.json"
        out.write_text(json.dumps(p, indent=1) + "\n")
        print(f"{out.name}: {len(p['actions'])} actions")
    reg_n = {r: sum(c["target"]["region"] == r for c in chosen) for r in PICK}
    ys = np.array([c["target"]["p"][1] for c in chosen])
    print(f"  presses by region {reg_n}; y < -1.5: {(ys < -1.5).sum()}, mid {(abs(ys) <= 1.5).sum()}, y > 1.5: {(ys > 1.5).sum()}")
    print(f"  fast presses {[c['id'] for c in fast_ok]}")
    print(f"  slides {[c['id'] for c in hard_sl]}")
    print(f"  turns {[c['id'] for c in turn_ok]}")
    print(f"  least margins: " + ", ".join(f"{c['id']} mount {per[c['id']].get('mount', 0):.2f}" for c in
                                           sorted(chosen + fast_ok + hard_sl + turn_ok + [ramp], key=lambda c: per[c['id']].get('mount', 99))[:4]))
    print(f"  dropped {len(dropped)}: " + "; ".join(f"{i} ({w})" for i, w in dropped if not i.startswith("hard-dense") and not i.startswith("hard-grid"))[:900])
    print(f"  presses dropped {sum(i.startswith(('hard-grid', 'hard-dense')) for i, _ in dropped)} of {len(presses)}; keys {sorted(next(iter(per.values())).keys())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    run(ap.parse_args())
