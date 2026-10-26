#!/usr/bin/env python3
"""Pass 6 of R24 on cb2t (v1 ID 2 on mount v7), after pass 5 (owner, 2026-10-25 16:3x: "improvisationally devise more
tests ... smoothness and accuracy of contact pose detection (more speeds, more variations, more locations/location
density), torsion, contact wrenches").

  density  up to 40 new press targets on the face, edges and flanks, by farthest-point sampling of the FE surface away
           from every target of passes 2b, 3 and 5 (at least 1.5 mm), both sides alike. On cb1r a GP's error at a twin
           fell from 2.10 to 0.70 mm (right side, 1.5 N and over) once the other twins were in training, so the error
           follows the coverage. Each target is pressed once, at one of four contact speeds taken in turn (0.15, 0.6,
           0.045 and 2.0 mm/s actual at the slider's 0.30), so that the speed varies over the face.
  slides   the same targets slid at 1 N, 0.1 and 0.2 mm in eight directions (turn_v 0.04, 0.69 deg/s actual): with the
           other slides in training, cb1r's shear came within 0.10 N and the force's direction within 5.7 deg.
  torsion  twist-0 and twist-3 at 2.5 and 5 N out to 20 deg and back (turn_v 0.016; at 25 deg joint 5 fell to 26 deg): on cb1r the half-difference was
           4.0 N mm at 5 N against 3.7 at 2.5 N at 15 deg, which suggests the contact had not slipped fully.

Turns about the vertical and the checks are those of scripts/ur5_tip_calibration_r24_pass3.py: joint 5 within 32 to 98
deg and joint 6 within -65 to +58 deg by the simulator's IK on cb2t's registration, and the margins of
scripts/ur5_tip_calibration_check.py --clearance on v7.

    .venv/bin/python scripts/ur5_tip_calibration_pass6.py [--n 40]
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
CB2T = LAB / "20261025-152920-cb2t"
LEVELS = dict(face=[0.25, 0.5, 1.0, 2.0, 3.5], edge=[0.25, 0.5, 1.0, 2.0, 3.0], flank=[0.25, 0.5, 1.0, 2.0])
SPEEDS = (0.0005, 0.002, 0.00015, 0.0067)            # m/s: 0.15, 0.6, 0.045 and 2.0 mm/s at the slider's 0.30


def targets(plans):
    pts = []
    for p in plans:
        for a in p["actions"]:
            t = a.get("target")
            if t and t.get("frame") == "P":
                pts.append(t["p"])
    return np.array(pts)


def run(a):
    plans = [json.loads((LAB / f"20261025-plan_sphere_lab_r24_{n}_cb2t.json").read_text()) for n in ("pass2b", "pass3", "pass5")]
    head = plans[2]
    geo = P3.C.Geometry(head)
    reg = P3.C.Registration(geo).load(json.loads((CB2T / "registration.json").read_text()))
    b = np.load(P3.BASIS)
    pts, nrm = b["pts"], b["normals"]
    have = targets(plans)
    reg_of = np.array([P3.region(n) for n in nrm], dtype=object)
    ok = np.array([r in LEVELS for r in reg_of]) & (np.min(np.linalg.norm(pts[:, None] - have[None], axis=2), axis=1) >= 1.5)
    proto = {x["id"]: x for x in plans[0]["actions"] if x["stage"] == "grid"}
    base = P3.clean(next(iter(proto.values())))
    cand, jt = [], {}
    for i in np.nonzero(ok)[0]:
        r = reg_of[i]
        t = dict(base, id=f"dense-{i:03d}", target=dict(frame="P", p=pts[i].tolist(), n=nrm[i].tolist(), index=int(i), region=r),
                 levels=LEVELS[r], stop_N=LEVELS[r][-1])
        for tw in P3.TURNS:
            j = P3.joints(geo, reg, t, tw)
            if j and P3.J5[0] <= j[0] and j[1] <= P3.J5[1] and P3.J6[0] <= j[2] and j[3] <= P3.J6[1]:
                c = dict(copy.deepcopy(t), id=f"{t['id']}-t{tw:+d}", twist=float(tw), src=int(i))
                cand.append(c)
                jt[c["id"]] = j
    # farthest-point order over the candidates' points, seeded by the existing targets
    idx = sorted({c["src"] for c in cand})
    P = pts[idx]
    d = np.min(np.linalg.norm(P[:, None] - have[None], axis=2), axis=1)
    order = []
    for _ in range(min(3 * a.n, len(idx))):
        k = int(np.argmax(d))
        order.append(idx[k])
        d = np.minimum(d, np.linalg.norm(P - P[k], axis=1))
    pool = [c for c in cand if c["src"] in set(order)]
    sl = P3.clean(next(x for x in plans[2]["actions"] if x["id"].startswith("slide-")))
    tw_src = {x["id"]: x for x in plans[1]["actions"] if x["stage"] == "twist"}
    p2 = json.loads((P3.EXP / "lab/20261023-plan_sphere_lab_r22_pass2.json").read_text())
    tw_src |= {x["id"]: x for x in p2["actions"] if x["stage"] == "twist"}
    twists = []
    for k in (0, 3):
        for F in (2.5, 5.0):
            t = P3.clean(tw_src[f"twist-{k}-2.5N"])
            t.update(id=f"twist-{k}-{F:.1f}N-20deg", press_N=F, hold_s=1.0, turn_v=0.016,
                     moves=[dict(degs=[5.0, 10.0, 15.0, 20.0, 15.0, 10.0, 5.0])])   # +25 deg put joint 5 at 26 deg
            twists.append(t)
    probe = {k: copy.deepcopy(v) for k, v in head.items() if k != "actions"}
    probe["actions"] = pool + twists
    per, under = P3.clearance(probe, a.check_session, "v7")
    best = {}
    for c in pool:
        if c["id"] in under or c["id"] not in per:
            continue
        s = min(per[c["id"]]["mount"] - 3.0, (jt[c["id"]][0] - 30.0) / 5.0)
        if c["src"] not in best or s > best[c["src"]][0]:
            best[c["src"]] = (s, c)
    # margins for an unattended run (simulator, 10-25: a slide and the 20 deg twist-0 crossed joint 5's 30 deg, with the
    # press poses at 32 to 33 deg): presses keep 4 deg, slides 8 deg (their turns move joint 5), twists 4 deg at their end
    chosen = [best[i][1] for i in order if i in best and jt[best[i][1]["id"]][0] >= 34.0][: a.n]
    acts = []
    for k, c in enumerate(chosen):
        c = {kk: v for kk, v in c.items() if kk != "src"}
        c["v"] = SPEEDS[k % len(SPEEDS)]
        acts.append(c)
    for c in chosen:
        if jt[c["id"]][0] < 38.0:
            continue
        s = dict(copy.deepcopy(sl), id=f"slide-{c['id']}-1N", target=copy.deepcopy(c["target"]), twist=c["twist"], press_N=1.0)
        acts.append(s)
    for t in twists:
        j_lo, j_hi = P3.joints(geo, reg, t, t.get("twist", 0.0) or 0.0), P3.joints(geo, reg, t, (t.get("twist", 0.0) or 0.0) + 20)
        if t["id"] in under or t["id"] not in per or not all(j and 34.0 <= j[0] and j[1] <= P3.J5[1] and P3.J6[0] <= j[2]
                                                             and j[3] <= P3.J6[1] for j in (j_lo, j_hi)):
            print(f"  {t['id']} dropped: clearance {per.get(t['id'])}, joints {j_lo} to {j_hi}")
            continue
        acts.append(t)
    fr = P3.clean(next(x for x in plans[2]["actions"] if x["id"] == "final-ref-p5"))
    fr["id"] = "final-ref-p6"
    acts.append(fr)
    probe["actions"] = acts
    probe["notes"] = list(head.get("notes") or [])[:-1] + [
        f"2026-10-25, workstation: R24 pass 6 for cb2t (scripts/ur5_tip_calibration_pass6.py): {len(chosen)} new press "
        f"targets by farthest-point sampling at least 1.5 mm from every target of passes 2b, 3 and 5, each at one of four "
        f"contact speeds in turn; the same targets slid at 1 N; twist-0 and twist-3 at 2.5 and 5 N to 20 deg; final-ref-p6. "
        f"Joints on cb2t's registration, clearance on v7 with {Path(a.check_session).name}'s."]
    out = LAB / "20261025-plan_sphere_lab_r24_pass6_cb2t.json"
    out.write_text(json.dumps(probe, indent=1) + "\n")
    y = np.array([c["target"]["p"][1] for c in chosen])
    print(f"{len(chosen)} targets (y < -1.5: {(y < -1.5).sum()}, |y| <= 1.5: {(abs(y) <= 1.5).sum()}, y > 1.5: {(y > 1.5).sum()}; "
          f"regions {dict(zip(*np.unique([c['target']['region'] for c in chosen], return_counts=True)))}), "
          f"{sum(x['id'].startswith('slide') for x in acts)} slides, {sum(x['stage'] == 'twist' for x in acts)} twists -> {out}")
    ms = [per[c["id"]]["mount"] for c in chosen]
    print(f"  mount {min(ms):.2f} mm at least; joint 5 {min(jt[c['id']][0] for c in chosen):.1f} deg at least")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--check-session", default=str(LAB / "20261023-203938-cb1t"))
    run(ap.parse_args())
