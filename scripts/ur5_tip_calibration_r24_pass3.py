#!/usr/bin/env python3
"""Pass 3 of R24 on the UR5 (owner, 2026-10-25): left-right twins of the grid, slides, twists at 5 N, and a final reference.

Owner, 11:4x: "more wrench calibration at the end of all this"; 11:5x: contact detection "pretty good on the fingertip's
right side, but pretty bad on the left ... the contact direction is in general pretty noisy". The builder's grid has 30
targets at y < -1.5 mm and 19 at y > 1.5 mm, so the two sides are calibrated unequally. Each builder grid target whose
mirror about y = 0 (the nearest surface point of the same region within 1 mm) has no target within 1 mm gets a twin press
at the same levels. The slides of the builder keep only their pre-sliding steps (0.1 to 0.3 mm), since with the slip
steps (0.5 and 1 mm) six of seven came within the margins on mount v6. Twists at 5 N repeat pass 2's at its face points,
with 1 s holds.

Each twin is tried at turns about the vertical from -20 to +40 deg. A turn is kept when the simulator's IK (the
program's geometry, a session's registration) puts joint 5 within 32 to 98 deg (the program's wrist window is 30 to 100)
and joint 6 within -65 to +58 deg (cb1t's run went from -65.8 to +56.7; the lab drops actions above +60) at the rest and
deep poses, and when scripts/ur5_tip_calibration_check.py --clearance finds every margin met. Of the kept turns, the one
with the larger of the two margins in their worst case is taken: the mount's clearance beyond 3.0 mm against a fifth of
joint 5's degrees beyond 30. On 10-25 the first version took the first turn that cleared the cell (+30 deg for 18 twins),
which put joint 5 at 17.7 to 27.6 deg on the robot.

    .venv/bin/python scripts/ur5_tip_calibration_r24_pass3.py --mount v6 --out lab/20261025-plan_sphere_lab_r24_pass3_cb2t.json
        [--joints-session lab/20261025-114653-cb1r] [--check-session lab/20261023-203938-cb1t]
        [--base-plan lab/20261025-plan_sphere_lab_r24_pass1.json]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs/experiments/20261022-ur5_tip_calibration"
BASIS = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-b_v2_surface_basis.npz"
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import ur5_tip_calibration as C  # noqa: E402
import ur5_tip_calibration_sim as S  # noqa: E402

KEEP = ("id", "stage", "kind", "target", "twist", "standoff", "press_N", "moves", "hold_s", "role", "levels", "beyond_mm",
        "v", "stop_N", "fit_N", "max_depth_mm")
TURNS = tuple(range(-20, 45, 5))
J5, J6 = (32.0, 98.0), (-65.0, 58.0)


def clean(a):
    return {k: copy.deepcopy(v) for k, v in a.items() if k in KEEP}


def region(n):
    """The builder's regions (scripts/ur5_tip_calibration_plan.py): face, edge, flank, tip end."""
    if n[2] > 0.7:
        return "tip"
    if abs(n[1]) > 0.7:
        return "flank"
    return "face" if n[0] > 0.9 else ("edge" if n[0] > 0.5 else None)


def twins(grid, pts, nrm):
    """For each target, its mirror about y = 0 on the surface, unless a target already lies within 1 mm of it."""
    have = np.array([a["target"]["p"] for a in grid])
    out = []
    for a in grid:
        p = np.array(a["target"]["p"]) * [1, -1, 1]
        d = np.linalg.norm(pts - p, axis=1)
        i = int(np.argmin(d))
        if d[i] > 1.0 or region(nrm[i]) != a["target"]["region"] or np.min(np.linalg.norm(have - pts[i], axis=1)) < 1.0:
            continue
        b = clean(a)
        b.update(id=f"twin-{a['id'][5:]}", twin_of=a["id"],
                 target=dict(frame="P", p=pts[i].tolist(), n=nrm[i].tolist(), index=i, region=a["target"]["region"]))
        out.append(b)
    return out


def joints(geo, reg, a, turn):
    """(least and largest joint 5, least and largest joint 6), deg, over the press's rest and deep poses."""
    tg, st = a["target"], float(a.get("standoff", 1.5))
    T_IP = geo.T_IP(tg["p"], tg["n"], float(turn), st, (0.0, 0.0), 0.0)
    qh = np.asarray(reg.q_home, float)
    j5, j6 = [], []
    for dz in (0.0, st + 1.0):
        q = S.ik(reg.T_BI @ C.Tm(T_IP[:3, :3], T_IP[:3, 3] - C.EZ * dz) @ C.inv(reg.T_EM @ reg.T_MP), qh)
        if q is None:
            return None
        j5.append(math.degrees(q[4]))
        j6.append(math.degrees(qh[5] + (q[5] - qh[5] + math.pi) % (2 * math.pi) - math.pi))
    return min(j5), max(j5), min(j6), max(j6)


def clearance(plan, session, mount):
    """scripts/ur5_tip_calibration_check.py --clearance on plan's actions: (per action {group: mm}, ids under a margin)."""
    with tempfile.TemporaryDirectory() as td:
        pp, out = Path(td) / "plan.json", Path(td) / "check.json"
        pp.write_text(json.dumps(plan))
        env = dict(os.environ, UR5_MOUNT=mount, NUMPY_MADVISE_HUGEPAGE="0")
        subprocess.run([sys.executable, str(ROOT / "scripts/ur5_tip_calibration_check.py"), str(session), "--clearance",
                        str(pp), "--clearance-stages", "grid,shear,twist,final", "--json", str(out)], env=env, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        c = json.loads(out.read_text())
    return c["clearance_per_action"], set(c["clearance_under"])


def run(a):
    base = json.loads(Path(a.base_plan).read_text())
    pass2 = json.loads((EXP / "lab/20261023-plan_sphere_lab_r22_pass2.json").read_text())
    full = json.loads((EXP / "plan/20261022-plan_sphere.json").read_text())
    geo = C.Geometry(base)
    reg = C.Registration(geo).load(json.loads((Path(a.joints_session) / "registration.json").read_text()))
    b = np.load(BASIS)
    grid = [x for x in full["actions"] if x["stage"] == "grid" and x.get("role") != "ref"]
    cand, jt = [], {}
    for t in twins(grid, b["pts"], b["normals"]):
        for tw in TURNS:
            j = joints(geo, reg, t, tw)
            if j and J5[0] <= j[0] and j[1] <= J5[1] and J6[0] <= j[2] and j[3] <= J6[1]:
                c = dict(copy.deepcopy(t), id=f"{t['id']}-t{tw:+d}", twist=float(tw))
                cand.append(c)
                jt[c["id"]] = j
    rest = []
    for x in full["actions"]:
        if x["stage"] == "shear":
            s = clean(x)
            s["moves"] = [m for m in s["moves"] if max(abs(d) for d in m["degs"]) < 4.0]      # 0.1 to 0.3 mm on R 5 mm
            rest.append(s)
    tw2 = {x["id"]: x for x in pass2["actions"] if x["stage"] == "twist"}
    twf = {x["id"]: x for x in full["actions"] if x["stage"] == "twist"}
    for k in range(4):
        t = clean(tw2.get(f"twist-{k}-2.5N") or twf[f"twist-{k}-2.5N"])
        t.update(id=f"twist-{k}-5.0N", press_N=5.0, hold_s=1.0, moves=[dict(degs=[5.0, 10.0, 15.0, 10.0, 5.0])])
        rest.append(t)
    fr = clean([x for x in pass2["actions"] if x["id"] == "final-ref"][0])
    fr["id"] = "final-ref-p3"                            # pass 2 runs a final-ref in the same session
    rest.append(fr)
    for x in rest:
        j = joints(geo, reg, x, x.get("twist", 0.0) or 0.0)
        jt[x["id"]] = j
    probe = {k: copy.deepcopy(v) for k, v in base.items() if k != "actions"}
    probe["actions"] = cand + rest
    per, under = clearance(probe, a.check_session, a.mount)
    best = {}
    for c in cand:
        if c["id"] in under or c["id"] not in per:
            continue
        j = jt[c["id"]]
        s = min(per[c["id"]]["mount"] - 3.0, (j[0] - 30.0) / 5.0)
        if c["twin_of"] not in best or s > best[c["twin_of"]][0]:
            best[c["twin_of"]] = (s, c)
    acts = [best[k][1] for k in sorted(best)]
    for x in rest:
        j = jt[x["id"]]
        if x["id"] in under or x["id"] not in per or j is None or not (J5[0] <= j[0] and j[1] <= J5[1] and J6[0] <= j[2]
                                                                      and j[3] <= J6[1]):
            print(f"  {x['id']} dropped: clearance {per.get(x['id'])}, joints {j}")
            continue
        acts.append(x)
    left = sorted({c["twin_of"] for c in cand} ^ set(best) | {t["twin_of"] for t in twins(grid, b["pts"], b["normals"])}
                  - {c["twin_of"] for c in cand})
    plan = {k: copy.deepcopy(v) for k, v in base.items() if k != "actions"}
    plan["actions"] = acts
    plan["notes"] = list(base.get("notes") or []) + [
        f"2026-10-25, workstation: R24 pass 3 for mount {a.mount} (scripts/ur5_tip_calibration_r24_pass3.py): {len(best)} "
        f"twins, each in the turn about the vertical that keeps joint 5 within {J5[0]:.0f} to {J5[1]:.0f} deg and joint 6 "
        f"within {J6[0]:.0f} to {J6[1]:.0f} deg ({Path(a.joints_session).name}'s registration, the simulator's IK) and the "
        f"margins of --clearance ({Path(a.check_session).name}'s registration); the builder's slides with their pre-sliding "
        f"steps only; twists at 5 N; final-ref-p3. Twins without such a turn: {', '.join(left) or 'none'}."]
    Path(a.out).write_text(json.dumps(plan, indent=1) + "\n")
    for c in acts:
        j, g = jt[c["id"]], per.get(c["id"], {})
        print(f"  {c['id']}: mount {g.get('mount', float('nan')):.2f} mm, finger {g.get('finger', float('nan')):.2f} mm, "
              f"joint 5 {j[0]:.1f}-{j[1]:.1f}, joint 6 {j[2]:.1f}-{j[3]:.1f} deg")
    print(f"{len(best)} twins, {sum(x['stage'] == 'shear' for x in acts)} slides, {sum(x['stage'] == 'twist' for x in acts)} "
          f"twists; no turn for {left} -> {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mount", default="v6")
    ap.add_argument("--joints-session", default=str(EXP / "lab/20261025-114653-cb1r"))
    ap.add_argument("--check-session", default=str(EXP / "lab/20261023-203938-cb1t"))
    ap.add_argument("--base-plan", default=str(EXP / "lab/20261025-plan_sphere_lab_r24_pass1.json"))
    ap.add_argument("--out", default=str(EXP / "lab/20261025-plan_sphere_lab_r24_pass3_cb2t.json"))
    run(ap.parse_args())
