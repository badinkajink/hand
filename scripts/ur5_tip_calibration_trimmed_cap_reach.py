#!/usr/bin/env python3
"""Reach of the tall indenter's presses with the trimmed tip cap (owner, 2026-10-27: "i trimmed the top cap so that there
is no "fingertip" coming out to the surface of the fingertip now ... what this means now is that we can calibrate the top
edge finally. and also we might have to be a little more ambitious with the flank presses").

For every FE surface point of the face, edges, flanks and tip end (ur5_tip_calibration_r24_pass3.BASIS and region), whether
a press reaches it on a session's registration with the checks of scripts/ur5_tip_calibration_tall_pass.py (Cell.press_ok):
a turn about the vertical in -20 to +40 deg (steps of 5) at which, at the rest and the deep pose, the simulator's IK keeps
joint 5 within 34 to 98 deg and joint 6 within -62 deg and the cap, the HEX-E's cable port stands 15 mm from the arm, and
the clearances to the tall indenter, the plan's other indenters, the board and the table are at least mount 3.0, finger m,
HEX-E 6.0, pad 0.3 and robot links 8.0 mm, the first three raised by the press's depth at its largest force less 1 mm
(face 5 N, edge 3 N, flank and tip 2 N, the plan builder's top levels). The combinations: finger margin m 0.8 or 1.5 mm,
joint 6 capped at +45 or +58 deg, the cap trimmed at x_F -0.10 (ur5_cb3_cell.CAP_TRIM_X) or as printed before. Each pose is
evaluated once; the finger's clearance is taken with either cap's samples. --validate N compares N points' verdicts with
Cell.press_ok itself.

The four M2 x 8 cheese heads (3.8 mm across, 1.3 mm tall, seated at x_F -0.10 on the cassette's walls and the cap's
blocks, as scripts/dip_b_v4.py screws_sweep models them) are not among the cell's tool parts. They stand 1.3 mm palmar of
the trimmed cap's face, so their clearance is reported beside the checks, and a third set of counts, "trimmed_heads", adds
them to the finger's group.

    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- env NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python \\
        scripts/ur5_tip_calibration_trimmed_cap_reach.py [--session lab/20261027-143529-tallv2id1] [--validate 12]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("UR5_MOUNT", "v7")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_cb3_cell as CELL  # noqa: E402
import ur5_tip_calibration_r24_pass3 as P3  # noqa: E402
import ur5_tip_calibration_tall_pass as TP  # noqa: E402

C = CELL.C
LAB = P3.EXP / "lab"
OUT = ROOT / "docs/experiments/20261026-tall_indenter/data/20261027-trimmed_cap_reach.json"
LEVELS = dict(TP.LEVELS, tip=[0.25, 0.5, 1.0, 2.0])         # the tip end's levels of scripts/ur5_tip_calibration_plan.py
REGIONS = ("face", "edge", "flank", "tip")
MARGINS = (0.8, 1.5)
J6_CAPS = (45.0, 58.0)
CAPS = ("trimmed", "old", "trimmed_heads")
FINGER = ("dip_link", "cover")                              # with the cap: the finger's group
MOUNT = ("adapter", "fork", "clamp_plate")
HEADS = dict(d=3.8, h=1.3, seat_x_F=-0.10)                 # DIN 84 M2: dk 3.8, k 1.3; dip_link_concept_b HEAD_D, HEAD_H


def heads_points(spacing=0.5):
    """Surface samples (M) of the four M2 cheese heads, axes along x_F at the holes of ur5_cb3_cell.TIP_F."""
    import trimesh
    out = []
    for y, z in CELL.TIP_F["holes"]:
        m = trimesh.creation.cylinder(radius=HEADS["d"] / 2, height=HEADS["h"], sections=36)
        m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0.0, 1.0, 0.0]))     # axis along x
        m.apply_translation([HEADS["seat_x_F"] - HEADS["h"] / 2, y, z])
        out.append(trimesh.sample.sample_surface_even(m, int(m.area / spacing ** 2), seed=0)[0])
    return C.apply(CELL.T_M_FROM_F, np.vstack(out))


def combo_key(cap, m, j6):
    return f"{cap}_finger{m:g}_j6max{j6:g}"


def _clear(cell, T_BE, P_M):
    T_IM = C.inv(cell.reg.T_BI) @ T_BE @ cell.reg.T_EM
    PI = C.apply(T_IM, P_M)
    return min(float(CELL.indenter_sdf(PI, cell.kind).min()), float(CELL.cell_sdf(PI, cell.reg.T_BI).min()),
               float(CELL.other_sdf(C.apply(T_BE @ cell.reg.T_EM, P_M)).min()))


def evaluate(cell, p, n, tw, old_cap, heads):
    """One turn of a press: the joints at the rest and deep poses and each group's least clearance over both, with the
    trimmed and with the old cap, and the screw heads'; or why it fails before the clearances (ik, j5, j6, port)."""
    js, worst = [], {}
    for T_BE in cell.deep(p, n, tw):
        q = CELL.SIM.ik(T_BE, cell.qh)
        if q is None:
            return dict(why="ik")
        j5 = math.degrees(q[4])
        j6 = math.degrees(cell.qh[5] + (q[5] - cell.qh[5] + math.pi) % (2 * math.pi) - math.pi)
        if not TP.J5[0] <= j5 <= TP.J5[1]:
            return dict(why="j5", j=[round(j5, 1), round(j6, 1)])
        if not cell.j6[0] <= j6 <= cell.j6[1]:
            return dict(why="j6", j=[round(j5, 1), round(j6, 1)])
        if CELL.self_clearance(q, cell.reg.T_EM)["port"] < 15.0:
            return dict(why="port", j=[round(j5, 1), round(j6, 1)])
        cl = CELL.clearance(q, T_BE, cell.reg.T_BI, cell.reg.T_EM, cell.kind, contact_P=p)
        cl["cap_old"] = _clear(cell, T_BE, old_cap)
        cl["heads"] = _clear(cell, T_BE, heads)
        js.append((j5, j6))
        for k, v in cl.items():
            worst[k] = min(worst.get(k, 1e9), float(v))
    g = dict(mount=min(worst[k] for k in MOUNT), hex_e=worst["hex_e"], pad=worst["pad"],
             robot=min(v for k, v in worst.items() if k not in MOUNT + FINGER + ("cap", "cap_old", "heads", "hex_e", "pad")),
             finger_trimmed=min(worst["dip_link"], worst["cover"], worst["cap"]),
             finger_old=min(worst["dip_link"], worst["cover"], worst["cap_old"]),
             cap_trimmed=worst["cap"], cap_old=worst["cap_old"], heads=worst["heads"])
    g["finger_trimmed_heads"] = min(g["finger_trimmed"], g["heads"])
    return dict(why=None, j5=[min(j[0] for j in js), max(j[0] for j in js)], j6=[min(j[1] for j in js), max(j[1] for j in js)],
                clear=g)


def passes(e, region, cap, m, j6cap):
    if e["why"] is not None or e["j6"][1] > j6cap:
        return False
    r = TP.extra(LEVELS[region][-1])
    c = e["clear"]
    need = dict(mount=TP.NEED["mount"] + r, hex_e=TP.NEED["hex_e"] + r, pad=TP.NEED["pad"], robot=TP.NEED["robot"])
    return all(c[k] >= v for k, v in need.items()) and c[f"finger_{cap}"] >= max(TP.NEED["finger"], m) + r


def failing(e, region, cap, m, j6cap):
    """Why a turn fails in one combination: the joint check, or the groups under their margins."""
    if e["why"] is not None:
        return e["why"]
    if e["j6"][1] > j6cap:
        return "j6"
    r = TP.extra(LEVELS[region][-1])
    c = e["clear"]
    need = dict(mount=TP.NEED["mount"] + r, hex_e=TP.NEED["hex_e"] + r, pad=TP.NEED["pad"], robot=TP.NEED["robot"],
                finger=max(TP.NEED["finger"], m) + r)
    bad = [k for k, v in need.items() if (c[f"finger_{cap}"] if k == "finger" else c[k]) < v]
    return ",".join(bad) or "ok"


def main(a):
    t0 = time.time()
    session = Path(a.session)
    base = json.loads(Path(a.base_plan or session / "plan.json").read_text())
    cell = TP.Cell(base, session, max(J6_CAPS), a.j6_min)
    old_cap = CELL.part_points("cap", CELL.finger_mesh("cap", trim=None))
    heads = heads_points()
    b = np.load(P3.BASIS)
    pts, nrm = b["pts"], b["normals"]
    reg = [P3.region(v) or "" for v in nrm]
    idx = [i for i in range(len(pts)) if reg[i] in REGIONS]
    if a.limit:
        idx = idx[::max(1, len(idx) // a.limit)]
    combos = [(cap, m, j6) for cap in CAPS for m in MARGINS for j6 in J6_CAPS]
    T_FP = np.linalg.inv(np.array([[-1.0, 0, 0, -2.05], [0, 1.0, 0, 0], [0, 0, -1.0, -77.05], [0, 0, 0, 1.0]]))
    rows = []
    for k, i in enumerate(idx):
        r = reg[i]
        ev = {tw: evaluate(cell, pts[i], nrm[i], tw, old_cap, heads) for tw in TP.TURNS}
        pF = C.apply(T_FP, pts[i][None])[0]
        row = dict(i=i, region=r, p_P=np.round(pts[i], 3).tolist(), x_F=round(float(pF[0]), 3), y_F=round(float(pF[1]), 3),
                   z_F=round(float(pF[2]), 3),
                   turns={combo_key(*c): [tw for tw in TP.TURNS if passes(ev[tw], r, *c)] for c in combos},
                   why={str(tw): failing(ev[tw], r, "trimmed", 0.8, max(J6_CAPS)) for tw in TP.TURNS},
                   finger_mm={str(tw): [round(ev[tw]["clear"][k], 2) for k in ("finger_trimmed", "finger_old", "heads")]
                              for tw in TP.TURNS if ev[tw]["why"] is None},
                   j6_deg={str(tw): [round(v, 1) for v in ev[tw]["j6"]] for tw in TP.TURNS if ev[tw]["why"] is None})
        rows.append(row)
        if k % 50 == 0:
            print(f"  {k}/{len(idx)} points, {time.time() - t0:.0f} s", flush=True)
    # counts by region, and the reached points thinned to 1 mm apart (flanks, then tip end, edges, face; as R29)
    summary, thinned = {}, {}
    for c in combos:
        key = combo_key(*c)
        ok = [q for q in rows if q["turns"][key]]
        summary[key] = {r: sum(q["region"] == r for q in ok) for r in REGIONS}
        order = sorted((q["i"] for q in ok), key=lambda i: ["flank", "tip", "edge", "face"].index(reg[i]))
        keep = TP.fps_spacing(pts, 1.0, order)
        thinned[key] = {r: sum(reg[i] == r for i in keep) for r in REGIONS}
    total = {r: sum(q["region"] == r for q in rows) for r in REGIONS}
    gained, lost_to_heads = {}, {}
    for m in MARGINS:
        for j6 in J6_CAPS:
            kt, ko, kh = combo_key("trimmed", m, j6), combo_key("old", m, j6), combo_key("trimmed_heads", m, j6)
            gained[f"finger{m:g}_j6max{j6:g}"] = {r: sum(q["region"] == r and bool(q["turns"][kt]) and not q["turns"][ko]
                                                         for q in rows) for r in REGIONS}
            lost_to_heads[f"finger{m:g}_j6max{j6:g}"] = {r: sum(q["region"] == r and bool(q["turns"][kt])
                                                                and not q["turns"][kh] for q in rows) for r in REGIONS}
    rec = dict(
        what="FE points of v2's surface basis that a press of the tall indenter reaches, by region, with the cap trimmed at "
             f"x_F {CELL.CAP_TRIM_X} and as printed before, on {session.name}'s registration",
        script="scripts/ur5_tip_calibration_trimmed_cap_reach.py", when=time.strftime("%Y-%m-%d %H:%M"),
        session=str(session.relative_to(ROOT)) if session.is_absolute() else str(session), mount=os.environ["UR5_MOUNT"],
        checks=dict(turns_deg=list(TP.TURNS), j5_deg=list(TP.J5), j6_min_deg=a.j6_min, j6_caps_deg=list(J6_CAPS),
                    finger_margins_mm=list(MARGINS), need_mm=TP.NEED,
                    raise_mm={r: round(TP.extra(LEVELS[r][-1]), 3) for r in REGIONS},
                    top_force_N={r: LEVELS[r][-1] for r in REGIONS}, port_mm=15.0, standoff_mm=1.5, deep_mm=2.5,
                    thinning_mm=1.0, cap_trim_x_F=CELL.CAP_TRIM_X, screw_heads=HEADS),
        points_by_region=total, reached=summary, reached_thinned_1mm=thinned, gained_by_trim=gained,
        lost_to_screw_heads=lost_to_heads,
        notes=["reached: points with at least one passing turn; trimmed and old are the cell's checks, trimmed_heads adds "
               "the four M2 heads to the finger's group",
               "turns: the turns (deg) that pass in each combination", "why: each turn's failure with the trimmed cap, "
               "finger 0.8 mm and joint 6 to +58 deg (ik, j5, j6, port, or the groups under their margins; ok)",
               "finger_mm: at each turn the finger's least clearance with the trimmed and the old cap, and the screw heads'",
               "j6_deg: joint 6 at the rest and deep poses"],
        points=rows, seconds=round(time.time() - t0, 1))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=0))
    print(json.dumps(dict(points_by_region=total, reached=summary, reached_thinned_1mm=thinned, gained_by_trim=gained,
                          lost_to_screw_heads=lost_to_heads), indent=1))
    print(f"-> {out} in {time.time() - t0:.0f} s")
    if a.validate:
        rng = np.random.default_rng(1)
        pick = [rows[j] for j in rng.choice(len(rows), min(a.validate, len(rows)), replace=False)]
        bad = 0
        for j6 in J6_CAPS:
            cv = TP.Cell(base, session, j6, a.j6_min)
            for q in pick:
                i, r = q["i"], q["region"]
                for m in MARGINS:
                    for tw in TP.TURNS:
                        ref = cv.press_ok(pts[i], nrm[i], tw, LEVELS[r][-1], m) is not None
                        mine = tw in q["turns"][combo_key("trimmed", m, j6)]
                        if ref != mine:
                            bad += 1
                            print(f"  validate: point {i} ({r}) turn {tw} finger {m} j6 {j6}: press_ok {ref}, here {mine}")
        print(f"validate: {len(pick)} points x {len(TP.TURNS)} turns x {len(MARGINS) * len(J6_CAPS)} combinations against "
              f"Cell.press_ok, {bad} differ ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", default=str(LAB / "20261027-143529-tallv2id1"))
    ap.add_argument("--base-plan", default=None, help="the plan whose checks (board, other indenters) apply; default the "
                    "session's plan.json")
    ap.add_argument("--j6-min", type=float, default=-62.0)
    ap.add_argument("--limit", type=int, default=0, help="about this many points (a quick look)")
    ap.add_argument("--validate", type=int, default=0, help="compare this many points with Cell.press_ok")
    ap.add_argument("--out", default=str(OUT))
    main(ap.parse_args())
