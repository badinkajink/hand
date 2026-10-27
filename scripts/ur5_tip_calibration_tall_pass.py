#!/usr/bin/env python3
"""R29 on the UR5: presses and slides where v2 ID 1's pad has no sphere press nearby and the tall indenter reaches, and
twists to 20 deg at face and edge points (owner, 2026-10-26: "seems like contact density is a big one. i will mount the
taller indenter tmw too"; the lab, 2026-10-27: the tall indenter's README drafts the request as R28, which QUEUE.md had
given to v2 ID 1's calibration).

Targets. The FE surface points (20261020-b_v2_surface_basis.npz) of the face, edges and flanks with no press of v2 ID 1's
sphere session (20261026-192158-v2id1) within --gap mm (1.0; on ID 2 the GP's error was 0.36 mm with the nearest press
1 mm away and 1.15 mm at 1.9 mm). A point is reached when some turn about the vertical in -20 to +40 deg (steps of 5)
keeps joint 5 within 34 to 98 deg and joint 6 within -65 deg and --j6-max (45: the cap for unattended runs since the
HEX-E's dropouts of 10-26; --j6-min -62, three degrees over the program's limit) at the rest and deep poses, by the
simulator's IK on the session's registration, and the clearances of the cell model (ur5_cb3_cell.clearance, mount v7, the
tall indenter and the base plan's other indenters as obstacles) exceed mount 3.0, finger 0.8 (1.5 on the flanks and in
the twists: --finger-flank), HEX-E 6.0, pad 0.3 and robot links 8.0 mm, the first three raised by the press's depth at its largest
force less 1 mm (cb2t: 0.461 F^0.782 mm past contact, plus 0.15 mm). Of the turns that pass, the one with the larger of the
mount's margin and a fifth of joint 5's degrees over 30 is taken (as R24 pass 3). The reached points are thinned by
farthest-point sampling until none lies within --spacing mm (1.0) of another.

Actions, at the plan builder's levels by region (face 0.25 to 5 N, edge 0.25 to 3 N, flank 0.25 to 2 N) and the grid's
speed (0.15 mm/s actual at the slider's 0.30):
  ref     the reference press at the face's centre first, after the presses and last (final-ref);
  press   each target at its turn; those whose joint 6 passes --cap (45 deg, unattended runs) first, then by joint 6;
  slide   each target at 1 N: eight directions, turns of 1.146 and 2.292 deg about the sphere's centre (0.1 and 0.2 mm), a
          slide's turned poses checked as its press;
  twist   to 20 deg in steps of 5 deg and back at 2.5 N and 0.28 deg/s actual (turn_v 0.016; the plan's default 0.07 deg/s
          takes 10 min a twist), at --twists face points and as many edge points, spread by farthest-point sampling
          over the points whose turn plus 20 deg keeps joint 6 under the cap.
Afterwards the plan is checked as the lab checks it, by scripts/ur5_tip_calibration_check.py --clearance on the session,
and actions under a margin are dropped.

    .venv/bin/python scripts/ur5_tip_calibration_tall_pass.py --session lab/<tall pass 1 session>
        [--base-plan lab/<its plan>.json] [--j6-max 45] [--out lab/20261027-plan_tall_lab_r29_v2id1.json]
"""
from __future__ import annotations

import argparse
import copy
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

C = CELL.C
EXP = P3.EXP
LAB = EXP / "lab"
TALL = ROOT / "docs/experiments/20261026-tall_indenter"
SPHERE_HOLDS = EXP / "data/20261026-192158-v2id1/holds.npz"
HEADER_FROM = LAB / "20261026-192158-v2id1/plan.json"          # payload, travel, slider and safety of v2 ID 1's runs
LEVELS = dict(face=[0.25, 0.5, 1.0, 2.0, 3.5, 5.0], edge=[0.25, 0.5, 1.0, 2.0, 3.0], flank=[0.25, 0.5, 1.0, 2.0])
TURNS = tuple(range(-20, 45, 5))
J5 = (34.0, 98.0)
NEED = dict(mount=3.0, finger=0.8, hex_e=6.0, pad=0.3, robot=8.0)
GROUPS = dict(mount=("adapter", "fork", "clamp_plate"), finger=("dip_link", "cover", "cap"), hex_e=("hex_e",), pad=("pad",))
SLIDE_DEGS = [1.146, 2.292]
TWIST_DEGS = [5.0, 10.0, 15.0, 20.0, 15.0, 10.0, 5.0]


def extra(F):
    return max(0.0, 0.461 * F ** 0.782 + 0.15 - 1.0)


def fps_spacing(P, spacing, order):
    """Indices of P kept so that no two lie within spacing mm, taken in the given order of preference."""
    keep = []
    for i in order:
        if all(np.linalg.norm(P[i] - P[k]) >= spacing for k in keep):
            keep.append(i)
    return keep


def fps(P, k):
    if len(P) <= k:
        return list(range(len(P)))
    out = [int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))]
    d = np.linalg.norm(P - P[out[0]], axis=1)
    while len(out) < k:
        out.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(P - P[out[-1]], axis=1))
    return out


class Cell:
    def __init__(self, base, session, j6_max, j6_min=-62.0, finger_flank=1.5):
        ck = base["checks"]
        CELL.BOARD.update(ck["board"])
        CELL.BOARD["half_mm"] = 400.0                 # the mat table under the wrist (top at z_B 32.8), as the reach map
        CELL.OTHER[:] = [(np.array(o["T_BI"], float), o["kind"]) for o in ck.get("other", [])]
        self.kind = base["indenter"]
        self.geo = C.Geometry(base)
        self.reg = C.Registration(self.geo).load(json.loads((Path(session) / "registration.json").read_text()))
        self.qh = np.asarray(self.reg.q_home, float)
        self.T_EP = self.reg.T_EM @ self.reg.T_MP
        self.j6 = (float(j6_min), float(j6_max))
        self.finger_flank = float(finger_flank)

    def pose(self, T_BE, contact_P, raise_mm, finger=None):
        """(ok, worst clearance per group, (joint 5, joint 6) deg) at flange pose T_BE."""
        q = CELL.SIM.ik(T_BE, self.qh)
        if q is None:
            return False, None, None
        j5 = math.degrees(q[4])
        j6 = math.degrees(self.qh[5] + (q[5] - self.qh[5] + math.pi) % (2 * math.pi) - math.pi)
        if not (J5[0] <= j5 <= J5[1] and self.j6[0] <= j6 <= self.j6[1]):
            return False, None, (j5, j6)
        if CELL.self_clearance(q, self.reg.T_EM)["port"] < 15.0:
            return False, None, (j5, j6)
        cl = CELL.clearance(q, T_BE, self.reg.T_BI, self.reg.T_EM, self.kind, contact_P=contact_P)
        worst = {}
        for k, v in cl.items():
            g = next((gn for gn, names in GROUPS.items() if k in names), "robot")
            worst[g] = min(worst.get(g, 1e9), v)
        need = {g: NEED[g] + (raise_mm if g in ("mount", "finger", "hex_e") else 0.0) for g in NEED}
        if finger is not None:
            need["finger"] = max(need["finger"], finger + raise_mm)
        return all(worst[g] >= need[g] for g in worst), worst, (j5, j6)

    def deep(self, p, n, turn, st=1.5):
        T_IP = self.geo.T_IP(p, n, float(turn), st)
        return [self.reg.T_BI @ C.Tm(T_IP[:3, :3], T_IP[:3, 3] - C.EZ * dz) @ C.inv(self.T_EP) for dz in (0.0, st + 1.0)]

    def press_ok(self, p, n, turn, top_N, finger=None):
        out, js = [], []
        for T in self.deep(p, n, turn):
            ok, worst, j = self.pose(T, p, extra(top_N), finger)
            if not ok:
                return None
            out.append(worst)
            js.append(j)
        worst = {g: min(w[g] for w in out) for g in out[0]}
        return worst, (min(j[0] for j in js), max(j[0] for j in js), min(j[1] for j in js), max(j[1] for j in js))

    def slide_ok(self, p, n, turn, F=1.0, finger=None):
        deep = self.deep(p, n, turn)[1]
        c_B = C.apply(self.reg.T_BI, self.geo.c_I)
        for d in range(0, 360, 45):
            u = self.reg.T_BI[:3, :3] @ np.cross(C.EZ, [math.cos(math.radians(d)), math.sin(math.radians(d)), 0.0])
            for dg in SLIDE_DEGS:
                if not self.pose(C.turn_about(deep, c_B, u, dg), p, extra(F), finger)[0]:
                    return False
        return True

    def twist_ok(self, p, n, turn, F=2.5):
        return all(self.press_ok(p, n, turn + t, F, self.finger_flank) is not None for t in (0.0, 10.0, 20.0))


def action(kind, aid, p, n, i, region, turn, **kw):
    tg = dict(frame="P", p=[float(v) for v in p], n=[float(v) for v in n], index=int(i), region=region)
    a = dict(id=aid, stage={"press": "grid", "slide": "shear", "twist": "twist"}[kind], kind=kind, target=tg,
             twist=float(turn), standoff=1.5)
    a.update(kw)
    return a


def run(a):
    t0 = time.time()
    base = json.loads(Path(a.base_plan).read_text())
    hdr = json.loads(HEADER_FROM.read_text())
    cell = Cell(base, a.session, a.j6_max, a.j6_min, a.finger_flank)
    b = np.load(P3.BASIS)
    pts, nrm = b["pts"], b["normals"]
    reg_of = [P3.region(v) or "" for v in nrm]
    H = np.load(SPHERE_HOLDS)
    pressed = H["p"][(H["kind"] == "press") & np.isfinite(H["p"]).all(1)]
    near = np.array([np.min(np.linalg.norm(pressed - p, axis=1)) for p in pts])
    cand = [i for i in range(len(pts)) if reg_of[i] in LEVELS and near[i] > a.gap]
    print(f"{len(cand)} uncovered points ({a.gap} mm from v2 ID 1's {len(pressed)} sphere press holds); "
          + ", ".join(f"{r} {sum(reg_of[i] == r for i in cand)}" for r in LEVELS), flush=True)
    reached = {}
    for k, i in enumerate(cand):
        r = reg_of[i]
        best = None
        for tw in TURNS:
            res = cell.press_ok(pts[i], nrm[i], tw, LEVELS[r][-1], cell.finger_flank if r == "flank" else None)
            if res is None:
                continue
            worst, j = res
            s = min(worst["mount"] - NEED["mount"] - extra(LEVELS[r][-1]), (j[0] - 30.0) / 5.0)
            if best is None or s > best[0]:
                best = (s, tw, worst, j)
        if best:
            reached[i] = best
        if k % 50 == 0:
            print(f"  {k}/{len(cand)} points, {len(reached)} reached, {time.time() - t0:.0f} s", flush=True)
    # thin to the spacing: flanks first, then edges, then the face; within a region the farthest from a press first
    order = sorted(reached, key=lambda i: (["flank", "edge", "face"].index(reg_of[i]), -near[i]))
    keep = fps_spacing(pts, a.spacing, order)
    presses, slides = [], []
    for i in keep:
        r = reg_of[i]
        s, tw, worst, j = reached[i]
        lv = LEVELS[r]
        key = (0 if j[3] > a.cap else 1, j[2])         # past the unattended cap first, then by joint 6
        presses.append((key, action("press", f"tall-{r}-{i:03d}", pts[i], nrm[i], i, r, tw, levels=lv, stop_N=lv[-1],
                                     max_depth_mm=2.5 if lv[-1] <= 2.0 else 3.0, hold_s=0.6)))
        if cell.slide_ok(pts[i], nrm[i], tw, finger=cell.finger_flank if r == "flank" else None):
            slides.append((key, action("slide", f"tall-slide-{r}-{i:03d}-1N", pts[i], nrm[i], i, r, tw, press_N=1.0,
                                        moves=[dict(dir_deg=float(d), degs=SLIDE_DEGS) for d in range(0, 360, 45)],
                                        hold_s=0.4, turn_v=0.04)))
    print(f"{len(reached)} reached, {len(keep)} kept at {a.spacing} mm, {len(slides)} slides, {time.time() - t0:.0f} s", flush=True)
    # twists at face and edge points (any point the tall indenter reaches, not only uncovered ones), under the unattended
    # cap, since they run late in the plan
    cell.j6 = (a.j6_min, min(a.j6_max, a.cap))
    tw_acts = []
    for r in ("face", "edge"):
        pool = [i for i in range(len(pts)) if reg_of[i] == r]
        P = pts[pool]
        tried = 0
        got = []
        for k in fps(P, len(P)):                     # farthest-point order over the whole region
            i = pool[k]
            if any(np.linalg.norm(pts[i] - pts[g]) < 2.0 for g, _ in got):
                continue
            for tw in TURNS:
                if cell.twist_ok(pts[i], nrm[i], tw):
                    got.append((i, tw))
                    break
            tried += 1
            if len(got) >= a.twists or tried > 60:
                break
        for i, tw in got:
            tw_acts.append(action("twist", f"tall-twist-{r}-{i:03d}-2.5N-20deg", pts[i], nrm[i], i, r, tw, press_N=2.5,
                                  hold_s=1.0, turn_v=0.016, moves=[dict(degs=TWIST_DEGS)]))
    print(f"twists: {len(tw_acts)}", flush=True)
    # references: the face's centre, as the sphere passes (ref-00, index 788)
    ref = action("press", "tall-ref-a", pts[788], nrm[788], 788, "face", 0.0, levels=[0.5, 1.0, 2.0], stop_N=2.0,
                 max_depth_mm=2.5, role="ref")
    refs = [dict(copy.deepcopy(ref), id=f"tall-ref-{c}") for c in "abc"]
    fin = dict(copy.deepcopy(ref), id="tall-final-ref", stage="final")
    acts = ([refs[0]] + [x for _, x in sorted(presses, key=lambda q: q[0])] + [refs[1]]
            + [x for _, x in sorted(slides, key=lambda q: q[0])] + tw_acts + [refs[2], fin])
    plan = {k: copy.deepcopy(v) for k, v in base.items() if k != "actions"}
    for k in ("payload", "travel", "speed_slider", "safety"):
        if k in hdr and k not in plan:
            plan[k] = copy.deepcopy(hdr[k])
    plan["actions"] = acts
    # the lab's check: --clearance on the session, each action's margins raised by its depth
    per, under = P3.clearance(plan, a.session, "v7")
    dropped = []
    kept = []
    for x in acts:
        m = per.get(x["id"])
        top = float(x.get("press_N") or x.get("stop_N") or 2.0)
        fl = a.finger_flank if (x["target"]["region"] == "flank" or x["kind"] == "twist") else NEED["finger"]
        need = dict(mount=NEED["mount"], finger=fl, hex_e=NEED["hex_e"])
        bad = x["id"] in under or m is None or any(g in m and m[g] < need[g] + extra(top) for g in need)
        (dropped if bad else kept).append(x if not bad else (x["id"], m))
    plan["actions"] = kept
    n_by = {r: sum(x["kind"] == "press" and x["target"]["region"] == r and x.get("role") != "ref" for x in kept) for r in LEVELS}
    ys = np.array([x["target"]["p"][1] for x in kept if x["kind"] == "press" and x.get("role") != "ref"])
    plan["notes"] = list(base.get("notes") or []) + [
        f"{time.strftime('%Y-%m-%d')}, workstation: R29 (scripts/ur5_tip_calibration_tall_pass.py) on "
        f"{Path(a.session).name}'s registration and the tall indenter: presses at {sum(n_by.values())} points with no "
        f"sphere press of v2 ID 1 within {a.gap} mm ({', '.join(f'{r} {v}' for r, v in n_by.items())}), "
        f"{sum(x['kind'] == 'slide' for x in kept)} slides at 1 N, {sum(x['kind'] == 'twist' for x in kept)} twists to 20 deg "
        f"at 2.5 N, reference presses; joint 5 >= {J5[0]:.0f} deg, joint 6 within -65 and {a.j6_max:g} deg; margins mount "
        f"3.0, finger 0.8, HEX-E 6.0 mm raised by the depth at the largest force less 1 mm."]
    out = Path(a.out)
    out.write_text(json.dumps(plan, indent=1) + "\n")
    per_kept = sorted(((per[x["id"]].get("mount", 99), x["id"]) for x in kept))
    print(f"{out.name}: {len(kept)} actions; presses {n_by}; y < -1.5: {(ys < -1.5).sum()}, mid {(abs(ys) <= 1.5).sum()}, "
          f"y > 1.5: {(ys > 1.5).sum()}; dropped by the check {len(dropped)}: {[d[0] for d in dropped][:12]}")
    print(f"  least mount margins: {per_kept[:4]}; keys {sorted(next(iter(per.values())).keys())}; {time.time() - t0:.0f} s")
    readme67 = None
    try:
        rt = json.loads((TALL / "data/20261026-tall_indenter_reach_tall_H160.json").read_text())["points"]
        rs = json.loads((TALL / "data/20261026-tall_indenter_reach_sphere.json").read_text())["points"]
        ss = {q["i"]: q for q in rs}
        readme67 = {q["i"] for q in rt if q["turns_ok"] and not ss.get(q["i"], {}).get("turns_ok")}
    except FileNotFoundError:
        pass
    if readme67:
        mine = {x["target"]["index"] for x in kept if x["kind"] == "press"}
        hit = sum(any(np.linalg.norm(pts[i] - pts[k]) < a.spacing for k in mine) for i in readme67)
        print(f"  of the README's {len(readme67)} points only the tall indenter reaches (cb3t), {hit} lie within "
              f"{a.spacing} mm of a target here")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", default=str(LAB / "20261026-192158-v2id1"))
    ap.add_argument("--base-plan", default=str(TALL / "plan/20261026-plan_tall_H160_check.json"))
    ap.add_argument("--gap", type=float, default=1.0)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--twists", type=int, default=6)
    ap.add_argument("--j6-max", type=float, default=45.0)
    ap.add_argument("--j6-min", type=float, default=-62.0, help="3 deg over the program's -65: the simulator took a "
                    "nominal -64.8 to -66.3 deg (lab, 10-27)")
    ap.add_argument("--finger-flank", type=float, default=1.5, help="the finger's margin (mm) on flank actions: the "
                    "simulator brought the cap within 0.17 mm of the tall indenter's sphere at a flank touch (lab, 10-27)")
    ap.add_argument("--cap", type=float, default=45.0, help="actions whose joint 6 passes this go first (owner present)")
    ap.add_argument("--out", default=str(LAB / "20261027-plan_tall_lab_r29_v2id1.json"))
    run(ap.parse_args())
