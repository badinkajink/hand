#!/usr/bin/env python3
"""Reorders a UR5 plan's actions so that no travel swings wrist 1 from negative to positive while wrist 2 rises (lab,
2026-10-30 00:13: three of the HEX-E's link losses came at the end of such travels in tall stage 2's grid, into
tall-face-702 on 10-27, tall-edge-627 on 10-28 and tall-edge-626 on 10-30; in the last, wrist 1 swung from -36.6 to
+19.3 deg and wrist 2 rose from 54.2 to 90.5 deg).

The joints along each travel come from replaying the plan as scripts/ur5_tip_calibration_holder_paths.py does: the
program's travel() and the inverse kinematics continued from home. A travel has the losses' signature when wrist 1
starts below 0 and passes above it while wrist 2 rises by more than --rise deg. Within each run of actions of one
stage, the order is searched that keeps every action within --places of its place and the references in theirs,
and that minimizes the wrist joints' travel (the sum of |dq4| + |dq5| + |dq6| between neighbouring rest poses, from
the pose the run starts at) with a penalty of 1000 per travel with the signature at the rest poses. The replay of the
new order counts the signatures along the whole travels.

    .venv/bin/python scripts/ur5_tip_calibration_wrist_order.py PLAN --session FOLDER --out OUT [--places 4]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import ur5_tip_calibration_holder_paths as HP

PENALTY = 1000.0


def replay(plan, session):
    """Per action replayed: (its index, the joints in deg at its rest pose, the joints in deg along its travel)."""
    HP.KIND = plan.get("indenter", "tiphold")
    mv = HP.Moves(plan, session)
    reg = mv.reg
    T = np.asarray(reg.T_BE_home, float)
    T = HP.C.Tm(T[:3, :3], T[:3, 3] + T[:3, :3] @ reg.T_EM[:3, :3] @ np.array([-12.0, 0.0, 0.0]))
    q = HP.SIM.ik(T, np.asarray(reg.q_home, float))
    out = []
    for i, act in enumerate(plan["actions"]):
        if act["kind"] not in ("touch", "hover", "press", "slide", "twist", "roll"):
            continue
        segs, T_end, _cp, i_own = mv.action_moves(act, T)
        tr = [np.degrees(q)]
        for si, seg in enumerate(segs):
            sm = HP.samples(seg, q)
            if sm is None:
                raise RuntimeError(f"no IK for {act['id']}, segment {si}")
            if si < i_own:
                tr += [np.degrees(x) for _, x in sm]
            q = sm[-1][1]
        out.append((i, tr[-1], np.array(tr)))
        T = T_end
    return out


def signature(tr, rise):
    """The travel's joints (n x 6, deg) swing wrist 1 from below 0 to above it while wrist 2 rises by more than rise."""
    k = int(np.argmax(tr[:, 3]))
    return bool(tr[0, 3] < 0.0 < tr[k, 3] and tr[k, 4] - tr[0, 4] > rise)


def search(q, q0, fixed, places, rise):
    """The order of one run (q: n x 6 rest joints in deg, in the run's order; q0: the joints the run starts from) that
    minimizes the wrists' travel plus PENALTY per signature between neighbours, each action within places of its place
    and the fixed ones in theirs: moves of one action to another place and swaps of two, kept while they lower the
    cost."""
    n = len(q)
    q = np.vstack([q, q0])                                  # index n: the start

    def cost(p):
        p = np.concatenate([[n], p])
        a, b = q[p[:-1]], q[p[1:]]
        sig = (a[:, 3] < 0.0) & (b[:, 3] > 0.0) & (b[:, 4] - a[:, 4] > rise)
        return float(np.abs(b[:, 3:] - a[:, 3:]).sum() + PENALTY * sig.sum())

    def ok(p):
        return all(abs(k - x) <= places and (not fixed[x] or k == x) for k, x in enumerate(p))

    p, c = list(range(n)), cost(np.arange(n))
    better = True
    while better:
        better = False
        for i in range(n):
            for j in range(max(0, i - 2 * places), min(n, i + 2 * places + 1)):
                for kind in ("move", "swap"):
                    t = p[:]
                    if kind == "move":
                        t.insert(j, t.pop(i))
                    else:
                        t[i], t[j] = t[j], t[i]
                    if t != p and ok(t):
                        ct = cost(np.array(t))
                        if ct < c - 1e-9:
                            p, c, better = t, ct, True
    return p


def reorder(plan, session, places=4, rise=10.0):
    """The plan with each stage's run of actions reordered (the references, ids with '-ref', in place), and a report:
    signatures and the wrists' travel before and after, and the largest move of an action."""
    rows = replay(plan, session)
    qrest = {i: qq for i, qq, _ in rows}
    q_prev = rows[0][2][0]                                  # the joints the plan starts from (home)
    acts = plan["actions"]
    order = list(range(len(acts)))
    k = 0
    while k < len(acts):                                    # runs of one stage
        e = k
        while e + 1 < len(acts) and acts[e + 1].get("stage") == acts[k].get("stage"):
            e += 1
        idx = [i for i in range(k, e + 1) if i in qrest]
        if len(idx) > 2:
            p = search(np.array([qrest[i] for i in idx]), q_prev, ["-ref" in acts[i]["id"] for i in idx], places, rise)
            for slot, j in zip(idx, p):
                order[slot] = idx[j]
        if idx:
            q_prev = qrest[order[idx[-1]]]
        k = e + 1
    new = copy.deepcopy(plan)
    new["actions"] = [copy.deepcopy(acts[i]) for i in order]
    rows2 = replay(new, session)

    def stats(rs, acts_):
        q = np.array([qq for _, qq, _ in rs])
        return dict(signatures=[acts_[i]["id"] for i, _, tr in rs if signature(tr, rise)],
                    wrist_travel_deg=round(float(np.abs(np.diff(q[:, 3:], axis=0)).sum()), 1))
    rep = dict(before=stats(rows, acts), after=stats(rows2, new["actions"]), moved=sum(1 for k, i in enumerate(order) if k != i),
               largest_move=max(abs(k - i) for k, i in enumerate(order)), places=places, rise_deg=rise)
    return new, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan")
    ap.add_argument("--session", required=True, help="a session folder whose registration.json places the tool and the "
                    "plan's indenter")
    ap.add_argument("--out", required=True)
    ap.add_argument("--places", type=int, default=4)
    ap.add_argument("--rise", type=float, default=10.0)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    new, rep = reorder(plan, a.session, a.places, a.rise)
    new.setdefault("notes", []).append(
        f"reordered by scripts/ur5_tip_calibration_wrist_order.py on {Path(a.session).name}: travels with wrist 1 from "
        f"below 0 to above it while wrist 2 rises more than {a.rise:g} deg, {len(rep['before']['signatures'])} -> "
        f"{len(rep['after']['signatures'])}; {rep['moved']} actions moved, at most {rep['largest_move']} places")
    Path(a.out).write_text(json.dumps(new, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    print("->", a.out)


if __name__ == "__main__":
    main()
