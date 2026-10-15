#!/usr/bin/env python3
"""Trajectory divergence between contact models (docs/handoff/20261015-curved_tips.md, step 4), from the traces of
scripts/curved_tips_traj.py (traj/traces.jsonl).

For each tip, hand and seed, two arms' traces are compared on their common 20 ms samples up to the end of the shorter,
with scripts/pad_builder_check.py's measures:
  dpos       |p_A - p_B|, the distance between the tool's positions, mm
  dang       the angle of the rotation between the tool's orientations, deg
  dN         per finger the difference of the summed normal force on the tool, N
  dT         per finger the length of the difference of the summed friction-force vectors on the tool, N
RMS over time (dN and dT over time and the three fingers), the final values (the last common sample), and the held
outcome and the end turn of each arm. The spread between placements of one arm is the same measure between two seeds of
one hand (every pair of seeds 0-2), the scale below which two models cannot be told apart by these placements. Pairs: plan-drake (the pads at the planner's numerics against the reference),
rl-plan (the RL trainer's numerics against the planner's under the same commands), rl-drake, floor-plan (friction
1.001 against 1.0: the divergence of a 0.1 % parameter change, the floor any model comparison is read against) and
dfloor-drake (the same in Drake). Per tip and pair: the median and the 25-75 % range over the placements (D1-D8, seeds
0-2), and the share of placements whose held outcome agrees and whose end turns lie within 3 deg.

    python3 scripts/curved_tips_traj_analysis.py
Out: docs/experiments/20261015-curved_tips/traj/summary.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/experiments/20261015-curved_tips/traj"
FINGERS = ("thumb", "index", "middle")
PAIRS = (("plan", "drake"), ("rl", "plan"), ("rl", "drake"), ("floor", "plan"), ("dfloor", "drake"))


def arrays(r):
    tr = r["trace"]
    t = np.array([s["t"] for s in tr])
    p = np.array([s["p"] for s in tr])
    q = np.array([s["q"] for s in tr])
    N = np.array([[s["N"][f] for f in FINGERS] for s in tr])
    T = np.array([[s["T"][f] for f in FINGERS] for s in tr])            # (samples, fingers, 3)
    return t, p, q, N, T


def compare(ra, rb):
    ta, pa, qa, Na, Ta = arrays(ra)
    tb, pb, qb, Nb, Tb = arrays(rb)
    # samples on the 20 ms grid by index (a 2 ms run's samples can lie up to one step after a 1 ms run's)
    _, ia, ib = np.intersect1d(np.round(ta / 0.02).astype(int), np.round(tb / 0.02).astype(int), return_indices=True)
    n = len(ia)
    if n < 5:
        return None
    ta, pa, qa, Na, Ta = ta[ia], pa[ia], qa[ia], Na[ia], Ta[ia]
    tb, pb, qb, Nb, Tb = tb[ib], pb[ib], qb[ib], Nb[ib], Tb[ib]
    dp = np.linalg.norm(pa - pb, axis=1)
    c = np.clip(np.abs(np.sum(qa * qb, axis=1)) / (np.linalg.norm(qa, axis=1) * np.linalg.norm(qb, axis=1)), 0, 1)
    ang = np.degrees(2 * np.arccos(c))
    dN, dT = Na - Nb, np.linalg.norm(Ta - Tb, axis=2)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))  # noqa: E731
    return {"n": int(n), "t_end": float(ta[-1]),
            "rms_dpos_mm": rms(dp), "final_dpos_mm": float(dp[-1]), "rms_dang_deg": rms(ang), "final_dang_deg": float(ang[-1]),
            "rms_dN_N": rms(dN), "rms_dT_N": rms(dT), "mean_N_B": float(Nb.mean()),
            "mean_T_B": float(np.linalg.norm(Tb, axis=2).mean()),
            "held_agree": bool(ra.get("held_end") == rb.get("held_end")),
            "turn_within_3": bool(abs(float(ra.get("turn_end_deg", np.nan)) - float(rb.get("turn_end_deg", np.nan))) <= 3.0
                                  and ra.get("held_end") == rb.get("held_end"))}


def main():
    rows = defaultdict(dict)
    for line in open(D / "traces.jsonl"):
        r = json.loads(line)
        if r.get("status") in ("complete", "ejected") and r.get("trace"):
            rows[(r["tip"], r["hand"], int(r["seed"]))][r["arm"]] = r
    tips = sorted({k[0] for k in rows})
    out = {"tips": {}, "pairs": [f"{a}-{b}" for a, b in PAIRS]}
    keys = ("rms_dpos_mm", "final_dpos_mm", "rms_dang_deg", "final_dang_deg", "rms_dN_N", "rms_dT_N")
    for tip in tips:
        T = {"pairs": {}}
        for a, b in PAIRS:
            per = []
            for (tp, hand, seed), arms in sorted(rows.items()):
                if tp != tip or a not in arms or b not in arms:
                    continue
                c = compare(arms[a], arms[b])
                if c:
                    per.append({"hand": hand, "seed": seed, **c})
            if not per:
                continue
            summ = {"n_placements": len(per)}
            for k in keys:
                v = np.array([x[k] for x in per])
                summ[k] = {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
                           "q75": float(np.percentile(v, 75)), "mean": float(v.mean())}
            summ["held_agree"] = int(sum(x["held_agree"] for x in per))
            summ["turn_within_3"] = int(sum(x["turn_within_3"] for x in per))
            summ["mean_N_ref"] = float(np.mean([x["mean_N_B"] for x in per]))
            summ["per_placement"] = per
            T["pairs"][f"{a}-{b}"] = summ
        T["spread"] = {}
        for arm in ("drake", "plan", "rl"):
            per = []
            for hand in sorted({k[1] for k in rows if k[0] == tip}):
                tr = [rows[k][arm] for k in sorted(rows) if k[0] == tip and k[1] == hand and arm in rows[k]]
                for i in range(len(tr)):
                    for j in range(i + 1, len(tr)):
                        c = compare(tr[i], tr[j])
                        if c:
                            per.append(c)
            if per:
                T["spread"][arm] = {"n_pairs": len(per), **{k: {"median": float(np.median([x[k] for x in per])),
                                                                 "q25": float(np.percentile([x[k] for x in per], 25)),
                                                                 "q75": float(np.percentile([x[k] for x in per], 75))}
                                                             for k in keys}}
        us = defaultdict(list)
        for (tp, _, _), arms in rows.items():
            if tp == tip:
                for arm, r in arms.items():
                    us[arm].append(r.get("us_per_step_median"))
        T["us_per_step_median"] = {k: float(np.median([x for x in v if x])) for k, v in us.items() if any(v)}
        out["tips"][tip] = T
    (D / "summary.json").write_text(json.dumps(out, indent=1))
    print(f"{'tip':30s} {'pair':12s} {'n':>3s} {'RMS pos mm':>16s} {'final pos':>16s} {'RMS ang deg':>16s} "
          f"{'RMS dN N':>16s} {'RMS dT N':>16s} {'turn<=3':>8s}")
    for tip, T in out["tips"].items():
        for pr, s in T["pairs"].items():
            f = lambda k: f"{s[k]['median']:.2f} [{s[k]['q25']:.2f},{s[k]['q75']:.2f}]"  # noqa: E731
            print(f"{tip:30s} {pr:12s} {s['n_placements']:3d} {f('rms_dpos_mm'):>16s} {f('final_dpos_mm'):>16s} "
                  f"{f('rms_dang_deg'):>16s} {f('rms_dN_N'):>16s} {f('rms_dT_N'):>16s} {s['turn_within_3']:>3d}/{s['n_placements']}")
        for arm, s in T["spread"].items():
            f = lambda k: f"{s[k]['median']:.2f} [{s[k]['q25']:.2f},{s[k]['q75']:.2f}]"  # noqa: E731
            print(f"{tip:30s} {'seeds ' + arm:12s} {s['n_pairs']:3d} {f('rms_dpos_mm'):>16s} {f('final_dpos_mm'):>16s} "
                  f"{f('rms_dang_deg'):>16s} {f('rms_dN_N'):>16s} {f('rms_dT_N'):>16s}")


if __name__ == "__main__":
    main()
