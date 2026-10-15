#!/usr/bin/env python3
"""Summaries of the fingertip co-design rows (docs/notes/20261014-tip_codesign_plan.md); each subcommand reads rows and
writes one JSON summary next to them.

  tolerance   plan step 2: per condition (friction mu, pad modulus E) the runs whose outcome changes against mu 1.0,
              E 10 MPa (held flip, or |turn change| > 3 deg), over the 84 non-bistable runs (64 replays D1-D8
              seeds 0-7 and 24 HOM turns D1-D8 seeds 0-2, less the replays D3 seeds 0-2 and D8 seed 1); tau, the
              relative friction change at which 5 % of them change (linear interpolation of the changed fraction
              against |mu - 1|, the worse of the two directions); the bistable replays at every condition.
  handfix     the pad friction fix on the hand: replays and HOM turns D1-D8 seeds 0-2 per variant against Drake
              (U2 rows): placements within 3 deg of Drake's turn and with Drake's held outcome.

    python3 scripts/tip_codesign_analysis.py tolerance
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/experiments/20261014-tip_codesign"
U2 = ROOT / "docs/experiments/20261014-pad_usability/agree"
BISTABLE = {("D3", 0), ("D3", 1), ("D3", 2), ("D8", 1)}
DTURN = 3.0


def rows(path):
    return [json.loads(l) for l in open(path) if l.strip()] if Path(path).exists() else []


def changed(a, b):
    return (bool(a["held"]) != bool(b["held"])) or abs(a["turn"] - b["turn"]) > DTURN


def outcome_replay(r):
    return {"turn": r["turn_end_deg"], "held": r["held_end"]}


def outcome_hom(r):
    return {"turn": r["turn_end_deg"], "held": r["held"]}


def tolerance():
    rep = [r for r in rows(D / "tolerance/replay.jsonl") if r.get("status") == "complete"]
    hom = [r for r in rows(D / "tolerance/hom.jsonl") if r.get("status") == "ok"]
    cond = lambda r: (float(r["mu"]), float(r["E_Pa"]))
    R = defaultdict(dict)
    for r in rep:
        R[cond(r)][("replay", r["hand"], int(r["seed"]))] = outcome_replay(r)
    for r in hom:
        R[cond(r)][("hom", r["hand"], int(r["seed"]))] = outcome_hom(r)
    ref = R[(1.0, 1e7)]
    out = {"n_ref": len(ref), "conditions": {}}
    for c, runs in sorted(R.items()):
        if c == (1.0, 1e7):
            continue
        keys = [k for k in runs if k in ref]
        nb = [k for k in keys if not (k[0] == "replay" and (k[1], k[2]) in BISTABLE)]
        ch = [k for k in nb if changed(runs[k], ref[k])]
        bis = {f"{k[1]} s{k[2]}": {"turn": runs[k]["turn"], "held": runs[k]["held"], "turn_ref": ref[k]["turn"]}
               for k in keys if k[0] == "replay" and (k[1], k[2]) in BISTABLE}
        out["conditions"][f"mu{c[0]:g}_E{c[1]:g}"] = {
            "mu": c[0], "E_Pa": c[1], "n_runs": len(keys), "n_nonbistable": len(nb), "n_changed": len(ch),
            "frac_changed": len(ch) / max(len(nb), 1), "changed": [f"{k[0]} {k[1]} s{k[2]}" for k in ch],
            "n_held_flip": sum(bool(runs[k]["held"]) != bool(ref[k]["held"]) for k in nb),
            "median_abs_dturn_deg": float(sorted(abs(runs[k]["turn"] - ref[k]["turn"]) for k in nb)[len(nb) // 2]) if nb else None,
            "bistable": bis}
    # tau: the worse direction, changed fraction against |mu - 1| (0 at mu 1)
    pts = defaultdict(list)
    for v in out["conditions"].values():
        if v["E_Pa"] == 1e7:
            pts[round(abs(v["mu"] - 1.0), 4)].append(v["frac_changed"])
    xs = sorted(pts)
    ys = [max(pts[x]) for x in xs]
    tau = None
    px, py = 0.0, 0.0
    for x, y in zip(xs, ys):
        if y >= 0.05:
            tau = px + (0.05 - py) * (x - px) / (y - py) if y > py else x
            break
        px, py = x, y
    out["tau_rel_friction"] = tau
    out["frac_by_dmu"] = {f"{x:g}": y for x, y in zip(xs, ys)}
    eE = {k: v["n_changed"] for k, v in out["conditions"].items() if v["mu"] == 1.0}
    out["E_changes"] = eE
    out["E_screening_variable"] = any(n > 4 for n in eE.values())
    (D / "tolerance/summary.json").write_text(json.dumps(out, indent=1))
    print(f"reference runs {out['n_ref']}; tau (relative friction) {tau}")
    for k, v in out["conditions"].items():
        print(f"{k:16s} changed {v['n_changed']:2d}/{v['n_nonbistable']} ({v['frac_changed']:.3f}), held flips "
              f"{v['n_held_flip']}, median |dturn| {v['median_abs_dturn_deg']:.2f} deg; bistable "
              + ", ".join(f"{b} {x['turn']:+.1f}{'' if x['held'] else ' dropped'}" for b, x in v["bistable"].items()))
    return out


def handfix():
    drk = {(r["hand"], int(r["seed"])): outcome_replay(r) for r in rows(U2 / "servo_drake.jsonl")
           if r.get("status") == "complete"}
    pads0 = {(r["hand"], int(r["seed"])): outcome_replay(r) for r in rows(U2 / "servo_mujoco.jsonl")
             if r.get("model") == "pads" and r.get("status") == "complete" and r.get("mu") in ("1.0", 1.0)}
    hd = {(r["hand"], int(r["seed"])): outcome_hom(r) for r in rows(U2 / "hom_servo.jsonl")
          if r.get("sim") == "drake" and r.get("status") == "ok"}
    hm0 = {(r["hand"], int(r["seed"])): outcome_hom(r) for r in rows(U2 / "hom_servo.jsonl")
           if r.get("sim") == "mujoco" and r.get("status") == "ok"}
    V = defaultdict(lambda: {"replay": {}, "hom": {}})
    for r in rows(D / "deterioration/handfix_replay.jsonl"):
        if r.get("status") == "complete":
            V[r["tag"] + f"_ir{float(r['impratio']):g}"]["replay"][(r["hand"], int(r["seed"]))] = outcome_replay(r)
    for r in rows(D / "deterioration/handfix_hom.jsonl"):
        if r.get("status") == "ok":
            V[r["tag"]]["hom"][(r["hand"], int(r["seed"]))] = outcome_hom(r)
    V["U2 pads (baseline)"] = {"replay": {k: v for k, v in pads0.items() if k[1] <= 2}, "hom": hm0}
    out = {}
    for name, d in V.items():
        res = {}
        for kind, ref in (("replay", drk), ("hom", hd)):
            ks = [k for k in d[kind] if k in ref]
            tol = 3.0 if kind == "replay" else 5.0
            agree = [k for k in ks if bool(d[kind][k]["held"]) == bool(ref[k]["held"])
                     and abs(d[kind][k]["turn"] - ref[k]["turn"]) <= tol]
            res[kind] = {"n": len(ks), "n_agree": len(agree), "tol_deg": tol,
                         "miss": {f"{k[0]} s{k[1]}": [round(d[kind][k]["turn"], 1), round(ref[k]["turn"], 1)]
                                  for k in ks if k not in agree}}
        out[name] = res
    (D / "deterioration/handfix_summary.json").write_text(json.dumps(out, indent=1))
    for name, res in out.items():
        print(f"{name:32s} replay {res['replay']['n_agree']}/{res['replay']['n']} within 3 deg of Drake; HOM "
              f"{res['hom']['n_agree']}/{res['hom']['n']} within 5 deg; replay misses {res['replay']['miss']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tolerance", "handfix"])
    a = ap.parse_args()
    {"tolerance": tolerance, "handfix": handfix}[a.cmd]()


if __name__ == "__main__":
    main()
