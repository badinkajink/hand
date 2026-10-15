#!/usr/bin/env python3
"""Summaries of the fingertip co-design rows (docs/notes/20261014-tip_codesign_plan.md); each subcommand reads rows and
writes one JSON summary next to them.

  tolerance   plan step 2: per condition (friction mu, pad modulus E) the runs whose outcome changes against mu 1.0,
              E 10 MPa (held flip, or |turn change| > 3 deg), over the 84 non-bistable runs (64 replays D1-D8
              seeds 0-7 and 24 HOM turns D1-D8 seeds 0-2, less the replays D3 seeds 0-2 and D8 seed 1). The floor is
              the changed fraction at |mu - 1| <= 0.5 %; tau is the relative friction change at which the changed
              fraction exceeds the floor by 5 points (linear interpolation against |mu - 1|, the worse direction);
              the plan's tau (5 % of runs) is kept as tau_rel_friction_plan. Also the bistable replays per condition.
  deterioration  plan step 3: per flat-family member and the sphere tip, pads (current and with the friction-pair fix)
              and Drake against the law: onset, approach, sliding arm by spin rate, brake, slip number.
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
    # tau of the plan: the worse direction's changed fraction against |mu - 1| reaches 5 %
    pts, med = defaultdict(list), defaultdict(list)
    for v in out["conditions"].values():
        if v["E_Pa"] == 1e7:
            pts[round(abs(v["mu"] - 1.0), 4)].append(v["frac_changed"])
            med[round(abs(v["mu"] - 1.0), 4)].append(v["median_abs_dturn_deg"])
    xs = sorted(pts)
    ys = [max(pts[x]) for x in xs]

    def crossing(level):
        px, py = 0.0, 0.0
        for x, y in zip(xs, ys):
            if y >= level:
                return px + (level - py) * (x - px) / (y - py) if y > py else x
            px, py = x, y
        return None
    out["tau_rel_friction_plan"] = crossing(0.05)
    # the floor: the mean changed fraction at |mu - 1| <= 0.5 % (outcome flips that any perturbation causes);
    # tau = the friction change at which the changed fraction exceeds the floor by 5 points
    fl = [y for x in xs if x <= 0.005 for y in pts[x]]
    out["floor_frac_changed"] = sum(fl) / len(fl) if fl else None
    out["tau_rel_friction"] = crossing(out["floor_frac_changed"] + 0.05) if fl else None
    out["frac_by_dmu"] = {f"{x:g}": y for x, y in zip(xs, ys)}
    out["median_abs_dturn_by_dmu"] = {f"{x:g}": max(med[x]) for x in xs}
    tau = out["tau_rel_friction"]
    eE = {k: v["n_changed"] for k, v in out["conditions"].items() if v["mu"] == 1.0}
    out["E_changes"] = eE
    out["E_screening_variable"] = any(n > 4 for n in eE.values())
    (D / "tolerance/summary.json").write_text(json.dumps(out, indent=1))
    print(f"reference runs {out['n_ref']}; floor {out['floor_frac_changed']}; tau (floor + 5 points) {tau}; "
          f"plan's tau (5 %) {out['tau_rel_friction_plan']}; median |dturn| by |dmu| {out['median_abs_dturn_by_dmu']}")
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


def deterioration():
    """Plan step 3: per member the pads' and Drake's static, onset, sliding and brake numbers against the law, the
    slip number at 1 rad/s and the fix's effect. Writes deterioration/summary.json."""
    law = {(r["tip"], r["field"]): r for r in rows(D / "deterioration/law.jsonl")}
    V = defaultdict(dict)
    for c in ("static", "T2", "spin", "T5"):
        for r in rows(D / f"deterioration/{c}.jsonl"):
            if r.get("status") != "complete":
                continue
            m = r["model"]
            if c == "static":
                V[m][("app", r["N"])] = r["approach_um"]
            elif c == "T2":
                V[m][("on", r["N"])] = r.get("rbar_onset_mm")
                V[m][("creep", r["N"])] = r.get("creep_deg_s")
            elif c == "spin":
                V[m][("w", r["omega"])] = r["rbar_spin_mm"]
                V[m][("slip", r["omega"])] = (r.get("share") or {}).get("slip")
            else:
                V[m][("N45", r["dt_ms"])] = r.get("N_at_45_N")
                V[m][("phi", r["dt_ms"])] = r.get("phi_end_deg")
    out = {"members": {}, "terms": {
        "onset_over_law": "pads' T2 onset arm at 1 N over the Drake-field law's arm at psi 0",
        "spin_over_law": "driven-spin arm at 1 N (line at 23-69 deg) over the law's quasi-static mean over psi 23/46/69",
        "N45_over_drake": "brake N at 45 deg, pads at 1 ms over Drake at 0.5 ms",
        "slip_number_1": "zeta t_r mu omega rho90 / delta at 1 rad/s and 1 N: the phantom penetration of a pad at the "
                         "patch's 90th-percentile radius over the approach (t_r 0.02 s, zeta 1, mu 1; law pads, psi 45)"}}
    for r_mm in ("0", "0.5", "1", "1.5", "2", "2.7", "3", "4", "6", "7.4", "sphere"):
        n = r_mm.replace(".", "p")
        if r_mm == "sphere":
            names = {"base": "mj_pads1", "fix": "mj_pads1_fix", "drake": "drake_hydro"}
            L = None
        else:
            names = {"base": f"mj_tpu{n}_pads1", "fix": f"mj_tpu{n}_pads1pf0p1_ir1000", "drake": f"drake_tpu{n}"}
            L = law.get((f"tpu{r_mm}", "drake"))
        m = {}
        lawsp = L["spin_arm_mm"]["1"] if L else V[names["drake"]].get(("w", 0.3))
        law0 = L["table"]["1|0"]["arm_mm"] if L else V[names["drake"]].get(("on", 1.0))
        for v, nm in names.items():
            d = V.get(nm, {})
            m[v] = {"approach_1N_um": d.get(("app", 1.0)), "onset_1N_mm": d.get(("on", 1.0)),
                    "onset_over_law": (d.get(("on", 1.0)) or float("nan")) / law0 if law0 else None,
                    "creep_1N_deg_s": d.get(("creep", 1.0)),
                    "spin_mm": {f"{w:g}": d.get(("w", w)) for w in (0.03, 0.3, 1.0, 3.0, 5.0) if d.get(("w", w))},
                    "spin_over_law": {f"{w:g}": d[("w", w)] / lawsp for w in (0.03, 0.3, 1.0, 3.0, 5.0)
                                      if d.get(("w", w)) and lawsp},
                    "slip_share_1": d.get(("slip", 1.0)),
                    "N45_1ms": d.get(("N45", 1.0)), "N45_05ms": d.get(("N45", 0.5)), "N45_025ms": d.get(("N45", 0.25)),
                    "phi_end_1ms": d.get(("phi", 1.0)), "phi_end_05ms": d.get(("phi", 0.5))}
        dr = m["drake"]
        for v in ("base", "fix"):
            m[v]["approach_over_drake"] = (m[v]["approach_1N_um"] / dr["approach_1N_um"]
                                           if m[v]["approach_1N_um"] and dr["approach_1N_um"] else None)
            ref = dr["N45_05ms"] or dr["N45_1ms"]
            m[v]["N45_over_drake"] = m[v]["N45_1ms"] / ref if m[v]["N45_1ms"] and ref else None
        if L:
            P = law[(f"tpu{r_mm}", "pads")]["table"]["1|45"]
            m["slip_number_1"] = 0.02 * 1.0 * P["rho90_mm"] / (P["delta_um"] * 1e-3)
            m["r_over_rs"] = float(r_mm) / 0.75
            m["law"] = {"onset_1N_mm": law0, "spin_1N_mm": lawsp, "N45": L["brake_N"]["N45"],
                        "pads_quasistatic_spin_1N_mm": law[(f"tpu{r_mm}", "pads")]["spin_arm_mm"]["1"],
                        "pads_quasistatic_onset_1N_mm": law[(f"tpu{r_mm}", "pads")]["table"]["1|0"]["arm_mm"]}
        out["members"][r_mm] = m
    (D / "deterioration/summary.json").write_text(json.dumps(out, indent=1))
    for k, m in out["members"].items():
        b, f, d = m["base"], m["fix"], m["drake"]
        print(f"r {k:6s} onset/law {b['onset_over_law']:.3f} (Drake {d['onset_over_law']:.3f}); spin/law at 1 rad/s "
              f"{b['spin_over_law'].get('1', float('nan')):.2f} fix {f['spin_over_law'].get('1', float('nan')):.2f} "
              f"Drake {d['spin_over_law'].get('1', float('nan')):.2f}; N45/Drake {b['N45_over_drake']:.2f} fix "
              f"{f['N45_over_drake']:.2f}; Pi {m.get('slip_number_1', float('nan')):.1f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tolerance", "handfix", "deterioration"])
    a = ap.parse_args()
    {"tolerance": tolerance, "handfix": handfix, "deterioration": deterioration}[a.cmd]()


if __name__ == "__main__":
    main()
