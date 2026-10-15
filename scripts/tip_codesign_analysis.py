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
  screen      plan step 7: J1 per candidate (stage 1, mu 0.7) from screen/scale/<key>/turn.jsonl, screen/hom.jsonl,
              with three-finger shares, the hold test and the fidelity rows; screen/summary_stage1.json.
  handfix     the pad friction fix on the hand: replays and HOM turns D1-D8 seeds 0-2 per variant against Drake
              (U2 rows): placements within 3 deg of Drake's turn and with Drake's held outcome.

    python3 scripts/tip_codesign_analysis.py tolerance
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
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


CELLS = (("cylinder", 25.0, 0.125), ("cylinder", 16.0, 0.625), ("cylinder", 10.0, 0.75), ("cylinder", 16.0, -0.375))


def held_turn(r):
    return float(r["turn_end_deg"]) if r.get("held") else 0.0


def stage_score(turn_rows, hom_rows, sel_seed=0, score_seeds=(1, 2)):
    """J of one candidate and condition: per cell the grasp with the best held turn on sel_seed, scored by its mean
    held turn on score_seeds; J = the mean of the four cell scores and the mean held turn of the D1-D8 HOM runs, each
    over 90 deg. Returns (J, details)."""
    cells = {}
    for c in CELLS:
        rs = [r for r in turn_rows if (r["shape"], float(r["d_mm"]), r.get("t")) == c and r.get("status") == "ok"]
        by = defaultdict(dict)
        for r in rs:
            by[int(r["cand"])][int(r["seed"])] = r
        sel = [(held_turn(v[sel_seed]), k) for k, v in by.items() if sel_seed in v]
        if not sel:
            cells[f"{c[1]:g}:{c[2]:g}"] = None
            continue
        best = max(sel)[1]
        sc = [held_turn(by[best][s]) for s in score_seeds if s in by[best]]
        cells[f"{c[1]:g}:{c[2]:g}"] = {"cand": best, "sel_turn": max(sel)[0], "score_deg": float(np.mean(sc)) if sc else None,
                                      "n_grasps": len(by)}
    hom = [held_turn(r) for r in hom_rows if r.get("status") == "ok"]
    parts = [v["score_deg"] / 90 for v in cells.values() if v and v["score_deg"] is not None]
    if hom:
        parts.append(float(np.mean(hom)) / 90)
    J = float(np.mean(parts)) if parts else None
    return J, {"cells": cells, "hom_mean_held_turn_deg": float(np.mean(hom)) if hom else None, "n_hom": len(hom),
               "n_parts": len(parts)}


def screen():
    """Plan step 7: J1 per candidate at mu 0.7 (stage 1), with the three-finger shares, the hold test and fidelity."""
    S = D / "screen"
    hom = rows(S / "hom.jsonl")
    fid = {c: rows(S / f"fidelity/{c}.jsonl") for c in ("static", "T2", "spin", "T5")}
    out = {"candidates": {}, "terms": {
        "J1": "mean over the four CAD-hand cells (the seed-0 best grasp's mean held turn on seeds 1-2) and the D1-D8 "
              "HOM mean held turn (24 runs), each over 90 deg; mu 0.7, servo plant, kp_model 0.72",
        "three_finger_frac": "share of turn runs ending with all three fingers loaded (scale cells; D1-D8 HOM)"}}
    import numpy as _np  # noqa: F401
    for kdir in sorted((S / "scale").glob("*")):
        key = kdir.name
        turn = [r for r in rows(kdir / "turn.jsonl")]
        hold = [r for r in rows(kdir / "hold.jsonl") if r.get("status") == "ok"]
        hk = [r for r in hom if r.get("tip") == key and str(r.get("mu")) in ("0.7",)]
        J, det = stage_score(turn, hk)
        ok = [r for r in turn if r.get("status") == "ok"]
        tf = float(np.mean([r.get("fingers_loaded_end") == 3 for r in ok])) if ok else None
        tfh = float(np.mean([r.get("fingers_end") == 3 for r in hk if r.get("status") == "ok"])) if hk else None
        dist = [np.median([v for k_, v in r["disturb"].items() if k_.startswith("F")]) for r in hold if r.get("disturb")]
        fd = {}
        for c, rr in fid.items():
            for r in rr:
                m = r["model"]
                if key not in m or r.get("status") != "complete":
                    continue
                src = "drake" if m.startswith("drake") else "pads"
                if c == "static":
                    fd[f"{src}_approach_1N_um"] = r["approach_um"]
                    if src == "pads":
                        fd["n_pads_side"] = (r.get("info") or {}).get("n_spheres")
                elif c == "T2":
                    fd[f"{src}_onset_1N_mm"] = r.get("rbar_onset_mm")
                    fd[f"{src}_kin_1N_mm"] = r.get("rbar_kin_mm")
                elif c == "spin":
                    fd["pads_spin_1N_1_mm"] = r["rbar_spin_mm"]
                    fd["pads_us_per_step_spin"] = r["us_per_step_median"]
                elif c == "T5":
                    fd["pads_N45_N"] = r.get("N_at_45_N")
        out["candidates"][key] = {"J1": J, **det, "n_turn_rows": len(ok), "three_finger_frac_cells": tf,
                                  "three_finger_frac_hom": tfh, "hold_held_frac": float(np.mean([r["held"] for r in hold])) if hold else None,
                                  "hold_median_force_disturbance_N": float(np.median(dist)) if dist else None,
                                  "fidelity": fd}
    (S / "summary_stage1.json").write_text(json.dumps(out, indent=1))
    rank = sorted(((v["J1"] or -1, k) for k, v in out["candidates"].items()), reverse=True)
    for J, k in rank:
        v = out["candidates"][k]
        print(f"{k:22s} J1 {J:6.3f}  cells " + " ".join(
            f"{c}:{(x or {}).get('score_deg') if x else None}" for c, x in v["cells"].items())
            + f"  HOM {v['hom_mean_held_turn_deg']} (n {v['n_hom']})  3f {v['three_finger_frac_cells']}"
            + f"  hold {v['hold_held_frac']}  F {v['hold_median_force_disturbance_N']}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tolerance", "handfix", "deterioration", "screen"])
    a = ap.parse_args()
    {"tolerance": tolerance, "handfix": handfix, "deterioration": deterioration, "screen": screen}[a.cmd]()


if __name__ == "__main__":
    main()
