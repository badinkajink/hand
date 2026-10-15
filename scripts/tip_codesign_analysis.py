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
  handv2      builder v1 against v2 on the as-built tip as printed (deterioration/handv2_summary.json).
  stage2      plan step 8: J2, the agreement gate and the finalists (screen/summary_stage2.json).
  stage3      plan step 9: J3 on fresh placements, the paired bootstrap interval against the as-built tip and the
              mu 0.5 three-finger hold (final/summary_stage3.json).
  builders    ledger step 27: builder v1 against v2 on the finalists (final/summary_builders.json).

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
    tcd = ("tcd_rs3_re3", "tcd_rs3_re3_p0.2", "tcd_rs5_re1", "tcd_rs3_re3_cr15", "tcd_rs3_re3_cr10.55")
    for r_mm in ("0", "0.5", "1", "1.5", "2", "2.7", "3", "4", "6", "7.4", "sphere") + tcd:
        n = r_mm.replace(".", "p")
        if r_mm.startswith("tcd_"):
            names = {"base": f"mj_{r_mm}_pads1", "fix": None, "drake": f"drake_{r_mm}"}
            L = law.get((r_mm, "drake"))
        elif r_mm == "sphere":
            names = {"base": "mj_pads1", "fix": "mj_pads1_fix", "drake": "drake_hydro"}
            L = None
        else:
            names = {"base": f"mj_tpu{n}_pads1", "fix": f"mj_tpu{n}_pads1pf0p1_ir1000", "drake": f"drake_tpu{n}"}
            L = law.get((f"tpu{r_mm}", "drake"))
        m = {}
        lawsp = L["spin_arm_mm"]["1"] if L else V[names["drake"]].get(("w", 0.3))
        law0 = L["table"]["1|0"]["arm_mm"] if L else V[names["drake"]].get(("on", 1.0))
        for v, nm in names.items():
            d = V.get(nm, {}) if nm else {}
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
            if not names.get(v):
                continue
            m[v]["approach_over_drake"] = (m[v]["approach_1N_um"] / dr["approach_1N_um"]
                                           if m[v]["approach_1N_um"] and dr["approach_1N_um"] else None)
            ref = dr["N45_05ms"] or dr["N45_1ms"]
            m[v]["N45_over_drake"] = m[v]["N45_1ms"] / ref if m[v]["N45_1ms"] and ref else None
        if L:
            tk = r_mm if r_mm.startswith("tcd_") else f"tpu{r_mm}"
            P = law[(tk, "pads")]["table"]["1|45"]
            m["slip_number_1"] = 0.02 * 1.0 * P["rho90_mm"] / (P["delta_um"] * 1e-3)
            if not r_mm.startswith("tcd_"):
                m["r_over_rs"] = float(r_mm) / 0.75
            m["law"] = {"onset_1N_mm": law0, "spin_1N_mm": lawsp, "N45": L["brake_N"]["N45"],
                        "pads_quasistatic_spin_1N_mm": law[(tk, "pads")]["spin_arm_mm"]["1"],
                        "pads_quasistatic_onset_1N_mm": law[(tk, "pads")]["table"]["1|0"]["arm_mm"]}
        out["members"][r_mm] = m
    (D / "deterioration/summary.json").write_text(json.dumps(out, indent=1))
    # builder tests (plan step 5): per member and variant the static errors; n* from the spacing variants
    B = rows(D / "deterioration/builders.jsonl")
    tau = json.loads((D / "tolerance/summary.json").read_text())["tau_rel_friction"]
    pts = []
    out["builders"] = {}
    for r in B:
        v = r["variants"]
        out["builders"][r["tip"]] = {k: {"arm0_over_law_1N": x["1"]["arm0_over_law"], "arm45_over_law_1N": x["1"]["arm45_over_law"],
                                         "delta_over_law_1N": x["1"]["delta_over_law"], "spin_over_law": x["spin_over_law"],
                                         "n_patch_1N": x["1"]["n_patch"], "n_pads": x["n_pads"]} for k, x in v.items()}
        out["builders"][r["tip"]]["ripple_1mm"] = r["ripple_1mm"]
        for k in ("pads_s0.5", "pads_s0.75", "pads_s1", "pads_s1.5"):
            for N in ("0.5", "1", "3"):
                pts.append((v[k][N]["n_patch"], abs(v[k][N]["arm0_over_law"] - 1), r["tip"], k, N))
    def n_star(P):
        P = sorted(P)
        for i, (n, e, *_rest) in enumerate(P):
            if all(q[1] < tau for q in P[i:]):
                return n
        return None
    smooth = [q for q in pts if q[2] not in ("tpu0", "tpu0.5", "tpu1") and not q[2].endswith("_p0.2")]
    nstar = n_star(pts)
    out["n_star"], out["n_star_smooth"] = nstar, n_star(smooth)
    out["n_star_smooth_members"] = "fillets of 1.5 mm or more, design surfaces (no r <= 1 mm, no as-printed)"
    out["n_star_rule"] = (f"the smallest loaded-pad count per side above which every spacing variant's 0.5-3 N psi-0 "
                          f"arm is within tau ({tau:.4f}) of the Drake-field law")
    (D / "deterioration/summary.json").write_text(json.dumps(out, indent=1))
    nan = float("nan")
    for k, m in out["members"].items():
        b, f, d = m["base"], m.get("fix") or {}, m["drake"]
        print(f"{k:20s} onset/law {b.get('onset_over_law') or nan:.3f} (Drake {d.get('onset_over_law') or nan:.3f}); "
              f"spin/law 1 rad/s {b.get('spin_over_law', {}).get('1', nan):.2f} fix {f.get('spin_over_law', {}).get('1', nan):.2f} "
              f"Drake {d.get('spin_over_law', {}).get('1', nan):.2f}; N45/Drake {b.get('N45_over_drake') or nan:.2f}; "
              f"Pi {m.get('slip_number_1', nan):.1f}")
    print("n* =", nstar, "; n* over smooth members =", out["n_star_smooth"], ";", out["n_star_rule"])
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


def handv2():
    """Plan step 5's hand check on the as-built tip as printed (tcd_rs3_re3_p0.2): builder v1 (h 8.5 mm, median K)
    against v2 (local h, per-pad K): replays D1-D8 seeds 0-4 and HOM seeds 0-2 at mu 1.0; each against Drake's replays
    of the same tip at seeds 0-2 (within 3 deg and the same held outcome). Writes deterioration/handv2_summary.json."""
    R = [r for r in rows(D / "deterioration/handv2_replay.jsonl") if r.get("status") == "complete"]
    Hm = [r for r in rows(D / "deterioration/handv2_hom.jsonl") if r.get("status") == "ok"]
    drk = {(r["hand"], int(r["seed"])): outcome_replay(r) for r in R if r["sim"] == "drake"}
    out = {}
    for tag in sorted({r["tag"] for r in R if r["sim"] == "mujoco"}):
        rep = {(r["hand"], int(r["seed"])): outcome_replay(r) for r in R if r["sim"] == "mujoco" and r["tag"] == tag}
        hom = {(r["hand"], int(r["seed"])): outcome_hom(r) for r in Hm if r["tag"] == tag}
        ks = [k for k in rep if k in drk]
        agree = [k for k in ks if bool(rep[k]["held"]) == bool(drk[k]["held"]) and abs(rep[k]["turn"] - drk[k]["turn"]) <= 3]
        out[tag] = {"n_replay": len(rep), "held_replay": sum(bool(v["held"]) for v in rep.values()),
                    "n_vs_drake": len(ks), "within3_of_drake": len(agree),
                    "hom_held": sum(bool(v["held"]) for v in hom.values()), "n_hom": len(hom),
                    "hom_mean_held_turn_deg": float(np.mean([v["turn"] if v["held"] else 0 for v in hom.values()])) if hom else None,
                    "replay_turns": {f"{k[0]} s{k[1]}": round(v["turn"], 2) for k, v in rep.items()}}
    tags = list(out)
    if len(tags) == 2:
        a, b = (out[t]["replay_turns"] for t in tags)
        out["v1_vs_v2_changed_replays"] = sum(abs(a[k] - b[k]) > 3 for k in a if k in b)
    (D / "deterioration/handv2_summary.json").write_text(json.dumps(out, indent=1))
    for t in tags:
        v = out[t]
        print(f"{t:28s} replays held {v['held_replay']}/{v['n_replay']}, within 3 deg of Drake {v['within3_of_drake']}/{v['n_vs_drake']};"
              f" HOM held {v['hom_held']}/{v['n_hom']}, mean held turn {v['hom_mean_held_turn_deg']}")
    print("v1 vs v2 replays differing by > 3 deg:", out.get("v1_vs_v2_changed_replays"))
    return out


SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
COND2 = (("1.0", SERVO), ("0.7", SERVO), ("0.5", SERVO), ("0.7", "kp0.5_kv0_fr0.18_fl0_dp0.0144"),
         ("0.7", "kp0.72_kv0_fr0.12_fl0_dp0.0144"))
BASE = "tcd_rs3_re3_p0.2"


def _radii(key):
    import re
    m = re.match(r"tcd_rs([0-9.]+)_re([0-9.]+)", key)
    return (float(m[1]), float(m[2])) if m else (2.7, 2.7)


def stage2():
    """Plan step 8: J2 = the minimum over the five conditions (mu 1.0/0.7/0.5 on the servo plant; kp 0.5 and torque
    limit 0.12 N m at mu 0.7) of the stage score (four CAD-hand cells and HOM on D1-D8 seeds 0-4); the agreement gate
    (D1-D8 replays seeds 0-2 at mu 1.0 within 3 deg of Drake and with its held outcome, at most two placements fewer than
    the as-built tip); finalists (1) highest J2 passing the gate, (2) highest J2 at no more than 0.8 of the as-built
    tip's us per replay step, (3) highest three-finger share at mu 0.5, (4) the as-built tip; a finalist that repeats an
    earlier one is replaced by the next candidate at least 1 mm away in r_side or r_end."""
    S = D / "screen"
    cands = json.loads((S / "stage2_candidates.json").read_text())["candidates"]
    hom = [r for r in rows(S / "stage2/hom.jsonl") if r.get("status") == "ok"]
    rep = [r for r in rows(S / "stage2/replay.jsonl") if r.get("status") == "complete"]
    out = {"candidates": {}, "conditions": [f"mu{m}_{p}" for m, p in COND2]}
    for k in cands:
        Js, det = {}, {}
        for mu, pl in COND2:
            if (mu, pl) == ("0.7", SERVO):
                turn = rows(S / f"scale/{k}/turn.jsonl")
            else:
                turn = rows(S / f"stage2/scale/{k}_mu{mu}_{pl}/turn.jsonl")
            hk = [r for r in hom if r.get("tip") == k and str(r.get("mu")) == mu and r.get("plant") == pl]
            J, dd = stage_score(turn, hk)
            Js[f"mu{mu}_{pl}"], det[f"mu{mu}_{pl}"] = J, dd
            if mu == "0.5":
                ok = [r for r in turn if r.get("status") == "ok"]
                det["three_finger_frac_mu0.5_cells"] = float(np.mean([r.get("fingers_loaded_end") == 3 for r in ok])) if ok else None
                det["three_finger_frac_mu0.5_hom"] = float(np.mean([r.get("fingers_end") == 3 for r in hk])) if hk else None
                hold = [r for r in rows(S / f"stage2/scale/{k}_mu{mu}_{pl}/hold.jsonl") if r.get("status") == "ok"]
                det["hold_held_frac_mu0.5"] = float(np.mean([r["held"] for r in hold])) if hold else None
        vals = [v for v in Js.values() if v is not None]
        mj = {(r["hand"], int(r["seed"])): r for r in rep if r.get("sim") == "mujoco" and r["tips" if "tips" in r else "tip"] == k}
        dr = {(r["hand"], int(r["seed"])): r for r in rep if r.get("sim") == "drake" and r["tip"] == k}
        ks = [x for x in mj if x in dr]
        agree = sum(bool(mj[x]["held_end"]) == bool(dr[x]["held_end"]) and abs(mj[x]["turn_end_deg"] - dr[x]["turn_end_deg"]) <= 3
                    for x in ks)
        us = [r["us_per_step_median"] for r in mj.values() if r.get("us_per_step_median")]
        out["candidates"][k] = {"J": Js, "J2": min(vals) if len(vals) == len(COND2) else None, "n_conditions": len(vals),
                                "agree_drake_3deg": agree, "n_vs_drake": len(ks),
                                "us_per_step_median": float(np.median(us)) if us else None, **det}
    C = out["candidates"]
    base = C.get(BASE, {})
    for k, v in C.items():
        v["gate_pass"] = (v["n_vs_drake"] >= 20 and base.get("n_vs_drake") and
                          v["agree_drake_3deg"] >= base["agree_drake_3deg"] - 2)
        v["cost_ratio"] = v["us_per_step_median"] / base["us_per_step_median"] if v.get("us_per_step_median") and base.get("us_per_step_median") else None
    order = lambda f, key: sorted((k for k in C if f(C[k]) and C[k].get(key) is not None), key=lambda k: -C[k][key])
    fin, why = [], {}

    def pick(lst, label):
        for k in lst:
            if k in fin or k == BASE:
                continue
            if any(max(abs(a - b) for a, b in zip(_radii(k), _radii(f))) < 1.0 for f in fin):
                continue
            fin.append(k)
            why[k] = label
            return k
    pick(order(lambda v: v["gate_pass"], "J2"), "(1) highest J2 passing the agreement gate")
    pick(order(lambda v: v["gate_pass"] and (v["cost_ratio"] or 9) <= 0.8, "J2"), "(2) highest J2 at <= 0.8 of the as-built cost")
    pick(order(lambda v: True, "three_finger_frac_mu0.5_cells"), "(3) highest three-finger share at mu 0.5")
    fin.append(BASE)
    why[BASE] = "(4) the as-built tip as printed, baseline"
    out["finalists"], out["finalist_rule"] = fin, why
    (S / "summary_stage2.json").write_text(json.dumps(out, indent=1))
    for k, v in sorted(C.items(), key=lambda kv: -(kv[1]["J2"] or -1)):
        print(f"{k:22s} J2 {v['J2'] if v['J2'] is None else round(v['J2'], 3)}  " + " ".join(
            f"{c.split('_')[0]}{'' if SERVO in c else ('/kp0.5' if 'kp0.5_' in c else '/fr0.12')}:{(x if x is None else round(x, 3))}"
            for c, x in v["J"].items())
            + f"  Drake {v['agree_drake_3deg']}/{v['n_vs_drake']} gate {v['gate_pass']} cost {v['cost_ratio'] and round(v['cost_ratio'], 2)}"
            + f"  3f@0.5 {v.get('three_finger_frac_mu0.5_cells')}")
    print("finalists:", why)
    return out


COND3 = COND2[1:]
SEEDS3_CELL, SEEDS3_HOM = (3, 4, 5), (5, 6, 7, 8, 9)
HANDS = tuple(f"D{i}" for i in range(1, 9))


def _sel_dir(k, mu, pl):
    """Stage 1/2 scale rows of one condition: the seed-0 runs there choose each cell's grasp."""
    S = D / "screen"
    return S / f"scale/{k}" if (mu, pl) == ("0.7", SERVO) else S / f"stage2/scale/{k}_mu{mu}_{pl}"


def _stage3_arrays(k, mu, pl, hom3):
    """Held turns (deg, 0 when dropped) of the stage-3 placements of one tip and condition: cells (4 x 3, the grasp
    chosen on seed 0 of stage 1/2 at seeds 3-5) and HOM (8 x 5, D1-D8 at seeds 5-9); NaN where a row is missing."""
    sel = [r for r in rows(_sel_dir(k, mu, pl) / "turn.jsonl") if r.get("status") == "ok" and int(r["seed"]) == 0]
    new = [r for r in rows(D / f"final/scale/{k}_mu{mu}_{pl}/turn.jsonl") if r.get("status") == "ok"]
    C = np.full((len(CELLS), len(SEEDS3_CELL)), np.nan)
    chosen = {}
    for i, c in enumerate(CELLS):
        cand = [(held_turn(r), int(r["cand"])) for r in sel if (r["shape"], float(r["d_mm"]), r.get("t")) == c]
        if not cand:
            continue
        best = max(cand)[1]
        chosen[f"{c[1]:g}:{c[2]:g}"] = best
        for r in new:
            if (r["shape"], float(r["d_mm"]), r.get("t")) == c and int(r["cand"]) == best and int(r["seed"]) in SEEDS3_CELL:
                C[i, SEEDS3_CELL.index(int(r["seed"]))] = held_turn(r)
    H = np.full((len(HANDS), len(SEEDS3_HOM)), np.nan)
    for r in hom3:
        if r.get("tip") == k and str(r.get("mu")) == mu and r.get("plant") == pl and int(r["seed"]) in SEEDS3_HOM:
            H[HANDS.index(r["hand"]), SEEDS3_HOM.index(int(r["seed"]))] = held_turn(r)
    return C, H, chosen


def _score3(C, H):
    """The stage score over arrays with a leading resample axis: the mean of the four cell means and the HOM mean,
    each over 90 deg (cells without a grasp drop out, as in stage_score)."""
    parts = np.concatenate([np.nanmean(C, axis=-1), np.nanmean(H.reshape(H.shape[0], -1), axis=-1)[:, None]], axis=1)
    return np.nanmean(parts / 90.0, axis=1)


def stage3(n_boot=10000, rng_seed=0):
    """Plan step 9: J3 per finalist = the minimum over the four stage-3 conditions (mu 0.7 and 0.5 on the servo plant,
    kp 0.5 and torque limit 0.12 N m at mu 0.7) of the stage score on fresh placements only (each cell's seed-0 grasp
    of stage 1/2 at seeds 3-5; HOM D1-D8 seeds 5-9). The paired difference against the as-built tip has a bootstrap
    interval: each resample draws, per cell, three of the seeds 3-5 and, per hand, five of the seeds 5-9 with
    replacement, the same draw for every tip and condition; 10,000 resamples, the 5th and 95th percentiles. A finalist
    beats the baseline when the interval lies above zero and its three-finger hold at mu 0.5 (stage-2 hold test, seed 0:
    held with all three fingers loaded) is at most two grasps below the baseline's. Writes final/summary_stage3.json."""
    F = D / "final"
    fin = json.loads((F / "stage3_tips.json").read_text())["finalists"]
    hom3 = [r for r in rows(F / "hom.jsonl") if r.get("status") == "ok"]
    rng = np.random.default_rng(rng_seed)
    ic = rng.integers(0, len(SEEDS3_CELL), size=(n_boot, len(CELLS), len(SEEDS3_CELL)))
    ih = rng.integers(0, len(SEEDS3_HOM), size=(n_boot, len(HANDS), len(SEEDS3_HOM)))
    rc, rh = np.arange(len(CELLS))[None, :, None], np.arange(len(HANDS))[None, :, None]
    out = {"finalists": fin, "conditions": [f"mu{m}_{p}" for m, p in COND3], "n_boot": n_boot, "candidates": {}}
    point, boot = {}, {}
    for k in fin:
        Jc, Jb, det = {}, [], {}
        for mu, pl in COND3:
            C, H, chosen = _stage3_arrays(k, mu, pl, hom3)
            name = f"mu{mu}_{pl}"
            Jc[name] = float(_score3(C[None], H[None])[0])
            Jb.append(_score3(C[rc, ic], H[rh, ih]))
            det[name] = {"cells_deg": {f"{c[1]:g}:{c[2]:g}": [None if np.isnan(x) else round(float(x), 2) for x in C[i]]
                                       for i, c in enumerate(CELLS)},
                         "chosen_grasp": chosen, "n_cell_rows": int(np.isfinite(C).sum()), "n_hom_rows": int(np.isfinite(H).sum()),
                         "hom_mean_held_turn_deg": float(np.nanmean(H)) if np.isfinite(H).any() else None,
                         "hom_held": int(np.nansum(H > 0))}
        point[k], boot[k] = min(Jc.values()), np.min(np.stack(Jb), axis=0)
        hold = [r for r in rows(_sel_dir(k, "0.5", SERVO) / "hold.jsonl") if r.get("status") == "ok"]
        out["candidates"][k] = {"J3": point[k], "J": Jc, **det, "hold3_mu0.5": sum(bool(r["held"]) and r.get("fingers_loaded") == 3
                                                                               for r in hold), "n_hold_mu0.5": len(hold)}
    base = out["candidates"].get(BASE)
    for k in fin:
        if k == BASE or base is None:
            continue
        d = boot[k] - boot[BASE]
        lo, hi = np.percentile(d, [5, 95])
        v = out["candidates"][k]
        v["dJ3_vs_base"] = point[k] - point[BASE]
        v["dJ3_ci90"] = [float(lo), float(hi)]
        v["hold_ok"] = v["hold3_mu0.5"] >= base["hold3_mu0.5"] - 2
        v["beats_base"] = bool(lo > 0 and v["hold_ok"])
    (F / "summary_stage3.json").write_text(json.dumps(out, indent=1))
    for k in fin:
        v = out["candidates"][k]
        print(f"{k:22s} J3 {v['J3']:.3f} " + " ".join(f"{c.split('_')[0]}{'' if SERVO in c else ('/kp0.5' if 'kp0.5_' in c else '/fr0.12')}"
                                                     f":{x:.3f}" for c, x in v["J"].items())
              + f"  hold3@0.5 {v['hold3_mu0.5']}/{v['n_hold_mu0.5']}"
              + (f"  dJ3 {v['dJ3_vs_base']:+.3f} [{v['dJ3_ci90'][0]:+.3f}, {v['dJ3_ci90'][1]:+.3f}] beats {v['beats_base']}"
                 if "dJ3_ci90" in v else "  (baseline)"))
    return out


def builders():
    """Ledger step 27: builder v1 (h 8.5 mm, median K) against v2 (local h, per-pad K) on each finalist: replays D1-D8
    seeds 0-4 and HOM seeds 0-2 at mu 1.0; replays at seeds 0-2 against Drake's stage-2 replays of the same tip (within
    3 deg and the same held outcome). Writes final/summary_builders.json."""
    F = D / "final"
    R = [r for r in rows(F / "builder_replay.jsonl") if r.get("status") == "complete"]
    Hm = [r for r in rows(F / "builder_hom.jsonl") if r.get("status") == "ok"]
    DR = [r for r in rows(D / "screen/stage2/replay.jsonl") if r.get("status") == "complete" and r.get("sim") == "drake"]
    out = {}
    for k in sorted({r["tip"] for r in R}):
        drk = {(r["hand"], int(r["seed"])): outcome_replay(r) for r in DR if r["tip"] == k}
        res = {}
        for tag in sorted({r["tag"] for r in R if r["tip"] == k}):
            rep = {(r["hand"], int(r["seed"])): outcome_replay(r) for r in R if r["tip"] == k and r["tag"] == tag}
            hom = {(r["hand"], int(r["seed"])): outcome_hom(r) for r in Hm if r["tip"] == k and r["tag"] == tag}
            ks = [x for x in rep if x in drk]
            agree = [x for x in ks if bool(rep[x]["held"]) == bool(drk[x]["held"]) and abs(rep[x]["turn"] - drk[x]["turn"]) <= DTURN]
            res[tag] = {"n_replay": len(rep), "held_replay": sum(bool(v["held"]) for v in rep.values()),
                        "n_vs_drake": len(ks), "within3_of_drake": len(agree), "n_hom": len(hom),
                        "hom_held": sum(bool(v["held"]) for v in hom.values()),
                        "hom_mean_held_turn_deg": float(np.mean([v["turn"] if v["held"] else 0 for v in hom.values()])) if hom else None,
                        "replay_turns": {f"{x[0]} s{x[1]}": round(v["turn"], 2) for x, v in rep.items()}}
        tags = list(res)
        if len(tags) == 2:
            a, b = (res[t]["replay_turns"] for t in tags)
            res["v1_vs_v2_changed_replays"] = sum(abs(a[x] - b[x]) > DTURN for x in a if x in b)
        out[k] = res
    (F / "summary_builders.json").write_text(json.dumps(out, indent=1))
    for k, res in out.items():
        for t, v in res.items():
            if isinstance(v, dict):
                print(f"{k:22s} {t:22s} replays held {v['held_replay']}/{v['n_replay']}, within 3 deg of Drake "
                      f"{v['within3_of_drake']}/{v['n_vs_drake']}; HOM held {v['hom_held']}/{v['n_hom']}, "
                      f"mean held turn {v['hom_mean_held_turn_deg']}")
        print(f"{k:22s} v1 vs v2 replays differing by > 3 deg: {res.get('v1_vs_v2_changed_replays')}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tolerance", "handfix", "deterioration", "screen", "handv2", "stage2", "stage3",
                                    "builders"])
    a = ap.parse_args()
    {"tolerance": tolerance, "handfix": handfix, "deterioration": deterioration, "screen": screen,
     "handv2": handv2, "stage2": stage2, "stage3": stage3, "builders": builders}[a.cmd]()


if __name__ == "__main__":
    main()
