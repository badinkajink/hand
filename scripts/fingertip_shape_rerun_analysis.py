#!/usr/bin/env python3
"""Summaries of the fingertip-shape rerun's rows (docs/handoff/20261103-fingertip_shape_rerun.md); each subcommand reads
rows under docs/experiments/20261104-fingertip_shape_rerun/ and writes one JSON summary beside them.

  bed      per tip the law (Drake's field on the 0.25 mm mesh at 2.5 MPa), the builder-v2 pads and Drake on the bed:
           approach, twist-onset arm, spin arm at 0.1-3 rad/s, brake force at 45 deg, touching pads and wall time per step
           (scripts/curved_tips_analysis.bed's fields; bed/summary.json)
  traj     the D1-D8 replays: per tip the pairs plan-drake and rl-plan and Drake's spread between seeds, with
           scripts/curved_tips_traj_analysis.compare's measures; the agreement gate's count (traj/summary.json)
  screen   stages 1 and 2 (tip_codesign_analysis.stage_score): J1 at friction 0.7 on the SCS0009 (cells scored on seeds 1-2,
           HOM seeds 0-2), J2 the minimum over the five conditions (HOM seeds 0-4); the gate (the pads within 3 deg of
           Drake's end turn with its held outcome, of 24, at most two below the v4b tip); the cost (median us per replay
           step over the v4b tip's); the three-finger share at friction 0.5; the share of the fingers' normal force on the
           cylinder through rigid parts (rigid_share); finalists (1) highest J2 passing the gate, (2) highest J2 at <= 0.8
           of the v4b tip's cost, (3) highest three-finger share at friction 0.5, (4) the next highest J2 passing the gate,
           up to four, among the eleven tips other than the v4b tip; stage 3 runs the finalists, the v4b tip (the
           baseline) and the as-built block (10-14 and 10-15's baseline) (screen/summary.json)
  stage3   J3 on fresh placements and the paired bootstrap against the v4b tip (10,000 resamples within cells and hands)
           (final/summary.json)
  cost     CPU step at the three-finger grip and 64-plan replanning (U1's protocol), GPU world-step at 2048 worlds
           (cost/summary.json)
  fe       per tip the probes' approach, force exponent and twist arm, the stiffness map under the 2 mm sphere, and the
           force resolution at the contact point with 05R4 (fe/summary.json)
  compute  wall time and core-hours per step from queue/jobs.jsonl and the ledger (compute/summary.json)
  all      every one of the above

    python3 scripts/fingertip_shape_rerun_analysis.py all
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curved_tips_analysis as CA  # noqa: E402
import fingertip_shape_rerun_common as C  # noqa: E402

D = C.D
rows, _pick, _ratio = CA.rows, CA._pick, CA._ratio
HANDS = tuple(f"D{i}" for i in range(1, 9))
SEEDS3_CELL, SEEDS3_HOM = (3, 4, 5), (5, 6, 7, 8, 9)
COND3 = [c for c in C.CONDS if c != ("1.0", C.SERVO)]
ASBUILT = C.BLOCKS[0]


def cname(mu, pl):
    return f"{mu}|{pl}"


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


# ------------------------------------------------------------------------------------------------ bed

def bed():
    out = {"tips": {}, "terms": {
        "law": "Drake's pressure field E (1 - g) on the tip's 0.25 mm mesh at 2.5 MPa, integrated quasi-statically",
        "pads": "MuJoCo 1 mm pads, builder v2, 2.5 MPa, relaxation 0.04 s, impratio 100, 1 ms",
        "drake": "Drake 1.57 compliant convex of the 0.25 mm mesh, 2.5 MPa, relaxation 0.01 s, SAP, 1 ms"}}
    for tip in C.BED:
        B = D / "bed" / tip
        law = rows(B / "law.jsonl")
        case = {c: rows(B / f"{c}.jsonl") for c in ("static", "T2", "spin", "T5")}
        L = {(r["field"], r.get("res_m")): r for r in law}
        lw = L.get(("drake", 0.00025))
        t = {"law": None, "pads": {}, "drake": {}}
        if lw:
            t["law"] = {"approach_um": lw["approach_um"], "onset_arm_mm": lw["onset_arm_mm"],
                        "spin_arm_mm": lw["spin_arm_mm"], "N45_N": lw["brake_N"]["N45"],
                        "pads_v2_qs": {k: (L.get(("pads_v2", None)) or {}).get(k) for k in ("approach_um", "onset_arm_mm",
                                                                                         "spin_arm_mm")},
                        "n_pads": (L.get(("pads_v2", None)) or {}).get("n_pads")}
        for kind in ("pads", "drake"):
            name, k = f"{kind}_{tip}", t[kind]
            k["approach_um"] = {f"{N:g}": (_pick(case["static"], model=name, N=N) or {}).get("approach_um")
                                for N in (0.5, 1.0, 2.0, 3.0)}
            s1 = _pick(case["static"], model=name, N=1.0) or {}
            k["touching_1N"], k["nefc_1N"] = s1.get("touching_per_side"), s1.get("nefc_mean")
            k["us_per_step_static_1N"] = s1.get("us_per_step_median")
            for N in (1.0, 3.0):
                r = _pick(case["T2"], model=name, N=N) or {}
                k[f"onset_arm_{N:g}N_mm"] = r.get("rbar_onset_mm")
                k[f"twist_creep_{N:g}N_deg_s"] = r.get("creep_deg_s")
            k["spin_arm_mm"] = {f"{w:g}": (_pick(case["spin"], model=name, N=1.0, omega=w) or {}).get("rbar_spin_mm")
                                for w in (0.1, 0.3, 1.0, 3.0)}
            k["N45_1ms_N"] = (_pick(case["T5"], model=name, dt_ms=1.0) or {}).get("N_at_45_N")
        P, Dk, lw_ = t["pads"], t["drake"], t["law"] or {}
        t["ratios"] = {
            "pads_approach_1N_over_law": _ratio(P["approach_um"]["1"], (lw_.get("approach_um") or {}).get("1")),
            "drake_approach_1N_over_law": _ratio(Dk["approach_um"]["1"], (lw_.get("approach_um") or {}).get("1")),
            "pads_onset_1N_over_law": _ratio(P["onset_arm_1N_mm"], (lw_.get("onset_arm_mm") or {}).get("1")),
            "drake_onset_1N_over_law": _ratio(Dk["onset_arm_1N_mm"], (lw_.get("onset_arm_mm") or {}).get("1")),
            "pads_spin_1_over_law": _ratio(P["spin_arm_mm"]["1"], (lw_.get("spin_arm_mm") or {}).get("1")),
            "drake_spin_1_over_law": _ratio(Dk["spin_arm_mm"]["1"], (lw_.get("spin_arm_mm") or {}).get("1")),
            "pads_spin_3_over_0.3": _ratio(P["spin_arm_mm"]["3"], P["spin_arm_mm"]["0.3"]),
            "pads_N45_over_drake": _ratio(P["N45_1ms_N"], Dk["N45_1ms_N"]),
            "pads_N45_over_law": _ratio(P["N45_1ms_N"], lw_.get("N45_N"))}
        out["tips"][tip] = t
    write(D / "bed" / "summary.json", out)
    f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"  # noqa: E731
    print(f"{'tip':28s} {'law appr':>8s} {'pads/law':>8s} {'drk/law':>8s} {'onset':>6s} {'pads':>6s} {'drk':>6s} "
          f"{'spin':>6s} {'pads':>6s} {'drk':>6s} {'p3/0.3':>6s} {'N45 p/d':>7s} {'touch':>5s}")
    for tip, t in out["tips"].items():
        lw_, R, P = t["law"] or {}, t["ratios"], t["pads"]
        print(f"{tip:28s} {f((lw_.get('approach_um') or {}).get('1'), 1):>8s} {f(R['pads_approach_1N_over_law']):>8s} "
              f"{f(R['drake_approach_1N_over_law']):>8s} {f((lw_.get('onset_arm_mm') or {}).get('1'), 2):>6s} "
              f"{f(R['pads_onset_1N_over_law'], 2):>6s} {f(R['drake_onset_1N_over_law'], 2):>6s} "
              f"{f((lw_.get('spin_arm_mm') or {}).get('1'), 2):>6s} {f(R['pads_spin_1_over_law'], 2):>6s} "
              f"{f(R['drake_spin_1_over_law'], 2):>6s} {f(R['pads_spin_3_over_0.3'], 2):>6s} "
              f"{f(R['pads_N45_over_drake'], 2):>7s} {f(P['touching_1N'], 0):>5s}")
    return out


# ------------------------------------------------------------------------------------------------ trajectories

def traj_rows():
    by = defaultdict(dict)
    for f in sorted((D / "traj").glob("*/*.jsonl")):
        for r in rows(f):
            if r.get("status") in ("complete", "ejected") and r.get("trace"):
                by[(r["tip"], r["hand"], int(r["seed"]))][r["arm"]] = r
    return by


def traj():
    import curved_tips_traj_analysis as TA
    by = traj_rows()
    keys = ("rms_dpos_mm", "rms_dang_deg", "final_dang_deg", "rms_dN_N", "rms_dT_N")
    out = {"tips": {}}
    for tip in C.TIPS:
        T = {"pairs": {}, "spread": {}}
        for a, b in (("plan", "drake"), ("rl", "plan")):
            per = []
            for (tp, hand, seed), arms in sorted(by.items()):
                if tp == tip and a in arms and b in arms:
                    c = TA.compare(arms[a], arms[b])
                    if c:
                        per.append({"hand": hand, "seed": seed, **c})
            if per:
                s = {"n_placements": len(per), "held_agree": int(sum(x["held_agree"] for x in per)),
                     "turn_within_3": int(sum(x["turn_within_3"] for x in per)),
                     **{k: {"median": float(np.median([x[k] for x in per])), "q25": float(np.percentile([x[k] for x in per], 25)),
                            "q75": float(np.percentile([x[k] for x in per], 75))} for k in keys},
                     "per_placement": per}
                T["pairs"][f"{a}-{b}"] = s
        per = []
        for hand in HANDS:
            tr = [by[k]["drake"] for k in sorted(by) if k[0] == tip and k[1] == hand and "drake" in by[k]]
            for i in range(len(tr)):
                for j in range(i + 1, len(tr)):
                    c = TA.compare(tr[i], tr[j])
                    if c:
                        per.append(c)
        if per:
            T["spread"]["drake"] = {"n_pairs": len(per), "rms_dang_deg": {"median": float(np.median([x["rms_dang_deg"] for x in per]))}}
        us = defaultdict(list)
        held = defaultdict(int)
        for (tp, _, _), arms in by.items():
            if tp == tip:
                for arm, r in arms.items():
                    us[arm].append(r.get("us_per_step_median"))
                    held[arm] += bool(r.get("held_end"))
        T["us_per_step_median"] = {k: float(np.median([x for x in v if x])) for k, v in us.items() if any(v)}
        T["held_end"] = dict(held)
        out["tips"][tip] = T
    write(D / "traj" / "summary.json", out)
    for tip, T in out["tips"].items():
        pd_, rp = T["pairs"].get("plan-drake"), T["pairs"].get("rl-plan")
        print(f"{tip:28s} plan-drake " + (f"{pd_['turn_within_3']}/{pd_['n_placements']} RMS ang {pd_['rms_dang_deg']['median']:.2f}"
                                          if pd_ else "-") + "  rl-plan " +
              (f"{rp['rms_dang_deg']['median']:.2f}" if rp else "-") +
              (f"  Drake spread {T['spread']['drake']['rms_dang_deg']['median']:.2f}" if T["spread"] else "") +
              f"  held {T['held_end']}")
    return out


# ------------------------------------------------------------------------------------------------ screen and stage 3

def scale_rows(base, k, mu, pl, what="turn"):
    return rows(base / f"scale/{k}_mu{mu}_{pl}/{what}.jsonl")


def hom_rows(base, k, mu, pl):
    return [r for r in rows(base / f"hom/{k}_mu{mu}_{pl}.jsonl") if r.get("status") == "ok"]


def rigid_stats(turn):
    v = [r["rigid_share"] for r in turn if r.get("status") == "ok" and r.get("rigid_share") is not None]
    parts = defaultdict(float)
    for r in turn:
        for p, s in (r.get("rigid_by_part") or {}).items():
            parts[p.split("_", 1)[1] if "_" in p else p] += s
    n = max(len(v), 1)
    return {"median": float(np.median(v)) if v else None, "q75": float(np.percentile(v, 75)) if v else None,
            "share_over_0.25": float(np.mean([x > 0.25 for x in v])) if v else None,
            "by_part_mean": {p: round(s / n, 4) for p, s in sorted(parts.items(), key=lambda kv: -kv[1])[:4]}}


def screen():
    import tip_codesign_analysis as AN
    S = D / "screen"
    tr = traj()["tips"]
    out = {"tips": {}, "conditions": [cname(*c) for c in C.CONDS], "reference": C.REF}
    for k in C.TIPS:
        T = {"conditions": {}}
        for mu, pl in C.CONDS:
            turn = scale_rows(S, k, mu, pl)
            hk = hom_rows(S, k, mu, pl)
            J, det = AN.stage_score(turn, hk)
            det["J"] = J
            det["n_turn_rows"] = sum(r.get("status") == "ok" for r in turn)
            det["rigid_share"] = rigid_stats(turn)
            det["hom_rigid_share_median"] = (float(np.median([r["rigid_share"] for r in hk if r.get("rigid_share") is not None]))
                                             if any(r.get("rigid_share") is not None for r in hk) else None)
            if (mu, pl) == ("0.7", C.SERVO):
                T["J1"], _ = AN.stage_score(turn, [r for r in hk if int(r["seed"]) in (0, 1, 2)])
            if (mu, pl) in (("0.5", C.SERVO), ("0.7", C.SERVO)):
                ok = [r for r in turn if r.get("status") == "ok"]
                det["three_finger_frac_cells"] = float(np.mean([r.get("fingers_loaded_end") == 3 for r in ok])) if ok else None
                hold = [r for r in scale_rows(S, k, mu, pl, "hold") if r.get("status") == "ok"]
                det["hold_held"] = int(sum(bool(r["held"]) for r in hold))
                det["hold3"] = int(sum(bool(r["held"]) and r.get("fingers_loaded") == 3 for r in hold))
                det["n_hold"] = len(hold)
            T["conditions"][cname(mu, pl)] = det
        vals = [v["J"] for v in T["conditions"].values() if v["J"] is not None]
        T["J2"] = min(vals) if len(vals) == len(C.CONDS) else None
        T["J2_condition"] = (min(T["conditions"], key=lambda c: T["conditions"][c]["J"] if T["conditions"][c]["J"] is not None else 9)
                             if vals else None)
        T["n_conditions"] = len(vals)
        pdk = (tr.get(k) or {}).get("pairs", {}).get("plan-drake")
        T["agree_drake_3deg"] = pdk["turn_within_3"] if pdk else None
        T["n_vs_drake"] = pdk["n_placements"] if pdk else 0
        T["us_per_step_median"] = ((tr.get(k) or {}).get("us_per_step_median") or {}).get("plan")
        T["three_finger_frac_mu0.5"] = T["conditions"][cname("0.5", C.SERVO)].get("three_finger_frac_cells")
        allturn = [r for mu, pl in C.CONDS for r in scale_rows(S, k, mu, pl)]
        T["rigid_share_all"] = rigid_stats(allturn)
        out["tips"][k] = T
    Cn, base = out["tips"], out["tips"].get(C.REF, {})
    for k, v in Cn.items():
        v["gate_pass"] = bool(v["n_vs_drake"] >= 20 and base.get("agree_drake_3deg") is not None and
                              v["agree_drake_3deg"] is not None and v["agree_drake_3deg"] >= base["agree_drake_3deg"] - 2)
        v["cost_ratio"] = _ratio(v.get("us_per_step_median"), base.get("us_per_step_median"))
    cand = [k for k in Cn if k != C.REF]
    order = lambda f, key: sorted((k for k in cand if f(Cn[k]) and Cn[k].get(key) is not None), key=lambda k: -Cn[k][key])  # noqa: E731
    fin, why = [], {}

    def pick(lst, label):
        for k in lst:
            if k not in fin:
                fin.append(k)
                why[k] = label
                return k
    pick(order(lambda v: v["gate_pass"], "J2"), "(1) highest J2 passing the agreement gate")
    pick(order(lambda v: v["gate_pass"] and (v["cost_ratio"] or 9) <= 0.8, "J2"), "(2) highest J2 at <= 0.8 of the v4b tip's cost")
    pick(order(lambda v: True, "three_finger_frac_mu0.5"), "(3) highest three-finger share at friction 0.5")
    while len(fin) < 4 and pick(order(lambda v: v["gate_pass"], "J2"), "(4) next highest J2 passing the gate"):
        pass
    out["finalists"], out["finalist_rule"] = fin, why
    out["stage3_tips"] = fin + [C.REF] + ([ASBUILT] if ASBUILT not in fin else [])
    write(S / "summary.json", out)
    r3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    for k, v in sorted(Cn.items(), key=lambda kv: -(kv[1]["J2"] or kv[1].get("J1") or -1)):
        print(f"{k:28s} J1 {r3(v.get('J1'))} J2 {r3(v['J2'])} ({v['n_conditions']}/5) "
              + " ".join(f"{c.split('|')[0]}{'' if C.SERVO in c else ('/kp0.5' if 'kp0.5_' in c else '/fr0.12')}:{r3(x['J'])}"
                         for c, x in v["conditions"].items())
              + f"  Drake {v['agree_drake_3deg']}/{v['n_vs_drake']} gate {v['gate_pass']} cost {r3(v['cost_ratio'])}"
              + f"  3f@0.5 {r3(v['three_finger_frac_mu0.5'])} rigid {r3(v['rigid_share_all']['median'])}")
    print("finalists:", why)
    return out


def stage3(n_boot=10000, rng_seed=0):
    import tip_codesign_analysis as AN
    S, F = D / "screen", D / "final"
    tips = json.loads((S / "summary.json").read_text())["stage3_tips"]
    rng = np.random.default_rng(rng_seed)
    ic = rng.integers(0, len(SEEDS3_CELL), size=(n_boot, len(AN.CELLS), len(SEEDS3_CELL)))
    ih = rng.integers(0, len(SEEDS3_HOM), size=(n_boot, len(HANDS), len(SEEDS3_HOM)))
    rc, rh = np.arange(len(AN.CELLS))[None, :, None], np.arange(len(HANDS))[None, :, None]
    out = {"tips": tips, "conditions": [cname(*c) for c in COND3], "n_boot": n_boot, "baseline": C.REF, "candidates": {}}
    point, boot = {}, {}
    for k in tips:
        Jc, Jb, det = {}, [], {}
        for mu, pl in COND3:
            sel = [r for r in scale_rows(S, k, mu, pl) if r.get("status") == "ok" and int(r["seed"]) == 0]
            new = [r for r in scale_rows(F, k, mu, pl) if r.get("status") == "ok"]
            Cm = np.full((len(AN.CELLS), len(SEEDS3_CELL)), np.nan)
            chosen = {}
            for i, c in enumerate(AN.CELLS):
                cands = [(AN.held_turn(r), int(r["cand"])) for r in sel if (r["shape"], float(r["d_mm"]), r.get("t")) == c]
                if not cands:
                    continue
                best = max(cands)[1]
                chosen[f"{c[1]:g}:{c[2]:g}"] = best
                for r in new:
                    if (r["shape"], float(r["d_mm"]), r.get("t")) == c and int(r["cand"]) == best and int(r["seed"]) in SEEDS3_CELL:
                        Cm[i, SEEDS3_CELL.index(int(r["seed"]))] = AN.held_turn(r)
            Hm = np.full((len(HANDS), len(SEEDS3_HOM)), np.nan)
            for r in hom_rows(F, k, mu, pl):
                if int(r["seed"]) in SEEDS3_HOM:
                    Hm[HANDS.index(r["hand"]), SEEDS3_HOM.index(int(r["seed"]))] = AN.held_turn(r)
            name = cname(mu, pl)
            Jc[name] = float(AN._score3(Cm[None], Hm[None])[0])
            Jb.append(AN._score3(Cm[rc, ic], Hm[rh, ih]))
            det[name] = {"chosen_grasp": chosen, "n_cell_rows": int(np.isfinite(Cm).sum()), "n_hom_rows": int(np.isfinite(Hm).sum()),
                         "hom_mean_held_turn_deg": float(np.nanmean(Hm)) if np.isfinite(Hm).any() else None,
                         "cells_mean_held_turn_deg": float(np.nanmean(Cm)) if np.isfinite(Cm).any() else None,
                         "rigid_share": rigid_stats(new)}
        point[k], boot[k] = min(Jc.values()), np.min(np.stack(Jb), axis=0)
        hold = [r for r in scale_rows(S, k, "0.5", C.SERVO, "hold") if r.get("status") == "ok"]
        out["candidates"][k] = {"J3": point[k], "J": Jc, **det,
                                "hold3_mu0.5": sum(bool(r["held"]) and r.get("fingers_loaded") == 3 for r in hold),
                                "n_hold_mu0.5": len(hold)}
    base = out["candidates"].get(C.REF)
    for k in tips:
        if k == C.REF or base is None:
            continue
        d = boot[k] - boot[C.REF]
        lo, hi = np.percentile(d, [5, 95])
        v = out["candidates"][k]
        v["dJ3_vs_base"] = point[k] - point[C.REF]
        v["dJ3_ci90"] = [float(lo), float(hi)]
        v["hold_ok"] = v["hold3_mu0.5"] >= base["hold3_mu0.5"] - 2
        v["beats_base"] = bool(lo > 0 and v["hold_ok"])
        v["below_base"] = bool(hi < 0)
    write(F / "summary.json", out)
    for k in tips:
        v = out["candidates"][k]
        print(f"{k:28s} J3 {v['J3']:.3f} " + " ".join(f"{c.split('|')[0]}:{x:.3f}" for c, x in v["J"].items())
              + f"  hold3@0.5 {v['hold3_mu0.5']}/{v['n_hold_mu0.5']}"
              + (f"  dJ3 {v['dJ3_vs_base']:+.3f} [{v['dJ3_ci90'][0]:+.3f}, {v['dJ3_ci90'][1]:+.3f}]" if "dJ3_ci90" in v
                 else "  (baseline)"))
    return out


# ------------------------------------------------------------------------------------------------ cost

def cost():
    Cd = D / "cost"
    out = {"tips": {}, "reference": C.REF}
    gx = {r["variant"]: r for r in rows(Cd / "gpu_export.jsonl") if r.get("status") == "ok"}
    gt = [r for r in rows(Cd / "gpu_timing.jsonl") if r.get("status") == "ok" and r.get("nworld") == 2048]
    for tip in C.TIPS:
        rs = [r for r in rows(Cd / f"cpu_{tip}.jsonl") if r.get("status") == "ok"]
        t = {"cpu": {}, "gpu": {}}
        for cell in ("cylinder:16:0.625", "cylinder:25:0.125"):
            for mdl in ("pads", "pt"):
                st = _pick(rs, cell=cell, model=mdl, state="grip", kind="step") or {}
                hl = _pick(rs, cell=cell, model=mdl, state="grip", kind="held") or {}
                ro = {n: (_pick(rs, cell=cell, model=mdl, state="grip", kind="rollout", nthread=n) or {}) for n in (1, 3)}
                t["cpu"][f"{cell}|{mdl}"] = {
                    "step_us": (st.get("step_us") or {}).get("median"), "ncon": st.get("ncon_mean"), "nefc": st.get("nefc_mean"),
                    "held": hl.get("held"), "solver_niter": hl.get("solver_niter_mean"),
                    "replan_1thread_s": (ro[1].get("wall_s") or {}).get("median"),
                    "replan_3thread_s": (ro[3].get("wall_s") or {}).get("median"), "n_pad_geoms": st.get("n_pad_geoms")}
        g = [r for r in gt if r["variant"] == tip]
        if g:
            t["gpu"] = {"us_per_world_step": float(np.median([r["us_per_world_step"] for r in g])),
                        "contacts_per_world": g[0].get("contacts_per_world"), "nefc_world": g[0].get("nefc_world_mean"),
                        "candidates_per_world": g[0].get("candidates_per_world"), "gpu_peak_mb": g[0].get("gpu_proc_peak_mb"),
                        "held_frac": g[0].get("held_frac"), "n_reps": len(g),
                        "export_held_worlds": (gx.get(tip) or {}).get("held_worlds"),
                        "export_ngeom": (gx.get(tip) or {}).get("ngeom")}
        out["tips"][tip] = t
    base = out["tips"].get(C.REF)
    for tip, t in out["tips"].items():
        if base:
            t["ratios"] = {k: _ratio(t["cpu"][k]["step_us"], base["cpu"][k]["step_us"]) for k in t["cpu"]}
            t["ratios"]["gpu"] = _ratio(t["gpu"].get("us_per_world_step"), base["gpu"].get("us_per_world_step"))
            t["ratios"]["gpu_mb"] = _ratio(t["gpu"].get("gpu_peak_mb"), base["gpu"].get("gpu_peak_mb"))
    write(Cd / "summary.json", out)
    f = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"  # noqa: E731
    for tip, t in out["tips"].items():
        a, b, g = t["cpu"]["cylinder:16:0.625|pads"], t["cpu"]["cylinder:25:0.125|pads"], t["gpu"]
        print(f"{tip:28s} 16mm {f(a['step_us'])} us ncon {f(a['ncon'], 0)} niter {f(a['solver_niter'], 2)} replan3 "
              f"{f(a['replan_3thread_s'], 2)} s | 25mm {f(b['step_us'])} us | GPU {f(g.get('us_per_world_step'), 2)} us "
              f"con {f(g.get('contacts_per_world'), 0)} {f(g.get('gpu_peak_mb'), 0)} MB")
    return out


# ------------------------------------------------------------------------------------------------ finite elements

def fe():
    F = D / "fe"
    fr = [r for r in rows(F / "fe.jsonl")]
    sens = {r["case"]: r for r in json.loads((F / "sensing.json").read_text())["rows"]} if (F / "sensing.json").exists() else {}
    sfe = {r["case"]: r for r in rows(F / "sense_fe.jsonl")}
    out = {"tips": {}}
    for tip in C.TIPS:
        P = {r["probe"]: r for r in fr if r.get("fe_key") == tip}
        if not P:
            continue
        ap = lambda p, F_="1": (P.get(p) or {}).get("per_force", {}).get(F_, {}).get("approach_um")  # noqa: E731
        arm = lambda p, F_="1": (P.get(p) or {}).get("per_force", {}).get(F_, {}).get("r_bar_mm")  # noqa: E731

        def expo(p):
            a, b = ap(p, "0.5"), ap(p, "3")
            return math.log(b / a) / math.log(6.0) if a and b else None
        smap = {p[2:]: ap(p) for p in P if p.startswith("s_")}
        apex = smap.get("apex")
        case = next(iter(P.values()))["case"]
        t = {"case": case, "body": next(iter(P.values())).get("body"), "magnets": len(next(iter(P.values())).get("magnets") or []),
             "sphere_apex_um": {F_: ap("s_apex", F_) for F_ in ("0.5", "1", "3", "6")},
             "rod_across_um": {F_: ap("cx12.5_z0", F_) for F_ in ("0.5", "1", "3", "6")},
             "rod_along_1N_um": ap("cz12.5_y0"), "exponent_sphere": expo("s_apex"), "exponent_rod": expo("cx12.5_z0"),
             "rod_over_sphere_1N": _ratio(ap("cx12.5_z0"), apex), "rod_along_over_across_1N": _ratio(ap("cz12.5_y0"), ap("cx12.5_z0")),
             "twist_arm_rod_mm": {F_: arm("cx12.5_z0", F_) for F_ in ("1", "3")},
             "sphere_map_1N_um": smap, "sphere_map_over_apex": {k: _ratio(v, apex) for k, v in smap.items()},
             "sphere_map_spread": (max(smap.values()) / min(smap.values()) if smap and min(smap.values()) else None),
             "n_dof": next(iter(P.values())).get("n_dof")}
        s = sens.get(case)
        if s:
            t["sensing"] = {"centre_force_N": s["gain"]["centre_force_N"], "centre_force_gain7_N": s["gain7"]["centre_force_N"],
                            "median_force_N": s["gain"]["median_force_N"], "B_max_component_mT": s["B_max_component_mT"],
                            "gain_sel": s["gain_sel"], "magnet_travel_um_per_N": s["magnet_travel_um_per_N"],
                            "saturated": s["gain"]["saturated"], "approach_1N_um": s["approach_1N_um"]}
        elif case in sfe:
            t["sensing"] = {"pending": True}
        out["tips"][tip] = t
    write(F / "summary.json", out)
    for tip, t in out["tips"].items():
        f = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"  # noqa: E731
        s = t.get("sensing") or {}
        print(f"{tip:28s} sphere {f(t['sphere_apex_um']['1'])} rod {f(t['rod_across_um']['1'])} along {f(t['rod_along_1N_um'])} um  "
              f"n {f(t['exponent_sphere'], 2)}/{f(t['exponent_rod'], 2)} rod/sph {f(t['rod_over_sphere_1N'], 2)} arm "
              f"{f(t['twist_arm_rod_mm']['1'], 2)} mm map spread {f(t['sphere_map_spread'], 2)}  "
              f"res {f(s.get('centre_force_N'), 3)} N B {f(s.get('B_max_component_mT'), 1)} mT")
    return out


# ------------------------------------------------------------------------------------------------ compute

def compute():
    """Wall time and core-hours per kind of job: each queue job ran on one core (FE and GPU rows from the ledger)."""
    jobs = [json.loads(l) for l in open(C.LOGS / "queue" / "jobs.jsonl")] if (C.LOGS / "queue" / "jobs.jsonl").exists() else []
    kind = defaultdict(lambda: {"n": 0, "core_h": 0.0, "first": None, "last": None})
    for j in jobs:
        w = j["job"].split()
        k = w[0] if w[0] != "traj" else f"traj {w[2]}"
        if w[0] in ("scale", "hom") and "/final/" in j["job"]:
            k = f"stage3 {w[0]}"
        e = kind[k]
        e["n"] += 1
        e["core_h"] += (j["end"] - j["start"]) / 3600
        e["first"] = min(e["first"] or j["start"], j["start"])
        e["last"] = max(e["last"] or j["end"], j["end"])
    out = {"jobs": {k: {"n": v["n"], "core_h": round(v["core_h"], 2), "wall_h": round((v["last"] - v["first"]) / 3600, 2)}
                    for k, v in kind.items()}}
    led = [json.loads(l) for l in open(C.LOGS / "compute.jsonl")] if (C.LOGS / "compute.jsonl").exists() else []
    out["ledger"] = led
    write(D / "compute" / "summary.json", out)
    for k, v in out["jobs"].items():
        print(f"{k:16s} {v['n']:4d} jobs {v['core_h']:6.2f} core-h over {v['wall_h']:5.2f} h")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["bed", "traj", "screen", "stage3", "cost", "fe", "compute", "all"])
    a = ap.parse_args()
    if a.what == "all":
        for f in (bed, screen, cost, fe, compute):
            f()
        if (D / "final").exists():
            stage3()
        return
    {"bed": bed, "traj": traj, "screen": screen, "stage3": stage3, "cost": cost, "fe": fe, "compute": compute}[a.what]()


if __name__ == "__main__":
    main()
