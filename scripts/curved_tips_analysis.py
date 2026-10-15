#!/usr/bin/env python3
"""Summaries of the curved-tip study's rows (docs/handoff/20261015-curved_tips.md); each subcommand reads rows under
docs/experiments/20261015-curved_tips/ and writes one JSON summary next to them.

  bed      step 2: per tip the law (Drake's field on the 0.25 mm mesh), the builder-v2 pads and Drake on the bed:
           approach at 0.5-3 N, twist-onset arm, spin arm at 0.1-3 rad/s, brake force at 45 deg, rotational and
           translational creep, touching pads, constraint rows and wall time per step (bed/summary.json)
  screen   step 5, stages 1 and 2 (the 2026-10-14 rules, tip_codesign_analysis.stage_score): per tip and condition the
           stage score; J1 at mu 0.7 on the servo plant (cells scored on seeds 1-2, HOM seeds 0-2), J2 the minimum over
           the five conditions (HOM seeds 0-4); the Drake agreement of the D1-D8 replays (traj/traces.jsonl: the pads
           at the planner's numerics within 3 deg of Drake's end turn with its held outcome, of 24) and its gate (at most
           two below the as-built block); the cost (median us per replay step over the as-built block's); the
           three-finger share at mu 0.5; finalists (1) highest J2 passing the gate, (2) highest J2 at <= 0.8 of the
           as-built cost, (3) highest three-finger share at mu 0.5, then the next highest J2 passing the gate up to four
           curved tips; the as-built block and the block finalist (3, 6) as references (screen/summary.json)
  stage3   step 5, stage 3: J3 on fresh placements and the paired bootstrap against the as-built block
           (tip_codesign_analysis.stage3's rules; final/summary.json)
  cost     step 3: CPU step at the three-finger grip and 64-plan replanning (U1's protocol), GPU world-step at the held
           RL state (cost/summary.json)

    python3 scripts/curved_tips_analysis.py bed
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
D = ROOT / "docs/experiments/20261015-curved_tips"


def rows(path):
    out = []
    for f in sorted(glob.glob(str(path))):
        for line in open(f):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _pick(rs, **kw):
    """The last complete (or 'ok') row matching every key=value."""
    out = [r for r in rs if r.get("status", "complete") in ("complete", "ok") and all(r.get(k) == v for k, v in kw.items())]
    return out[-1] if out else None


def _ratio(a, b):
    return float(a / b) if (a is not None and b not in (None, 0)) else None


def bed():
    B = D / "bed"
    law = rows(B / "law.p*.jsonl")              # law.jsonl is their concatenation (the film's law lookup)
    case = {c: rows(B / f"{c}.jsonl") for c in ("static", "T2", "spin", "T5", "T1")}
    tips = sorted({r["tip"] for r in law})
    out = {"tips": {}, "terms": {
        "law": "Drake's pressure field E (1 - g) on the tip's 0.25 mm mesh, integrated quasi-statically (curved_tips_law.py)",
        "pads": "MuJoCo 1 mm pads, builder v2, relaxation 0.02 s, impratio 100, 1 ms (curved_tips_bed.py)",
        "drake": "Drake 1.57 compliant convex of the 0.25 mm mesh, relaxation 0.01 s, SAP, 1 ms"}}
    for tip in tips:
        L = {(r["field"], r.get("res_m")): r for r in law if r["tip"] == tip}
        lw = L.get(("drake", 0.00025))
        t = {"law": None, "pads": {}, "drake": {}}
        if lw:
            t["law"] = {"approach_um": lw["approach_um"], "onset_arm_mm": lw["onset_arm_mm"],
                        "spin_arm_mm": lw["spin_arm_mm"], "N45_N": lw["brake_N"]["N45"],
                        "approach_1N_um_res0.5": (L.get(("drake", 0.0005)) or {}).get("approach_um", {}).get("1"),
                        "local_approach_1N_um": (L.get(("local", None)) or {}).get("approach_um", {}).get("1"),
                        "pads_v2_qs": {k: (L.get(("pads_v2", None)) or {}).get(k) for k in ("approach_um", "onset_arm_mm",
                                                                                         "spin_arm_mm")},
                        "pads_v1_qs_onset_1N_mm": (L.get(("pads_v1", None)) or {}).get("onset_arm_mm", {}).get("1"),
                        "n_pads": (L.get(("pads_v2", None)) or {}).get("n_pads")}
        for kind in ("pads", "drake"):
            name = f"{kind}_{tip}"
            k = t[kind]
            k["approach_um"] = {f"{N:g}": (_pick(case["static"], model=name, N=N) or {}).get("approach_um")
                                for N in (0.5, 1.0, 2.0, 3.0)}
            s1 = _pick(case["static"], model=name, N=1.0) or {}
            k["touching_1N"] = s1.get("touching_per_side")
            k["nefc_1N"] = s1.get("nefc_mean")
            k["ncon_1N"] = s1.get("ncon_mean")
            k["us_per_step_static_1N"] = s1.get("us_per_step_median")
            k["n_spheres"] = (s1.get("info") or {}).get("n_spheres")
            for N in (1.0, 3.0):
                r = _pick(case["T2"], model=name, N=N) or {}
                k[f"onset_arm_{N:g}N_mm"] = r.get("rbar_onset_mm")
                k[f"kin_arm_{N:g}N_mm"] = r.get("rbar_kin_mm")
                k[f"twist_creep_{N:g}N_deg_s"] = r.get("creep_deg_s")
            k["spin_arm_mm"] = {f"{w:g}": (_pick(case["spin"], model=name, N=1.0, omega=w) or {}).get("rbar_spin_mm")
                                for w in (0.1, 0.3, 1.0, 3.0)}
            k["us_per_step_spin_1"] = (_pick(case["spin"], model=name, N=1.0, omega=1.0) or {}).get("us_per_step_median")
            sp = _pick(case["spin"], model=name, N=1.0, omega=1.0) or {}
            k["spin_slip_share_1"] = (sp.get("share") or {}).get("slip")
            for dt in (1.0, 0.5):
                k[f"N45_{dt:g}ms_N"] = (_pick(case["T5"], model=name, dt_ms=dt) or {}).get("N_at_45_N")
            r = _pick(case["T1"], model=name, N=1.0) or {}
            k["pull_mu_eff"] = r.get("mu_eff")
            k["pull_creep_mm_s"] = r.get("creep_mm_s")
        P, Dk, lw_ = t["pads"], t["drake"], t["law"] or {}
        t["ratios"] = {
            "pads_approach_1N_over_law": _ratio(P["approach_um"]["1"], (lw_.get("approach_um") or {}).get("1")),
            "drake_approach_1N_over_law": _ratio(Dk["approach_um"]["1"], (lw_.get("approach_um") or {}).get("1")),
            "pads_onset_1N_over_law": _ratio(P["onset_arm_1N_mm"], (lw_.get("onset_arm_mm") or {}).get("1")),
            "drake_onset_1N_over_law": _ratio(Dk["onset_arm_1N_mm"], (lw_.get("onset_arm_mm") or {}).get("1")),
            "pads_spin_1_over_law": _ratio(P["spin_arm_mm"]["1"], (lw_.get("spin_arm_mm") or {}).get("1")),
            "drake_spin_1_over_law": _ratio(Dk["spin_arm_mm"]["1"], (lw_.get("spin_arm_mm") or {}).get("1")),
            "pads_spin_3_over_0.3": _ratio(P["spin_arm_mm"]["3"], P["spin_arm_mm"]["0.3"]),
            "drake_spin_3_over_0.3": _ratio(Dk["spin_arm_mm"]["3"], Dk["spin_arm_mm"]["0.3"]),
            "pads_N45_over_drake_0.5ms": _ratio(P["N45_1ms_N"], Dk["N45_0.5ms_N"] or Dk["N45_1ms_N"]),
            "pads_cost_over_drake": _ratio(P["us_per_step_static_1N"], Dk["us_per_step_static_1N"])}
        out["tips"][tip] = t
    (D / "bed" / "summary.json").write_text(json.dumps(out, indent=1))
    hdr = ("tip", "law appr", "pads/law", "drk/law", "onset law", "pads/law", "drk/law", "spin law", "pads/law",
           "drk/law", "pads 3/0.3", "N45 pads/drk", "touch", "nefc", "us pads", "us drk")
    print(" ".join(f"{h:>10s}" for h in hdr))
    f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"  # noqa: E731
    for tip, t in out["tips"].items():
        lw_, R, P, Dk = t["law"] or {}, t["ratios"], t["pads"], t["drake"]
        print(f"{tip[:28]:>28s} " + " ".join(f"{v:>10s}" for v in (
            f((lw_.get("approach_um") or {}).get("1"), 1), f(R["pads_approach_1N_over_law"]), f(R["drake_approach_1N_over_law"]),
            f((lw_.get("onset_arm_mm") or {}).get("1")), f(R["pads_onset_1N_over_law"]), f(R["drake_onset_1N_over_law"]),
            f((lw_.get("spin_arm_mm") or {}).get("1")), f(R["pads_spin_1_over_law"]), f(R["drake_spin_1_over_law"]),
            f(R["pads_spin_3_over_0.3"]), f(R["pads_N45_over_drake_0.5ms"]), f(P["touching_1N"], 0), f(P["nefc_1N"], 0),
            f(P["us_per_step_static_1N"], 0), f(Dk["us_per_step_static_1N"], 0))))
    return out


SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
CONDS = [("0.7", SERVO), ("1.0", SERVO), ("0.5", SERVO), ("0.7", "kp0.5_kv0_fr0.18_fl0_dp0.0144"),
         ("0.7", "kp0.72_kv0_fr0.12_fl0_dp0.0144")]
COND3 = [c for c in CONDS if c != ("1.0", SERVO)]
ASBUILT, FINALIST = "tcd_rs3_re3_p0.2", "tcd_rs3_re6_p0.2"
HANDS = tuple(f"D{i}" for i in range(1, 9))
SEEDS3_CELL, SEEDS3_HOM = (3, 4, 5), (5, 6, 7, 8, 9)


def cname(mu, pl):
    return f"{mu}|{pl}"


def agreement():
    """Per tip, from traj/traces.jsonl: the D1-D8 replays (seeds 0-2) whose pads run at the planner's numerics ends
    within 3 deg of Drake's turn with Drake's held outcome, and the median us per replay step of those pads runs."""
    by = {}
    for r in rows(D / "traj" / "traces.jsonl"):
        if r.get("status") in ("complete", "ejected") and r["arm"] in ("plan", "drake") and int(r["seed"]) in (0, 1, 2):
            by.setdefault(r["tip"], {}).setdefault((r["hand"], int(r["seed"])), {})[r["arm"]] = r
    out = {}
    for tip, d in by.items():
        ks = [k for k, v in d.items() if "plan" in v and "drake" in v]
        ag = sum(bool(d[k]["plan"].get("held_end")) == bool(d[k]["drake"].get("held_end"))
                 and abs(float(d[k]["plan"].get("turn_end_deg", np.nan)) - float(d[k]["drake"].get("turn_end_deg", np.nan))) <= 3
                 for k in ks)
        us = [v["plan"]["us_per_step_median"] for v in d.values() if "plan" in v and v["plan"].get("us_per_step_median")]
        out[tip] = {"agree_drake_3deg": int(ag), "n_vs_drake": len(ks), "us_per_step_median": float(np.median(us)) if us else None}
    return out


def screen():
    import tip_codesign_analysis as AN
    S = D / "screen"
    hom = [r for r in rows(S / "hom.jsonl") if r.get("status") == "ok"]
    ag = agreement()
    tips = sorted({p.name.split("_mu")[0] for p in (S / "scale").glob("*")})
    out = {"tips": {}, "conditions": [cname(*c) for c in CONDS]}
    for k in tips:
        T = {"conditions": {}}
        for mu, pl in CONDS:
            turn = rows(S / f"scale/{k}_mu{mu}_{pl}/turn.jsonl")
            hk = [r for r in hom if r.get("tip") == k and str(r.get("mu")) == mu and r.get("plant") == pl]
            J, det = AN.stage_score(turn, hk)
            det["J"] = J
            det["n_turn_rows"] = sum(r.get("status") == "ok" for r in turn)
            if (mu, pl) == ("0.7", SERVO):
                J1, _ = AN.stage_score(turn, [r for r in hk if int(r["seed"]) in (0, 1, 2)])
                T["J1"] = J1
            if (mu, pl) in (("0.5", SERVO), ("0.7", SERVO)):
                ok = [r for r in turn if r.get("status") == "ok"]
                det["three_finger_frac_cells"] = float(np.mean([r.get("fingers_loaded_end") == 3 for r in ok])) if ok else None
                hold = [r for r in rows(S / f"scale/{k}_mu{mu}_{pl}/hold.jsonl") if r.get("status") == "ok"]
                det["hold_held"] = int(sum(bool(r["held"]) for r in hold))
                det["hold3"] = int(sum(bool(r["held"]) and r.get("fingers_loaded") == 3 for r in hold))
                det["n_hold"] = len(hold)
            T["conditions"][cname(mu, pl)] = det
        vals = [v["J"] for v in T["conditions"].values() if v["J"] is not None]
        T["J2"] = min(vals) if len(vals) == len(CONDS) else None
        T["n_conditions"] = len(vals)
        T.update(ag.get(k, {"agree_drake_3deg": None, "n_vs_drake": 0, "us_per_step_median": None}))
        T["three_finger_frac_mu0.5"] = T["conditions"][cname("0.5", SERVO)].get("three_finger_frac_cells")
        out["tips"][k] = T
    Cn = out["tips"]
    base = Cn.get(ASBUILT, {})
    for k, v in Cn.items():
        v["gate_pass"] = bool(v["n_vs_drake"] >= 20 and base.get("n_vs_drake") and
                              v["agree_drake_3deg"] >= base["agree_drake_3deg"] - 2)
        v["cost_ratio"] = (v["us_per_step_median"] / base["us_per_step_median"]
                           if v.get("us_per_step_median") and base.get("us_per_step_median") else None)
    curved = [k for k in Cn if k not in (ASBUILT, FINALIST)]
    order = lambda f, key: sorted((k for k in curved if f(Cn[k]) and Cn[k].get(key) is not None), key=lambda k: -Cn[k][key])  # noqa: E731
    fin, why = [], {}

    def pick(lst, label):
        for k in lst:
            if k not in fin:
                fin.append(k)
                why[k] = label
                return k
    pick(order(lambda v: v["gate_pass"], "J2"), "(1) highest J2 passing the agreement gate")
    pick(order(lambda v: v["gate_pass"] and (v["cost_ratio"] or 9) <= 0.8, "J2"), "(2) highest J2 at <= 0.8 of the as-built cost")
    pick(order(lambda v: True, "three_finger_frac_mu0.5"), "(3) highest three-finger share at mu 0.5")
    while len(fin) < 4 and pick(order(lambda v: v["gate_pass"], "J2"), "(4) next highest J2 passing the gate"):
        pass
    out["finalists"], out["finalist_rule"] = fin, why
    out["stage3_tips"] = fin + [ASBUILT, FINALIST]
    (S / "summary.json").write_text(json.dumps(out, indent=1))
    for k, v in sorted(Cn.items(), key=lambda kv: -(kv[1]["J2"] or kv[1].get("J1") or -1)):
        r3 = lambda x: None if x is None else round(x, 3)  # noqa: E731
        print(f"{k:28s} J1 {r3(v.get('J1'))} J2 {r3(v['J2'])} ({v['n_conditions']}/5) "
              + " ".join(f"{c.split('|')[0]}{'' if SERVO in c else ('/kp0.5' if 'kp0.5_' in c else '/fr0.12')}:{r3(x['J'])}"
                         for c, x in v["conditions"].items())
              + f"  Drake {v['agree_drake_3deg']}/{v['n_vs_drake']} gate {v['gate_pass']} cost {r3(v['cost_ratio'])}"
              + f"  3f@0.5 {r3(v['three_finger_frac_mu0.5'])}")
    print("finalists:", why)
    return out


def stage3(n_boot=10000, rng_seed=0):
    import tip_codesign_analysis as AN
    S, F = D / "screen", D / "final"
    fin = json.loads((S / "summary.json").read_text())["stage3_tips"]
    hom3 = [r for r in rows(F / "hom.jsonl") if r.get("status") == "ok"]
    rng = np.random.default_rng(rng_seed)
    ic = rng.integers(0, len(SEEDS3_CELL), size=(n_boot, len(AN.CELLS), len(SEEDS3_CELL)))
    ih = rng.integers(0, len(SEEDS3_HOM), size=(n_boot, len(HANDS), len(SEEDS3_HOM)))
    rc, rh = np.arange(len(AN.CELLS))[None, :, None], np.arange(len(HANDS))[None, :, None]
    out = {"tips": fin, "conditions": [cname(*c) for c in COND3], "n_boot": n_boot, "candidates": {}}
    point, boot = {}, {}
    for k in fin:
        Jc, Jb, det = {}, [], {}
        for mu, pl in COND3:
            sel = [r for r in rows(S / f"scale/{k}_mu{mu}_{pl}/turn.jsonl") if r.get("status") == "ok" and int(r["seed"]) == 0]
            new = [r for r in rows(F / f"scale/{k}_mu{mu}_{pl}/turn.jsonl") if r.get("status") == "ok"]
            Cm = np.full((len(AN.CELLS), len(SEEDS3_CELL)), np.nan)
            chosen = {}
            for i, c in enumerate(AN.CELLS):
                cand = [(AN.held_turn(r), int(r["cand"])) for r in sel if (r["shape"], float(r["d_mm"]), r.get("t")) == c]
                if not cand:
                    continue
                best = max(cand)[1]
                chosen[f"{c[1]:g}:{c[2]:g}"] = best
                for r in new:
                    if (r["shape"], float(r["d_mm"]), r.get("t")) == c and int(r["cand"]) == best and int(r["seed"]) in SEEDS3_CELL:
                        Cm[i, SEEDS3_CELL.index(int(r["seed"]))] = AN.held_turn(r)
            Hm = np.full((len(HANDS), len(SEEDS3_HOM)), np.nan)
            for r in hom3:
                if r.get("tip") == k and str(r.get("mu")) == mu and r.get("plant") == pl and int(r["seed"]) in SEEDS3_HOM:
                    Hm[HANDS.index(r["hand"]), SEEDS3_HOM.index(int(r["seed"]))] = AN.held_turn(r)
            name = cname(mu, pl)
            Jc[name] = float(AN._score3(Cm[None], Hm[None])[0])
            Jb.append(AN._score3(Cm[rc, ic], Hm[rh, ih]))
            det[name] = {"chosen_grasp": chosen, "n_cell_rows": int(np.isfinite(Cm).sum()), "n_hom_rows": int(np.isfinite(Hm).sum()),
                         "hom_mean_held_turn_deg": float(np.nanmean(Hm)) if np.isfinite(Hm).any() else None,
                         "cells_mean_held_turn_deg": float(np.nanmean(Cm)) if np.isfinite(Cm).any() else None}
        point[k], boot[k] = min(Jc.values()), np.min(np.stack(Jb), axis=0)
        hold = [r for r in rows(S / f"scale/{k}_mu0.5_{SERVO}/hold.jsonl") if r.get("status") == "ok"]
        out["candidates"][k] = {"J3": point[k], "J": Jc, **det,
                                "hold3_mu0.5": sum(bool(r["held"]) and r.get("fingers_loaded") == 3 for r in hold),
                                "n_hold_mu0.5": len(hold)}
    base = out["candidates"].get(ASBUILT)
    for k in fin:
        if k == ASBUILT or base is None:
            continue
        d = boot[k] - boot[ASBUILT]
        lo, hi = np.percentile(d, [5, 95])
        v = out["candidates"][k]
        v["dJ3_vs_base"] = point[k] - point[ASBUILT]
        v["dJ3_ci90"] = [float(lo), float(hi)]
        v["hold_ok"] = v["hold3_mu0.5"] >= base["hold3_mu0.5"] - 2
        v["beats_base"] = bool(lo > 0 and v["hold_ok"])
    F.mkdir(parents=True, exist_ok=True)
    (F / "summary.json").write_text(json.dumps(out, indent=1))
    for k in fin:
        v = out["candidates"][k]
        print(f"{k:28s} J3 {v['J3']:.3f} " + " ".join(f"{c.split('|')[0]}{'' if SERVO in c else ('/kp0.5' if 'kp0.5_' in c else '/fr0.12')}"
                                                     f":{x:.3f}" for c, x in v["J"].items())
              + f"  hold3@0.5 {v['hold3_mu0.5']}/{v['n_hold_mu0.5']}"
              + (f"  dJ3 {v['dJ3_vs_base']:+.3f} [{v['dJ3_ci90'][0]:+.3f}, {v['dJ3_ci90'][1]:+.3f}] beats {v['beats_base']}"
                 if "dJ3_ci90" in v else "  (baseline)"))
    return out


def cost():
    """Step 3: per tip the CPU step at the servo plant's three-finger grip (pad_usability_timing, U1's protocol; cells
    16 and 25 mm, state 'grip', pads and point contact), its contacts and constraint rows, the 64-plan replanning time on
    1 and 3 threads and the plans reset; the GPU world-step at the held RL state (2048 worlds) with its contacts, rows,
    broadphase candidates and GPU memory (cost/summary.json)."""
    Cd = D / "cost"
    out = {"tips": {}}
    gx = {r["variant"]: r for r in rows(Cd / "gpu_export.jsonl") if r.get("status") == "ok"}
    gt = [r for r in rows(Cd / "gpu_timing.jsonl") if r.get("status") == "ok" and r.get("nworld") == 2048]
    for f in sorted(Cd.glob("cpu_*.jsonl")):
        tip = f.stem[4:]
        rs = [r for r in rows(f) if r.get("status") == "ok"]
        t = {"cpu": {}, "gpu": {}}
        for cell in ("cylinder:16:0.625", "cylinder:25:0.125"):
            for mdl in ("pads", "pt"):
                st = _pick(rs, cell=cell, model=mdl, state="grip", kind="step") or {}
                hl = _pick(rs, cell=cell, model=mdl, state="grip", kind="held") or {}
                ro = {n: (_pick(rs, cell=cell, model=mdl, state="grip", kind="rollout", nthread=n) or {}) for n in (1, 3)}
                t["cpu"][f"{cell}|{mdl}"] = {
                    "step_us": (st.get("step_us") or {}).get("median"), "solve_us": (st.get("constraint_us") or {}).get("median"),
                    "ncon": st.get("ncon_mean"), "nefc": st.get("nefc_mean"), "held": hl.get("held"),
                    "solver_niter": hl.get("solver_niter_mean"), "obj_disp_mm": hl.get("obj_disp_mm"),
                    "F_fingers_N": st.get("F_fingers_N"),
                    "replan_1thread_s": (ro[1].get("wall_s") or {}).get("median"),
                    "replan_3thread_s": (ro[3].get("wall_s") or {}).get("median"), "plans_reset": ro[3].get("plans_reset"),
                    "n_pad_geoms": st.get("n_pad_geoms")}
        g = [r for r in gt if r["variant"] == tip]
        if g:
            t["gpu"] = {"us_per_world_step": float(np.median([r["us_per_world_step"] for r in g])),
                        "contacts_per_world": g[0].get("contacts_per_world"), "nefc_world": g[0].get("nefc_world_mean"),
                        "candidates_per_world": g[0].get("candidates_per_world"), "gpu_peak_mb": g[0].get("gpu_proc_peak_mb"),
                        "held_frac": g[0].get("held_frac"), "n_reps": len(g),
                        "export_held_worlds": (gx.get(tip) or {}).get("held_worlds"),
                        "export_ngeom": (gx.get(tip) or {}).get("ngeom")}
        out["tips"][tip] = t
    base = out["tips"].get(ASBUILT)
    for tip, t in out["tips"].items():
        if base:
            t["ratios"] = {k: _ratio(t["cpu"][k]["step_us"], base["cpu"][k]["step_us"]) for k in t["cpu"]}
            t["ratios"]["replan_3thread_16"] = _ratio(t["cpu"]["cylinder:16:0.625|pads"]["replan_3thread_s"],
                                                      base["cpu"]["cylinder:16:0.625|pads"]["replan_3thread_s"])
            t["ratios"]["replan_3thread_25"] = _ratio(t["cpu"]["cylinder:25:0.125|pads"]["replan_3thread_s"],
                                                      base["cpu"]["cylinder:25:0.125|pads"]["replan_3thread_s"])
            t["ratios"]["gpu"] = _ratio(t["gpu"].get("us_per_world_step"), base["gpu"].get("us_per_world_step"))
    (Cd / "summary.json").write_text(json.dumps(out, indent=1))
    f = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"  # noqa: E731
    print(f"{'tip':28s} {'16mm us':>8s} {'ncon':>5s} {'nefc':>5s} {'niter':>5s} {'disp':>5s} {'replan3':>7s} | {'25mm us':>8s} "
          f"{'ncon':>5s} {'replan3':>7s} | {'pt16 us':>7s} | {'gpu us':>6s} {'con':>5s} {'cand':>5s} {'MB':>6s}")
    for tip, t in out["tips"].items():
        a, b, c = t["cpu"]["cylinder:16:0.625|pads"], t["cpu"]["cylinder:25:0.125|pads"], t["cpu"]["cylinder:16:0.625|pt"]
        g = t["gpu"]
        print(f"{tip:28s} {f(a['step_us']):>8s} {f(a['ncon'], 0):>5s} {f(a['nefc'], 0):>5s} {f(a['solver_niter'], 2):>5s} "
              f"{f(a['obj_disp_mm'], 2):>5s} {f(a['replan_3thread_s'], 2):>7s} | {f(b['step_us']):>8s} {f(b['ncon'], 0):>5s} "
              f"{f(b['replan_3thread_s'], 2):>7s} | {f(c['step_us']):>7s} | {f(g.get('us_per_world_step'), 2):>6s} "
              f"{f(g.get('contacts_per_world'), 0):>5s} {f(g.get('candidates_per_world'), 0):>5s} {f(g.get('gpu_peak_mb'), 0):>6s}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["bed", "screen", "stage3", "cost"])
    a = ap.parse_args()
    {"bed": bed, "screen": screen, "stage3": stage3, "cost": cost}[a.what]()


if __name__ == "__main__":
    main()
