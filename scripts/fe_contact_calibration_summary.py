#!/usr/bin/env python3
"""Summary of the finite-element calibration (docs/experiments/20261018-fe_contact_calibration/): finite elements, fits,
bed, the iCEM command replays (D1) and the D1-D8 replays (D2), with the plan's criteria applied per tip and calibration.

Pairs of runs (pads against Drake at one calibration, one plan and seed; or one model's seeds 1 and 2 against seed 0):
  held agreement   both keep the tool or both drop it
  same branch      both keep the tool and end within 15 deg of each other
  rms_angle_deg    RMS over the 20 ms samples of the angle between the two tool orientations
  rms_dN_N         RMS over samples and fingers of the difference of each finger's summed normal force on the tool
  rms_dT_N         the same for the magnitude of each finger's friction force on the tool
Per tip and calibration: held agreement over the runs, medians over same-branch pairs, and the pads' seed spread (the
median rms_angle_deg of the same-branch pairs of seeds 1 and 2 against seed 0).

    .venv/bin/python scripts/fe_contact_calibration_summary.py
Out: docs/experiments/20261018-fe_contact_calibration/summary.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/experiments/20261018-fe_contact_calibration"
TIPS = ["block4_g30", "cap_g30", "ell16_g30", "ell11_g30"]
FINGERS = ("thumb", "index", "middle")
SCRIPT = "scripts/fe_contact_calibration_summary.py"


def rows(path):
    out = []
    if Path(path).exists():
        with open(path) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    return out


def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else None


def med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


def tr_arrays(tr):
    t = np.array([e["t"] for e in tr])
    q = np.array([e["q"] for e in tr])
    N = np.array([[e["N"].get(f, 0.0) for f in FINGERS] for e in tr])
    T = np.array([[np.linalg.norm(e["T"].get(f, [0, 0, 0])) for f in FINGERS] for e in tr])
    return t, q, N, T


def pair(A, B, turn_key="turn_end_deg", held_key="held"):
    ta, qa, Na, Ta = tr_arrays(A["trace"])
    tb, qb, Nb, Tb = tr_arrays(B["trace"])
    n = min(len(qa), len(qb))
    if n == 0:
        return None
    ang = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(qa[:n] * qb[:n], 1)), -1, 1)))
    sep = np.where(ang > 10)[0]
    ha, hb = bool(A[held_key]), bool(B[held_key])
    same = ha and hb and abs(A[turn_key] - B[turn_key]) <= 15.0
    return {"held": [ha, hb], "turn_end_deg": [A[turn_key], B[turn_key]], "same_branch": same,
            "rms_angle_deg": rms(ang), "t_apart_10deg_s": float(ta[sep[0]]) if len(sep) else None,
            "rms_dN_N": rms(Na[:n] - Nb[:n]), "rms_dT_N": rms(Ta[:n] - Tb[:n]), "n": int(n)}


def summarize(pairs):
    pairs = [p for p in pairs if p]
    same = [p for p in pairs if p["same_branch"]]
    return {"n": len(pairs), "held_agree": sum(p["held"][0] == p["held"][1] for p in pairs),
            "same_branch": len(same),
            "median_rms_angle_same_deg": med([p["rms_angle_deg"] for p in same]),
            "median_rms_angle_all_deg": med([p["rms_angle_deg"] for p in pairs]),
            "median_rms_dN_same_N": med([p["rms_dN_N"] for p in same]),
            "median_rms_dT_same_N": med([p["rms_dT_N"] for p in same])}


# ------------------------------------------------------------------------------------------ D1: iCEM command replays

def d1():
    R = {}
    for r in rows(D / "replays" / "icem.jsonl"):
        if r.get("status") == "ok":
            R[(r["plan"], r["calib"], r["plant"], int(r["seed"]))] = r
    out = {}
    plans = sorted({k[0] for k in R})
    for tip in TIPS:
        for own in (False, True):
            tp = [p for p in plans if (p.startswith("own_") == own) and R.get(next((k for k in R if k[0] == p), None), {}).get("tip") == tip]
            for calib in ("U", "M1", "M2", "M3"):
                pd, sp, runs = [], [], []
                for p in tp:
                    for s in (0, 1, 2):
                        A, B = R.get((p, calib, "pads", s)), R.get((p, calib, "drake", s))
                        if A:
                            runs.append({"plan": p, "seed": s, "plant": "pads", "held": A["held"], "turn": A["turn_end_deg"]})
                        if B:
                            runs.append({"plan": p, "seed": s, "plant": "drake", "held": B["held"], "turn": B["turn_end_deg"]})
                        if A and B:
                            x = pair(A, B)
                            if x:
                                x.update(plan=p, seed=s)
                                pd.append(x)
                    for s in (1, 2):
                        A0, A1 = R.get((p, calib, "pads", 0)), R.get((p, calib, "pads", s))
                        if A0 and A1:
                            sp.append(pair(A0, A1))
                if not runs:
                    continue
                key = f"{tip}|{'own' if own else 'ready'}|{calib}"
                s_pd, s_sp = summarize(pd), summarize(sp)
                bound = max(3.0, s_sp["median_rms_angle_same_deg"] or 0.0)
                need = 3 if own else 5
                ok = (s_pd["n"] > 0 and s_pd["held_agree"] >= (need if own else min(need, s_pd["n"] - 1 if s_pd["n"] < 6 else need))
                      and (s_pd["median_rms_angle_same_deg"] is None or s_pd["median_rms_angle_same_deg"] <= bound)
                      and (own or ((s_pd["median_rms_dN_same_N"] or 0) <= 0.25 and (s_pd["median_rms_dT_same_N"] or 0) <= 0.35)))
                out[key] = {"tip": tip, "own": own, "calib": calib, "pads_vs_drake": s_pd, "pads_seed_spread": s_sp,
                            "angle_bound_deg": bound, "pass": bool(ok), "pairs": pd, "runs": runs,
                            "held_pads": sum(1 for r in runs if r["plant"] == "pads" and r["held"]),
                            "held_drake": sum(1 for r in runs if r["plant"] == "drake" and r["held"]),
                            "n_pads": sum(1 for r in runs if r["plant"] == "pads"),
                            "n_drake": sum(1 for r in runs if r["plant"] == "drake")}
    return out


# ------------------------------------------------------------------------------------------ D2: D1-D8 replays

def d2():
    R = {}
    for r in rows(D / "replays" / "d1d8.jsonl"):
        if r.get("status") in ("complete", "ejected"):
            R[(r["tip"], r["calib"], r["arm"], r["hand"], int(r["seed"]))] = dict(
                r, held=bool(r.get("held_end")) and r.get("status") == "complete")
    out = {}
    hands = [f"D{k}" for k in range(1, 9)]
    for tip in TIPS:
        for calib in ("U", "M1", "M2"):
            arms = sorted({k[2] for k in R if k[0] == tip and k[1] == calib})
            if not arms:
                continue
            res = {"held": {}, "pairs": {}}
            for arm in arms:
                res["held"][arm] = {s: sum(1 for h in hands if R.get((tip, calib, arm, h, s), {}).get("held"))
                                    for s in (0, 1, 2) if any((tip, calib, arm, h, s) in R for h in hands)}
                res["held"][arm]["n"] = {s: sum(1 for h in hands if (tip, calib, arm, h, s) in R) for s in (0, 1, 2)}
                res["held"][arm]["unstable"] = sum(1 for k, r in R.items() if k[:3] == (tip, calib, arm)
                                                   and r.get("status") == "ejected")
            sp = [pair(R[(tip, calib, "plan", h, 0)], R[(tip, calib, "plan", h, s)], "turn_end_deg", "held")
                  for h in hands for s in (1, 2) if (tip, calib, "plan", h, 0) in R and (tip, calib, "plan", h, s) in R]
            res["pairs"]["plan_seed_spread"] = summarize(sp)
            for other in ("drake", "rl", "ir1000", "pads05", "spring"):
                pp = [pair(R[(tip, calib, "plan", h, s)], R[(tip, calib, other, h, s)], "turn_end_deg", "held")
                      for h in hands for s in (0, 1, 2) if (tip, calib, "plan", h, s) in R and (tip, calib, other, h, s) in R]
                if pp:
                    res["pairs"][f"plan_vs_{other}"] = summarize(pp)
            out[f"{tip}|{calib}"] = res
    return out


# ------------------------------------------------------------------------------------------ bed

def bed():
    out = {}
    for case in ("static", "T2", "spin", "T1"):
        for r in rows(D / "bed" / f"{case}.jsonl"):
            if r.get("status") != "complete":
                continue
            k = f"{r['tip']}|{r['calib']}|{r['arm']}"
            e = out.setdefault(k, {"tip": r["tip"], "calib": r["calib"], "arm": r["arm"], "E_MPa": r["E_Pa"] / 1e6})
            N = f"{r['N']:g}"
            if case == "static":
                e.setdefault("approach_um", {})[N] = r["approach_um"]
                e.setdefault("touching", {})[N] = r.get("touching_per_side")
            elif case == "T2":
                e.setdefault("onset_arm_mm", {})[N] = r.get("rbar_onset_mm")
                e.setdefault("creep_deg_s", {})[N] = r.get("creep_deg_s")
            elif case == "spin":
                e["spin_arm_mm"] = r.get("rbar_spin_mm")
            else:
                e["pull_mu_eff"] = r.get("mu_eff")
                e["pull_creep_mm_s"] = r.get("creep_mm_s")
    return out


def main():
    fe = [r for r in rows(D / "fe" / "fe.jsonl")]
    meta = rows(D / "fe" / "meta.jsonl")
    fits = json.loads((D / "fit" / "fits.json").read_text()) if (D / "fit" / "fits.json").exists() else {}
    out = {"fe": {f"{r['tip']}|{r['grid']}|{r['probe']}": {k: r[k] for k in ("tip", "grid", "probe", "kind", "R_mm", "y_mm",
                                                                            "z_mm", "per_force", "n_dof")} for r in fe},
           "fe_meta": meta, "fits": fits.get("tips", {}), "bed": bed(), "d1": d1(), "d2": d2(),
           "script": SCRIPT, "when": time.strftime("%Y-%m-%d %H:%M")}
    tmp = D / "summary.json.tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, D / "summary.json")
    for k, v in out["d1"].items():
        s = v["pads_vs_drake"]
        print(f"D1 {k:28s} held agree {s['held_agree']}/{s['n']} same {s['same_branch']} angle {s['median_rms_angle_same_deg']} "
              f"dN {s['median_rms_dN_same_N']} dT {s['median_rms_dT_same_N']} spread {v['pads_seed_spread']['median_rms_angle_same_deg']} "
              f"pass {v['pass']}")
    for k, v in out["d2"].items():
        print("D2", k, json.dumps(v["held"]), {a: (p["held_agree"], p["n"], p["median_rms_angle_same_deg"]) for a, p in v["pairs"].items()})


if __name__ == "__main__":
    main()
