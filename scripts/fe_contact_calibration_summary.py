#!/usr/bin/env python3
"""Summary of the finite-element calibration (docs/experiments/20261018-fe_contact_calibration/): finite elements, fits,
bed, the iCEM command replays (D1) and the D1-D8 replays (D2), with the plan's criteria applied per tip and calibration.

Pairs of runs (pads against Drake at one calibration, one plan and seed; or one model's seeds 1 and 2 against seed 0):
  held agreement   both keep the tool or both drop it
  same branch      both keep the tool and end within 15 deg of each other (cylinder replays: the angle between the two
                   tools' axes at the end; D1-D8 replays: the difference of the turns, see pair())
  angle_first_deg  the angle between the two tools at the first common sample (cylinder replays: 20 ms after the grip)
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


def axis_of(q):
    """The body z axis (a cylinder's marked axis) of unit quaternions (n, 4), w first."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], 1)


def pair(A, B, turn_key="turn_end_deg", held_key="held", axis=False):
    """axis: compare the tools' axes (a cylinder is symmetric about its own axis, so its spin about it is no part of the
    turn); otherwise the full orientations (the D1-D8 screwdriver).

    Same branch (plan.json terms): both runs keep the tool and end within 15 deg of each other. On the cylinder replays
    (axis) that is the angle between the two tools' axes at the last common sample: each replay measures its turn from
    its own tool pose after the 0.8 s grip, and two simulators can leave the grip with the axes apart (angle_first_deg,
    the first sample, 20 ms after the grip), so a difference of turns there compares angles from different references.
    On the D1-D8 replays the traces start 20 ms into the grip and both turns are measured from the shaft's tilt at the
    grip (the readiness protocol), so the difference of the turns is kept."""
    ta, qa, Na, Ta = tr_arrays(A["trace"])
    tb, qb, Nb, Tb = tr_arrays(B["trace"])
    n = min(len(qa), len(qb))
    if n == 0:
        return None
    if axis:
        ang = np.degrees(np.arccos(np.clip(np.sum(axis_of(qa[:n]) * axis_of(qb[:n]), 1), -1, 1)))
    else:
        ang = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(qa[:n] * qb[:n], 1)), -1, 1)))
    sep = np.where(ang > 10)[0]
    ha, hb = bool(A[held_key]), bool(B[held_key])
    if axis:
        same = ha and hb and float(ang[n - 1]) <= 15.0
    else:
        same = ha and hb and abs(A[turn_key] - B[turn_key]) <= 15.0
    return {"held": [ha, hb], "turn_end_deg": [A[turn_key], B[turn_key]], "same_branch": same,
            "rms_angle_deg": rms(ang), "angle_first_deg": float(ang[0]), "angle_end_deg": float(ang[n - 1]),
            "t_first_s": float(ta[0]), "t_apart_10deg_s": float(ta[sep[0]]) if len(sep) else None,
            "rms_dN_N": rms(Na[:n] - Nb[:n]), "rms_dT_N": rms(Ta[:n] - Tb[:n]), "n": int(n)}


def summarize(pairs):
    pairs = [p for p in pairs if p]
    same = [p for p in pairs if p["same_branch"]]
    rng = lambda k: [min(p[k] for p in pairs), max(p[k] for p in pairs)] if pairs else None  # noqa: E731
    return {"n": len(pairs), "held_agree": sum(p["held"][0] == p["held"][1] for p in pairs),
            "same_branch": len(same),
            "median_rms_angle_same_deg": med([p["rms_angle_deg"] for p in same]),
            "median_rms_angle_all_deg": med([p["rms_angle_deg"] for p in pairs]),
            "median_rms_dN_same_N": med([p["rms_dN_N"] for p in same]),
            "median_rms_dT_same_N": med([p["rms_dT_N"] for p in same]),
            "angle_first_range_deg": rng("angle_first_deg"), "angle_end_range_deg": rng("angle_end_deg")}


# ------------------------------------------------------------------------------------------ D1: iCEM command replays

def d1(seeds=(0, 1, 2)):
    """Pads against Drake on the logged iCEM commands, per tip, plan family (the readiness plans or the tip's own M2
    plans) and calibration; 'pads' is rule R4's pad model and 'pads1' the 1 mm pads. `seeds`: the placements compared (the
    plan's criteria count 3 per plan; more seeds, as on 2026-10-20, report the same statistics over more pairs)."""
    R = {}
    for r in rows(D / "replays" / "icem.jsonl"):
        if r.get("status") == "ok":
            R[(r["plan"], r["calib"], r["plant"], int(r["seed"]))] = r
    tip_of = {k[0]: r["tip"] for k, r in R.items()}
    out = {}
    for tip in TIPS:
        for own in (False, True):
            plans = sorted({p for p in tip_of if tip_of[p] == tip and p.startswith("own_") == own})
            for calib in ("U", "M1", "M2", "M3"):
                for padk in ("pads", "pads1", "pads_ir1000"):
                    pd, sp, runs = [], [], []
                    for p in plans:
                        for s in seeds:
                            A, B = R.get((p, calib, padk, s)), R.get((p, calib, "drake", s))
                            for plant, X in ((padk, A), ("drake", B)):
                                if X:
                                    runs.append({"plan": p, "seed": s, "plant": plant, "held": X["held"],
                                                 "turn": X["turn_end_deg"], "unstable": X.get("unstable", False)})
                            if A and B:
                                x = pair(A, B, axis=True)
                                if x:
                                    x.update(plan=p, seed=s)
                                    pd.append(x)
                        for s in seeds[1:]:
                            A0, A1 = R.get((p, calib, padk, seeds[0])), R.get((p, calib, padk, s))
                            if A0 and A1:
                                sp.append(pair(A0, A1, axis=True))
                    if not pd:
                        continue
                    s_pd, s_sp = summarize(pd), summarize(sp)
                    bound = max(3.0, s_sp["median_rms_angle_same_deg"] or 0.0)
                    n = s_pd["n"]
                    need = n if own else n - 1                  # 3 of 3 (own plans), 5 of 6 (readiness plans)
                    ok_angle = (s_pd["median_rms_angle_same_deg"] is not None
                                and s_pd["median_rms_angle_same_deg"] <= bound)
                    ok_force = own or ((s_pd["median_rms_dN_same_N"] or 9) <= 0.25 and (s_pd["median_rms_dT_same_N"] or 9) <= 0.35)
                    out[f"{tip}|{'own' if own else 'ready'}|{calib}|{padk}"] = {
                        "tip": tip, "own": own, "calib": calib, "pads": padk, "pads_vs_drake": s_pd, "pads_seed_spread": s_sp,
                        "angle_bound_deg": bound, "complete": n == len(seeds) * (1 if own else 2),
                        "pass": bool(s_pd["held_agree"] >= need and ok_angle and ok_force),
                        "pass_parts": {"held": s_pd["held_agree"] >= need, "angle": ok_angle, "force": ok_force},
                        "pairs": pd, "runs": runs,
                        "held_pads": sum(1 for r in runs if r["plant"] == padk and r["held"]),
                        "held_drake": sum(1 for r in runs if r["plant"] == "drake" and r["held"])}
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


def model_at(cv, tip, model, probe, E, F):
    """(approach um, r_bar mm) of a phase-B curve at modulus E (Pa) and load F (N)."""
    r = cv.get((tip, model, probe))
    if r is None:
        return None, None
    d, Fg = np.array(r["delta_um"]) * 1e-6, np.array(r["F_N"]) * E / r["E0_Pa"]
    ok = Fg > 0
    lf, ld = np.log(Fg[ok]), np.log(d[ok])
    keep = np.r_[True, np.diff(lf) > 0]
    dd = float(np.exp(np.interp(math.log(F), lf[keep], ld[keep])))
    rb = np.nan_to_num(np.array(r["r_bar_mm"], float))[ok][keep]
    return dd * 1e6, float(np.interp(math.log(dd), np.log(d[ok][keep]), rb))


def bed_checks(bed, fits, fe):
    """Per tip and calibration: the simulators' bed static approach against the phase-B curve at the same E
    (implementation, within 5 %), the bed onset arm against the finite elements' r_bar under the 12.5 mm rod
    (0.70-1.15 at 1 and 3 N), and the pads at the RL numerics against the planner numerics (within 10 %)."""
    cv = {(r["tip"], r["model"], r["probe"]): r for r in rows(D / "fit" / "curves.jsonl") if r["probe"] == "cx12.5_z0"}
    out = {}
    for k, e in bed.items():
        tip, calib, arm = e["tip"], e["calib"], e["arm"]
        t = fits.get(tip, {})
        if arm == "drake":
            model = "drake"
        else:
            model = (t.get("pad_model") or {}).get(calib, "v2")
        E = e["E_MPa"] * 1e6
        app = {}
        # the bed counts the approach from the design reach; the model's own surface (the pads' envelope, the hull)
        # first touches the tool after first_touch_shift_um more travel, which the curve leaves out
        shift = (cv.get((tip, model, "cx12.5_z0")) or {}).get("first_touch_shift_um", 0.0)
        for N, a in (e.get("approach_um") or {}).items():
            c = model_at(cv, tip, model, "cx12.5_z0", E, float(N))[0]
            app[N] = {"bed_um": a, "first_touch_shift_um": shift, "curve_um": c,
                      "err": ((a - shift) / c - 1) if c else None}
        arms = {}
        for N, a in (e.get("onset_arm_mm") or {}).items():
            fr = (fe.get(f"{tip}|base|cx12.5_z0") or {}).get("per_force", {}).get(N, {}).get("r_bar_mm")
            cr = model_at(cv, tip, model, "cx12.5_z0", E, float(N))[1]
            arms[N] = {"bed_mm": a, "fe_r_bar_mm": fr, "ratio": (a / fr) if (a and fr) else None, "curve_r_bar_mm": cr,
                       "bed_over_curve": (a / cr) if (a and cr) else None}
        errs = [abs(v["err"]) for v in app.values() if v["err"] is not None]
        rats = [v["ratio"] for v in arms.values() if v["ratio"] is not None]
        out[k] = {"tip": tip, "calib": calib, "arm": arm, "model": model, "E_MPa": e["E_MPa"], "approach": app,
                  "onset": arms, "creep_deg_s": e.get("creep_deg_s"), "spin_arm_mm": e.get("spin_arm_mm"),
                  "impl_ok": bool(errs) and max(errs) <= 0.05, "impl_max_err": max(errs) if errs else None,
                  "arm_ok": bool(rats) and min(rats) >= 0.70 and max(rats) <= 1.15,
                  "arm_ratio_range": [min(rats), max(rats)] if rats else None}
    for k, v in list(out.items()):
        if v["arm"] == "rl":
            p = out.get(f"{v['tip']}|{v['calib']}|pads")
            if p:
                da = [abs(v["approach"][N]["bed_um"] / p["approach"][N]["bed_um"] - 1) for N in v["approach"] if N in p["approach"]]
                do = [abs(v["onset"][N]["bed_mm"] / p["onset"][N]["bed_mm"] - 1) for N in v["onset"]
                      if N in p["onset"] and v["onset"][N]["bed_mm"] and p["onset"][N]["bed_mm"]]
                v["rl_vs_planner"] = {"approach_max": max(da) if da else None, "onset_max": max(do) if do else None,
                                      "ok": bool(da and do and max(da) <= 0.10 and max(do) <= 0.10)}
    return out


def verdicts(fits, bchk, d1s, d2s):
    """Per tip, against the plan's criteria: approach and twist arm (phase-B curves, both simulators), the bed checks,
    consistency on the iCEM commands and the spring check."""
    out = {}
    for tip in TIPS:
        t = fits.get(tip)
        if not t:
            continue
        v = {"calibrations": {}}
        for c in ("M1", "M3"):
            cc = (t.get("calib_of") or {}).get(c, c) if c == "M1" else c
            pm = (t.get("pad_model") or {}).get(c, "v2")
            dr, pa = t["models"]["drake"].get(cc), t["models"].get(pm, {}).get(cc)
            if not dr or not pa:
                continue
            traj = "M1" if c == "M1" else "M2"
            bd = bchk.get(f"{tip}|{traj}|drake", {})
            bp = bchk.get(f"{tip}|{traj}|pads", {})
            d1 = d1s.get(f"{tip}|ready|{traj}|pads")
            d1o = d1s.get(f"{tip}|own|M2|pads") if c == "M3" else None
            # M3 replayed with the series spring in both simulators (2026-10-19): its own rows decide consistency and M2
            # no longer stands in for it, so the spring check is reported but not required
            m3_direct = c == "M3" and bool((d1s.get(f"{tip}|ready|M3|pads") or {}).get("complete"))
            if m3_direct:
                d1 = d1s.get(f"{tip}|ready|M3|pads")
                d1o = d1s.get(f"{tip}|own|M3|pads") or d1o
            spring = (d2s.get(f"{tip}|M2") or {}).get("pairs", {}).get("plan_vs_spring") if c == "M3" else None
            spring_ok = None
            if spring and spring.get("median_rms_angle_same_deg") is not None:
                spring_ok = spring["median_rms_angle_same_deg"] <= 1.0 and (spring["median_rms_dN_same_N"] or 9) <= 0.05
            def both(curve_ok, bed_ok):
                return None if bed_ok is None else bool(curve_ok and bed_ok)
            parts = {"approach_drake": both(dr["pass_approach"], bd.get("impl_ok")),
                     "approach_pads": both(pa["pass_approach"], bp.get("impl_ok")),
                     "arm_drake": both(dr["pass_arm"], bd.get("arm_ok")), "arm_pads": both(pa["pass_arm"], bp.get("arm_ok")),
                     "approach_curve_drake": dr["pass_approach"], "approach_curve_pads": pa["pass_approach"],
                     "arm_curve_drake": dr["pass_arm"], "arm_curve_pads": pa["pass_arm"],
                     "bed_impl_drake": bd.get("impl_ok"), "bed_impl_pads": bp.get("impl_ok"),
                     "bed_arm_drake": bd.get("arm_ok"), "bed_arm_pads": bp.get("arm_ok"),
                     "consistency_ready": (d1 or {}).get("pass"), "consistency_own": (d1o or {}).get("pass") if c == "M3" else None,
                     "spring_check": spring_ok}
            need = ["approach_drake", "approach_pads", "arm_drake", "arm_pads", "consistency_ready"]
            if c == "M3":
                need += ["consistency_own"] + ([] if m3_direct else ["spring_check"])
            ok = all(parts.get(k) is True for k in need)
            v["calibrations"][c] = {"pad_model": pm, "parts": parts, "required": need, "pass": ok,
                                    "consistency_rows": ("M3 with the spring in both simulators" if m3_direct else
                                                         ("M2 standing in" if c == "M3" else c)),
                                    "failed": [k for k in need if parts.get(k) is not True],
                                    "drake": {k: dr[k] for k in ("E_MPa", "k_s_N_per_mm", "approach_rms_pct", "approach_max_pct",
                                                                 "arm_min", "arm_max")},
                                    "pads": {k: pa[k] for k in ("E_MPa", "k_s_N_per_mm", "approach_rms_pct", "approach_max_pct",
                                                                "arm_min", "arm_max")}}
        v["modelled_well"] = any(x["pass"] for x in v["calibrations"].values())
        out[tip] = v
    return out


def main():
    fe = [r for r in rows(D / "fe" / "fe.jsonl")]
    meta = rows(D / "fe" / "meta.jsonl")
    fits = json.loads((D / "fit" / "fits.json").read_text()) if (D / "fit" / "fits.json").exists() else {}
    out = {"fe": {f"{r['tip']}|{r['grid']}|{r['probe']}": {k: r[k] for k in ("tip", "grid", "probe", "kind", "R_mm", "y_mm",
                                                                            "z_mm", "per_force", "n_dof")} for r in fe},
           "fe_meta": meta, "fits": fits.get("tips", {}), "bed": bed(), "d1": d1(), "d2": d2(),
           "script": SCRIPT, "when": time.strftime("%Y-%m-%d %H:%M")}
    out["bed_checks"] = bed_checks(out["bed"], out["fits"], out["fe"])
    out["verdicts"] = verdicts(out["fits"], out["bed_checks"], out["d1"], out["d2"])
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
    for k, v in out["bed_checks"].items():
        print("bed", k, "impl", v["impl_ok"], v["impl_max_err"], "arm", v["arm_ok"], v["arm_ratio_range"], v.get("rl_vs_planner"))
    for tip, v in out["verdicts"].items():
        print("verdict", tip, v["modelled_well"], {c: (x["pass"], x["failed"]) for c, x in v["calibrations"].items()})


if __name__ == "__main__":
    main()
