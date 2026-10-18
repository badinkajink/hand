#!/usr/bin/env python3
"""Numbers behind the finite-element calibration plan (docs/experiments/20261018-fe_contact_calibration/plan.json),
computed from rows of earlier studies; no new simulation.

  fits     From the 95A 20 % stiffness rows of 2026-10-15 (docs/experiments/20261015-curved_tips/stiffness/fe.jsonl and
           models.jsonl: finite elements and the Drake field, pads v2 and v1 at 10 MPa under a 2 mm-radius sphere at 7
           palm points and the 12.5 mm-radius rod at z 0 and +-4 mm, 0.5, 1 and 3 N), what one modulus per tip leaves:
             rod       E fitted (log-RMS) over the rod's 3 positions x 3 forces; error at the rod and under the sphere
             rod_1N    E fitted at the rod's z 0 at 1 N; error at 0.5 and 3 N
             all       E fitted over every probe and force
             spring    a series spring k_s (N/mm) in front of the model at a local modulus E_loc, both fitted over every
                       probe and force: approach = F / k_s + the model's approach at E_loc
             spring_rod  the same fitted over the rod's probes only
  patch    The contact patch under the rod at z 0 at 1 N: the finite elements' pressure-weighted RMS radius (rows) against
           the Drake field's and the v2 pads' at the rod-fitted modulus, and the field's at the series spring's E_loc.
           Loads each tip's hull: run it through resguard.
  replays  From the 85A readiness replays (docs/experiments/20261016-sim2real_fingertip_readiness/replays/traces.jsonl,
           streamed): pads (planner numerics) against Drake per hand at seed 0 (held outcome, RMS tool-angle difference
           over the run and over the first 2 s, the time the two first differ by 10 deg, per-finger normal and friction
           force differences); the pads' own seed pairs; the same plans at another modulus (block 7.55 against 10 and
           16.4 MPa, curved tips 30 % against 50 % infill); the time the tool starts turning on the block.
  bed      The readiness bed rows (summary.json) of the four 30 % variants and the E x0.5 and x2 arms.

Rows: docs/experiments/20261018-fe_contact_calibration/prior/{fits95,patch95,replays,bed85}.json.

    .venv/bin/python scripts/fe_contact_calibration_prior.py fits
    ~/.claude/bin/resguard.sh run --mem 3G --cpu 100 -- taskset -c 4 env OMP_NUM_THREADS=1 \\
        .venv/bin/python scripts/fe_contact_calibration_prior.py patch
    ~/.claude/bin/resguard.sh run --mem 2G --cpu 100 -- taskset -c 4 .venv/bin/python \\
        scripts/fe_contact_calibration_prior.py replays
    .venv/bin/python scripts/fe_contact_calibration_prior.py bed
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "docs/experiments/20261018-fe_contact_calibration/prior"
STIFF = ROOT / "docs/experiments/20261015-curved_tips/stiffness"
READY = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
SCRIPT = "scripts/fe_contact_calibration_prior.py"
FORCES = (0.5, 1.0, 3.0)
TIPS = ["tcd_rs3_re3", "tcd_cap8.5", "tcd_ell7.4x8.5x16x32", "tcd_qell7.4x8.5x11x22"]
FINGERS = ("thumb", "index", "middle")


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
        fh.flush()
        os.fsync(fh.fileno())
    print(f"wrote {path.relative_to(ROOT)}", flush=True)


# ------------------------------------------------------------------------------------------ fits (95A rows)

def fe_table(fe, tip):
    return {r["probe"]: (r["kind"], {F: r["per_force"][f"{F:g}"]["approach_um"] for F in FORCES})
            for r in fe if r["tip"] == tip and r["kind"] != "patch"}


def curves(mo, tip, model):
    return {r["probe"]: (np.array(r["curve"]["delta_um"]), np.array(r["curve"]["F_N"]))
            for r in mo if r["tip"] == tip and r["model"] == model}


def approach(curve, F, scale):
    """Approach (um) of a model whose force curve was computed at 10 MPa, at modulus scale * 10 MPa and load F."""
    dg, Fg = curve
    Fs = Fg * scale
    ok = Fs > 0
    lf, ld = np.log(Fs[ok]), np.log(dg[ok])
    keep = np.r_[True, np.diff(lf) > 0]
    return float(np.exp(np.interp(math.log(F), lf[keep], ld[keep])))


def log_err(fet, cur, probes, scale, ks=None, forces=FORCES):
    out = []
    for p in probes:
        for F in forces:
            d = approach(cur[p], F, scale) + (1e3 * F / ks if ks else 0.0)
            out.append(math.log(d / fet[p][1][F]))
    return np.array(out)


def pct_rms(e):
    return 100 * float(np.sqrt(np.mean((np.exp(e) - 1) ** 2)))


def pct_mean(e):
    return 100 * float(np.mean(np.exp(e) - 1))


def cmd_fits(a):
    from scipy.optimize import minimize
    fe = [json.loads(l) for l in open(STIFF / "fe.jsonl")]
    mo = [json.loads(l) for l in open(STIFF / "models.jsonl")]
    res = {"source": ["docs/experiments/20261015-curved_tips/stiffness/fe.jsonl",
                      "docs/experiments/20261015-curved_tips/stiffness/models.jsonl"],
           "recipe": "w95_20 (Generic FLEX 95A, solid 30 MPa, gyroid 20 %), one 4.76 x 1.59 mm magnet 5 mm deep",
           "terms": {"E_MPa": "modulus of the model (Drake hydroelastic modulus; pads: E in K_i = E a_i / h_i)",
                     "rms_pct": "RMS of (model approach / finite-element approach - 1) over the probes and forces named",
                     "mean_pct": "mean of the same", "k_s_N_per_mm": "series spring stiffness",
                     "E_loc_MPa": "local modulus of the model behind the series spring"},
           "tips": {}}
    for tip in TIPS:
        fet = fe_table(fe, tip)
        rod = [p for p in fet if fet[p][0] == "cyl"]
        sph = [p for p in fet if fet[p][0] == "sphere"]
        res["tips"][tip] = {}
        for m in ("drake", "v2", "v1"):
            cur = curves(mo, tip, m)

            def fit1(probes, forces=FORCES, x0=0.5):
                f = lambda x: float(np.sum(log_err(fet, cur, probes, math.exp(x[0]), forces=forces) ** 2))
                return math.exp(minimize(f, [x0], method="Nelder-Mead").x[0])

            def fit2(probes):
                f = lambda x: float(np.sum(log_err(fet, cur, probes, math.exp(x[0]), math.exp(x[1])) ** 2))
                best = None
                for x0 in ([1.0, 2.0], [2.0, 2.5], [3.0, 2.0], [4.0, 1.5]):
                    r = minimize(f, x0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-10, maxiter=4000))
                    best = r if best is None or r.fun < best.fun else best
                return math.exp(best.x[0]), math.exp(best.x[1])

            s_rod = fit1(rod)
            s_1n = fit1(["cyl0"], forces=(1.0,))
            s_all = fit1(rod + sph, x0=1.0)
            sc, kc = fit2(rod + sph)
            sd, kd = fit2(rod)
            res["tips"][tip][m] = {
                "rod": {"E_MPa": 10 * s_rod, "rod_rms_pct": pct_rms(log_err(fet, cur, rod, s_rod)),
                        "sphere_rms_pct": pct_rms(log_err(fet, cur, sph, s_rod)),
                        "sphere_mean_pct": pct_mean(log_err(fet, cur, sph, s_rod))},
                "rod_1N": {"E_MPa": 10 * s_1n,
                           "rod_z0_err_pct": {f"{F:g}": 100 * (approach(cur["cyl0"], F, s_1n) / fet["cyl0"][1][F] - 1)
                                              for F in FORCES}},
                "all": {"E_MPa": 10 * s_all, "rod_rms_pct": pct_rms(log_err(fet, cur, rod, s_all)),
                        "sphere_rms_pct": pct_rms(log_err(fet, cur, sph, s_all))},
                "spring": {"E_loc_MPa": 10 * sc, "k_s_N_per_mm": kc,
                           "rod_rms_pct": pct_rms(log_err(fet, cur, rod, sc, kc)),
                           "sphere_rms_pct": pct_rms(log_err(fet, cur, sph, sc, kc)),
                           "max_pct": 100 * float(np.max(np.abs(np.exp(log_err(fet, cur, rod + sph, sc, kc)) - 1)))},
                "spring_rod": {"E_loc_MPa": 10 * sd, "k_s_N_per_mm": kd,
                               "rod_rms_pct": pct_rms(log_err(fet, cur, rod, sd, kd)),
                               "sphere_rms_pct": pct_rms(log_err(fet, cur, sph, sd, kd))},
            }
            v = res["tips"][tip][m]
            print(f"{tip:24s} {m:5s} rod E {v['rod']['E_MPa']:5.1f} MPa rod {v['rod']['rod_rms_pct']:5.1f} % sphere "
                  f"{v['rod']['sphere_rms_pct']:5.1f} % | spring k_s {kc:5.1f} N/mm E_loc {10 * sc:6.0f} MPa rod "
                  f"{v['spring']['rod_rms_pct']:4.1f} % sphere {v['spring']['sphere_rms_pct']:4.1f} %", flush=True)
        fexp = {p: math.log(fet[p][1][3.0] / fet[p][1][0.5]) / math.log(6.0) for p in ("apex", "cyl0")}
        res["tips"][tip]["fe"] = {"exponent_sphere_apex": fexp["apex"], "exponent_rod_z0": fexp["cyl0"],
                                  "rod_over_sphere_1N": fet["cyl0"][1][1.0] / fet["apex"][1][1.0]}
    res.update(script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"))
    write("fits95.json", res)


# ------------------------------------------------------------------------------------------ patch (95A rows)

def cmd_patch(a):
    import curved_tips_stiffness as CS
    import tip_codesign_geometry as TG
    fits = json.loads((OUT / "fits95.json").read_text())["tips"]
    fe = {(r["tip"], r["probe"]): r for r in map(json.loads, open(STIFF / "fe.jsonl")) if r["kind"] != "patch"}

    def bisect(force, lo=1e-6, hi=2e-3):
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            lo, hi = (mid, hi) if force(mid) < 1.0 else (lo, mid)
        return math.sqrt(lo * hi)

    def weighted(yz, w):
        c = (w[:, None] * yz).sum(0) / w.sum()
        r = np.linalg.norm(yz - c, axis=1)
        return float(np.sqrt((w * r ** 2).sum() / w.sum())), float((w * r).sum() / w.sum())

    out = {"source": "docs/experiments/20261015-curved_tips/stiffness/fe.jsonl (patch_r_rms_mm, the rod at z 0, 1 N)",
           "terms": {"r_rms_mm": "pressure-weighted RMS distance of the contact pressure from its centre",
                     "r_bar_mm": "pressure-weighted mean distance from the centre: the full-slip twist arm, torque / (mu N)"},
           "tips": {}}
    part = OUT / "patch95.json"
    if part.exists():                                      # resume: keep the tips already written
        out["tips"].update(json.loads(part.read_text()).get("tips", {}))
    for tip in ("tcd_cap8.5", "tcd_ell7.4x8.5x16x32", "tcd_rs3_re3"):
        if tip in out["tips"]:
            continue
        t0 = time.time()
        sh = TG.shape(tip)
        field = CS.NearField(sh)
        pr = CS.Probe(sh, "cyl", 0.0, 0.0)
        field.restrict("cyl", pr.c0 - np.array([CS.R_CYL * 1e-3, 0, 0]), 0.008)
        idx = np.where(field.p(pr.X0 - np.array([CS.DGRID[-1], 0.0, 0.0])) > 0)[0]

        def field_patch(E):
            d = bisect(lambda dd: pr.field_force(field, dd, idx) * E / 1e7)
            X = pr.X0[idx] - np.array([d, 0.0, 0.0])
            w = field.p(X) * pr.dA * (-pr.u[idx, 0])
            m = w > 0
            rr, rb = weighted(pr.X0[idx][m][:, 1:], w[m])
            return {"E_MPa": E / 1e6, "approach_um": d * 1e6, "r_rms_mm": rr * 1e3, "r_bar_mm": rb * 1e3}

        C, Nn, A, K = sh.pads(1e7, h="local", K="each", full=True)
        rs = 0.00075

        def pad_patch(E):
            Ks = K * E / 1e7
            d = bisect(lambda dd: pr.pad_force(C, Ks, rs, dd))
            c0 = pr.c0 - np.array([d, 0.0, 0.0])
            v = np.column_stack([c0[0] - C[:, 0], np.zeros(len(C)), c0[2] - C[:, 2]])
            dist = np.linalg.norm(v, axis=1)
            m = rs + CS.R_CYL * 1e-3 - dist > 0
            f = Ks[m] * (rs + CS.R_CYL * 1e-3 - dist[m]) * v[m, 0] / dist[m]
            P = C[m] + rs * v[m] / dist[m, None]
            rr, rb = weighted(P[:, 1:], f)
            return {"E_MPa": E / 1e6, "approach_um": d * 1e6, "r_rms_mm": rr * 1e3, "r_bar_mm": rb * 1e3,
                    "n_pads": int(m.sum())}

        f1 = fe[(tip, "cyl0")]["per_force"]["1"]
        dk, vk = fits[tip]["drake"], fits[tip]["v2"]
        sp = field_patch(dk["spring"]["E_loc_MPa"] * 1e6)
        sp.update(k_s_N_per_mm=dk["spring"]["k_s_N_per_mm"],
                  total_approach_um=sp["approach_um"] + 1e3 / dk["spring"]["k_s_N_per_mm"])
        row = {"fe": {"approach_um": f1["approach_um"], "r_rms_mm": f1["patch_r_rms_mm"], "n_contact": f1["n_contact"]},
               "drake_rod_fit": field_patch(dk["rod_1N"]["E_MPa"] * 1e6),
               "drake_spring": sp,
               "v2_rod_fit": pad_patch(vk["rod_1N"]["E_MPa"] * 1e6),
               "wall_s": time.time() - t0}
        out["tips"][tip] = row
        print(tip, json.dumps(row, default=lambda x: round(x, 3)), flush=True)
        out.update(script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"))
        write("patch95.json", out)                       # after every tip, so a killed run resumes


# ------------------------------------------------------------------------------------------ replays (85A rows)

def cmd_replays(a):
    keep = {"block4_g30", "block4_g50", "block_e10", "cap_g30", "cap_g50", "ell16_g30", "ell16_g50", "ell11_g30",
            "ell11_g50"}
    R = {}
    with open(READY / "replays/traces.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("variant") not in keep or r.get("status") not in ("complete", "ejected"):
                continue
            tr = r.get("trace") or []
            R[(r["variant"], r["arm"], r["hand"], int(r["seed"]))] = dict(
                held=bool(r["held_end"]), turn=float(r["turn_end_deg"]), t=np.array([e["t"] for e in tr]),
                q=np.array([e["q"] for e in tr]), p=np.array([e["p"] for e in tr]),
                N=np.array([[e["N"].get(f, 0.0) for f in FINGERS] for e in tr]),
                T=np.array([[np.linalg.norm(e["T"].get(f, [0, 0, 0])) for f in FINGERS] for e in tr]))

    def ang(q1, q2):
        n = min(len(q1), len(q2))
        return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(q1[:n] * q2[:n], 1)), -1, 1)))

    def rms(x):
        return float(np.sqrt(np.mean(np.square(x))))

    def pair(A, B):
        a = ang(A["q"], B["q"])
        n = min(len(A["N"]), len(B["N"]))
        sep = np.where(a > 10)[0]
        first2 = A["t"][:len(a)] <= 2.0
        return {"held": [A["held"], B["held"]], "turn_end_deg": [round(A["turn"], 2), round(B["turn"], 2)],
                "same_branch": A["held"] == B["held"] and abs(A["turn"] - B["turn"]) <= 15.0,
                "rms_angle_deg": rms(a), "rms_angle_first_2s_deg": rms(a[first2]),
                "t_apart_10deg_s": float(A["t"][sep[0]]) if len(sep) else None,
                "rms_dN_N": rms(A["N"][:n] - B["N"][:n]), "rms_dT_N": rms(A["T"][:n] - B["T"][:n]),
                "rms_dpos_mm": rms(np.linalg.norm(A["p"][:n] - B["p"][:n], axis=1))}

    def summary(pairs):
        same = [x for x in pairs if x["same_branch"] and all(x["held"])]
        return {"n": len(pairs), "held_agree": sum(1 for x in pairs if x["held"][0] == x["held"][1]),
                "both_held_same_branch": len(same),
                "median_rms_angle_all_deg": float(np.median([x["rms_angle_deg"] for x in pairs])) if pairs else None,
                "median_rms_angle_same_branch_deg": float(np.median([x["rms_angle_deg"] for x in same])) if same else None,
                "median_rms_dN_same_branch_N": float(np.median([x["rms_dN_N"] for x in same])) if same else None,
                "median_rms_dT_same_branch_N": float(np.median([x["rms_dT_N"] for x in same])) if same else None,
                "median_rms_dpos_same_branch_mm": float(np.median([x["rms_dpos_mm"] for x in same])) if same else None}

    out = {"source": "docs/experiments/20261016-sim2real_fingertip_readiness/replays/traces.jsonl",
           "terms": {"rms_angle_deg": "RMS over the run (20 ms samples) of the angle between the two tool orientations",
                     "same_branch": "both runs held or both dropped, final turns within 15 deg",
                     "t_apart_10deg_s": "first time the two tool orientations differ by more than 10 deg",
                     "rms_dN_N": "RMS difference of each finger's summed normal force on the tool",
                     "rms_dT_N": "RMS difference of each finger's friction-force magnitude on the tool",
                     "held": "the replay's held_end: tool not dropped and at least two fingers loaded at the end"},
           "pads_vs_drake": {}, "pads_seed_pairs": {}, "modulus_pairs": {}}
    for v in sorted(keep):
        hands = sorted({k[2] for k in R if k[0] == v})
        pd = {h: pair(R[(v, "plan", h, 0)], R[(v, "drake", h, 0)]) for h in hands
              if (v, "plan", h, 0) in R and (v, "drake", h, 0) in R}
        if pd:
            out["pads_vs_drake"][v] = {"per_hand": pd, "summary": summary(list(pd.values())),
                                       "drake_held_all_pad_seeds_dropped": [
                                           h for h in hands if pd.get(h) and R[(v, "drake", h, 0)]["held"]
                                           and all((v, "plan", h, s) in R and not R[(v, "plan", h, s)]["held"]
                                                   for s in (0, 1, 2))]}
        sp = [pair(R[(v, "plan", h, 0)], R[(v, "plan", h, s)]) for h in hands for s in (1, 2)
              if (v, "plan", h, 0) in R and (v, "plan", h, s) in R]
        out["pads_seed_pairs"][v] = summary(sp)
    for va, vb, why in (("block4_g30", "block_e10", "modulus 7.55 against 10 MPa (+32 %); tip 3.73 against 6.33 g; board "
                                                    "against none"),
                        ("block4_g30", "block4_g50", "modulus 7.55 against 16.4 MPa (x2.17); tip 3.73 against 4.49 g"),
                        ("cap_g30", "cap_g50", "modulus 13.4 against 29.2 MPa (x2.17)"),
                        ("ell16_g30", "ell16_g50", "modulus 15.0 against 32.6 MPa (x2.17)")):
        pp = [pair(R[(va, "plan", h, s)], R[(vb, "plan", h, s)]) for (v, arm, h, s) in R
              if v == va and arm == "plan" and (vb, "plan", h, s) in R]
        out["modulus_pairs"][f"{va}|{vb}"] = dict(summary(pp), change=why)
    onset = {}
    for (v, arm, h, s), A in R.items():
        if v == "block4_g30" and arm == "plan" and s == 0:
            i0 = int(np.searchsorted(A["t"], 1.5))
            a = np.degrees(2 * np.arccos(np.clip(np.abs(A["q"] @ A["q"][i0]), -1, 1)))
            k5 = np.where((a > 5) & (A["t"] > 1.5))[0]
            onset[h] = float(A["t"][k5[0]]) if len(k5) else None
    out["block_turn_onset_5deg_s"] = onset
    for v, x in out["pads_vs_drake"].items():
        print(v, json.dumps(x["summary"], default=lambda z: round(z, 3)), "Drake held, pads dropped every seed:",
              x["drake_held_all_pad_seeds_dropped"], flush=True)
    for k, x in out["modulus_pairs"].items():
        print(k, json.dumps(x, default=lambda z: round(z, 3) if isinstance(z, float) else z), flush=True)
    out.update(script=SCRIPT, when=time.strftime("%Y-%m-%d %H:%M"))
    write("replays.json", out)


# ------------------------------------------------------------------------------------------ bed (85A rows)

def cmd_bed(a):
    s = json.loads((READY / "summary.json").read_text())
    rows = {k: v for k, v in s["bed"].items() if v["variant"] in ("block4_g30", "cap_g30", "ell16_g30", "ell11_g30")}
    fe = [r for r in s["fe"] if r["layout"] == "four_d3x2_d5" and r["recipe"] == "w85_30"]
    write("bed85.json", {"source": "docs/experiments/20261016-sim2real_fingertip_readiness/summary.json (bed, fe)",
                         "block_fe_w85_30": fe[0] if fe else None, "bed": rows, "script": SCRIPT,
                         "when": time.strftime("%Y-%m-%d %H:%M")})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fits", "patch", "replays", "bed"])
    a = ap.parse_args()
    {"fits": cmd_fits, "patch": cmd_patch, "replays": cmd_replays, "bed": cmd_bed}[a.cmd](a)


if __name__ == "__main__":
    main()
