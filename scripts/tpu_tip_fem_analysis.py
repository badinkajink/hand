#!/usr/bin/env python3
"""Tables for the TPU tip finite-element page (steps 4 and 11): the pressure-law modulus that represents each printed
tip, how well one modulus does over loads and indenters, the spreads with print orientation and contact location, and
the bench prediction.

Pressure law: the pads' Winkler law on the tip block, p = E d / h_f with d the local overlap and h_f = 8.5 mm
(fingertip_geometry.FOUNDATION), integrated on the same face nodes as the finite-element model.

    python3 scripts/tpu_tip_fem_analysis.py            # prints the tables, writes summary.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "docs/experiments/20261009-tpu_tip_fem"
H_F = 8.5
FORCES = (0.5, 1.0, 3.0)


def rows(name):
    p = DIR / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def law_f(A, g0, delta):
    return float((A * np.clip(delta - g0, 0, None)).sum() / H_F)


def law_delta(A, g0, F, E):
    lo, hi = 0.0, 5.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if E * law_f(A, g0, mid) < F:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def recipe_table():
    R = rows("tip_runs.jsonl")
    out = {}
    for name in dict.fromkeys(r["recipe"] for r in R):
        rr = [r for r in R if r["recipe"] == name]
        d_path = DIR / "data" / f"{name}.npz"
        D = np.load(d_path) if d_path.exists() else None
        rec = dict(name=name, E_s=rr[0]["E_s"], core=rr[0]["core"], build=rr[0].get("build"), rho=rr[0].get("rho"))
        cyl = {r["F_target"]: r for r in rr if r["indenter"] == "cyl"}
        pla = {r["F_target"]: r for r in rr if r["indenter"] == "plate"}
        if not all(F in cyl for F in FORCES):
            continue
        Fs = sorted(cyl)
        dl = np.array([cyl[F]["delta_mm"] for F in Fs])
        n_cyl = np.polyfit(np.log(dl), np.log(Fs), 1)[0]
        rec.update(delta_um={F: cyl[F]["delta_mm"] * 1e3 for F in FORCES}, force_exponent=float(n_cyl),
                   k_1N=float(n_cyl * 1.0 / cyl[1.0]["delta_mm"]), E_fit_cyl={F: cyl[F]["E_law_fit_MPa"] for F in FORCES},
                   arm={F: cyl[F]["arm"] for F in FORCES}, outside_overlap_1N=cyl[1.0]["outside_overlap"],
                   presliding_um={F: cyl[F].get("u_presliding_mm", float("nan")) * 1e3 for F in FORCES},
                   loop_uJ={F: cyl[F].get("loop_area_mJ", float("nan")) * 1e3 for F in FORCES})
        if pla:
            rec["E_fit_plate"] = pla[1.0]["E_law_fit_MPa"]
            rec["k_plate"] = 1.0 / pla[1.0]["delta_mm"]
        if D is not None:
            # contact half-width: pressure-weighted RMS distance from the contact line along z
            b = {}
            for F in FORCES:
                p, X = D[f"cyl_{F:g}__p"], D[f"cyl_{F:g}__X"]
                cz = (p * X[:, 2]).sum() / p.sum()
                b[F] = float(np.sqrt((p * (X[:, 2] - cz) ** 2).sum() / p.sum()))
            rec["b_rms_mm"] = b
            rec["b_exponent"] = float(np.polyfit(np.log(FORCES), np.log([b[F] for F in FORCES]), 1)[0])
            # best single E over the cylinder at 0.5/1/3 N and the plate at 1 N: RMS of log approach error
            cases = [("cyl", F) for F in FORCES] + ([("plate", 1.0)] if pla else [])
            Es = np.geomspace(0.5, 500, 400)
            err = []
            for E in Es:
                e = []
                for ind, F in cases:
                    A, g0 = D[f"{ind}_{F:g}__A"], D[f"{ind}_{F:g}__g0"]
                    dfe = (cyl if ind == "cyl" else pla)[F]["delta_mm"]
                    e.append(math.log(law_delta(A, g0, F, E) / dfe))
                err.append(e)
            err = np.array(err)
            k = int(np.argmin((err ** 2).sum(1)))
            rec["E_best"] = float(Es[k])
            rec["E_best_errors_pct"] = {f"{ind} {F:g} N": float((math.exp(v) - 1) * 100) for (ind, F), v in zip(cases, err[k])}
            rec["E_best_max_err_pct"] = float(np.max(np.abs(np.exp(err[k]) - 1)) * 100)
            # coupling length from the face profile along z at 1 N (outside the contact)
            if "profile_cyl_z__c" in D.files:
                c, u = D["profile_cyl_z__c"], D["profile_cyl_z__u"]
                zc = cyl[1.0]["cz"]
                s = np.abs(c - zc)
                m = (s > 3 * b[1.0]) & (s < 3 * b[1.0] + 4.0) & (u > 0)
                if m.sum() > 3:
                    sl = np.polyfit(s[m], np.log(u[m]), 1)[0]
                    rec["coupling_length_mm"] = float(-1 / sl) if sl < 0 else float("inf")
        out[name] = rec
    return out


def gpu_table():
    G = rows("gpu_runs.jsonl")
    out = {}
    for tag in dict.fromkeys(r["tag"] for r in G):
        rr = sorted([r for r in G if r["tag"] == tag and r["indenter"] == "cyl"], key=lambda r: r["delta_mm"])
        if len(rr) < 2:
            continue
        d = np.array([r["delta_mm"] for r in rr])
        F = np.array([r["F"] for r in rr])
        n = np.polyfit(np.log(d), np.log(F), 1)[0]
        at = lambda target: float(np.exp(np.interp(math.log(target), np.log(F), np.log(d))))  # noqa: E731
        out[tag] = dict(delta_um={f: at(f) * 1e3 for f in FORCES if F.min() <= f <= F.max()}, force_exponent=float(n),
                        F_at={float(r["delta_mm"]) * 1e3: r["F"] for r in rr}, arm={float(r["F"]): r["arm"] for r in rr},
                        t_per_delta_s=float(np.mean([r["t_solve_s"] for r in rr])), n_el=rr[0]["n_el"], n_dof=rr[0]["n_dof"],
                        gpu_mem_GB=max(r.get("gpu_mem_GB", 0) for r in rr), cg=int(np.mean([r["cg_iterations"] for r in rr])))
    return out


def bed_table():
    out = {}
    for t in ("T1", "T2"):
        for r in rows(f"bed/tip_{t}.jsonl"):
            E = float(r["model"].split("_E")[-1].replace("p", "."))
            out.setdefault(E, {})[(t, r["N"])] = r
    return out


def main():
    T = recipe_table()
    for k, r in T.items():
        print(f"{k:14s} E_s {r['E_s']:4.0f}: approach {r['delta_um'][0.5]:6.1f}/{r['delta_um'][1.0]:6.1f}/{r['delta_um'][3.0]:6.1f} um, "
              f"F~d^{r['force_exponent']:.2f}, k(1N) {r['k_1N']:6.1f} N/mm, E_fit cyl {r['E_fit_cyl'][0.5]:.1f}/{r['E_fit_cyl'][1.0]:.1f}/"
              f"{r['E_fit_cyl'][3.0]:.1f}, plate {r.get('E_fit_plate', float('nan')):.1f}, E_best {r.get('E_best', float('nan')):.1f} "
              f"(max err {r.get('E_best_max_err_pct', float('nan')):.0f} %), b exp {r.get('b_exponent', float('nan')):.2f}, arm "
              f"{r['arm'][1.0]:.2f}/{r['arm'][3.0]:.2f}, presl {r['presliding_um'][1.0]:.1f} um, l {r.get('coupling_length_mm', float('nan')):.2f} mm")
    G = gpu_table()
    for k, g in G.items():
        print(f"GPU {k:12s}: approach at 0.5/1/3 N {g['delta_um']}, F~d^{g['force_exponent']:.2f}, {g['n_el']} el, "
              f"{g['t_per_delta_s']:.0f} s per approach, CG {g['cg']}")
    B = bed_table()
    for E in sorted(B):
        a = {N: B[E].get(("T2", N), {}).get("rbar_onset_mm") for N in (1.0, 3.0)}
        print(f"pads E {E:5.1f} MPa: T2 onset arm 1 N {a[1.0]}, 3 N {a[3.0]}; T1 mu_eff {B[E].get(('T1', 1.0), {}).get('mu_eff')}")
    json.dump(dict(recipes=T, gpu=G), open(DIR / "summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
