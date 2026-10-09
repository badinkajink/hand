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
# contact gaps from the STL surface (tpu_tip_fem_tip.py run --surface, tpu_tip_fem_gpu.py --surface): the rows the page
# uses; the voxel-top and grid-node gaps of the first runs stay in tip_runs.jsonl, data/ and the GPU tags without _surf
TIP_ROWS, TIP_DATA, SUFFIX = "tip_runs_surf.jsonl", DIR / "data/surf", "_surf"


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


def recipe_table(rows_file=None, data_dir=None):
    rows_file, data_dir = rows_file or TIP_ROWS, Path(data_dir or TIP_DATA)
    R = rows(rows_file)
    out = {}
    for name in dict.fromkeys(r["recipe"] for r in R):
        rr = [r for r in R if r["recipe"] == name]
        d_path = data_dir / f"{name}.npz"
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


def bed_table(pull_ok=0.9):
    """Bed rows of the pads per modulus {E: {(task, N): row}}; moduli whose pads fail the pull test (T1 mu_eff below
    pull_ok at any load) are dropped."""
    out = {}
    for t in ("T1", "T2"):
        for r in rows(f"bed/tip_{t}.jsonl"):
            E = float(r["model"].split("_E")[-1].replace("p", "."))
            out.setdefault(E, {})[(t, r["N"])] = r
    return {E: v for E, v in out.items()
            if all(r.get("mu_eff", 1.0) >= pull_ok for (t, _), r in v.items() if t == "T1")}


def bed_failed(pull_ok=0.9):
    """Moduli whose pads fail the pull test, with their mu_eff per load."""
    out = {}
    for r in rows("bed/tip_T1.jsonl"):
        E = float(r["model"].split("_E")[-1].replace("p", "."))
        out.setdefault(E, {})[r["N"]] = r.get("mu_eff")
    return {E: v for E, v in out.items() if min(v.values()) < pull_ok}


def pads_arm_at(B, E, N=3.0):
    """Pads' T2 onset arm at load N, interpolated in log E between the bed's moduli (None outside their range)."""
    pts = sorted((e, v[("T2", N)]["rbar_onset_mm"]) for e, v in B.items()
                 if ("T2", N) in v and v[("T2", N)].get("rbar_onset_mm") is not None)
    if not pts or not pts[0][0] <= E <= pts[-1][0]:
        return None
    return float(np.interp(math.log(E), [math.log(e) for e, _ in pts], [a for _, a in pts]))


NOMINAL = "g20_across"   # homogenized face of the block: nodes, areas and gaps are the same for every recipe


def law_unit():
    """The pressure law at unit modulus on the nominal block face: {indenter: (F/E grid, delta grid)} for the cylinder
    and the plate (the law's approach depends on F/E only)."""
    D = np.load(TIP_DATA / f"{NOMINAL}.npz")
    out = {}
    dl = np.geomspace(1e-5, 3.0, 600)
    for ind in ("cyl", "plate"):
        A, g0 = D[f"{ind}_1__A"], D[f"{ind}_1__g0"]
        out[ind] = (np.array([law_f(A, g0, d) for d in dl]), dl)
    return out


def _at(F, y, target, extrap=0.1):
    """y at force target by log-log interpolation over the rows (F ascending); None beyond extrap past the ends."""
    F, y = np.asarray(F, float), np.asarray(y, float)
    if target < F[0] * (1 - extrap) or target > F[-1] * (1 + extrap) or len(F) < 2:
        return None
    lf, ly = np.log(F), np.log(y)
    if target > F[-1]:
        s = (ly[-1] - ly[-2]) / (lf[-1] - lf[-2])
        return float(np.exp(ly[-1] + s * (math.log(target) - lf[-1])))
    if target < F[0]:
        s = (ly[1] - ly[0]) / (lf[1] - lf[0])
        return float(np.exp(ly[0] + s * (math.log(target) - lf[0])))
    return float(np.exp(np.interp(math.log(target), lf, ly)))


def resolved_table(suffix=None):
    """Per resolved tip (GPU rows whose tag ends in suffix, keyed without it): stiffness, best single pressure-law
    modulus on the nominal block face (cylinder at 0.5, 1 and 3 N and the plate at 1 N, each where the rows reach it),
    twist arm, edge load, width exponent."""
    suffix = SUFFIX if suffix is None else suffix
    G = rows("gpu_runs.jsonl")
    law = law_unit()
    out = {}
    tags = [t for t in dict.fromkeys(r["tag"] for r in G) if t.endswith(suffix) and (suffix or not t.endswith("_surf"))]
    for tag in tags:
        rec = {}
        for ind in ("cyl", "plate"):
            rr = sorted({r["delta_mm"]: r for r in G if r["tag"] == tag and r["indenter"] == ind}.values(),
                        key=lambda r: r["delta_mm"])
            if len(rr) < 2:
                continue
            F = np.array([r["F"] for r in rr])
            d = np.array([r["delta_mm"] for r in rr])
            n = float(np.polyfit(np.log(d), np.log(F), 1)[0])
            at = {f: _at(F, d, f) for f in FORCES}
            rec[ind] = dict(F=F, delta=d, n=n, delta_at=at, rows=rr)
        if "cyl" not in rec:
            continue
        c = rec["cyl"]
        cases = [("cyl", f, c["delta_at"][f]) for f in FORCES if c["delta_at"][f]]
        if "plate" in rec and rec["plate"]["delta_at"][1.0]:
            cases.append(("plate", 1.0, rec["plate"]["delta_at"][1.0]))
        Es = np.geomspace(0.05, 500, 600)
        err = np.array([[math.log(np.exp(np.interp(math.log(f / E), np.log(law[ind][0][law[ind][0] > 0]),
                                                   np.log(law[ind][1][law[ind][0] > 0]))) / dfe)
                         for ind, f, dfe in cases] for E in Es])
        k = int(np.argmin((err ** 2).sum(1)))
        r1 = dict(cases=[f"{ind} {f:g} N" for ind, f, _ in cases], E_best=float(Es[k]),
                  E_best_errors_pct={f"{ind} {f:g} N": float((math.exp(v) - 1) * 100) for (ind, f, _), v in zip(cases, err[k])},
                  E_best_max_err_pct=float(np.max(np.abs(np.exp(err[k]) - 1)) * 100),
                  force_exponent=c["n"], delta_um={f: (v * 1e3 if v else None) for f, v in c["delta_at"].items()})
        d1 = c["delta_at"][1.0]
        r1["k_1N"] = float(c["n"] / d1) if d1 else None
        r1["k_plate"] = float(1.0 / rec["plate"]["delta_at"][1.0]) if "plate" in rec and rec["plate"]["delta_at"][1.0] else None
        r1["arm"] = {f: _at(c["F"], [r["arm"] for r in c["rows"]], f) for f in FORCES}
        r1["outside_overlap_1N"] = _at(c["F"], [max(r["outside_overlap"], 1e-9) for r in c["rows"]], 1.0)
        path = DIR / "data" / f"gpu_{tag}_cyl.npz"
        if path.exists():
            D = np.load(path)
            X = D["X_face"]
            es, bb, FF = [], [], []
            for r in c["rows"]:
                key = f"p_{r['delta_mm']:g}"
                if key not in D.files:
                    continue
                p = D[key]
                es.append(max(float(p[np.abs(X[:, 1]) > EDGE_Y].sum() / p.sum()), 1e-9))
                cz = (p * X[:, 2]).sum() / p.sum()
                bb.append(float(np.sqrt((p * (X[:, 2] - cz) ** 2).sum() / p.sum())))
                FF.append(r["F"])
            if len(FF) >= 2:
                o = np.argsort(FF)
                FF, es, bb = np.array(FF)[o], np.array(es)[o], np.array(bb)[o]
                r1["edge_share_1N"] = _at(FF, es, 1.0)
                r1["b_rms_mm"] = {f: _at(FF, bb, f) for f in FORCES}
                m = (FF >= 0.3) & (FF <= 4.5)
                if m.sum() >= 2:
                    r1["b_exponent"] = float(np.polyfit(np.log(FF[m]), np.log(bb[m]), 1)[0])
        out[tag[:len(tag) - len(suffix)] if suffix else tag] = r1
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
    for k, r in resolved_table().items():
        print(f"RES {k:12s}: k(1N) {r['k_1N'] or float('nan'):6.1f} N/mm, plate {r['k_plate'] or float('nan'):6.0f} N/mm, "
              f"E_best {r['E_best']:.2f} (max err {r['E_best_max_err_pct']:.0f} % over {r['cases']}), arm1 {r['arm'][1.0] or float('nan'):.2f}, "
              f"edge {r.get('edge_share_1N', float('nan')):.2f}, b exp {r.get('b_exponent', float('nan')):.2f}")
    B = bed_table()
    for E in sorted(B):
        a = {N: B[E].get(("T2", N), {}).get("rbar_onset_mm") for N in (1.0, 3.0)}
        print(f"pads E {E:5.1f} MPa: T2 onset arm 1 N {a[1.0]}, 3 N {a[3.0]}; T1 mu_eff {B[E].get(('T1', 1.0), {}).get('mu_eff')}")
    json.dump(dict(recipes=T, gpu=G, resolved=resolved_table()), open(DIR / "summary.json", "w"), indent=1, default=float)




def stiffness_map_law(name, forces=FORCES):
    """Pads with a per-node Winkler stiffness taken from the finite-element plate solve at 1 N (k_i = p_i / (A_i delta),
    N/mm^3): approach and twist arm under the cylinder against the finite-element ones."""
    path = TIP_DATA / f"{name}.npz"
    if not path.exists():
        return None
    D = np.load(path)
    if "plate_1__p" not in D.files:
        return None
    Xp, pp, Ap, dp = D["plate_1__X"], D["plate_1__p"], D["plate_1__A"], float(D["plate_1__delta"])
    kmap = {(round(x[1], 6), round(x[2], 6)): p / (a * dp) for x, p, a in zip(Xp, pp, Ap)}
    R = {r["F_target"]: r for r in rows(TIP_ROWS) if r["recipe"] == name and r["indenter"] == "cyl"}
    out = {}
    for F in forces:
        X, A_, g0 = D[f"cyl_{F:g}__X"], D[f"cyl_{F:g}__A"], D[f"cyl_{F:g}__g0"]
        k = np.array([kmap.get((round(x[1], 6), round(x[2], 6)), np.nan) for x in X])
        k = np.where(np.isfinite(k), k, np.nanmedian(k))
        lo, hi = 0.0, 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if (k * A_ * np.clip(mid - g0, 0, None)).sum() < F:
                lo = mid
            else:
                hi = mid
        d = 0.5 * (lo + hi)
        p = k * A_ * np.clip(d - g0, 0, None)
        c = (p[:, None] * X[:, 1:]).sum(0) / p.sum()
        arm = float((p * np.linalg.norm(X[:, 1:] - c, axis=1)).sum() / p.sum())
        out[F] = dict(delta_um=d * 1e3, arm=arm, delta_err_pct=(d / R[F]["delta_mm"] - 1) * 100,
                      arm_err_pct=(arm / R[F]["arm"] - 1) * 100)
    return out


EDGE_Y = 3.5      # mm: the outer 1.2 mm of the 9.4 mm flat palmar face, and the fillets beyond it


def edge_share(name, F=1.0):
    """Share of the cylinder's force on face nodes with |y| > EDGE_Y (the face edges over the fillet corners), for the
    finite-element tip and for the pressure law on the same nodes at the same force (uniform modulus)."""
    path = TIP_DATA / f"{name}.npz"
    if not path.exists():
        return None
    D = np.load(path)
    p, X, A_, g0 = D[f"cyl_{F:g}__p"], D[f"cyl_{F:g}__X"], D[f"cyl_{F:g}__A"], D[f"cyl_{F:g}__g0"]
    fe = float(p[np.abs(X[:, 1]) > EDGE_Y].sum() / p.sum())
    d = law_delta(A_, g0, F, 10.0)
    pl = A_ * np.clip(d - g0, 0, None)
    law = float(pl[np.abs(X[:, 1]) > EDGE_Y].sum() / pl.sum())
    return dict(fe=fe, law=law)


def edge_share_gpu(tag, delta=0.04):
    path = DIR / "data" / f"gpu_{tag}_cyl.npz"
    if not path.exists():
        return None
    D = np.load(path)
    key = f"p_{delta:g}"
    if key not in D.files:
        return None
    p, X = D[key], D["X_face"]
    return float(p[np.abs(X[:, 1]) > EDGE_Y].sum() / p.sum())


if __name__ == "__main__":
    main()
