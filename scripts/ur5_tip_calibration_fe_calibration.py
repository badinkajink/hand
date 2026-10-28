#!/usr/bin/env python3
"""The FE pad calibrated to the real tips on the UR5 (owner, 2026-10-28: "rebuild the FE surface basis ... and yea, let's
try to calibrate the FE to the real tips"; "i am pretty sure the magnet positions are pretty locked in in the print").

Two tips with one pad body and one print recipe, different magnets: v2 ID 1 (five N52, FE case b_v2) and v3 ID 1 (four
N52 in pad 05R4, FE case b_05R4, built 2026-10-28 on the same grid and depth). For each, the press holds of the sphere's
grid actions that both ran (matched by id, as scripts/ur5_tip_calibration_four_vs_five.py), and:

  fit     predict.Physical on all the holds up to --max-n newtons: the magnets' strength k and the board's offsets dx, dy,
          dz (priors 5 % and 0.2 mm around the rest fit) and the compliance scale c of the FE basis, free. The FE is
          linear and its TPU one material, so c scales every displacement: E_fit = E_FE / c is the modulus the board's
          change asks for (the FE's 12 MPa). One c for both tips is a property of the print; two are the tips'.
  cv      the same fit under five folds grouped by action, each held-out hold estimated by the fitted model: median
          location error along the surface (mm) and |f_n| error (N and relative), beside the untrained model (the FE
          and the rest fit alone, c = 1).
The four magnets' rest field leaves their strength k and the board's offset dx nearly interchangeable (the free fit put
k at its bound 2.0 and dx at 2.69 mm); v3's rest fit is redone with dx within --dx-tol of v2 ID 1's, the same board mount.
  depth   the registered indentation (depth, mm: the sphere's radius less its distance from the pad's centre of
          curvature) against f_n by region, a line fitted per region; the FE's own give at the contact is the normal
          component of the basis's load-weighted displacement per newton (load_u, um/N). The difference of the slopes is
          the compliance the FE does not have (series and local).

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 500 -- .venv/bin/python scripts/ur5_tip_calibration_fe_calibration.py \\
        V3_HOLDS_DIR [--max-n 3.5] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src/morphohand/driver/reskin5x/host"))
import fingertip_wrench_model as W  # noqa: E402
import ur5_tip_calibration_cb1r_checks as K  # noqa: E402
import ur5_tip_calibration_four_vs_five as FV  # noqa: E402
import ur5_tip_calibration_hard_extrapolation as HX  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402

E_FE_MPA = 12.0
BINS = ((0.0, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 8.0))
REGIONS = ("face", "edge", "flank")


BOARD_CAL = ROOT / "docs/experiments/20261020-fingertip_contact_estimation/data/20261020-calib_pad_as_found.json"


def tip(name, holds_dir, case, dx_range=None):
    """The tip's holds, rest calibration (refitted with the board's offset bounded when dx_range is given) and basis."""
    H = FV.holds(holds_dir)
    cal0 = W.Calib.load(Path(holds_dir) / "calib_rest.json")
    if dx_range is not None:
        free = cal0
        cal0 = W.calibrate_frames_fixed(np.asarray(free.b_rest_uT, float).reshape(5, 3), W.Calib.load(BOARD_CAL), case=case,
                                        poles=free.poles, dx_range=dx_range)
        print(f"{name}: rest fit k {free.k:.3f}, dx {free.dx:.2f} mm (rms {free.rms_mT:.3f} mT) -> with dx in "
              f"{dx_range[0]:.2f}-{dx_range[1]:.2f}: k {cal0.k:.3f}, dx {cal0.dx:.2f}, dy {cal0.dy:.2f}, dz {cal0.dz:.2f} mm "
              f"(rms {cal0.rms_mT:.3f} mT)", flush=True)
    basis = W.load_basis(W.basis_path(case))
    return dict(name=name, H=H, cal0=cal0, basis=basis, case=case)


def loc_fn(H, idx, P, Fn):
    n = H["n"][idx]
    dp = P - H["p"][idx]
    loc = np.linalg.norm(dp - np.sum(dp * n, axis=1)[:, None] * n, axis=1)
    return loc, np.abs(Fn - H["f_n"][idx]), np.abs(Fn - H["f_n"][idx]) / H["f_n"][idx]


def table(H, loc, fe, fr):
    out = {}
    for key, m in [("all", np.ones(len(loc), bool))] + [(f"{lo:g}-{hi:g}", HX.in_bin(H, lo, hi)) for lo, hi in BINS] \
            + [(r, H["region"] == r) for r in REGIONS]:
        ok = m & np.isfinite(loc)
        if m.any():
            out[key] = dict(n=int(m.sum()), detected=float(ok.sum() / m.sum()),
                            loc_mm=float(np.median(loc[ok])) if ok.any() else None,
                            fn_err_N=float(np.median(fe[ok])) if ok.any() else None,
                            fn_rel=float(np.median(fr[ok])) if ok.any() else None)
    return out


def depth_fit(H, basis):
    """Per region: the registered indentation against f_n (slope um/N, intercept mm) and the FE's give at the contact."""
    from scipy.spatial import cKDTree
    tree = cKDTree(basis["pts"])
    _, k = tree.query(H["p"])
    lu = basis["load_u"][k]                                   # (n, 3 dirs, 3) um per N along +x, +y, +z of the tip frame
    n = H["n"]
    fe_give = np.abs(np.einsum("ni,nij,nj->n", n, lu, n))     # the normal give per newton of a normal load (um/N)
    out = {}
    for r in REGIONS:
        m = (H["region"] == r) & np.isfinite(H["depth"]) & (H["f_n"] > 0.2)
        if m.sum() < 10:
            continue
        A = np.c_[H["f_n"][m], np.ones(m.sum())]
        (slope, icpt), *_ = np.linalg.lstsq(A, H["depth"][m], rcond=None)
        out[r] = dict(n=int(m.sum()), slope_um_per_N=float(1e3 * slope), intercept_mm=float(icpt),
                      fe_give_um_per_N=float(np.median(fe_give[m])),
                      extra_um_per_N=float(1e3 * slope - np.median(fe_give[m])))
    return out


def run_tip(t, max_n, ids):
    t0 = time.time()
    H = K.sub(t["H"], np.isin(t["H"]["id"].astype(str), ids))
    res = dict(case=t["case"], holds=int(len(H["id"])))
    tr = np.nonzero(H["f_n"] <= max_n)[0]
    ph = PR.Physical(t["cal0"], t["basis"]).fit(H, tr)
    th = ph.theta
    res["fit"] = dict(train_holds=int(len(tr)), max_n=max_n, k=float(th[0]), k_rest=float(t["cal0"].k),
                      dx_mm=float(th[1] - t["cal0"].dx), dy_mm=float(th[2] - t["cal0"].dy), dz_mm=float(th[3] - t["cal0"].dz),
                      c=float(math.exp(th[4])), E_fit_MPa=float(E_FE_MPA / math.exp(th[4])), rms_uT=ph.rms_uT)
    print(f"{t['name']}: fit on {len(tr)} holds <= {max_n} N: k {th[0]:.3f} (rest {t['cal0'].k:.3f}), offsets "
          f"{th[1] - t['cal0'].dx:+.3f} {th[2] - t['cal0'].dy:+.3f} {th[3] - t['cal0'].dz:+.3f} mm, c {math.exp(th[4]):.3f} "
          f"(E {E_FE_MPA / math.exp(th[4]):.1f} MPa), rms {ph.rms_uT:.1f} uT, {time.time() - t0:.0f} s", flush=True)
    # untrained: the FE basis and the rest calibration alone (c = 1)
    idx = np.arange(len(H["id"]))
    P0, F0 = PR.Physical(t["cal0"], t["basis"]).predict(H, idx)
    l0, e0, r0 = loc_fn(H, idx, P0, F0)
    res["untrained"] = table(H, l0, e0, r0)
    res["rest"] = dict(k=float(t["cal0"].k), dx=float(t["cal0"].dx), dy=float(t["cal0"].dy), dz=float(t["cal0"].dz),
                       rms_mT=float(t["cal0"].rms_mT))
    # cv
    fo = HX.folds(H)
    P, Fn = np.full((len(idx), 3), np.nan), np.full(len(idx), np.nan)
    cs = []
    for k in range(HX.FOLDS):
        te = np.nonzero(fo == k)[0]
        trk = np.nonzero((fo != k) & (H["f_n"] <= max_n))[0]
        phk = PR.Physical(t["cal0"], t["basis"]).fit(H, trk)
        cs.append(float(math.exp(phk.theta[4])))
        P[te], Fn[te] = phk.predict(H, te)
        print(f"  fold {k}: c {cs[-1]:.3f}, {time.time() - t0:.0f} s", flush=True)
    l1, e1, r1 = loc_fn(H, idx, P, Fn)
    res["cv"] = dict(c_folds=cs, table=table(H, l1, e1, r1))
    res["depth"] = depth_fit(H, t["basis"])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("v3")
    ap.add_argument("--max-n", type=float, default=3.5)
    ap.add_argument("--dx-tol", type=float, default=0.3, help="v3's board offset within v2 ID 1's +- this (mm): the same "
                    "board mount")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t2 = tip("five (v2 ID 1)", FV.V2, "b_v2")
    dx2 = float(t2["cal0"].dx)
    t3 = tip("four (v3 ID 1)", a.v3, "four_05R4", dx_range=(dx2 - a.dx_tol, dx2 + a.dx_tol))
    ids = sorted(set(t3["H"]["id"].astype(str)) & set(t2["H"]["id"].astype(str)))
    res = dict(script="scripts/ur5_tip_calibration_fe_calibration.py", v3=str(a.v3), v2=str(FV.V2.relative_to(ROOT)),
               actions=len(ids), E_FE_MPa=E_FE_MPA, made=time.strftime("%Y-%m-%d %H:%M"))
    res["four_v3id1"] = run_tip(t3, a.max_n, ids)
    res["five_v2id1"] = run_tip(t2, a.max_n, ids)
    for name in ("four_v3id1", "five_v2id1"):
        r = res[name]
        for key in ("all", "0-1", "1-2", "2-3.5", "3.5-8"):
            u, c = r["untrained"].get(key), r["cv"]["table"].get(key)
            if u and c and u["loc_mm"] is not None and c["loc_mm"] is not None:
                print(f"{name} {key:6s} untrained loc {u['loc_mm']:.2f} mm, f_n {u['fn_rel'] * 100:.0f} % (detected "
                      f"{u['detected'] * 100:.0f} %); fitted loc {c['loc_mm']:.2f} mm, f_n {c['fn_rel'] * 100:.0f} %")
        for reg, d in r["depth"].items():
            print(f"{name} depth {reg}: {d['slope_um_per_N']:.0f} um/N + {d['intercept_mm']:.2f} mm; FE give "
                  f"{d['fe_give_um_per_N']:.0f} um/N; extra {d['extra_um_per_N']:.0f} um/N")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
        print("->", a.out)


if __name__ == "__main__":
    main()
