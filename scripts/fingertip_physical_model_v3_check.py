#!/usr/bin/env python3
"""The physical model of the printed fingertip on the second four-magnet tip, v3 ID 2, beside v3 ID 1 and the five-magnet
v2 ID 1 (2026-10-29, after v3 ID 2's night suite of 10-28/29). Three checks on the low sphere's sessions:

  rest    v3 ID 2's rest calibration as v3 ID 1's was made: the free fit (the analysis's calib_rest.json), the fit with the
          board's offset dx within v2 ID 1's +- 0.3 mm (the same board mount), and, for both v3 tips, the fit with dx held
          at 0.70-0.80 mm, where the presses put v3 ID 1's board (the physical-model page's and the overview's open item).
  fit     scripts/ur5_tip_calibration_fe_calibration.py's run_tip on each tip, on the press actions that all three tips
          ran: the strength k, the board's offsets and the FE's compliance scale c fitted to the holds up to 3.5 N; the
          untrained model (FE and rest calibration, c = 1) and the fitted one under five folds grouped by action.
  gain    the board's signal per newton, |d_board| / f_n, by region of the pad, measured and predicted. The groups are
          scripts/ur5_tip_calibration_gain_by_region.py's (the reference presses at the face centre, FE node 788; the
          grid stage's face, edge and flank presses on the shared actions; the top-edge pass's tip-end presses). The model
          predicts each hold's change from its registered contact point and the HEX-E's force: the rest-calibrated model
          with c = 1 (untrained) and the model fitted in 'fit'. Medians over the holds of 0.5 to 3.5 N.

Writes docs/experiments/20261026-fingertip_physical_model/data/20261029-v3_tips_model_check.json.

    ~/.claude/bin/resguard.sh run --mem 3G --cpu 500 -- taskset -c 0-3 .venv/bin/python \\
        scripts/fingertip_physical_model_v3_check.py
"""
from __future__ import annotations

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
import ur5_tip_calibration_fe_calibration as FC  # noqa: E402
import ur5_tip_calibration_four_vs_five as FV  # noqa: E402
import ur5_tip_calibration_gain_by_region as GB  # noqa: E402
import ur5_tip_calibration_predict as PR  # noqa: E402

OUT = ROOT / "docs/experiments/20261026-fingertip_physical_model/data/20261029-v3_tips_model_check.json"
TIPS = (("v3id1", "v3 ID 1", ROOT / "logs/20261028-spherev3id1_twists", "four_05R4"),
        ("v3id2", "v3 ID 2", ROOT / "logs/20261029-spherev3id2", "four_05R4"),
        ("v2id1", "v2 ID 1", FV.V2, "b_v2"))
DX_TOL = 0.3            # the v3 tips' rest offset within v2 ID 1's +- this (mm), as the coordinator's fe_calibration
DX_PRESS = (0.70, 0.80)  # where v3 ID 1's presses put its board (0.74-0.75 mm in 20261028-fe_calibration.json)
MAX_N = 3.5
MID = (0.5, 3.5)
GROUPS = ("face centre, other plans", "grid, face", "top-edge pass, tip", "grid, edge", "grid, flank")


def rel(p):
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_absolute() else str(p)


def predicted_gain(t, theta, H):
    """|predicted d_board| / f_n of each hold, from its registered point, the HEX-E's force and the model at theta."""
    ph = PR.Physical(t["cal0"], t["basis"])
    M = ph.rbf(H["p"]).reshape(len(H["p"]), 3, ph.nm * 6)
    Jm = W.Model(ph.cal(theta)).motion_jacobian()
    pred = math.exp(theta[4]) * np.einsum("ij,kfj,kf->ki", Jm, M, H["f"])
    return np.linalg.norm(pred, axis=1) / H["f_n"]


def median_in(gain, f, m):
    mm = m & (f > MID[0]) & (f <= MID[1])
    return (float(np.median(gain[mm])), int(mm.sum())) if mm.any() else (None, 0)


def main():
    t0 = time.time()
    raw = {key: np.load(Path(d) / "holds.npz") for key, _, d, _ in TIPS}
    shared = set.intersection(*(set(H["id"][H["kind"] == "press"].astype(str)) for H in raw.values()))
    grid = set.intersection(*(set(H["id"][(H["kind"] == "press") & (H["stage"] == "grid")].astype(str))
                              for H in raw.values()))
    print(f"{len(shared)} press actions in all three sessions, {len(grid)} of them in the grid stage", flush=True)
    out = dict(script="scripts/fingertip_physical_model_v3_check.py", made=time.strftime("%Y-%m-%d %H:%M"),
               sessions={key: rel(d) for key, _, d, _ in TIPS}, shared_actions=len(shared), shared_grid_actions=len(grid),
               max_n=MAX_N, gain_band_N=list(MID), rest={}, fit={}, gain={})

    dx2 = float(W.Calib.load(Path(FV.V2) / "calib_rest.json").dx)
    tips = {}
    for key, label, d, case in TIPS:
        if case == "b_v2":
            tips[key] = FC.tip(label, d, case)
            continue
        tips[key] = FC.tip(label, d, case, dx_range=(dx2 - DX_TOL, dx2 + DX_TOL))
        free = W.Calib.load(Path(d) / "calib_rest.json")
        held = W.calibrate_frames_fixed(np.asarray(free.b_rest_uT, float).reshape(5, 3),
                                        W.Calib.load(FC.BOARD_CAL), case=case, poles=free.poles, dx_range=DX_PRESS)
        b = tips[key]["cal0"]
        out["rest"][key] = {name: dict(k=c.k, dx=c.dx, dy=c.dy, dz=c.dz, rms_mT=c.rms_mT, rms_next_mT=c.rms_next_mT,
                                       poles=list(c.poles))
                            for name, c in (("free", free), ("bounded", b), ("press_offset", held))}
        out["rest"][key]["dx_bounds"] = dict(bounded=[dx2 - DX_TOL, dx2 + DX_TOL], press_offset=list(DX_PRESS))
        print(f"{label} rest: free k {free.k:.3f} dx {free.dx:.2f} ({free.rms_mT:.3f} mT); bounded k {b.k:.3f} dx "
              f"{b.dx:.2f} ({b.rms_mT:.3f} mT); dx in {DX_PRESS}: k {held.k:.3f} dx {held.dx:.2f} ({held.rms_mT:.3f} mT)",
              flush=True)

    ids = sorted(shared)
    for key, label, _, _ in TIPS:
        r = FC.run_tip(tips[key], MAX_N, ids)
        r["rest_used"] = dict(k=float(tips[key]["cal0"].k), dx=float(tips[key]["cal0"].dx))
        out["fit"][key] = r
        th = np.array([r["fit"]["k"], r["fit"]["dx_mm"] + tips[key]["cal0"].dx, r["fit"]["dy_mm"] + tips[key]["cal0"].dy,
                       r["fit"]["dz_mm"] + tips[key]["cal0"].dz, math.log(r["fit"]["c"])])
        # the rest fit again with the board held where the presses put it (+- 0.02 mm), the tip's own chip frames kept
        d_tip = dict((k, d) for k, _, d, _ in TIPS)[key]
        free = W.Calib.load(Path(d_tip) / "calib_rest.json")
        at = W.calibrate_frames_fixed(np.asarray(free.b_rest_uT, float).reshape(5, 3), free, case=tips[key]["case"],
                                      poles=free.poles, dx_range=(th[1] - 0.02, th[1] + 0.02))
        out["rest"].setdefault(key, {})
        out["rest"][key]["free"] = out["rest"][key].get("free") or dict(k=free.k, dx=free.dx, dy=free.dy, dz=free.dz,
                                                                         rms_mT=free.rms_mT, rms_next_mT=free.rms_next_mT,
                                                                         poles=list(free.poles))
        out["rest"][key]["at_press_board"] = dict(k=at.k, dx=at.dx, dy=at.dy, dz=at.dz, rms_mT=at.rms_mT,
                                                  rms_next_mT=at.rms_next_mT, poles=list(at.poles))
        print(f"{label} rest with the board at the presses' {th[1]:.2f} mm: k {at.k:.3f} (rms {at.rms_mT:.3f} mT; free "
              f"{free.rms_mT:.3f} mT at k {free.k:.3f}, dx {free.dx:.2f})", flush=True)
        c0 = tips[key]["cal0"]
        th0 = np.array([c0.k, c0.dx, c0.dy, c0.dz, 0.0])
        H = FV.holds(Path(dict((k, d) for k, _, d, _ in TIPS)[key]))
        f = H["f_n"]
        g_meas = np.linalg.norm(H["d_board"], axis=1) / f
        g_un, g_fit = predicted_gain(tips[key], th0, H), predicted_gain(tips[key], th, H)
        rows = {}
        for name, m in GB.groups(H, "sph-top-", grid).items():
            if name not in GROUPS:
                continue
            (gm, n), (gu, _), (gf, _) = median_in(g_meas, f, m), median_in(g_un, f, m), median_in(g_fit, f, m)
            rows[name] = dict(holds=n, measured=gm, untrained=gu, fitted=gf)
            if gm is not None:
                print(f"  {label} {name:28s} {n:4d} holds: measured {gm:5.1f}, untrained {gu:5.1f}, fitted {gf:5.1f} uT/N",
                      flush=True)
        out["gain"][key] = rows
        print(f"{label}: c {r['fit']['c']:.3f} (E {r['fit']['E_fit_MPa']:.1f} MPa), k {r['fit']['k']:.3f}, board at "
              f"{th[1]:.2f} mm; untrained loc {r['untrained']['all']['loc_mm']:.2f} mm ({time.time() - t0:.0f} s)",
              flush=True)

    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"-> {rel(OUT)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
