#!/usr/bin/env python3
"""Calibrate a printed SR2 fingertip back to the sphere pads (2026-10-12).

`predict` prints what the finite-element model expects a print to measure on the bench. `fit` takes bench measurements
of one printed tip and returns its solid modulus E_s, the pads' modulus E* for that tip, its friction coefficient, and
the command that writes a scene with those values.

The model is linear elastic and its contact is frictionless, so at a given force the approach scales as 1/E_s and the
magnet's travel as 1/E_s; E* scales as E_s. Rows: docs/experiments/20261009-tpu_tip_fem/magnet_runs.jsonl
(scripts/tpu_tip_magnet.py); field model: scripts/tpu_tip_print_plan.py.

Measurements, each optional:
  --indent F:d ...   force (N) and approach (um) of the screwdriver shaft (or a 12.5 mm rod, axis across the finger)
                     pressed on the palmar face at the contact point, from first touch
  --field F:dB ...   force (N) on the face and the change of the field magnitude (uT) at the magnetometer
  --pull N:F         pinch force per pad (N) and the axial pull (N) at which the tool slips out of a two-pad pinch
  --twist N:T        pinch force per pad (N) and the torque (N mm) at which the tool spins in a two-pad pinch

    .venv/bin/python scripts/tpu_tip_calibrate.py predict --tip w95_20 --magnet d4.76x1.59@5
    .venv/bin/python scripts/tpu_tip_calibrate.py fit --tip w95_20 --magnet d4.76x1.59@5 --indent 0.5:22 1:41 3:118 --pull 2:3.1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_magnet as TM  # noqa: E402
import tpu_tip_print_plan as PP  # noqa: E402

OUT_DIR = ROOT / "docs/experiments/20261009-tpu_tip_fem/print"
BASE_SCENE = ROOT / "results/phase1/real_v1/20261011-sv1_u0308_b050_servo_tip"


def row_for(tip, magnet):
    mag, dep = TM.parse_magnet(magnet)
    r = PP.rows().get((tip, mag or "none", dep))
    if r is None:
        raise SystemExit(f"no finite-element row for {tip} {magnet}: run scripts/tpu_tip_magnet.py run --recipes {tip} "
                         f"--magnets {magnet}")
    return r


def fe_curve(r):
    F = np.array(sorted(float(k) for k in r["per_force"]))
    d = np.array([r["per_force"][f"{f:g}"]["delta_um"] for f in F])
    return F, d


def approach_at(r, F):
    """FE approach (um) at force F: log-log between the rows' forces, extended along the end segments outside them (the
    page's calculator, scripts/tpu_tip_widget.py, does the same)."""
    Fs, d = fe_curve(r)
    x, xs, ys = math.log(F), np.log(Fs), np.log(d)
    i = int(np.clip(np.searchsorted(xs, x) - 1, 0, len(xs) - 2))
    return float(np.exp(ys[i] + (ys[i + 1] - ys[i]) * (x - xs[i]) / (xs[i + 1] - xs[i])))


def predict(r, gap=PP.GAPS[0]):
    out = dict(recipe=r["recipe"], magnet=r["magnet"], depth_mm=r["depth_mm"], E_s_MPa=r["E_s"], E_star_MPa=r["E_best_MPa"],
               approach_um={k: v["delta_um"] for k, v in r["per_force"].items()},
               arm_mm={k: v["arm_mm"] for k, v in r["per_force"].items()})
    if r["magnet"] != "none":
        f = PP.field(r, gap)
        out.update(chip_gap_mm=gap, field_at_chip_mT=f["B0_mT"], field_per_N_normal_uT=f["S_normal_norm"],
                   field_per_N_shear_across_uT=f["S_shear_y_norm"], force_noise_normal_N=f["sigF_normal_N"])
    return out


def pairs(vals):
    return [tuple(float(x) for x in v.split(":")) for v in vals or []]


def fit(r, indent=None, field_pts=None, pull=None, twist=None, gap=PP.GAPS[0]):
    E0 = r["E_s"]
    est = []
    for F, d in pairs(indent):
        est.append(("indent", E0 * approach_at(r, F) / d))
    if field_pts and r["magnet"] != "none":
        S = PP.field(r, gap)["S_normal_norm"]                    # uT/N at E0, from the travel at 1 N
        for F, dB in pairs(field_pts):
            est.append(("field", E0 * S * F / dB))
    out = dict(recipe=r["recipe"], magnet=r["magnet"], depth_mm=r["depth_mm"], E_s_nominal_MPa=E0)
    if est:
        vals = np.array([e for _, e in est])
        E_s = float(np.exp(np.mean(np.log(vals))))
        out.update(E_s_MPa=E_s, E_s_estimates=[dict(kind=k, E_s_MPa=e) for k, e in est],
                   E_star_MPa=r["E_best_MPa"] * E_s / E0)
    if pull:
        (N, Fs), = pairs(pull)
        out["mu"] = Fs / (2 * N)
    if twist:
        (N, T), = pairs(twist)
        mu = out.get("mu")
        out["twist_arm_mm"] = T / (2 * mu * N) if mu else None
    if "E_star_MPa" in out:
        cmd = (f"python scripts/make_pad_morphology_run.py --src {BASE_SCENE.relative_to(ROOT)} --out "
               f"results/phase1/real_v1/<run>_tpu2.7pads1_E{out['E_star_MPa']:.1f} --model pads --E {out['E_star_MPa'] * 1e6:.4g}")
        if out.get("mu"):
            cmd += f" --mu {out['mu']:.3g}"
        out["scene_command"] = cmd
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["predict", "fit"])
    ap.add_argument("--tip", required=True, choices=list(TM.RECIPES))
    ap.add_argument("--magnet", default="d4.76x1.59@5", help="none, or <magnet>@<depth mm> as in scripts/tpu_tip_magnet.py")
    ap.add_argument("--gap", type=float, default=PP.GAPS[0], help="chip's distance behind the back face, mm")
    ap.add_argument("--indent", nargs="*")
    ap.add_argument("--field", nargs="*")
    ap.add_argument("--pull")
    ap.add_argument("--twist")
    ap.add_argument("--save", action="store_true", help="write the result to docs/experiments/20261009-tpu_tip_fem/print/")
    a = ap.parse_args()
    r = row_for(a.tip, a.magnet)
    res = predict(r, a.gap) if a.cmd == "predict" else fit(r, a.indent, a.field, [a.pull] if a.pull else None,
                                                             [a.twist] if a.twist else None, a.gap)
    print(json.dumps(res, indent=1))
    if a.save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        name = f"calibration_{a.tip}_{a.magnet.replace('@', '_d').replace('.', 'p')}.json"
        json.dump(res, open(OUT_DIR / name, "w"), indent=1)
        print("wrote", OUT_DIR / name)
