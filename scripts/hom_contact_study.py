#!/usr/bin/env python3
"""Run the 2026-10-01 contact-patch study on the two-pad pinch rig (scripts/hom_contact_rig.py).

Phases, each resumable (a row whose key is already in the output file is skipped):
  torsion  friction torque at steady spin against pinch force, every contact model
  brake    gravity swing while the pinch is lowered 6 -> 0.2 N (the paper's load step)
  plate    Drake only: the pad pressed on a rigid plate, penetration and patch against the foundation law
  cop      kinematic centre-of-pressure sweep (the paper's GCF origin)
  films    one tiled film of the brake for eight models

The condim-4 MuJoCo variants are calibrated from the Drake reference rows (relaxation 0.01 s):
constant mu_t = Drake's torsion arm at 1 N, scheduled mu_t = the power law fitted to all of them.
The sphere pads take only the modulus E and the relaxation time; nothing is fitted to Drake.

    PY=logs/20261001-hom_contact/venv/bin/python
    MUJOCO_GL=egl OMP_NUM_THREADS=1 $PY scripts/hom_contact_study.py --out docs/experiments/20261001-hom_contact_patch
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hom_contact_rig as H  # noqa: E402

DRAKE_DEF = "drake:hydro:E1e7:r1"              # SAP default relaxation time 0.1 s
DRAKE_REF = "drake:hydro:E1e7:r1:rt0.01"       # elastic reference: torsion is rate-independent below ~0.01 s
DRAKE_RT02 = "drake:hydro:E1e7:r1:rt0.02"      # matched to the sphere pads' relaxation time
DRAKE_PT = "drake:point"
MJ_P3 = "mj:point3"
SPHERES = [f"mj:spheres:s{s}:rs0.75:ir100:tr0.02" for s in ("2", "1", "0.5")]
SP_TR01 = "mj:spheres:s0.5:rs0.75:ir100:tr0.1"
SP_IR10 = "mj:spheres:s0.5:rs0.75:tr0.02"
FORCES = (0.25, 0.5, 1.0, 2.0, 4.0)


def key(r):
    return "|".join(str(r.get(k)) for k in ("exp", "spec", "N", "omega", "d_mm", "T_ramp", "delta_mm"))


def load(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def run_rows(path, jobs, fn):
    have = {key(r) for r in load(path)}
    for kw in jobs:
        probe = dict(kw)
        probe.setdefault("exp", fn.__name__.replace("exp_", ""))
        if "d_cg" in probe:
            probe["d_mm"] = probe.pop("d_cg") * 1e3
        if key(probe) in have:
            continue
        t0 = time.time()
        row = fn(**kw)
        H.append_row(path, row)
        print(f"  {row['exp']:8s} {row['spec']:42s} {json.dumps({k: kw[k] for k in kw if k != 'spec'})}"
              f"  {time.time() - t0:5.1f} s", flush=True)


def fit_law(rows):
    N = np.array([r["N"] for r in rows])
    rb = np.array([r["rbar_per_pad_mm"] for r in rows]) * 1e-3
    p, lc = np.polyfit(np.log(N), np.log(rb), 1)
    return float(np.exp(lc)), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/experiments/20261001-hom_contact_patch")
    ap.add_argument("--phases", default="plate,torsion,brake,cop,films")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    phases = a.phases.split(",")
    tpath, bpath, cpath = out / "torsion.jsonl", out / "brake.jsonl", out / "cop.jsonl"

    # ---- torsion: the Drake reference first, it calibrates the condim-4 variants
    print("torsion: Drake", flush=True)
    jobs = [dict(spec=s, N=N, omega=w) for s in (DRAKE_REF, DRAKE_DEF, DRAKE_RT02) for w in (0.5, 2.0) for N in FORCES]
    jobs += [dict(spec=DRAKE_PT, N=N, omega=2.0) for N in FORCES]
    if "torsion" in phases:
        run_rows(tpath, jobs, H.exp_torsion)
    ref = [r for r in load(tpath) if r["spec"] == DRAKE_REF]
    c, pw = fit_law(ref)
    c1 = float(np.mean([r["rbar_per_pad_mm"] for r in ref if r["N"] == 1.0])) * 1e-3
    calib = {"fit_c_m": c, "fit_p": pw, "mu_t_at_1N_m": c1}
    (out / "calibration.json").write_text(json.dumps(calib, indent=1))
    print(f"  Drake reference law rbar = {c * 1e3:.4f} mm * N^{pw:.4f}; mu_t(1 N) = {c1 * 1e3:.4f} mm", flush=True)
    MJ_P4 = f"mj:point4:mt{c1:.6g}:ir100"
    MJ_P4_IR10 = f"mj:point4:mt{c1:.6g}"
    MJ_P4S = f"mj:point4s:fit{c:.6g}x{pw:.4f}:ir100"
    mj_specs = [MJ_P4, MJ_P4_IR10, MJ_P4S] + SPHERES + [SP_TR01, SP_IR10]
    if "torsion" in phases:
        print("torsion: MuJoCo", flush=True)
        jobs = [dict(spec=MJ_P3, N=N, omega=2.0) for N in FORCES]
        jobs += [dict(spec=s, N=N, omega=w) for s in mj_specs for w in (0.5, 2.0) for N in FORCES]
        run_rows(tpath, jobs, H.exp_torsion)

    brake_specs = [DRAKE_DEF, DRAKE_REF, DRAKE_RT02, DRAKE_PT, MJ_P3, MJ_P4, MJ_P4_IR10, MJ_P4S] + SPHERES + [SP_TR01, SP_IR10]
    if "brake" in phases:
        print("brake", flush=True)
        jobs = [dict(spec=s, d_cg=d * 1e-3, T_ramp=4.0, traces=True) for d in (15.0, 30.0) for s in brake_specs]
        jobs += [dict(spec=s, d_cg=0.015, T_ramp=12.0, traces=True) for s in (DRAKE_RT02, MJ_P4, SPHERES[1], SPHERES[2])]
        run_rows(bpath, jobs, H.exp_brake)

    if "plate" in phases:
        print("plate", flush=True)
        jobs = [dict(E=E, res_mm=0.5, F=F) for E in (1e6, 1e7) for F in (0.25, 1.0, 4.0)]
        run_rows(out / "plate.jsonl", jobs, H.exp_plate)

    if "cop" in phases:
        print("cop", flush=True)
        jobs = [dict(spec=s, delta_mm=0.21) for s in
                (DRAKE_REF, "drake:hydro:E1e7:r0.5", MJ_P3) + tuple(SPHERES)]
        run_rows(cpath, jobs, H.exp_cop)

    if "films" in phases:
        print("films", flush=True)
        fdir = out / "media"
        fdir.mkdir(exist_ok=True)
        # columns: point contact | elastic patch | patch at 0.1 s relaxation | extra; rows: Drake, MuJoCo
        tiles = [DRAKE_PT, DRAKE_REF, DRAKE_DEF, DRAKE_RT02, MJ_P3, SPHERES[2], SP_TR01, MJ_P4S]
        allf = []
        for s in tiles:
            r = H.exp_brake(s, 0.015, T_ramp=4.0, keep_frames=True, film_wh=(400, 300))
            allf.append(r["_frames"])
            print(f"  film {s}: {len(r['_frames'])} frames, phi_end {r['phi_end']:.1f}", flush=True)
        n = min(len(f) for f in allf)
        grid = []
        for k in range(n):
            row1 = np.concatenate([allf[i][k] for i in range(4)], axis=1)
            row2 = np.concatenate([allf[i][k] for i in range(4, 8)], axis=1)
            grid.append(np.concatenate([row1, row2], axis=0))
        H.write_mp4(grid, fdir / "20261001-brake_eight_models.mp4", fps=25)
        # a still of the moment the reference passes 45 deg, for the page poster
        from PIL import Image
        Image.fromarray(grid[min(n - 1, 62)]).save(fdir / "20261001-brake_eight_models_poster.png")
        (out / "films.json").write_text(json.dumps({"tiles": tiles, "frames": n, "fps": 25,
                                                    "file": "media/20261001-brake_eight_models.mp4"}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
