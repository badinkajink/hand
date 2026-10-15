#!/usr/bin/env python3
"""Contact-bed cases of the curved-tip study (docs/handoff/20261015-curved_tips.md, step 2): the two-pad pinch rig of
the contact bed (two force-controlled rails squeeze the 25 mm, 24.5 g screwdriver, mu 1, 1 ms) with a curved tip on each
rail, through tip_codesign_bed's tcd hooks.

Models per tip:
  pads    the 1 mm pads of builder v2 (each pad's K = E a_i / h_i, h_i its normal distance to the volume centroid),
          relaxation 0.02 s, impratio 100, elliptic cone; spec fields hloc:Keach
  drake   Drake 1.57's compliant convex of the tip's 0.25 mm mesh, E 10 MPa, relaxation 0.01 s, SAP
  drake05 the same on the 0.5 mm mesh (the hull-resolution check)

Cases (one fsynced row each in docs/experiments/20261015-curved_tips/bed/<case>.jsonl, resumable by key):
  static N    approach (rail travel from first touch, mean over the last 0.2 s of 0.8 s) at pinch N per pad; the
              touching pads (Drake: contact-surface faces), contacts and constraint rows of the last step, and the
              median wall time per 1 ms step
  T2 N        contact_bed_twist: the onset arm under a torque ramp scaled to the law's onset arm (the Drake-field law
              on the 0.25 mm mesh, bed/law.p*.jsonl), and the rotational creep at half the onset torque over a 1 s hold
  spin N w    tip_codesign_bed.case_spin: the tool driven about the pinch axis at w rad/s, the arm the mean torque over
              the turn 0.4-1.2 rad over 2 mu N
  T5          contact_bed_brake: the pinch force at which the tool, released 15 mm off the pinch, stops at 45 deg
  T1 N        contact_bed_pull: the onset force of a 2 N/s tangential pull and the creep speed at half of it over 1 s

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/curved_tips_bed.py --tips tcd_cap6_u0.1 --models pads drake --cases static T2 spin T5 T1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tip_codesign_bed as B  # noqa: E402  (installs the tcd hooks into the bed rigs)
import contact_bed_common as CB  # noqa: E402
import contact_bed_pull as P  # noqa: E402
import curved_tips_common as C  # noqa: E402
import hom_contact_rig as H  # noqa: E402
import pad_usability_backends as PUB  # noqa: E402
import tpu_tip_rig as T  # noqa: E402
import tip_codesign_geometry as TG  # noqa: E402
import functools  # noqa: E402

TG.shape = functools.lru_cache(maxsize=None)(TG.shape)   # one shape per process: its hulls and pads take 2-18 s
SCRIPT = "scripts/curved_tips_bed.py"
OUT = C.D / "bed"
T.MESH_DIR = C.LOGS / "meshes"
PADS_X, DRAKE_X = "hloc:Keach:mres0.00025", "mres0.00025"
N_STATIC, N_T2, N_T1 = (0.5, 1.0, 2.0, 3.0), (1.0, 3.0), (1.0,)
OMEGA = (0.1, 0.3, 1.0, 3.0)


KINDS = {"pads": PADS_X, "drake": DRAKE_X, "drake05": "mres0.0005"}   # drake05: Drake on the 0.5 mm mesh (hull check)


def model(key: str, kind: str) -> str:
    """Register and return the bed model name of a tip: kind 'pads' (builder v2), 'drake' or 'drake05'."""
    _, v = B.tcd_model(key, KINDS[kind], drake=kind != "pads")
    name = f"{kind}_{key}"
    B.register({name: v})
    return name


def law_onset(key: str) -> dict:
    """The Drake-field law's onset arm (m) at psi 0 per N, on the 0.25 mm mesh."""
    for f in sorted(OUT.glob("law*.jsonl")):
        for r in CB.read_rows(f):
            if r["tip"] == key and r["field"] == "drake" and abs((r.get("res_m") or 0) - 0.00025) < 1e-9:
                return {float(n): v * 1e-3 for n, v in r["onset_arm_mm"].items()}
    raise KeyError(f"no 0.25 mm Drake-field law row for {key} in {OUT}/law*.jsonl")


def case_static(name, N, dt_ms):
    spec = CB.MODELS[name][0]
    with T.tpu_rigs():
        H.DT = dt_ms * 1e-3
        rig = H.make_rig(spec, d_cg=0.0, gravity=False)
        rig.set_pad_force(N)
        rig.set_tool_wrench(np.zeros(3), np.zeros(3))
        W, X, NE, NC = [], [], [], []
        n = int(round(0.8 / H.DT))
        for k in range(n):
            w0 = time.perf_counter()
            rig.step(H.DT)
            W.append(time.perf_counter() - w0)
            if k >= n - int(round(0.2 / H.DT)):
                X.append(H.tool_touch(rig.sp) - abs(rig.pad_x()["L"]))
                if rig.sim == "mujoco":
                    NE.append(rig.d.nefc), NC.append(rig.d.ncon)
        pt = T.patch_of(rig)
    X = np.array(X) * 1e6
    row = {"L": pt["L"], "R": pt["R"], "info": getattr(rig, "info", {}), "sim": rig.sim,
           "approach_um": float(X.mean()), "approach_sd_um": float(X.std()), "T_static_s": 0.8,
           "us_per_step_median": float(np.median(W) * 1e6), "touching_per_side": pt["L"].get("n")}
    if NE:
        row.update(nefc_mean=float(np.mean(NE)), ncon_mean=float(np.mean(NC)))
    return row


def case_T1(name, N, dt_ms):
    with T.tpu_rigs():
        return P.run_case(CB.MODELS[name][1], N, dt_ms)


def key_of(r):
    return (r["case"], r["model"], r.get("N"), r.get("omega"), r["dt_ms"])


def run(tips, kinds, cases, dts=(1.0,), drake_omega=None):
    for key in tips:
        for kind in kinds:
            name = model(key, kind)
            for case in cases:
                out = OUT / f"{case}.jsonl"
                have = {key_of(r) for r in CB.read_rows(out) if "case" in r}
                Ns = {"static": N_STATIC if kind == "pads" else (1.0, 3.0), "T2": N_T2 if kind == "pads" else (1.0,),
                      "spin": (1.0,), "T1": N_T1, "T5": (None,)}[case]
                oms = (OMEGA if kind == "pads" else (drake_omega or OMEGA)) if case == "spin" else (None,)
                dl = (dts if case != "T5" or kind == "pads" else (1.0, 0.5))
                for dt in dl:
                    for N in Ns:
                        for om in oms:
                            k = (case, name, N, om, dt)
                            if k in have:
                                continue
                            t0 = time.time()
                            try:
                                if case == "static":
                                    r = case_static(name, N, dt)
                                elif case == "T2":
                                    r = B.case_T2(name, N, dt, law_onset(key)[N])
                                elif case == "spin":
                                    r = B.case_spin(name, N, om, dt, decomp=kind == "pads")
                                elif case == "T5":
                                    r = B.case_T5(name, dt)
                                else:
                                    r = case_T1(name, N, dt)
                                r.setdefault("status", "complete")
                            except Exception as e:                  # a failed case is a row
                                r = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()[-1500:]}
                            r.update(case=case, model=name, tip=key, kind=kind, rig_spec=CB.MODELS[name][0], N=N,
                                     omega=om, dt_ms=dt, script=SCRIPT, git_rev=CB.git_rev(),
                                     wall_s=time.time() - t0, when=time.strftime("%Y-%m-%d %H:%M"))
                            PUB.append_row(out, r)
                            keys = ("approach_um", "touching_per_side", "nefc_mean", "rbar_onset_mm", "creep_deg_s",
                                    "rbar_spin_mm", "N_at_45_N", "mu_eff", "creep_mm_s", "us_per_step_median")
                            print(case, name, N, om, dt, r.get("status"),
                                  {q: round(r[q], 4) for q in keys if isinstance(r.get(q), (int, float))},
                                  r.get("error", ""), f"{time.time() - t0:.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tips", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["pads", "drake"], choices=list(KINDS))
    ap.add_argument("--cases", nargs="+", default=["static", "T2", "spin", "T5", "T1"])
    ap.add_argument("--drake-omega", nargs="+", type=float, default=None)
    a = ap.parse_args()
    run(a.tips, a.models, a.cases, drake_omega=a.drake_omega)


if __name__ == "__main__":
    main()
