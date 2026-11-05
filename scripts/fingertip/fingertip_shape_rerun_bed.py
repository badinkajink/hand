#!/usr/bin/env python3
"""The contact bed of the fingertip-shape rerun (docs/handoff/20261103-fingertip_shape_rerun.md, ledger step 3): the
cases of 2026-10-15 (scripts/fingertip/curved_tips_bed.py, curved_tips_law.py) at 2.5 MPa, the scenes' modulus since 2026-10-30.

  law   curved_tips_law's fields at E = 2.5 MPa (Drake's field on the 0.5 and 0.25 mm meshes, Winkler, local, pads v1 and
        v2): approach at 0.5-3 N, the all-slip arms at psi 0-69 deg, the quasi-static brake force at 45 deg
        -> docs/experiments/<run>/bed/<tip>/law.jsonl
  bed   curved_tips_bed's cases with the pads of builder v2 at 2.5 MPa and the relaxation time pad_tr(2.5 MPa) = 0.04 s
        (impratio 100, 1 ms), and Drake's compliant convex of the 0.25 mm mesh at 2.5 MPa (relaxation 0.01 s, SAP):
        static (approach), T2 (twist-onset arm under a torque ramp scaled to the law's onset arm), spin (the arm at
        0.1-3 rad/s; Drake at 1 rad/s), T5 (the brake force at 45 deg, 1 ms) -> docs/experiments/<run>/bed/<tip>/<case>.jsonl

Each tip writes its own folder, so tips run in parallel processes.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/fingertip/fingertip_shape_rerun_bed.py law --tips tcd_cap6_u0.1
    $PY scripts/fingertip/fingertip_shape_rerun_bed.py bed --tips tcd_cap6_u0.1 --models pads drake --cases static T2 spin T5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import fingertip_shape_rerun_common as C  # noqa: E402

OUT = C.D / "bed"
SCRIPT = "scripts/fingertip/fingertip_shape_rerun_bed.py"


def cmd_law(a):
    import curved_tips_law as CL
    CL.SCRIPT = f"{SCRIPT} law (curved_tips_law)"
    for k in a.tips:
        sys.argv = ["curved_tips_law.py", "--tips", k, "--E", f"{C.E_PA:g}", "--out", str(OUT / k / "law.jsonl")]
        CL.main()


def cmd_bed(a):
    import contact_bed_common as CB
    import curved_tips_bed as CBD
    import tpu_tip_rig as T
    import tip_codesign_geometry as TG
    T.MESH_DIR = C.LOGS / "meshes"
    CBD.SCRIPT = f"{SCRIPT} bed (curved_tips_bed)"
    E, tr, rt = f"E{C.E_PA:g}", f"tr{C.TR_S:g}", f"rt{C.DRAKE_RT:g}"

    def model(key, kind):
        """The bed model of a tip at 2.5 MPa: tip_codesign_bed.tcd_model's specs with E (and the pads' tr) set."""
        t = f"tpu{TG.shape(key).a * 1e3:g}:tcd={key}"
        if kind == "pads":
            ex = ":" + CBD.PADS_X
            v = (f"mj:spheres:s1:rs0.75:ir100:{tr}:{E}{ex}:{t}", f"mj:spheres:s1:rs0.75:{tr}:{E}{ex}:{t}")
        else:
            ex = ":" + CBD.DRAKE_X
            v = (f"drake:hydro:{E}:r1:{rt}:{t}{ex}", f"drake:hydro:{rt}:{E}:{t}{ex}")
        name = f"{kind}_{key}"
        CBD.B.register({name: v})
        return name

    def law_onset(key):
        for r in CB.read_rows(OUT / key / "law.jsonl"):
            if r["tip"] == key and r["field"] == "drake" and abs((r.get("res_m") or 0) - 0.00025) < 1e-9:
                if abs(r["E_Pa"] - C.E_PA) > 1.0:
                    raise ValueError(f"{key}: law row at {r['E_Pa']:g} Pa")
                return {float(n): v * 1e-3 for n, v in r["onset_arm_mm"].items()}
        raise KeyError(f"no 0.25 mm Drake-field law row for {key} in {OUT / key}/law.jsonl")

    CBD.model, CBD.law_onset = model, law_onset
    for k in a.tips:
        CBD.OUT = OUT / k
        CBD.OUT.mkdir(parents=True, exist_ok=True)
        run(CBD, CB, k, a.models, a.cases)


# the cases per model: (loads N, spin rates rad/s, time steps ms); the pads' as on 10-15, Drake's the step's four metrics
# at one load and rate (Drake's spin arm moved 3-11 % from the law over 0.1-3 rad/s on 10-15, and it runs 2.5 times
# slower at 2.5 MPa than at 10 MPa)
PLAN = {"pads": {"static": ((0.5, 1.0, 2.0, 3.0), (None,), (1.0,)), "T2": ((1.0, 3.0), (None,), (1.0,)),
                 "spin": ((1.0,), (0.1, 0.3, 1.0, 3.0), (1.0,)), "T5": ((None,), (None,), (1.0,))},
        "drake": {"static": ((1.0, 3.0), (None,), (1.0,)), "T2": ((1.0,), (None,), (1.0,)),
                  "spin": ((1.0,), (1.0,), (1.0,)), "T5": ((None,), (None,), (1.0,))}}


def run(CBD, CB, key, kinds, cases):
    """curved_tips_bed.run for one tip over PLAN's loads, rates and steps (its row format and resume keys)."""
    import time
    import traceback
    for kind in kinds:
        name = CBD.model(key, kind)
        for case in cases:
            out = CBD.OUT / f"{case}.jsonl"
            have = {CBD.key_of(r) for r in CB.read_rows(out) if "case" in r}
            Ns, oms, dts = PLAN[kind][case]
            for dt in dts:
                for N in Ns:
                    for om in oms:
                        if (case, name, N, om, dt) in have:
                            continue
                        t0 = time.time()
                        try:
                            if case == "static":
                                r = CBD.case_static(name, N, dt)
                            elif case == "T2":
                                r = CBD.B.case_T2(name, N, dt, CBD.law_onset(key)[N])
                            elif case == "spin":
                                r = CBD.B.case_spin(name, N, om, dt, decomp=kind == "pads")
                            else:
                                r = CBD.B.case_T5(name, dt)
                            r.setdefault("status", "complete")
                        except Exception as e:                  # a failed case is a row
                            r = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()[-1500:]}
                        r.update(case=case, model=name, tip=key, kind=kind, rig_spec=CB.MODELS[name][0], N=N, omega=om,
                                 dt_ms=dt, E_Pa=C.E_PA, script=CBD.SCRIPT, git_rev=CB.git_rev(), wall_s=time.time() - t0,
                                 when=time.strftime("%Y-%m-%d %H:%M"))
                        CBD.PUB.append_row(out, r)
                        keys = ("approach_um", "touching_per_side", "nefc_mean", "rbar_onset_mm", "creep_deg_s",
                                "rbar_spin_mm", "N_at_45_N", "us_per_step_median")
                        print(case, name, N, om, dt, r.get("status"),
                              {q: round(r[q], 4) for q in keys if isinstance(r.get(q), (int, float))},
                              r.get("error", ""), f"{time.time() - t0:.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lw = sub.add_parser("law")
    lw.add_argument("--tips", nargs="+", required=True)
    bd = sub.add_parser("bed")
    bd.add_argument("--tips", nargs="+", required=True)
    bd.add_argument("--models", nargs="+", default=["pads", "drake"], choices=["pads", "drake"])
    bd.add_argument("--cases", nargs="+", default=["static", "T2", "spin", "T5"])
    a = ap.parse_args()
    {"law": cmd_law, "bed": cmd_bed}[a.cmd](a)


if __name__ == "__main__":
    main()
