#!/usr/bin/env python3
"""Contact on the bed for the printable 85A fingertips (docs/handoff/20261016-sim2real_fingertip_readiness.md, step 4).

The contact bed of the curved-tip study (scripts/curved_tips_bed.py: two force-controlled rails squeeze the 25 mm, 24.5 g
screwdriver between two copies of the tip, friction 1.0) with each variant's tip shape at the variant's pad modulus E
(scripts/sim2real_fingertip_scenes.py: the 85A finite elements of the magnet block, solid 12 MPa; curved tips scaled by
their 2026-10-15 tool-fit ratio). Arms, each a separate model on the bed:

  v2       1 mm pads, builder v2 (hloc:Keach), impratio 100, 1 ms: the planner's and evaluation's numerics
  v1       the same pads with builder v1 (one stiffness per tip, h 8.5 mm)
  rl       builder v2 at the RL trainer's numerics: impratio 10, 2 ms step
  drake    Drake 1.57 hydroelastic, the tip's compliant convex at E, relaxation 0.01 s, SAP, 1 ms
  E0.5x, E2x   builder v2 at half and twice E: Siraya's solid modulus at 6 or 24 MPa in place of the 12 MPa proxy

Friction and impratio stay separate from the normal stiffness: every arm keeps friction 1.0, and only the rl arm changes
impratio. Cases (curved_tips_bed): static (approach at 0.5, 1, 3 N), T2 (twist-onset arm and creep at 1 and 3 N),
spin (arm at 1 rad/s, 1 N), T1 (pull onset and creep at 1 N). Rows: docs/experiments/20261016-sim2real_fingertip_readiness/
bed/<case>.jsonl with the variant and arm added, resumable.

    PY=logs/20261001-hom_contact/venv/bin/python
    ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- $PY scripts/sim2real_fingertip_bed.py --variants block4_g30 --arms v2 drake
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curved_tips_bed as CTB  # noqa: E402
import contact_bed_common as CB  # noqa: E402
import pad_usability_backends as PUB  # noqa: E402
import tip_codesign_bed as B  # noqa: E402

D = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
OUT = D / "bed"
ASSETS = ROOT / "assets/mjcf/experimental/20261016-sim2real_fingertip_readiness"
SCRIPT = "scripts/sim2real_fingertip_bed.py"
ARMS = {"v2": ("hloc:Keach", 1.0, 1.0), "v1": ("", 1.0, 1.0), "rl": ("hloc:Keach:ir10", 2.0, 1.0),
        "drake": (None, 1.0, 1.0), "E0.5x": ("hloc:Keach", 1.0, 0.5), "E2x": ("hloc:Keach", 1.0, 2.0)}
NS = {"static": (0.5, 1.0, 3.0), "T2": (1.0, 3.0), "spin": (1.0,), "T1": (1.0,)}
LAW_DIR = ROOT / "docs/experiments/20261015-curved_tips/bed"


def law_onset(key: str) -> dict:
    """The 10 MPa Drake-field law's onset arm per N (2026-10-15 rows), used only to scale T2's torque ramp."""
    for f in sorted(LAW_DIR.glob("law*.jsonl")):
        for r in CB.read_rows(f):
            if r["tip"] == key and r["field"] == "drake" and abs((r.get("res_m") or 0) - 0.00025) < 1e-9:
                return {float(n): v * 1e-3 for n, v in r["onset_arm_mm"].items()}
    raise KeyError(key)


def model_of(key, arm, E):
    extra, dt, scale = ARMS[arm]
    Ee = E * scale
    if arm == "drake":
        name, v = B.tcd_model(key, f"mres0.00025:E{Ee:.6g}", drake=True)
        name = f"drake_{key}_E{Ee:.4g}"
    else:
        ex = ":".join(x for x in (extra, "mres0.00025", f"E{Ee:.6g}") if x)
        name, v = B.tcd_model(key, ex)
    B.register({name: v})
    return name, dt, Ee


def run(variants, arms, cases):
    for variant in variants:
        spec = json.loads((ASSETS / variant / "README.json").read_text())
        key, E = spec["key"], spec["pad_E_Pa"]
        for arm in arms:
            name, dt, Ee = model_of(key, arm, E)
            for case in cases:
                out = OUT / f"{case}.jsonl"
                have = {(r["case"], r["model"], r.get("N"), r.get("omega"), r["dt_ms"]) for r in CB.read_rows(out)
                        if "case" in r and r.get("status") == "complete"} if out.exists() else set()
                Ns = NS[case] if arm != "drake" or case != "static" else (0.5, 1.0, 3.0)
                oms = (1.0,) if case == "spin" else (None,)
                for N in Ns:
                    for om in oms:
                        if (case, name, N, om, dt) in have:
                            continue
                        t0 = time.time()
                        try:
                            if case == "static":
                                r = CTB.case_static(name, N, dt)
                            elif case == "T2":
                                r = B.case_T2(name, N, dt, law_onset(key)[N])
                            elif case == "spin":
                                r = B.case_spin(name, N, om, dt, decomp=arm != "drake")
                            else:
                                r = CTB.case_T1(name, N, dt)
                            r.setdefault("status", "complete")
                        except Exception as e:
                            r = {"status": "failed", "error": repr(e), "traceback": traceback.format_exc()[-1500:]}
                        r.update(case=case, model=name, variant=variant, arm=arm, tip=key, E_Pa=Ee, rig_spec=CB.MODELS[name][0],
                                 N=N, omega=om, dt_ms=dt, script=SCRIPT, git_rev=CB.git_rev(), wall_s=time.time() - t0,
                                 when=time.strftime("%Y-%m-%d %H:%M"))
                        r.pop("trace", None) if case == "static" else None
                        PUB.append_row(out, r)
                        keys = ("approach_um", "touching_per_side", "rbar_onset_mm", "creep_deg_s", "rbar_spin_mm",
                                "mu_eff", "creep_mm_s", "us_per_step_median")
                        print(case, variant, arm, N, om, r.get("status"),
                              {q: round(r[q], 4) for q in keys if isinstance(r.get(q), (int, float))}, r.get("error", ""),
                              f"{time.time() - t0:.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", default=["v2", "v1", "drake"], choices=list(ARMS))
    ap.add_argument("--cases", nargs="+", default=["static", "T2", "spin", "T1"], choices=list(NS))
    a = ap.parse_args()
    CTB.T.MESH_DIR = ROOT / "logs/20261016-sim2real_fingertip_readiness/meshes"
    OUT.mkdir(parents=True, exist_ok=True)
    run(a.variants, a.arms, a.cases)


if __name__ == "__main__":
    main()
