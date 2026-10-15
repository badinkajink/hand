#!/usr/bin/env python3
"""Summaries of the curved-tip study's rows (docs/handoff/20261015-curved_tips.md); each subcommand reads rows under
docs/experiments/20261015-curved_tips/ and writes one JSON summary next to them.

  bed      step 2: per tip the law (Drake's field on the 0.25 mm mesh), the builder-v2 pads and Drake on the bed:
           approach at 0.5-3 N, twist-onset arm, spin arm at 0.1-3 rad/s, brake force at 45 deg, rotational and
           translational creep, touching pads, constraint rows and wall time per step (bed/summary.json)

    python3 scripts/curved_tips_analysis.py bed
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/experiments/20261015-curved_tips"


def rows(path):
    out = []
    for f in sorted(glob.glob(str(path))):
        for line in open(f):
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _pick(rs, **kw):
    """The last complete row matching every key=value."""
    out = [r for r in rs if r.get("status", "complete") == "complete" and all(r.get(k) == v for k, v in kw.items())]
    return out[-1] if out else None


def _ratio(a, b):
    return float(a / b) if (a is not None and b not in (None, 0)) else None


def bed():
    B = D / "bed"
    law = rows(B / "law*.jsonl")
    case = {c: rows(B / f"{c}.jsonl") for c in ("static", "T2", "spin", "T5", "T1")}
    tips = sorted({r["tip"] for r in law})
    out = {"tips": {}, "terms": {
        "law": "Drake's pressure field E (1 - g) on the tip's 0.25 mm mesh, integrated quasi-statically (curved_tips_law.py)",
        "pads": "MuJoCo 1 mm pads, builder v2, relaxation 0.02 s, impratio 100, 1 ms (curved_tips_bed.py)",
        "drake": "Drake 1.57 compliant convex of the 0.25 mm mesh, relaxation 0.01 s, SAP, 1 ms"}}
    for tip in tips:
        L = {(r["field"], r.get("res_m")): r for r in law if r["tip"] == tip}
        lw = L.get(("drake", 0.00025))
        t = {"law": None, "pads": {}, "drake": {}}
        if lw:
            t["law"] = {"approach_um": lw["approach_um"], "onset_arm_mm": lw["onset_arm_mm"],
                        "spin_arm_mm": lw["spin_arm_mm"], "N45_N": lw["brake_N"]["N45"],
                        "approach_1N_um_res0.5": (L.get(("drake", 0.0005)) or {}).get("approach_um", {}).get("1"),
                        "local_approach_1N_um": (L.get(("local", None)) or {}).get("approach_um", {}).get("1"),
                        "pads_v2_qs": {k: (L.get(("pads_v2", None)) or {}).get(k) for k in ("approach_um", "onset_arm_mm",
                                                                                         "spin_arm_mm")},
                        "pads_v1_qs_onset_1N_mm": (L.get(("pads_v1", None)) or {}).get("onset_arm_mm", {}).get("1"),
                        "n_pads": (L.get(("pads_v2", None)) or {}).get("n_pads")}
        for kind in ("pads", "drake"):
            name = f"{kind}_{tip}"
            k = t[kind]
            k["approach_um"] = {f"{N:g}": (_pick(case["static"], model=name, N=N) or {}).get("approach_um")
                                for N in (0.5, 1.0, 2.0, 3.0)}
            s1 = _pick(case["static"], model=name, N=1.0) or {}
            k["touching_1N"] = s1.get("touching_per_side")
            k["nefc_1N"] = s1.get("nefc_mean")
            k["ncon_1N"] = s1.get("ncon_mean")
            k["us_per_step_static_1N"] = s1.get("us_per_step_median")
            k["n_spheres"] = (s1.get("info") or {}).get("n_spheres")
            for N in (1.0, 3.0):
                r = _pick(case["T2"], model=name, N=N) or {}
                k[f"onset_arm_{N:g}N_mm"] = r.get("rbar_onset_mm")
                k[f"kin_arm_{N:g}N_mm"] = r.get("rbar_kin_mm")
                k[f"twist_creep_{N:g}N_deg_s"] = r.get("creep_deg_s")
            k["spin_arm_mm"] = {f"{w:g}": (_pick(case["spin"], model=name, N=1.0, omega=w) or {}).get("rbar_spin_mm")
                                for w in (0.1, 0.3, 1.0, 3.0)}
            k["us_per_step_spin_1"] = (_pick(case["spin"], model=name, N=1.0, omega=1.0) or {}).get("us_per_step_median")
            sp = _pick(case["spin"], model=name, N=1.0, omega=1.0) or {}
            k["spin_slip_share_1"] = (sp.get("share") or {}).get("slip")
            for dt in (1.0, 0.5):
                k[f"N45_{dt:g}ms_N"] = (_pick(case["T5"], model=name, dt_ms=dt) or {}).get("N_at_45_N")
            r = _pick(case["T1"], model=name, N=1.0) or {}
            k["pull_mu_eff"] = r.get("mu_eff")
            k["pull_creep_mm_s"] = r.get("creep_mm_s")
        P, Dk, lw_ = t["pads"], t["drake"], t["law"] or {}
        t["ratios"] = {
            "pads_approach_1N_over_law": _ratio(P["approach_um"]["1"], (lw_.get("approach_um") or {}).get("1")),
            "drake_approach_1N_over_law": _ratio(Dk["approach_um"]["1"], (lw_.get("approach_um") or {}).get("1")),
            "pads_onset_1N_over_law": _ratio(P["onset_arm_1N_mm"], (lw_.get("onset_arm_mm") or {}).get("1")),
            "drake_onset_1N_over_law": _ratio(Dk["onset_arm_1N_mm"], (lw_.get("onset_arm_mm") or {}).get("1")),
            "pads_spin_1_over_law": _ratio(P["spin_arm_mm"]["1"], (lw_.get("spin_arm_mm") or {}).get("1")),
            "drake_spin_1_over_law": _ratio(Dk["spin_arm_mm"]["1"], (lw_.get("spin_arm_mm") or {}).get("1")),
            "pads_spin_3_over_0.3": _ratio(P["spin_arm_mm"]["3"], P["spin_arm_mm"]["0.3"]),
            "drake_spin_3_over_0.3": _ratio(Dk["spin_arm_mm"]["3"], Dk["spin_arm_mm"]["0.3"]),
            "pads_N45_over_drake_0.5ms": _ratio(P["N45_1ms_N"], Dk["N45_0.5ms_N"] or Dk["N45_1ms_N"]),
            "pads_cost_over_drake": _ratio(P["us_per_step_static_1N"], Dk["us_per_step_static_1N"])}
        out["tips"][tip] = t
    (D / "bed" / "summary.json").write_text(json.dumps(out, indent=1))
    hdr = ("tip", "law appr", "pads/law", "drk/law", "onset law", "pads/law", "drk/law", "spin law", "pads/law",
           "drk/law", "pads 3/0.3", "N45 pads/drk", "touch", "nefc", "us pads", "us drk")
    print(" ".join(f"{h:>10s}" for h in hdr))
    f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"  # noqa: E731
    for tip, t in out["tips"].items():
        lw_, R, P, Dk = t["law"] or {}, t["ratios"], t["pads"], t["drake"]
        print(f"{tip[:28]:>28s} " + " ".join(f"{v:>10s}" for v in (
            f((lw_.get("approach_um") or {}).get("1"), 1), f(R["pads_approach_1N_over_law"]), f(R["drake_approach_1N_over_law"]),
            f((lw_.get("onset_arm_mm") or {}).get("1")), f(R["pads_onset_1N_over_law"]), f(R["drake_onset_1N_over_law"]),
            f((lw_.get("spin_arm_mm") or {}).get("1")), f(R["pads_spin_1_over_law"]), f(R["drake_spin_1_over_law"]),
            f(R["pads_spin_3_over_0.3"]), f(R["pads_N45_over_drake_0.5ms"]), f(P["touching_1N"], 0), f(P["nefc_1N"], 0),
            f(P["us_per_step_static_1N"], 0), f(Dk["us_per_step_static_1N"], 0))))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["bed"])
    a = ap.parse_args()
    {"bed": bed}[a.what]()


if __name__ == "__main__":
    main()
