#!/usr/bin/env python3
"""Deterioration sweep of the sphere pads over fingertip geometry (fingertip co-design study, plan steps 3 and 5).

Per member (a tip key of tip_codesign_law: tpu<r>, a tip_codesign_geometry tcd key, or 'sphere' for the 10.55 mm real_v1
sphere tip) and variant:

  base     1 mm pads as built today (relaxation 0.02 s, impratio 100)
  fix      the same pads with explicit pad-tool pairs, solreffriction (0.1 s, 1), impratio 1000 (tip_codesign_bed.py)
  drake    Drake's compliant convex (E 10 MPa, relaxation 0.01 s, SAP)

it runs the bed cases of tip_codesign_bed.py: static at 0.5, 1 and 3 N; T2 at 1 and 3 N with the ramp scaled to the
Drake-field law's onset arm (deterioration/law.jsonl); the driven spin at 1 N and omega 0.03, 0.3, 1, 3 and 5 rad/s
(Drake: 0.3, 1 and 3); T5 at 1 ms (Drake: 1 and 0.5 ms, and 0.25 ms with --fine-drake). Rows:
deterioration/{static,T2,spin,T5}.jsonl, one fsynced line per case.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/tip_codesign_sweep.py --members tpu2.7 tpu6 --variants base fix drake
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tip_codesign_bed as B  # noqa: E402

D = ROOT / "docs/experiments/20261014-tip_codesign/deterioration"
FIX = "pf0.1:ir1000"
SPHERE = {"base": ("mj_pads1", ("mj:spheres:s1:rs0.75:ir100:tr0.02", "mj:spheres:s1:rs0.75:tr0.02")),
          "fix": ("mj_pads1_fix", (f"mj:spheres:s1:rs0.75:ir1000:tr0.02:pf0.1", "mj:spheres:s1:rs0.75:tr0.02:pf0.1")),
          "drake": ("drake_hydro", ("drake:hydro:E1e7:r1:rt0.01", "drake:hydro:rt0.01"))}
SPHERE_C_LAW = 0.996e-3          # m N^-1/4: the sphere's elastic-foundation onset arm (contact_bed_twist.C_LAW)


def law_arms(member, Ns=(1.0, 3.0)):
    """The Drake-field law's onset arm at psi 0 (m) per N, from deterioration/law.jsonl."""
    if member == "sphere":
        return {N: SPHERE_C_LAW * N ** 0.25 for N in Ns}
    for r in map(json.loads, open(D / "law.jsonl")):
        if r["tip"] == member and r["field"] == "drake":
            return {N: r["table"][f"{N:g}|0"]["arm_mm"] * 1e-3 for N in Ns}
    raise KeyError(f"no law row for {member}")


def model_of(member, variant):
    if member == "sphere":
        name, v = SPHERE[variant]
    elif member.startswith("tcd_"):
        name, v = B.tcd_model(member, FIX if variant == "fix" else "", drake=variant == "drake")
    else:
        r_mm = float(member[3:])
        if variant == "drake":
            name, v = B.tpu_model(r_mm, drake=True)
        else:
            name, v = B.tpu_model(r_mm, FIX if variant == "fix" else "")
    B.register({name: v})
    return name


def run_member(member, variant, fine_drake=False):
    name = model_of(member, variant)
    laws = {(name, N): a for N, a in law_arms(member).items()}
    dr = variant == "drake"
    B.run([name], ["static"], [0.5, 1.0, 3.0] if not dr else [1.0], [None], [1.0], D / "static.jsonl")
    B.run([name], ["T2"], [1.0, 3.0] if not dr else [1.0], [None], [1.0], D / "T2.jsonl", laws=laws)
    B.run([name], ["spin"], [1.0], [0.3, 1.0, 3.0] if dr else [0.03, 0.3, 1.0, 3.0, 5.0], [1.0], D / "spin.jsonl")
    B.run([name], ["T5"], [None], [None], [1.0, 0.5] + ([0.25] if fine_drake else []) if dr else [1.0],
          D / "T5.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--variants", nargs="+", default=["base", "fix", "drake"])
    ap.add_argument("--fine-drake", action="store_true", help="also Drake T5 at 0.25 ms")
    ap.add_argument("--law", action="store_true", help="first the law rows of members that lack them")
    a = ap.parse_args()
    if a.law:
        import tip_codesign_law as L
        have = {r["tip"] for r in map(json.loads, open(D / "law.jsonl"))} if (D / "law.jsonl").exists() else set()
        for m in a.members:
            if m not in have and m != "sphere":
                rows = L.law_rows(m)
                with open(D / "law.jsonl", "a") as fh:
                    for r in rows:
                        fh.write(json.dumps(r) + "\n")
                        fh.flush()
                        os.fsync(fh.fileno())
    for m in a.members:
        for v in a.variants:
            run_member(m, v, a.fine_drake)


if __name__ == "__main__":
    main()
