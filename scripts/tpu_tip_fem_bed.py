#!/usr/bin/env python3
"""Bed tasks T1 (pull) and T2 (twist) for the 1 mm sphere pads on the printed tip (tpu2.7) at the moduli fitted to the
finite-element tip (step 4 of the TPU tip job). Registers models mj_tpu2p7_pads1_E<MPa> with the pads' stiffness
K_s = E / 8.5 mm * A_s and runs tpu_tip_rig.bed on them; rows go to <out-dir>/tip_T1.jsonl and tip_T2.jsonl.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/tpu_tip_fem_bed.py --E 7.5 10 14 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tpu_tip_rig as TR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--E", type=float, nargs="+", required=True, help="pad modulus, MPa")
    ap.add_argument("--N", type=float, nargs="+", default=[0.5, 1.0, 3.0])
    ap.add_argument("--dt", type=float, nargs="+", default=[1.0])
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2"])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs/experiments/20261009-tpu_tip_fem/bed")
    a = ap.parse_args()
    orig = TR.tpu_models
    extra = {}
    for E in a.E:
        tag = f"{E:g}".replace(".", "p")
        e = f"E{E * 1e6:g}"
        extra[f"mj_tpu2p7_pads1_E{tag}"] = (f"mj:spheres:s1:rs0.75:ir100:tr0.02:{e}:tpu2.7",
                                            f"mj:spheres:s1:rs0.75:tr0.02:{e}:tpu2.7")

    def models(radii=(6.0, 2.7, 0.0)):
        out = orig(radii)
        out.update(extra)
        return out
    TR.tpu_models = models
    # hom_contact_rig.MjRig.pad_x (used only to report the pad position in exp_torsion) reads self.x0, which
    # TpuMjRig.__init__ does not set; give the block rig the sphere rig's first-touch value
    if not hasattr(TR.TpuMjRig, "x0"):
        TR.TpuMjRig.x0 = TR.H.X0
    a.out_dir.mkdir(parents=True, exist_ok=True)
    TR.bed(list(extra), a.tasks, a.N, a.dt, a.out_dir)


if __name__ == "__main__":
    main()
