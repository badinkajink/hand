#!/usr/bin/env python3
"""Two-pad block bed at the RL trainer's numerics (experiment U2 (g) of the 2026-10-14 pad-usability study).

Measures, on the contact bed with the printed TPU block (2.7 mm fillets) in place of the sphere (tpu_tip_rig): the
effective friction coefficient and pre-slip creep of a pinched tool under an axial pull (T1; creep in mm/s at half the
slip force), the twist-onset torque and arm (T2; arm = onset torque / (mu x pinch force), mm) and the braked swing (T5)
for two pad models the earlier bed rows lack:

  mj_tpu2p7_pads1_ir10   1 mm pads, impratio 10 (the trainer's)       mj:spheres:s1:rs0.75:ir10:tr0.02:tpu2.7
  mj_tpu2p7_pads2        2 mm pads of 1.5 mm radius, impratio 100      mj:spheres:s2:rs1.5:ir100:tr0.02:tpu2.7
  mj_tpu2p7_pads1        1 mm pads, impratio 100 (tpu_tip_rig's), rerun at 1 and 2 ms as the reference

at pinch forces of 1 and 3 N per pad and steps of 1 and 2 ms. The impratio-100 rows at 1 and 5 ms, and Drake's, are
docs/experiments/20261006-fingertip_backends/tip_T1.jsonl, tip_T2.jsonl and tip_T5.jsonl; mj_tpu2p7_pads1 also reruns
at 1 ms to check that those rows reproduce with the current bed code. It registers the models in
contact_bed_common.MODELS and contact_bed_pull.MODELS, as tpu_tip_rig.bed does for its own, and calls tpu_tip_rig.bed.

Since commit 7282267e (2026-10-07) hom_contact_rig's rigs carry `x0`, the pad-centre distance from the pinch axis at
first touch (tool_touch(spec)), which replaced the module constant X0; tpu_tip_rig's subclasses never set it and fail in
pad_x(). This script sets it before their constructors run, which gives the value X0 had for these specs.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/pad_usability_bed.py                       # all cases (resumable by model, N, dt)
    $PY scripts/pad_usability_bed.py --tasks T2 --N 1 --dt 1 --models mj_tpu2p7_pads1_ir10

Rows: docs/experiments/20261014-pad_usability/agree/bed/tip_T1.jsonl, tip_T2.jsonl, tip_T5.jsonl (fsynced per case).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import contact_bed_common as CB  # noqa: E402
import contact_bed_pull as P  # noqa: E402
import hom_contact_rig as H  # noqa: E402
import tpu_tip_rig as T  # noqa: E402


def _with_x0(cls):
    init = cls.__init__

    def __init__(self, sp, d_cg, gravity, kinematic=False):
        self.x0 = H.tool_touch(sp)
        init(self, sp, d_cg, gravity, kinematic)
    cls.__init__ = __init__


_with_x0(T.TpuMjRig)
_with_x0(T.TpuDrakeRig)

OUT_DIR = ROOT / "docs/experiments/20261014-pad_usability/agree/bed"
NEW = {   # model name -> (rig spec, chain spec), as tpu_tip_rig.tpu_models
    "mj_tpu2p7_pads1_ir10": ("mj:spheres:s1:rs0.75:ir10:tr0.02:tpu2.7", "mj:spheres:s1:rs0.75:tr0.02:ir10:tpu2.7"),
    "mj_tpu2p7_pads2": ("mj:spheres:s2:rs1.5:ir100:tr0.02:tpu2.7", "mj:spheres:s2:rs1.5:tr0.02:tpu2.7"),
}


def register():
    for k, v in NEW.items():
        CB.MODELS.setdefault(k, v)
        P.MODELS.setdefault(v[1], v[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(NEW))
    ap.add_argument("--tasks", nargs="+", default=["T1", "T2", "T5"])
    ap.add_argument("--N", nargs="+", type=float, default=[1.0, 3.0])
    ap.add_argument("--dt", nargs="+", type=float, default=[1.0, 2.0])
    ap.add_argument("--ref-dt", nargs="*", type=float, default=[1.0, 2.0],
                    help="steps at which mj_tpu2p7_pads1 (impratio 100) also runs: 1 ms reproduces the 10-06 rows, "
                         "2 ms is the step-matched reference")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    register()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    T.bed(a.models, a.tasks, a.N, a.dt, a.out_dir)
    if a.ref_dt:
        T.bed(["mj_tpu2p7_pads1"], a.tasks, a.N, a.ref_dt, a.out_dir)


if __name__ == "__main__":
    main()
