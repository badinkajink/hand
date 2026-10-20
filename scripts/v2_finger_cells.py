#!/usr/bin/env python3
"""The readiness study's control tasks (scripts/sim2real_fingertip_cells.py: HOM on every candidate grasp of the four cells,
pinned iCEM on each cell's best HOM grasp) on the v2 finger, 9.75 mm longer (scripts/v2_finger_scenes.py), so that the
2026-10-16 rows of the same variants on the CAD finger are its comparison: does the v2 DIP link change the first bench task?

The scale study's kinematics take the pad centre 36.36 mm past the PIP axis (hand_object_scale_kin.P, CHAIN and the distal
capsule CAP_DIST lengthened by the neck); the cells' finger layouts, the servo plant, the pads and every planner setting
are the readiness study's. Rows: docs/experiments/20261019-v2_finger/cells/<tag>.jsonl (the readiness columns); caches
under logs/20261019-v2_finger/.

    SCALE_HAND=cad_v1 ~/.claude/bin/resguard.sh run --mem 3G --cpu 700 -- env NUMPY_MADVISE_HUGEPAGE=0 .venv/bin/python \\
        scripts/v2_finger_cells.py hom --variants ell11_g30_v2a block5_g30_v2a
    ... scripts/v2_finger_cells.py icem --variants ell11_g30_v2a block5_g30_v2a --cells cylinder:16:0.625 cylinder:25:0.125
    V2_FINGER_BASE=1 ... scripts/v2_finger_cells.py hom --variants ell11_g30 block5_g30     # the CAD finger, same code
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["SCALE_HAND"] = "cad_v1"
os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hand_object_scale_kin as K  # noqa: E402
import sim2real_fingertip_cells as SC  # noqa: E402
import v2_finger_scenes as VS  # noqa: E402

BASE = os.environ.get("V2_FINGER_BASE") == "1"      # 1: the readiness study's CAD finger, rows to cells_base/ (the comparison)
if not BASE:
    K.P = 0.026610 + VS.NECK
    K.CHAIN = K.J2 + K.P
    K.CAP_DIST = 0.016060 + VS.NECK
    SC.ASSETS = VS.OUT
SC.LOGS = ROOT / ("logs/20261019-v2_finger" + ("_base" if BASE else ""))
SC.OUT = ROOT / "docs/experiments/20261019-v2_finger" / ("cells_base" if BASE else "cells")
SC.SCRIPT = "scripts/v2_finger_cells.py" + (" (V2_FINGER_BASE=1)" if BASE else "")

if __name__ == "__main__":
    SC.main()
