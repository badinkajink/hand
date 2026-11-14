#!/usr/bin/env python3
"""HOM's inputs at the moment its least squares fails ("SVD did not converge"), for the stopped rollouts of
steel_hand_compare.py (2026-11-13): runs that script's `hom` command with HomController.step wrapped, and appends one
record per failure (the measured joints' finiteness, the object's pose, the centres of pressure, each finger's frame) to
OUT. In all 11 records of docs/experiments/20261113-steel_solver/rows/hom_stopped_inputs.jsonl the sphere lay on the
floor, 0.31-0.34 m from its start, with no fingertip contact.

    CUDA_VISIBLE_DEVICES="" logs/20261113-steel/venv/bin/python scripts/contact/steel_hom_stop_inputs.py OUT.jsonl \
        hom --plant steel_pt_pf --seeds 0 1 --pairs 6 --chunk 12 --tag _diag
    .venv/bin/python scripts/contact/steel_hom_stop_inputs.py OUT.jsonl hom --plant mjc_pt --seeds 0 1 --tag _diag
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import steel_hand_compare as S  # noqa: E402
from sr2hand.control import hom as H  # noqa: E402
orig_step = H.HomController.step
orig_gcf = H.Mirror.gcf_witness
out = open(sys.argv[1], "a")
def fin(x):
    return bool(np.all(np.isfinite(np.asarray(x, float))))
def step(self, meas, cmd):
    try:
        return orig_step(self, meas, cmd)
    except Exception as e:
        rec = {"err": f"{type(e).__name__}: {e}",
               "q_finite": {f: fin(v) for f, v in meas.q.items()},
               "obj_pos": np.asarray(meas.obj_pos).tolist(), "obj_quat": np.asarray(meas.obj_quat).tolist(),
               "cop": {f: (np.asarray(c[0]).tolist(), float(c[1])) for f, c in meas.cop.items()}}
        g = {}
        for f in self.mr.fingers:
            try:
                gg = self.gcf(f, meas)
                g[f] = {"src": gg.source, "d": gg.d if hasattr(gg, "d") else None, "R1_finite": fin(gg.R1),
                        "p1": gg.p1.tolist(), "p0": gg.p0.tolist()}
            except Exception as e2:
                g[f] = {"exc": f"{type(e2).__name__}: {e2}"}
        rec["gcf"] = g
        out.write(json.dumps(rec, default=str) + "\n"); out.flush()
        raise
H.HomController.step = step
sys.argv = ["steel_hand_compare.py"] + sys.argv[2:]
sys.exit(S.main())
