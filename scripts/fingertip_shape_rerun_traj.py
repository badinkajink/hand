#!/usr/bin/env python3
"""The D1-D8 bench-plan replays of the fingertip-shape rerun (ledger step 5, the Drake agreement gate): scripts/
curved_tips_traj.py's arms on the SCS0009 plant at friction 1.0, with the pads (builder v2) and Drake's compliant convex
tip both at 2.5 MPa (reorient_backends.E_TPU; the pads' relaxation time 0.04 s, Drake's 0.01 s), scenes under
logs/<run>/scenes/, rows in docs/experiments/<run>/traj/traces.jsonl (one fsynced row per arm, tip, hand and seed).

The gate (10-14 and 10-15): a replay agrees when the pads at the planner's numerics end within 3 deg of Drake's turn with
Drake's held outcome; the rerun takes the v4b tip in place of the as-built block as the reference of the gate.

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/fingertip_shape_rerun_traj.py --tips tcd_cap6_u0.1 --arms plan rl drake --hands D1 D2 --seeds 0 1 2
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_shape_rerun_common as C  # noqa: E402
import curved_tips_traj as CT  # noqa: E402

CT.PUB.SCN = C.LOGS / "scenes"
CT.OUT = C.D / "traj" / "traces.jsonl"
CT.SCRIPT = "scripts/fingertip_shape_rerun_traj.py (curved_tips_traj)"
_append = CT.PUB.append_row


def append(path, row):
    row.update(E_Pa=C.E_PA, pad_tr_s=C.TR_S, drake_rt_s=C.DRAKE_RT, study=f"{C.DATE}-fingertip_shape_rerun")
    _append(path, row)


CT.PUB.append_row = append

if __name__ == "__main__":
    if "--out" not in sys.argv:
        sys.argv += ["--out", str(CT.OUT)]
    CT.main()
