#!/usr/bin/env python3
"""Whether ID 2's data stand in for a recalibration of v1 ID 1 (owner, 2026-10-26: "you think it's worth putting the v1
id 1 fingertip back on after all this added calibration?"). ID 1 ran cb1t and cb1r on mount v3 with 19 twins and a few
slides; ID 2 ran cb2t on v7 with the twins, 87 slides and 37 more press targets.

  coverage  location error (mm, along the surface) at ID 1's twins (cb1r) of a GP trained on ID 1's other presses
            (cb1t and cb1r), with ID 2's twins at the same points of the pad added, with all of ID 2's presses added,
            and on ID 2's presses alone.
  shear     shear error (N) and the angle between the estimated and the measured force (deg) on ID 1's slides (cb1r),
            of a GP trained on ID 1's presses, with ID 2's slides added, with ID 2's presses and slides added, and on
            ID 2's presses and slides alone.

    .venv/bin/python scripts/ur5_tip_calibration_pooled_checks.py [--out data/20261025-114653-cb1r/pooled_checks.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import ur5_tip_calibration_cb1r_checks as K  # noqa: E402

D = K.EXP / "data"


def ids(H):
    return H["id"].astype(str)


def run(out):
    A, B, C = (K.load(D / n) for n in ("20261023-203938-cb1t", "20261025-114653-cb1r", "20261025-152920-cb2t"))
    tw1 = np.char.startswith(ids(B), "twin")
    T1 = K.sub(B, tw1)
    own = K.cat(K.sub(A, A["kind"] == "press"), K.sub(B, (B["kind"] == "press") & ~tw1))
    own_all = K.cat(own, T1)
    C_p = K.sub(C, C["kind"] == "press")
    C_tw = K.sub(C, np.char.startswith(ids(C), "twin"))
    C_sl = K.sub(C, C["kind"] == "slide")

    def loc(train):
        return K.by_side(T1, K.loc_err(T1, K.gp(train).predict(T1["d_board"])[:, :3]))

    S1 = K.sub(B, B["kind"] == "slide")
    n, ft = S1["n"], K.shear(S1["f"], S1["n"])

    def sh(train):
        F = K.gp(train).predict(S1["d_board"])[:, 3:]
        return dict(shear_err_N=K.med(np.linalg.norm(K.shear(F, n) - ft, axis=1)), angle_deg=K.med(K.angle(F, S1["f"])))

    res = dict(
        coverage=dict(n_twins=int(len(np.unique(ids(T1)))), n_holds=int(len(T1["id"])), n_id2_twins=int(len(np.unique(ids(C_tw)))),
                      id1=loc(own), id1_and_id2_twins=loc(K.cat(own, C_tw)), id1_and_id2_presses=loc(K.cat(own, C_p)),
                      id2_presses=loc(C_p)),
        shear=dict(n_slides=int(len(np.unique(ids(S1)))), n_holds=int(len(S1["id"])),
                   hexe_shear_N=K.med(np.linalg.norm(ft, axis=1)), physical=sh_phys(S1, n, ft),
                   id1_presses=sh(own_all), id1_presses_and_id2_slides=sh(K.cat(own_all, C_sl)),
                   id1_presses_and_id2_presses_slides=sh(K.cat(own_all, C_p, C_sl)), id2_presses_slides=sh(K.cat(C_p, C_sl))))
    Path(out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=lambda x: round(float(x), 3)))


def sh_phys(S, n, ft):
    F = S["est_f"]
    return dict(shear_err_N=K.med(np.linalg.norm(K.shear(F, n) - ft, axis=1)), angle_deg=K.med(K.angle(F, S["f"])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(D / "20261025-114653-cb1r/pooled_checks.json"))
    run(ap.parse_args().out)
