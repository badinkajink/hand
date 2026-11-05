#!/usr/bin/env python3
"""Finite elements of the fingertip-shape rerun's tips (docs/handoff/20261103-fingertip_shape_rerun.md, ledger step 6),
through the machinery of the 10-19/20 checks of the 11/22 (scripts/v2_tip_fe_checks.py) with each tip in place of the 11/22:
the model of the 10-18 calibration (two 0.45 mm perimeters and four skins of solid TPU at 12 MPa over the homogenized 30 %
gyroid of Siraya 85A, bonded magnets, frictionless rigid probes), in the tip's print orientation (upright, or face up for
the blocks and the hemisphere).

Bodies:
  pad      the tip cut where the v4b pad is cut (13.45 mm under the apex, x_F 2.90), the cut face fixed on the cassette and
           the mount face held where the tip reaches it, as case b_05R4 of the checks; with pad 05R4's four 3 x 2 mm
           magnets (a 5.0 mm square about the contact point, tops 7.5 mm deep) where they fit the shape (mount/mount.json)
  qell     a quarter-ellipsoid on its own back plane, which is fixed (the board side), without magnets: its back lies in
           front of the cassette, and 05R4's magnets would stand out of it

Per tip:
  probes   the calibration's 18 probes at 0.5, 1, 3 and 6 N (the 2 mm-radius sphere at the apex, across the width at
           y 2.6, 4.1, 5.2 mm and along the finger at z +4, -4, -7 mm; rods of radius 5, 8 and 12.5 mm across the finger at
           z 0 and +-4 mm and of radius 8 and 12.5 mm along it): approach, stiffness, patch and twist arm -> fe/fe.jsonl
  sense    with magnets: unit normal and shear patch loads (Gaussian, sigma 0.6 mm) at y -3, 0, 3 and z_c -4..+4 mm with
           the magnets' rigid motions -> fe/sense_fe.jsonl; then the Cramer-Rao bound of the normal force at the contact
           point from one 141 Hz sample of the board's five chips with the contact position and a uniform background field
           unknown, at GAIN_SEL 7 and at the gain the static field needs (tpu_tip_magnet_sensing) -> fe/sensing.json

    ~/.claude/bin/resguard.sh run --mem 5G --cpu 500 -- taskset -c 12-15 env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \\
        NUMPY_MADVISE_HUGEPAGE=0 ~/miniconda3/envs/femtip/bin/python scripts/fingertip_shape_rerun_fe.py --tips <tip>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import fingertip_shape_rerun_common as C  # noqa: E402
import fe_contact_calibration_fe as FE  # noqa: E402
import tip_codesign_geometry as TG  # noqa: E402
import v2_tip_fe_checks as V  # noqa: E402

V.OUT = C.D / "fe"
V.SCRIPT = "scripts/fingertip_shape_rerun_fe.py (v2_tip_fe_checks)"
_grid = FE.grid


def grid(kind="base"):
    """FE.grid; 'qell:<x_back>' the base grid's y and z zones with x zones that start at the quarter-ellipsoid's back
    plane (insert frame), so that its nodes there exist and are fixed."""
    if not str(kind).startswith("qell:"):
        return _grid(kind)
    xb = float(kind.split(":")[1])
    g = _grid("base")
    g["x"] = ([(xb, 1.0, 0.5), (1.0, 5.4, 0.55), (5.4, 8.5, 0.25)] if xb < 0.9 else
              [(xb, 5.4, 0.4), (5.4, 8.5, 0.25)] if xb < 5.0 else [(xb, 8.5, 0.25)])
    return g


FE.grid = grid


def case_of(tip, mount):
    """(case name, case dict) of a tip: the v4b pad body with 05R4 where its magnets fit, the quarter-ellipsoid on its
    back plane without magnets."""
    sh = TG.shape(tip)
    if getattr(sh, "family", "") == "qell":
        xb = round((sh.xc - (TG.REACH - 0.0085)) * 1e3, 6)
        return f"{tip}|qell", dict(grid=f"qell:{xb}", depth=7.5, mags="none")
    fits = mount["tips"][tip]["magnets_fit"]
    return f"{tip}|pad_{'05R4' if fits else 'none'}", dict(grid="base", depth=7.5, mags="05R4" if fits else "none", body="b")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tips", nargs="+", default=C.TIPS)
    ap.add_argument("--what", nargs="+", default=["probes", "sense"], choices=["probes", "sense", "resolution"])
    ap.add_argument("--max-dof", type=int, default=320000)
    a = ap.parse_args()
    mount = json.loads((C.D / "mount" / "mount.json").read_text())
    V.OUT.mkdir(parents=True, exist_ok=True)
    for tip in a.tips:
        name, case = case_of(tip, mount)
        V.CASES[name] = case
        V.KEY, V.BUILD = tip, C.build_axis(tip)
        if case.get("body") == "b":
            V.SENSE_GAP[name] = 3.32 - 6.45            # the board's sensing plane, x_F 3.32 (the DIP study's level)
        if "probes" in a.what:
            V.run_case(name, a.max_dof)
        if "sense" in a.what and case["mags"] != "none":
            V.run_sense(name)
    if "sense" in a.what or "resolution" in a.what:
        for r in FE.rows(V.OUT / "sense_fe.jsonl"):
            V.SENSE_GAP.setdefault(r["case"], 3.32 - 6.45)
        V.sense()


if __name__ == "__main__":
    main()
