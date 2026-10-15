#!/usr/bin/env python3
"""Finite elements of the finalist fingertips (fingertip co-design study, plan step 10): the whole printed tip with its
magnet (scripts/tpu_tip_magnet.py, 358k DOF, CHOLMOD) with tpu_tip_design's geometry hooks (inside_body, surface_x)
replaced by a tip_codesign_geometry shape's insert-frame functions. The apex stays at x 8.5 mm and the contact point at
z 10.86 mm (tpu_tip_design.X_FACE, Z_C), so tpu_tip_magnet's gaps and pocket apply unchanged; the tool line lies across
the tip (psi 0). One model per process (two reached 11 GB); run alone on four cores:

    ~/.claude/bin/resguard.sh run --mem 6G --cpu 400 -- taskset -c 4-7 ~/miniconda3/envs/femtip/bin/python \\
        scripts/tip_codesign_fe.py --tips tcd_rs3_re3_p0.2 --recipes w95_20 --magnet d4.76x1.59@5

Rows: docs/experiments/20261014-tip_codesign/final/fe.jsonl (tpu_tip_magnet's row with the tip key added).
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import tip_codesign_geometry as TG  # noqa: E402
import tpu_tip_design as TD  # noqa: E402
import tpu_tip_magnet as M  # noqa: E402

OUT = ROOT / "docs/experiments/20261014-tip_codesign/final/fe.jsonl"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tips", nargs="+", required=True)
    ap.add_argument("--recipes", nargs="+", default=["w95_20"])
    ap.add_argument("--magnet", default="d4.76x1.59@5", help="magnet@depth_mm, or none")
    ap.add_argument("--forces", nargs="+", type=float, default=[0.5, 1.0, 3.0])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    mag, depth = (None, None) if a.magnet == "none" else (a.magnet.split("@")[0], float(a.magnet.split("@")[1]))
    have = set()
    if a.out.exists():
        have = {(r["tip"], r["recipe"], r["magnet"]) for r in map(json.loads, open(a.out))}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for key in a.tips:
        sh = TG.shape(key)
        TD.inside_body, TD.surface_x = sh.fe_inside, sh.fe_surface_x
        for rec in a.recipes:
            if (key, rec, mag or "none") in have:
                continue
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / "row.jsonl"
                M.run(rec, M.RECIPES[rec], mag, depth, tuple(a.forces), out=tmp)
                row = json.loads(tmp.read_text().splitlines()[-1])
            row.update(tip=key, study="20261014-tip_codesign", script_wrapper="scripts/tip_codesign_fe.py")
            with open(a.out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                os.fsync(fh.fileno())


if __name__ == "__main__":
    main()
