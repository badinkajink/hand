#!/usr/bin/env python3
"""Pad builders v1 and v2 against Drake on the CAD hand's 10 mm cylinder cell, per printable variant
(docs/handoff/20261016-sim2real_fingertip_readiness.md, step 4).

scripts/curved_tips_cell_drake.py (CD) with the variant's fingers (scripts/sim2real_fingertip_scenes.py: CAD links, printed
tip mass and magnets, the board on the DIP link) and the variant's modulus E for the pads of both builders and for Drake's
compliant tip; everything else as on 2026-10-15: cell cylinder:10:0.75, friction 0.7, the servo plant, HOM with kp_model
0.72 and the post removed after the squeeze, candidates 0-8, seeds 0-5. Rows: docs/experiments/
20261016-sim2real_fingertip_readiness/cell10/<variant>.jsonl.

    PY=logs/20261001-hom_contact/venv/bin/python
    ~/.claude/bin/resguard.sh run --mem 3G --cpu 200 -- $PY scripts/sim2real_fingertip_cell10.py --variant block4_g30 \\
        --arms v1 v2 drake --seeds 0 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curved_tips_cell_drake as CD  # noqa: E402  (sets SCALE_HAND=cad_v1 before importing the scale study)

D = ROOT / "docs/experiments/20261016-sim2real_fingertip_readiness"
ASSETS = ROOT / "assets/mjcf/experimental/20261016-sim2real_fingertip_readiness"
LOGS = ROOT / "logs/20261016-sim2real_fingertip_readiness"


def main():
    variant = sys.argv[sys.argv.index("--variant") + 1]
    del sys.argv[sys.argv.index("--variant"):sys.argv.index("--variant") + 2]
    spec = json.loads((ASSETS / variant / "README.json").read_text())
    E = float(spec["pad_E_Pa"])

    def scenes_for(arm, tip, mu):
        h, K = CD.BUILDERS["v1" if arm == "v1" else "v2"]
        ns = argparse.Namespace(E=E, pf=None, tr=CD.RB.PAD_TR, tag=None, h=h, K=K)
        CD.R.install(ns, force_tip=tip)
        CD.S.CAD_DIR = ASSETS / variant / "cad_v1"
        CD.S.GEN = LOGS / "gen" / variant
        CD.S.SCENES = LOGS / "cell10" / f"{variant}_{'v1' if arm == 'v1' else 'v2'}_mu{mu:g}"
        CD.S.MU, CD.S.PLANT = mu, CD.C.SERVO
        CD.S.BUILDER = "v1" if arm == "v1" else "v2"
        return CD.R.tag_of(ns)
    CD.scenes_for = scenes_for
    CD.CellDrake = functools.partial(CD.CellDrake, E=E)
    CD.C.LOGS = LOGS
    orig = CD.append

    def append(path, row):
        row.update(variant=variant, E_Pa=E, wrapper="scripts/sim2real_fingertip_cell10.py")
        if row.get("arm") == "drake":
            row["tip_tag"] = row["tip_tag"].replace("E1e+07", f"E{E:.4g}")
        orig(path, row)
    CD.append = append
    sys.argv += ["--tip", spec["key"], "--out", str(D / "cell10" / f"{variant}.jsonl")]
    CD.main()


if __name__ == "__main__":
    main()
