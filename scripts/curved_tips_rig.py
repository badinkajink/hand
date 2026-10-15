#!/usr/bin/env python3
"""tip_codesign_rig (hand-level runs: plan replays, HOM turns on D1-D8, the scale study's CAD-hand cells, films and RL
scenes with a chosen tip) for the curved-tip study, with its scene caches under logs/20261015-curved_tips/rig/ in place
of the 2026-10-14 study's folders. Arguments are tip_codesign_rig's; the study runs every tip with builder v2
(--h local --K each).

    PY=logs/20261001-hom_contact/venv/bin/python
    $PY scripts/curved_tips_rig.py hom --tip tcd_cap6_u0.1 --seeds 0 1 2 --mu 0.7 --h local --K each --set kp_model=0.72 \\
        --out docs/experiments/20261015-curved_tips/screen/hom.jsonl
    SCALE_HAND=cad_v1 .venv/bin/python scripts/curved_tips_rig.py scale --tip tcd_cap6_u0.1 --mu 0.7 --h local --K each ...
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import curved_tips_common as C  # noqa: E402
import tip_codesign_rig as R  # noqa: E402

R.SCN = C.LOGS / "rig" / "scenes"          # cmd_scale and cmd_icem put cell scenes in R.SCN.parent / "scale"
R.SCRIPT = "scripts/curved_tips_rig.py (tip_codesign_rig)"
_extra = R._row_extra
R._row_extra = lambda a: dict(_extra(a), study="20261015-curved_tips")

if __name__ == "__main__":
    R.main()
