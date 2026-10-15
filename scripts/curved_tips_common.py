#!/usr/bin/env python3
"""Members, labels and paths of the curved-fingertip study (docs/handoff/20261015-curved_tips.md): capsule and
half-ellipsoid tips (tip_codesign_geometry.CurvedTip) against the printed block of the 2026-10-14 co-design study.

Every curved tip is printed upright (mount face on the bed, 0.1 mm layers, key suffix _u0.1); the blocks are printed
palmar face up at 0.2 mm (_p0.2), as on 2026-10-14. tcd_sph10.55 is the whole real_v1 sphere tip (no envelope), the
limit against which the machinery is tested; tcd_ell10.55x10.55x10.55 is the half-ellipsoid family's member at that
limit, cut by the envelope.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "docs/experiments/20261015-curved_tips"
LOGS = ROOT / "logs/20261015-curved_tips"
ASSETS = ROOT / "assets/mjcf/experimental/20261015-curved_tips"
PY = "logs/20261001-hom_contact/venv/bin/python"     # MuJoCo 3.6.0 and pydrake
VENV = ".venv/bin/python"                            # MuJoCo and trimesh
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
DATE = "20261015"

PRINT = "_u0.1"
CAPSULES = ["tcd_cap6", "tcd_cap7.4", "tcd_cap8.5"]
ELLIPSOIDS = ["tcd_ell7.4x8.5x11x22", "tcd_ell7.4x5x11x22", "tcd_ell7.4x8.5x16x32"]
CURVED = [k + PRINT for k in CAPSULES + ELLIPSOIDS]           # the printed candidates
LITERAL = "tcd_ell7.4x8.5x11"                                 # the symmetric half-ellipsoid (no flat mount face)
SPHERE, SPHERE_ENV = "tcd_sph10.55", "tcd_ell10.55x10.55x10.55"
ASBUILT, FINALIST = "tcd_rs3_re3_p0.2", "tcd_rs3_re6_p0.2"   # the printed block and the 2026-10-14 best finalist
BLOCKS = [ASBUILT, FINALIST]
BED_MEMBERS = CURVED + [LITERAL, SPHERE, SPHERE_ENV] + BLOCKS


def label(key: str) -> str:
    """Plain name of a tip key for tables, figures and film labels."""
    if key.startswith("tpu"):
        return f"block r {key[3:]} mm"
    m = re.match(r"tcd_cap([0-9.]+)", key)
    if m:
        return f"capsule r {m[1]} mm" + (", design surface" if PRINT not in key else "")
    m = re.match(r"tcd_ell([0-9.]+)x([0-9.]+)x([0-9.]+)(?:x([0-9.]+))?", key)
    if m:
        s = f"half-ellipsoid {m[1]} x {m[2]} x {m[3]}" + (f"/{m[4]}" if m[4] else "") + " mm"
        return s + (", design surface" if PRINT not in key else "")
    m = re.match(r"tcd_sph([0-9.]+)", key)
    if m:
        return f"sphere R {m[1]} mm"
    if key == ASBUILT:
        return "as-built block, printed"
    m = re.match(r"tcd_rs([0-9.]+)_re([0-9.]+)", key)
    if m:
        return f"block ({m[1]}, {m[2]})" + (", printed" if "_p" in key else ", design surface")
    return key
