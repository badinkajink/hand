#!/usr/bin/env python3
"""Members, labels, paths and settings of the fingertip-shape rerun (docs/handoff/20261103-fingertip_shape_rerun.md): the
screens of 2026-10-14 (blocks) and 2026-10-15 (capsules, half- and quarter-ellipsoids) and the FE checks of 2026-10-18/19
repeated on the current hand and models, with a hemisphere added.

What changed since 10-15, and is set here or in the wrappers that import this module:
  hand     the CAD cells run on the v4b hand (SCALE_HAND=cad_v2: DIP link b v4, cassette 3T, cap TW15H3 trimmed at
           x_F -0.10), whose tip body lies 36.36 mm from the PIP axis; the D1-D8 hands stay real_v1
  pads     2.5 MPa (reorient_backends.E_TPU) with the relaxation time pad_tr(E) = 0.04 s, builder v2; Drake at 2.5 MPa
  plant    the SCS0009 (kp0.72_kv0_fr0.18_fl0_dp0.0144), as on 10-15
  tip      the reference is the v4b tip, the 11/22 half-ellipsoid's design surface (cad_v2's key); it is also the 11/22
           candidate of 10-15, which ran as printed upright (_u0.1, within 4.5 um of the design over the contact region)

The other curved tips keep their 10-15 keys (printed upright in 0.1 mm layers), the blocks theirs (printed face up in
0.2 mm layers). The hemisphere tcd_ell7.4x7.4x7.4 has semi-axes of 7.4 mm, the envelope's half-width, in every direction:
a dome on a round post that runs straight back to the back face. It reaches neither the mount face nor the end face, so
it has no upright print and is simulated on its design surface. tcd_sph10.55, the whole real_v1 sphere tip, runs on the
bed only, as on 10-14 and 10-15.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import _paths  # noqa: E402,F401  every folder of scripts/ and src/ on sys.path
import curved_tips_common as CC  # noqa: E402

DATE = "20261104"
D = ROOT / f"docs/experiments/{DATE}-fingertip_shape_rerun"
LOGS = ROOT / f"logs/{DATE}-fingertip_shape_rerun"
PY = "logs/20261001-hom_contact/venv/bin/python"     # MuJoCo 3.6.0 and pydrake
VENV = ".venv/bin/python"                            # MuJoCo and trimesh
FEM = str(Path.home() / "miniconda3/envs/femtip/bin/python")   # trimesh, scipy sparse solvers, no MuJoCo
SERVO = "kp0.72_kv0_fr0.18_fl0_dp0.0144"
E_PA, TR_S = 2.5e6, 0.04                             # reorient_backends.E_TPU and pad_tr(E_TPU)
DRAKE_RT = 0.01                                      # Drake's relaxation time, as reorient_backends.DrakeBench

REF = "tcd_ell7.4x8.5x11x22"                         # the v4b pad's dome (hand_object_scale_sim.TIP on cad_v2)
BLOCKS = ["tcd_rs3_re3_p0.2", "tcd_rs3_re6_p0.2"]   # the printed as-built block and the 10-14 finalist (3, 6)
CAPSULES = [k + CC.PRINT for k in CC.CAPSULES]
HALF = ["tcd_ell7.4x5x11x22_u0.1", "tcd_ell7.4x8.5x16x32_u0.1"]
QUARTERS = [k + CC.PRINT for k in CC.QUARTERS]
HEMI = "tcd_ell7.4x7.4x7.4"
SPHERE = CC.SPHERE
TIPS = [REF] + BLOCKS + CAPSULES + HALF + QUARTERS + [HEMI]          # the task screen's twelve
BED = TIPS + [SPHERE]
OLD_KEY = {REF: REF + CC.PRINT}                      # the key a tip ran under on 10-15, where it differs

# the five conditions of 10-14 and 10-15: friction 0.7, 1.0 and 0.5 on the SCS0009, and at friction 0.7 the servo's gain
# at 0.5 N m/rad and its torque limit at 0.12 N m
CONDS = [("0.7", SERVO), ("1.0", SERVO), ("0.5", SERVO), ("0.7", "kp0.5_kv0_fr0.18_fl0_dp0.0144"),
         ("0.7", "kp0.72_kv0_fr0.12_fl0_dp0.0144")]
CELLS = ["cylinder:25:0.125", "cylinder:16:0.625", "cylinder:10:0.75", "cylinder:16:-0.375"]


def label(key: str) -> str:
    """Plain name of a tip key for tables, figures and film labels."""
    if key == REF:
        return "half-ellipsoid 7.4 x 8.5 x 11/22 mm (v4b)"
    if key == HEMI:
        return "hemisphere r 7.4 mm"
    return re.sub(r", printed$", "", CC.label(key))


def short(key: str) -> str:
    """A short name for figure ticks."""
    if key == REF:
        return "11/22 (v4b)"
    if key == HEMI:
        return "hemisphere 7.4"
    if key == SPHERE:
        return "sphere 10.55"
    m = re.match(r"tcd_cap([0-9.]+)", key)
    if m:
        return f"capsule {m[1]}"
    m = re.match(r"tcd_(qell|ell)([0-9.]+)x([0-9.]+)x([0-9.]+)x([0-9.]+)", key)
    if m:
        return f"{'quarter' if m[1] == 'qell' else 'half'} {m[3]} {m[4]}/{m[5]}"
    if key == BLOCKS[0]:
        return "block (3, 3)"
    if key == BLOCKS[1]:
        return "block (3, 6)"
    return key


def family(key: str) -> str:
    if key.startswith("tcd_rs"):
        return "block"
    if key.startswith("tcd_cap"):
        return "capsule"
    if key.startswith("tcd_qell"):
        return "quarter-ellipsoid"
    if key == HEMI:
        return "hemisphere"
    if key.startswith("tcd_sph"):
        return "sphere"
    return "half-ellipsoid"


def build_axis(key: str) -> str:
    """The print's layer normal in tpu_tip_design's frame for the FE's walls and skins: x for the blocks (palmar face up)
    and the hemisphere (on its back face), z for the upright prints."""
    return "x" if key.startswith("tcd_rs") or key == HEMI else "z"
