#!/usr/bin/env python3
"""A morphology run whose frozen scene has the printed TPU fingertip instead of the legacy box tip.

The calibrated RL runs (`make_cal_morphology_run.py`, `make_tipmesh_morphology_run.py`) train on the 10.6 x 21.2 x
15 mm box tips of the deploy scenes: MuJoCo point contact, one to four contacts per tip. This copies a run directory
and rewrites its frozen_scene.xml with each tip replaced by the TPU block of `fingertip_geometry.py` (2.7 mm fillets
by default), either as sphere pads (`--model pads`: spacing `--spacing`, pad stiffness E/h A_s reached through solimp
d0, as in `reorient_backends.replace_tips`) or as one convex mesh (`--model pt`). Everything else in the scene (plant,
tool, floor, solver options) is unchanged, so a trainer pointed at the copy differs from the source only in the tip.
The pads collide with the tool only; the block's convex mesh takes the floor and finger-finger contacts (without that,
a 2 mm pad tip resting on the floor makes up to 738 contacts per world).

  `--region front` keeps the pads on the front half of the block (palmar side of its mid-depth plane).

  uv run --extra rl python scripts/make_pad_morphology_run.py \\
      --src results/phase1/real_v1/20260920-sv1_u0308_b050_cal_tip \\
      --out results/phase1/real_v1/20261006-sv1_u0308_b050_cal_tip_tpu2.7pads1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import fingertip_geometry as G  # noqa: E402
import reorient_backends as RB  # noqa: E402


def front_region(sign):
    """Pads on the front half of the block (palmar side of its mid-depth plane): the palmar face, its fillets, the
    front half of the side and end faces. The tool has not touched the back half in any bench replay."""
    mid = G.REACH - G.DEPTH / 2.0

    def keep(C, N):
        return sign * C[:, 0] >= mid
    return keep


def build(src_xml: Path, out_xml: Path, tip: str, model: str, spacing: float, rs: float, region: str,
          mu: float | None = None, tool_only: bool = True) -> dict:
    root = ET.parse(src_xml).getroot()
    if mu is None:                   # the friction of the tip geoms being replaced
        g = root.find(".//body[@name='thumb_tip']/geom")
        mu = float((g.get("friction") or "1").split()[0]) if g is not None else 1.0
    reg = None
    if region == "front":
        def reg(C, N, sign):     # the block's palmar face normal is +x for the thumb, -x for index/middle
            return front_region(sign)(C, N)
    # the mass trap (scene_mutate): pin the reshaped bodies to the source's compiled inertials before the tips change,
    # so only the contact surface differs from the source run and the pad calibration sees the final masses
    from morphohand.studies.scene_mutate import compiled_inertials
    inertials = compiled_inertials(src_xml)
    for f in RB.FINGERS:
        for bn in (f"{f}_tip", f"{f}_pip_frame"):
            b = root.find(f".//body[@name='{bn}']")
            old_in = b.find("inertial")
            if old_in is not None:
                b.remove(old_in)
            b.insert(0, ET.Element("inertial", inertials[bn]))
    meshes, meta = RB.replace_tips(root, tip, model, mu, out_xml.parent / "assets", tool=RB.OBJ, s=spacing, rs=rs,
                                   region=reg, tool_only=tool_only)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_xml)
    meta.update(tip=tip, model=model, spacing=spacing, pad_radius=rs, region=region, mu=mu, tool_only=tool_only)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tip", default="tpu2.7")
    ap.add_argument("--model", default="pads", choices=["pads", "pt"])
    ap.add_argument("--spacing", type=float, default=RB.PAD_S)
    ap.add_argument("--radius", type=float, default=None, help="pad radius (default 0.75 x spacing)")
    ap.add_argument("--region", default="all", choices=["all", "front"])
    a = ap.parse_args()
    src, out = a.src.resolve(), a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("frozen_scene"):
            shutil.copy(f, out / f.name)
    rs = a.radius if a.radius else 0.75 * a.spacing
    meta = build(src / "frozen_scene.xml", out / "frozen_scene.xml", a.tip, a.model, a.spacing, rs, a.region)
    s = json.load(open(out / "summary.json"))
    s["frozen_scene_xml"] = str(out / "frozen_scene.xml")
    s["fingertip"] = {k: (v if not isinstance(v, dict) else {kk: float(vv) for kk, vv in v.items()})
                      for k, v in meta.items()}
    s["derived_from"] = str(src)
    json.dump(s, open(out / "summary.json", "w"), indent=1)
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(out / "frozen_scene.xml"))
    print(f"{out}: {m.ngeom} geoms ({meta.get('n_pads')}), pad d0 {meta.get('pad_d0')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
